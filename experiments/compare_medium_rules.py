"""Compare two rules for a trope's medium on one map, as two colormaps.

    50% rule   features.py: the medium holding at least half of a trope's examples
    lift rule  a medium whose share of the trope's examples is at least 3x its
               share of all media examples on the wiki, and at least 25% of the
               trope's; when several qualify, the one furthest above its usual share

The 50% rule favours the media with the most examples overall: crossing 50% is far
easier for Video Games (15% of all examples) than for Comic Books (5%). The lift
rule measures each trope against the wiki's own mix of media instead.

Both colormaps give each medium the same colour, so switching between them in the
colormap menu shows only which tropes change. The hovercard shows both verdicts.

Writes data/maps/medium_rules_<variant>.html; the real map is untouched.

    uv run experiments/compare_medium_rules.py
"""

import html
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import datamapplot
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA, TROPES_PARQUET  # noqa: E402
from embed import MAP_VARIANT  # noqa: E402
from features import (  # noqa: E402
    FEATURES_PARQUET,
    FEW,
    MIN_MEDIA_EXAMPLES,
    MIN_TROPES_PER_MEDIUM,
    OTHER,
    SPREAD,
    UNKNOWN,
    media_counts,
)
from render import (  # noqa: E402
    CUSTOM_CSS,
    DOT,
    FIELD_NAME,
    FONT,
    MAPS,
    color_mapping,
    snippet,
    wiki_words,
)

MIN_LIFT = 3.0
MIN_SHARE = 0.25
NON_MEDIA = {SPREAD, FEW, UNKNOWN, OTHER}

HOVER_TEMPLATE = f"""
<div style="max-width: 340px; white-space: normal; font-weight: 400; color: #1f2328;">
  <div style="font-weight: 600; font-size: 15px; line-height: 1.25;">{{title}}</div>
  <div style="display: grid; grid-template-columns: 74px minmax(0, 1fr);
    gap: 3px 10px; margin-top: 6px; font-size: 12px; line-height: 1.35;">
    <div style="{FIELD_NAME}">Media</div><div>{{media}}</div>
    <div style="{FIELD_NAME}">50% rule</div>
    <div><span style="{DOT.format(color="{majority_dot}")}"></span>{{majority_text}}</div>
    <div style="{FIELD_NAME}">Lift rule</div>
    <div><span style="{DOT.format(color="{lift_dot}")}"></span>{{lift_text}}</div>
  </div>
  <div style="font-size: 13px; margin-top: 8px; padding-top: 7px;
    border-top: 1px solid #e5e7eb; line-height: 1.4;">{{snippet}}</div>
</div>
"""


def lift_medium(counts: Counter, baseline: dict[str, float]) -> tuple[str, float]:
    """The lift rule's medium for one trope, and how far above its usual share."""
    total = sum(counts.values())
    if total == 0:
        return UNKNOWN, 0.0
    if total < MIN_MEDIA_EXAMPLES:
        return FEW, 0.0
    lifts = {
        m: (n / total) / baseline[m]
        for m, n in counts.items()
        if n / total >= MIN_SHARE and (n / total) / baseline[m] >= MIN_LIFT
    }
    if not lifts:
        return SPREAD, 0.0
    medium = max(lifts, key=lifts.get)
    return medium, lifts[medium]


def fold_rare(medium: pd.Series) -> pd.Series:
    """Media chosen for too few tropes become OTHER, as in features.py."""
    tally = medium.value_counts()
    rare = set(tally[tally < MIN_TROPES_PER_MEDIUM].index) - NON_MEDIA
    return medium.where(~medium.isin(rare), OTHER)


def restrict(mapping: dict[str, str], values: pd.Series) -> dict[str, str]:
    """The shared mapping, limited to the categories one colormap uses."""
    present = set(values)
    return {k: c for k, c in mapping.items() if k in present}


def main():
    tropes = pd.read_parquet(
        TROPES_PARQUET,
        columns=["page", "title", "description", "aliases", "url"]
        + ["folder_labels", "folder_items"],
    )
    features = pd.read_parquet(
        FEATURES_PARQUET,
        columns=["page", "primary_medium", "primary_medium_share", "media_summary"],
    )
    labels = pd.read_parquet(DATA / f"labels_{MAP_VARIANT}.parquet")
    df = tropes.merge(features, on="page", how="left", validate="1:1")
    df = df.merge(labels, on="page", how="left", validate="1:1")
    assert len(df) == len(tropes), "merge changed the row count"
    assert df["primary_medium"].notna().all(), (
        "features out of sync; re-run features.py"
    )
    assert df["x"].notna().all(), "some tropes have no coordinates"

    counts = [
        media_counts(folder_labels, items)
        for folder_labels, items in zip(df["folder_labels"], df["folder_items"])
    ]
    everything = Counter()
    for c in counts:
        everything.update(c)
    n_examples = sum(everything.values())
    baseline = {m: n / n_examples for m, n in everything.items()}
    raw_lift, lift = zip(*(lift_medium(c, baseline) for c in counts))
    raw_lift = pd.Series(raw_lift, index=df.index)
    df["lift_medium"] = fold_rare(raw_lift)

    majority, by_lift = df["primary_medium"], df["lift_medium"]
    print(
        pd.DataFrame(
            {"50% rule": majority.value_counts(), "lift rule": by_lift.value_counts()}
        )
        .fillna(0)
        .astype(int)
        .sort_values("lift rule", ascending=False)
        .to_string()
    )
    changed = majority != by_lift
    print(f"{changed.sum():,} tropes change category; most common moves:")
    moves = (majority[changed] + " -> " + by_lift[changed]).value_counts().head(12)
    print(moves.to_string())

    shared = color_mapping(pd.concat([majority, by_lift]))
    top = [c.most_common(1)[0][0] if c else None for c in counts]
    majority_text = [
        f"{t} · {s:.0%} of its examples" if m not in NON_MEDIA - {OTHER} else m
        for m, t, s in zip(majority, top, df["primary_medium_share"])
    ]
    lift_text = [
        f"{r} · {x:.1f}× its usual share" if r not in NON_MEDIA else r
        for r, x in zip(raw_lift, lift)
    ]
    media = df["media_summary"].replace("", "—")
    aliases = df["aliases"].map(lambda a: [wiki_words(x) for x in a])
    extra = pd.DataFrame(
        {
            # DataMapPlot puts these into the card as raw HTML, so escape them all.
            "title": df["title"].map(html.escape),
            "media": media.map(html.escape),
            "majority_text": pd.Series(majority_text).map(html.escape),
            "majority_dot": majority.map(shared),
            "lift_text": pd.Series(lift_text).map(html.escape),
            "lift_dot": by_lift.map(shared),
            "snippet": df["description"].map(snippet).map(html.escape),
            "url": df["url"],
            "search_text": df["title"] + " " + aliases.map(" ".join),
        }
    )

    label_cols = sorted(
        (c for c in df.columns if c.startswith("label_layer_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    label_layers = [df[c].fillna("Unlabelled").to_numpy() for c in label_cols]
    n_finest = pd.Series(label_layers[0]).nunique()
    assert n_finest >= pd.Series(label_layers[-1]).nunique(), (
        "layers must be finest-first"
    )

    fig = datamapplot.create_interactive_plot(
        df[["x", "y"]].to_numpy(),
        *label_layers,
        hover_text=df["title"].to_numpy(),
        extra_point_data=extra,
        hover_text_html_template=HOVER_TEMPLATE,
        on_click="window.open(`{url}`)",
        enable_search=True,
        search_field="search_text",
        colormap_rawdata=[majority.to_numpy(), by_lift.to_numpy()],
        colormap_metadata=[
            {
                "field": "rule_majority",
                "description": "Primary medium (50% rule)",
                "kind": "categorical",
                "color_mapping": restrict(shared, majority),
            },
            {
                "field": "rule_lift",
                "description": "Skews toward (lift rule)",
                "kind": "categorical",
                "color_mapping": restrict(shared, by_lift),
                # 22 categories; past 20, DataMapPlot hides the legend unless asked.
                "show_legend": True,
            },
        ],
        cvd_safer=True,
        title="TV Tropes Map: two medium rules",
        sub_title="Switch between the rules in the colormap menu, bottom left.",
        title_font_size=24,
        sub_title_font_size=13,
        font_family=FONT,
        tooltip_font_family=FONT,
        tooltip_font_weight=400,
        custom_css=CUSTOM_CSS,
        noise_label="Unlabelled",
    )
    output = MAPS / f"medium_rules_{MAP_VARIANT}.html"
    fd, tmp = tempfile.mkstemp(dir=MAPS, suffix=".html")
    os.close(fd)
    try:
        fig.save(tmp)
        os.replace(tmp, output)
    except BaseException:
        os.unlink(tmp)
        raise
    print(f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
