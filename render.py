"""Render a variant's interactive datamap to data/maps/map_<variant>.html.

These pages stay local for now. The hovercards show the opening of each trope's
own description, which a public version will need to settle: TV Tropes text is
CC BY-NC-SA, so it can be shown with credit, or replaced by generated summaries.

    uv run render.py                                    # the map's variant
    uv run render.py --variant plain --variant instructed
"""

import argparse
import html
import os
import re
import tempfile

import datamapplot
import pandas as pd

from common import DATA, DUMP_SNAPSHOT, TROPES_PARQUET, write_bytes_atomic
from embed import INSTRUCTIONS, MAP_VARIANT

MAPS = DATA / "maps"
TITLE = "TV Tropes Map"
# No digit-leading token: DataMapPlot writes the family name unquoted into CSS and
# into deck.gl's canvas font string, where one is invalid.
FONT = "IBM Plex Sans"
SNIPPET_CHARS = 240

# DataMapPlot sets line-height 0.95 on its panels, so a wrapped title's descenders
# run into the subtitle. The pills match sep-map's.
CUSTOM_CSS = """
#main-title { line-height: 1.1 !important; letter-spacing: -0.01em; text-wrap: balance; }
#main-title + br + span {
  display: inline-block; margin-top: 4px; line-height: 1.3; text-wrap: pretty;
}
.title-pill {
  display: inline-block; margin: 8px 6px 0 0; padding: 3px 10px;
  font-size: 11px; font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase;
  color: #636c76; background: rgba(99, 108, 118, 0.08);
  border: 1px solid rgba(99, 108, 118, 0.15); border-radius: 12px;
}
"""

# Same card grammar as the other maps: title, labelled fields in fixed places, then
# the description, then a muted credit line.
FIELD_NAME = "color: #6b7280; font-size: 11px; padding-top: 1px;"
HOVER_TEMPLATE = f"""
<div style="max-width: 340px; white-space: normal; font-weight: 400; color: #1f2328;">
  <div style="font-weight: 600; font-size: 15px; line-height: 1.25;">{{title}}</div>
  <div style="display: grid; grid-template-columns: 62px minmax(0, 1fr);
    gap: 3px 10px; margin-top: 6px; font-size: 12px; line-height: 1.35;">
    <div style="{FIELD_NAME}">Also called</div><div>{{aliases}}</div>
    <div style="{FIELD_NAME}">Most in</div><div>{{media}}</div>
  </div>
  <div style="font-size: 13px; margin-top: 8px; padding-top: 7px;
    border-top: 1px solid #e5e7eb; line-height: 1.4;">{{snippet}}</div>
  <div style="color: #6b7280; font-size: 11px; margin-top: 7px;">TV Tropes · CC BY-NC-SA 3.0</div>
</div>
"""


def wiki_words(name: str) -> str:
    """'AbsentDad' -> 'Absent Dad': aliases are stored as wiki page names."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)


def snippet(description: str) -> str:
    text = " ".join((description or "").split())
    if len(text) <= SNIPPET_CHARS:
        return text
    return text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"


def top_media(labels, items, media: set[str], n: int = 3) -> str:
    ranked = sorted(
        ((label, count) for label, count in zip(labels, items) if label in media),
        key=lambda lc: -lc[1],
    )
    return " · ".join(label for label, count in ranked[:n] if count > 0) or "—"


def media_folders(tropes: pd.DataFrame, min_tropes: int = 1000) -> set[str]:
    # Most pages folder their examples by medium, but some fold by franchise or by
    # subtype ("Pretty Cure"). Media folders recur across thousands of pages and
    # the others rarely do, so a frequency floor keeps only media. The spelling
    # variants ("Anime and Manga", "Anime & Manga") still need merging before
    # medium can be a colormap.
    counts = pd.Series(
        [label for labels in tropes["folder_labels"] for label in set(labels)]
    ).value_counts()
    return set(counts[counts >= min_tropes].index)


def title_pills(variant: str) -> str:
    year, month = DUMP_SNAPSHOT.split("-")
    month_name = pd.Timestamp(year=int(year), month=int(month), day=1).strftime("%B")
    pills = [f"{month_name} {year} snapshot", f"{variant} embeddings", "Unofficial"]
    spans = "".join(f'<span class="title-pill">{html.escape(p)}</span>' for p in pills)
    return f"<div>{spans}</div>"


def render(variant: str, tropes: pd.DataFrame) -> None:
    labels = pd.read_parquet(DATA / f"labels_{variant}.parquet")
    df = tropes.merge(labels, on="page", how="left", validate="1:1")
    assert len(df) == len(tropes), "merge changed the row count"
    assert df["x"].notna().all(), "some tropes have no coordinates"

    aliases = df["aliases"].map(lambda a: [wiki_words(x) for x in a])
    media = media_folders(df)
    extra = pd.DataFrame(
        {
            # DataMapPlot puts these into the card as raw HTML, so escape them all.
            "title": df["title"].map(html.escape),
            "aliases": aliases.map(
                lambda a: html.escape(", ".join(a[:3])) if a else "—"
            ),
            "media": [
                html.escape(top_media(names, counts, media))
                for names, counts in zip(df["folder_labels"], df["folder_items"])
            ],
            "snippet": df["description"].map(snippet).map(html.escape),
            "url": df["url"],
            # Alternate names are how many tropes get searched for ("Absent Dad").
            "search_text": df["title"] + " " + aliases.map(" ".join),
        }
    )

    label_cols = sorted(
        (c for c in df.columns if c.startswith("label_layer_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),  # numeric: _10 must sort after _2
    )
    label_layers = [df[c].fillna("Unlabelled").to_numpy() for c in label_cols]
    n_finest = pd.Series(label_layers[0]).nunique()
    n_coarsest = pd.Series(label_layers[-1]).nunique()
    assert n_finest >= n_coarsest, "label layers must be finest-first for DataMapPlot"

    fig = datamapplot.create_interactive_plot(
        df[["x", "y"]].to_numpy(),
        *label_layers,
        hover_text=df["title"].to_numpy(),
        extra_point_data=extra,
        hover_text_html_template=HOVER_TEMPLATE,
        on_click="window.open(`{url}`)",
        enable_search=True,
        search_field="search_text",
        cvd_safer=True,
        title=TITLE,
        # Nothing else says clicking works: deck.gl shows a grab cursor on points.
        sub_title=f"{len(df):,} tropes, placed by the meaning of their descriptions. Click one to open it.",
        title_font_size=24,
        sub_title_font_size=13,
        font_family=FONT,
        tooltip_font_family=FONT,
        tooltip_font_weight=400,
        custom_css=CUSTOM_CSS,
        noise_label="Unlabelled",
    )

    output = MAPS / f"map_{variant}.html"
    MAPS.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=MAPS, suffix=".html")
    os.close(fd)
    try:
        fig.save(tmp)
        text = open(tmp, encoding="utf-8").read()
        # Pills go inside the title panel, after the subtitle; the panel holds only
        # spans, so its first closing </div> is its own.
        text, n = re.subn(
            r'(<div\s+id="title-container"[^>]*>.*?)(</div>)',
            lambda m: m.group(1) + title_pills(variant) + m.group(2),
            text,
            count=1,
            flags=re.DOTALL,
        )
        assert n == 1, "title panel not found for the pills"
        write_bytes_atomic(text.encode("utf-8"), output)
    finally:
        os.unlink(tmp)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(INSTRUCTIONS), action="append")
    args = parser.parse_args()
    tropes = pd.read_parquet(
        TROPES_PARQUET,
        columns=[
            "page",
            "title",
            "description",
            "aliases",
            "folder_labels",
            "folder_items",
            "url",
        ],
    )
    for variant in args.variant or [MAP_VARIANT]:
        render(variant, tropes)


if __name__ == "__main__":
    main()
