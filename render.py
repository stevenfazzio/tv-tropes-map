"""Render the interactive datamap.

The map's own variant goes to docs/index.html, which GitHub Pages serves at
stevenfazzio.com/tv-tropes-map/, with its data in zip files beside it: inlined,
the page is several megabytes, too heavy for phones and for link-preview
crawlers. Other variants render as single files to data/maps/map_<variant>.html,
for comparing locally.

The hovercards quote the opening of each trope's description. TV Tropes text is
CC BY-NC-SA 3.0, so each card credits it, the title panel links the licence, and
the published map is shared under the same licence.

Colormaps, after DataMapPlot's own region colours:

    Primary medium  the medium holding at least half of a trope's examples
                    (features.py); most tropes are spread, in pale grey, so the
                    colours show where medium-specific tropes are
    Prominence      links from other tropes, as a percentile of all tropes. Not
                    also ranked within regions: link counts barely depend on
                    region (it explains ~8% of the variance in log links), so a
                    regional ranking came out almost the same (Spearman 0.95).
                    The hovercard gives each trope's rank in its region.

    uv run render.py                                    # the map's variant
    uv run social_preview.py                            # its link-card image
    uv run render.py --variant plain --variant instructed
"""

import argparse
import html
import os
import re
import tempfile

import datamapplot
import glasbey
import pandas as pd

from common import DATA, DUMP_SNAPSHOT, ROOT, TROPES_PARQUET, write_bytes_atomic
from embed import INSTRUCTIONS, MAP_VARIANT
from features import FEATURES_PARQUET, FEW, OTHER, SPREAD, UNKNOWN

MAPS = DATA / "maps"
DOCS = ROOT / "docs"
OUTPUT = DOCS / "index.html"
# DataMapPlot names the data files <prefix>_point_data_0.zip and so on, and the page
# fetches them by basename, so they sit beside index.html.
DATA_PREFIX = "tropes"
SOCIAL_PREVIEW = DOCS / "social-preview.png"
PUBLIC_URL = "https://stevenfazzio.com/tv-tropes-map/"
REPO_URL = "https://github.com/stevenfazzio/tv-tropes-map"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/3.0/"
TITLE = "TV Tropes Map"
# No digit-leading token: DataMapPlot writes the family name unquoted into CSS and
# into deck.gl's canvas font string, where one is invalid.
FONT = "IBM Plex Sans"
SNIPPET_CHARS = 240
PROMINENCE_CMAP = "viridis"
# DataMapPlot fixes deck.gl's scroll-zoom speed at 0.01 in datamap.js and offers no
# option, so the page resets it once the map exists. deck.gl 9.1 scales the view by
# 2 / (1 + e^-|delta * speed|) per wheel event, at most one zoom level: a mouse notch
# (delta ~100) zooms 0.55 levels at 0.01 and 0.82 at 0.02, and a trackpad, whose
# events carry small deltas, zooms about twice as fast.
ZOOM_SPEED = 0.02
SET_ZOOM_SPEED = (
    "datamap.deckgl.setProps({controller: "
    f"{{scrollZoom: {{speed: {ZOOM_SPEED}, smooth: true}}}}}});"
)

# Non-answers get greys so the media carry the colour; each grey differs, because
# the legend selects points by matching colours. "Spread across media" is most of
# the map, so it gets the palest.
NEUTRAL_COLORS = {
    SPREAD: "#d9d9d9",
    UNKNOWN: "#ececec",
    FEW: "#bdbdbd",
    OTHER: "#8c8c8c",
}

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
.title-credit { margin-top: 8px; font-size: 11px; line-height: 1.3; color: #6b7280; }
.title-credit a { color: inherit; text-decoration-color: rgba(107, 114, 128, 0.45); }
.title-credit a:hover { color: #1f2328; }
"""

# Same card grammar as the other maps: title, labelled fields in fixed places, then
# the description, then a muted credit line. The media dot uses the Primary medium
# colormap's colour, so a card reads against that legend.
DOT = (
    "display: inline-block; width: 8px; height: 8px; border-radius: 50%; "
    "margin-right: 6px; box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.15); "
    "background: {color};"
)
FIELD_NAME = "color: #6b7280; font-size: 11px; padding-top: 1px;"
HOVER_TEMPLATE = f"""
<div style="max-width: 340px; white-space: normal; font-weight: 400; color: #1f2328;">
  <div style="font-weight: 600; font-size: 15px; line-height: 1.25;">{{title}}</div>
  <div style="display: grid; grid-template-columns: 74px minmax(0, 1fr);
    gap: 3px 10px; margin-top: 6px; font-size: 12px; line-height: 1.35;">
    <div style="{FIELD_NAME}">Also called</div><div>{{aliases}}</div>
    <div style="{FIELD_NAME}">Media</div>
    <div><span style="{DOT.format(color="{medium_dot}")}"></span>{{media}}</div>
    <div style="{FIELD_NAME}">Linked from</div><div>{{linked}}</div>
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


def tropes_count(n: int) -> str:
    return f"{n:,} trope" + ("" if n == 1 else "s")


def ordinal(n: int) -> str:
    suffix = (
        "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    )
    return f"{n:,}{suffix}"


def categorical_palette(n_colors: int, avoid: list[str]) -> list[str]:
    """n_colors glasbey colours, each also kept distinct from the `avoid` colours."""
    # Seeded with the greys already in use, so glasbey doesn't hand out a muted
    # colour that reads as one of them.
    palette = glasbey.extend_palette(
        avoid,
        palette_size=len(avoid) + n_colors,
        colorblind_safe=True,
        # glasbey's default, moderate red-green CVD.
        cvd_severity=50.0,
        # A floor of 35: near-black colours are distinct as swatches but read as
        # the same colour as small dots.
        lightness_bounds=(35, 75),
        chroma_bounds=(20, 90),
        # Always explicit: left unset, glasbey takes the hue range from a seed
        # palette of four or more colours, and four greys span no hues.
        hue_bounds=(0, 360),
    )
    return palette[len(avoid) :]


def color_mapping(values: pd.Series) -> dict[str, str]:
    """Category -> colour, assigned most frequent first (glasbey's earliest colours
    are its most distinct)."""
    order = values.value_counts().index.tolist()
    named = [c for c in order if c not in NEUTRAL_COLORS]
    neutrals = {c: NEUTRAL_COLORS[c] for c in order if c in NEUTRAL_COLORS}
    mapping = dict(zip(named, categorical_palette(len(named), list(neutrals.values()))))
    mapping.update(neutrals)
    # The legend selects points by colour, so no two categories may share one.
    assert len(set(mapping.values())) == len(mapping), (
        f"two categories share a colour: {mapping}"
    )
    return mapping


def smallest_region(df: pd.DataFrame, label_cols: list[str]) -> pd.Series:
    """Each trope's finest named region, keyed by (layer, name); None if unnamed at
    every layer. Names repeat across layers, so the layer is part of the key."""
    region = pd.Series(None, index=df.index, dtype=object)
    for i, col in enumerate(label_cols):  # finest first
        unset = region.isna() & (df[col] != "Unlabelled")
        region[unset] = f"{i}:" + df.loc[unset, col]
    return region


def title_extras(variant: str) -> str:
    """Pills, then a credit line linking TV Tropes, the licence and the code."""
    year, month = DUMP_SNAPSHOT.split("-")
    month_name = pd.Timestamp(year=int(year), month=int(month), day=1).strftime("%B")
    pills = [f"{month_name} {year} snapshot", "Unofficial"]
    if variant != MAP_VARIANT:  # only local comparison pages need to say which
        pills.insert(1, f"{variant} embeddings")
    spans = "".join(f'<span class="title-pill">{html.escape(p)}</span>' for p in pills)
    # The cards credit TV Tropes too, but a hovercard can't hold a link.
    credit = (
        'Trope text from <a href="https://tvtropes.org/">TV Tropes</a> · '
        f'Map and text <a href="{LICENSE_URL}">CC BY-NC-SA 3.0</a> · '
        f'<a href="{REPO_URL}">Source code</a>'
    )
    return f'<div>{spans}</div><div class="title-credit">{credit}</div>'


def open_graph_tags(n_tropes: int) -> str:
    og = {
        "og:title": TITLE,
        "og:description": (
            f"A data map of {n_tropes:,} tropes from the TV Tropes wiki, placed by "
            "what each trope's description means and named at four zoom levels. "
            "Pan, zoom, hover and search."
        ),
        "og:type": "website",
        "og:url": PUBLIC_URL,
        # Declared unconditionally: social_preview.py screenshots the rendered page,
        # so the image is always produced after the HTML that names it.
        "og:image": PUBLIC_URL + SOCIAL_PREVIEW.name,
    }
    tags = [
        f'<meta property="{k}" content="{html.escape(v, quote=True)}">'
        for k, v in og.items()
    ]
    tags.append(
        f'<meta name="description" content="{html.escape(og["og:description"])}">'
    )
    tags.append('<meta name="twitter:card" content="summary_large_image">')
    return "\n".join(tags)


def render(variant: str, tropes: pd.DataFrame) -> None:
    public = variant == MAP_VARIANT
    labels = pd.read_parquet(DATA / f"labels_{variant}.parquet")
    df = tropes.merge(labels, on="page", how="left", validate="1:1")
    assert len(df) == len(tropes), "merge changed the row count"
    assert df["x"].notna().all(), "some tropes have no coordinates"

    label_cols = sorted(
        (c for c in df.columns if c.startswith("label_layer_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),  # numeric: _10 must sort after _2
    )
    label_layers = [df[c].fillna("Unlabelled").to_numpy() for c in label_cols]
    n_finest = pd.Series(label_layers[0]).nunique()
    n_coarsest = pd.Series(label_layers[-1]).nunique()
    assert n_finest >= n_coarsest, "label layers must be finest-first for DataMapPlot"

    medium_colors = color_mapping(df["primary_medium"])
    links = df["inbound_links"]
    region = smallest_region(df, label_cols)
    in_region = region.notna()
    grouped = links[in_region].groupby(region[in_region])
    region_rank = grouped.rank(ascending=False, method="min").astype(int)
    region_size = grouped.transform("size")
    prominence = 100 * links.rank(pct=True)

    linked = links.map(tropes_count)
    linked[in_region] = [
        f"{tropes_count(n)} · {ordinal(r)} of {s:,} in its region"
        for n, r, s in zip(links[in_region], region_rank, region_size)
    ]
    media = df["media_summary"].replace("", "—")
    media[df["primary_medium"] == FEW] += " (few examples)"

    aliases = df["aliases"].map(lambda a: [wiki_words(x) for x in a])
    extra = pd.DataFrame(
        {
            # DataMapPlot puts these into the card as raw HTML, so escape them all.
            "title": df["title"].map(html.escape),
            "aliases": aliases.map(
                lambda a: html.escape(", ".join(a[:3])) if a else "—"
            ),
            "media": media.map(html.escape),
            "medium_dot": df["primary_medium"].map(medium_colors),
            "linked": linked.map(html.escape),
            "snippet": df["description"].map(snippet).map(html.escape),
            "url": df["url"],
            # Alternate names are how many tropes get searched for ("Absent Dad").
            "search_text": df["title"] + " " + aliases.map(" ".join),
        }
    )

    colormap_rawdata = [
        df["primary_medium"].to_numpy(),
        prominence.to_numpy(),
    ]
    colormap_metadata = [
        {
            "field": "medium",
            "description": "Primary medium",
            "kind": "categorical",
            # An explicit mapping, so the hovercard dots match the map exactly.
            "color_mapping": medium_colors,
        },
        {
            "field": "prominence",
            "description": "Prominence (percentile)",
            "kind": "continuous",
            "cmap": PROMINENCE_CMAP,
            # The colourbar spaces its ticks evenly between the range's ends. Left
            # to the data, the bottom is 0.7 (the tie of unlinked tropes) and the
            # ticks land on 25.5, 50.3, ...; pinned, they read 0, 25, 50, 75, 100.
            "vmin": 0,
            "vmax": 100,
        },
    ]

    fig = datamapplot.create_interactive_plot(
        df[["x", "y"]].to_numpy(),
        *label_layers,
        hover_text=df["title"].to_numpy(),
        extra_point_data=extra,
        hover_text_html_template=HOVER_TEMPLATE,
        on_click="window.open(`{url}`)",
        enable_search=True,
        search_field="search_text",
        colormap_rawdata=colormap_rawdata,
        colormap_metadata=colormap_metadata,
        cvd_safer=True,
        title=TITLE,
        # Nothing else says clicking works: deck.gl shows a grab cursor on points.
        # Short enough to stay on one line at 1280 px, where a second line makes
        # the panel cover the top region label.
        sub_title=f"{len(df):,} tropes, placed by what they mean. Click one to open it.",
        title_font_size=24,
        sub_title_font_size=13,
        font_family=FONT,
        tooltip_font_family=FONT,
        tooltip_font_weight=400,
        custom_css=CUSTOM_CSS,
        custom_js=SET_ZOOM_SPEED,
        noise_label="Unlabelled",
        # The public page fetches its data from files beside it; DataMapPlot writes
        # them here, during this call.
        inline_data=not public,
        offline_data_path=str(DOCS / DATA_PREFIX) if public else None,
    )

    output = OUTPUT if public else MAPS / f"map_{variant}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output.parent, suffix=".html")
    os.close(fd)
    try:
        fig.save(tmp)
        text = open(tmp, encoding="utf-8").read()
        if public:
            # After the charset declaration, which browsers only honour in the
            # first 1024 bytes of the document.
            anchor = re.search(r"<meta[^>]*charset[^>]*>", text) or re.search(
                "<head>", text
            )
            assert anchor is not None, "no <head> in the rendered page"
            at = anchor.end()
            text = text[:at] + "\n" + open_graph_tags(len(df)) + text[at:]
        # Pills go inside the title panel, after the subtitle; the panel holds only
        # spans, so its first closing </div> is its own.
        text, n = re.subn(
            r'(<div\s+id="title-container"[^>]*>.*?)(</div>)',
            lambda m: m.group(1) + title_extras(variant) + m.group(2),
            text,
            count=1,
            flags=re.DOTALL,
        )
        assert n == 1, "title panel not found for the pills"
        write_bytes_atomic(text.encode("utf-8"), output)
    finally:
        os.unlink(tmp)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB)")
    if public:
        # A smaller corpus writes fewer chunks; leftovers from a bigger one would
        # sit in docs/ unused, so keep only the files this page fetches.
        used = set(re.findall(rf"{DATA_PREFIX}_\w+?\.zip", text))
        for path in DOCS.glob(f"{DATA_PREFIX}_*.zip"):
            if path.name not in used:
                path.unlink()
        assert used and all((DOCS / f).exists() for f in used), "data files missing"
        data_mb = sum((DOCS / f).stat().st_size for f in used) / 1e6
        print(f"  + {len(used)} data files ({data_mb:.2f} MB): {sorted(used)}")
        print(f"  next: uv run social_preview.py to refresh {SOCIAL_PREVIEW.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(INSTRUCTIONS), action="append")
    args = parser.parse_args()
    tropes = pd.read_parquet(
        TROPES_PARQUET, columns=["page", "title", "description", "aliases", "url"]
    )
    features = pd.read_parquet(FEATURES_PARQUET)
    tropes = tropes.merge(features, on="page", how="left", validate="1:1")
    assert tropes["primary_medium"].notna().all(), (
        "features out of sync with tropes; re-run features.py"
    )
    for variant in args.variant or [MAP_VARIANT]:
        render(variant, tropes)


if __name__ == "__main__":
    main()
