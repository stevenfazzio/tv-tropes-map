"""Derive per-trope fields for the colormaps and hovercards from the parsed pages.

    primary medium  the medium holding at least half of a trope's examples, read
                    from its example folders. Most tropes have none and are
                    "Spread across media", which is the point: the colormap
                    shows where medium-specific tropes are.
    inbound links   how many other tropes link to it (TV Tropes' "wicks"),
                    counted from trope pages only, with links made through an
                    alternate name resolved to the trope they reach.

Big tropes that keep their examples on per-medium subpages have one folder link
per medium rather than their examples, so they mostly read as spread; the fresh
crawl, which fetches those subpages, fixes that.

Writes data/trope_features.parquet, row-aligned with tropes.parquet.

    uv run features.py
"""

from collections import Counter

import pandas as pd

from common import DATA, MAIN_PAGES_PARQUET, TROPES_PARQUET, write_parquet_safely

FEATURES_PARQUET = DATA / "trope_features.parquet"

# Example-folder label -> medium. Pages spell the same folder several ways; these
# cover every media label used by at least 1,000 tropes. The long tail of other
# labels (franchises, subtypes: "Pretty Cure", "Played Straight") is not a medium.
MEDIA = {
    "Anime & Manga": "Anime & Manga",
    "Anime and Manga": "Anime & Manga",
    "Manga": "Anime & Manga",
    "Anime": "Anime & Manga",
    "Comic Books": "Comic Books",
    "Comics": "Comic Books",
    "Comic Strips": "Comic Strips",
    "Newspaper Comics": "Comic Strips",
    "Webcomics": "Webcomics",
    "Web Comics": "Webcomics",
    "Films — Live-Action": "Film (live-action)",
    "Film — Live-Action": "Film (live-action)",
    "Film": "Film (live-action)",
    "Films — Animation": "Film (animated)",
    "Films — Animated": "Film (animated)",
    "Film — Animated": "Film (animated)",
    "Film — Animation": "Film (animated)",
    "Live-Action TV": "Live-Action TV",
    "Live Action TV": "Live-Action TV",
    "Western Animation": "Western Animation",
    "Asian Animation": "Asian Animation",
    "Web Animation": "Web Animation",
    "Web Original": "Web Original",
    "Web Videos": "Web Original",
    "Web Video": "Web Original",
    "Literature": "Literature",
    "Fan Works": "Fan Works",
    "Tabletop Games": "Tabletop Games",
    "Video Games": "Video Games",
    "Visual Novels": "Video Games",
    "Music": "Music",
    "Theatre": "Theatre",
    "Theater": "Theatre",
    "Professional Wrestling": "Pro Wrestling",
    "Pro Wrestling": "Pro Wrestling",
    "Radio": "Radio & Podcasts",
    "Podcasts": "Radio & Podcasts",
    "Myths & Religion": "Myths & Religion",
    "Mythology and Religion": "Myths & Religion",
    "Mythology": "Myths & Religion",
    "Advertising": "Advertising",
    "Toys": "Toys",
    "Puppet Shows": "Puppet Shows",
    "Real Life": "Real Life",
}
PRIMARY_MEDIUM_SHARE = 0.5
# Below this many examples in media folders a "primary medium" is noise (1 of 2).
MIN_MEDIA_EXAMPLES = 5
# Main media this rare go to "Other medium", so the legend stays readable.
MIN_TROPES_PER_MEDIUM = 30
SPREAD, FEW, UNKNOWN, OTHER = (
    "Spread across media",
    "Few examples",
    "Unknown",
    "Other medium",
)


def media_counts(labels, items) -> Counter:
    counts = Counter()
    for label, n in zip(labels, items):
        if label in MEDIA:
            counts[MEDIA[label]] += n
    return counts


def primary_medium(counts: Counter) -> tuple[str, float]:
    total = sum(counts.values())
    if total == 0:
        return UNKNOWN, 0.0
    medium, n = counts.most_common(1)[0]
    if total < MIN_MEDIA_EXAMPLES:
        return FEW, n / total
    return (medium if n / total >= PRIMARY_MEDIUM_SHARE else SPREAD), n / total


def media_summary(counts: Counter, top: int = 3) -> str:
    total = sum(counts.values())
    return (
        " · ".join(f"{m} {n / total:.0%}" for m, n in counts.most_common(top))
        if total
        else ""
    )


def inbound_links(tropes: pd.DataFrame, pages: pd.DataFrame) -> pd.Series:
    """For each trope, how many other tropes link to it."""
    # Links use whichever name the writer typed, often an alternate one, and wiki
    # names aren't always cased the same way, so resolve through a lowercase index.
    to_page = {p.lower(): p for p in pages["page"]}
    for page, aliases in zip(pages["page"], pages["aliases"]):
        to_page.update({a.lower(): page for a in aliases})
    is_trope = set(tropes["page"])
    counts, n_links, n_resolved = Counter(), 0, 0
    for source, links in zip(tropes["page"], tropes["main_links"]):
        targets = set()
        for link in links:
            n_links += 1
            if (target := to_page.get(link.lower())) is not None:
                n_resolved += 1
                if target in is_trope and target != source:
                    targets.add(target)
        counts.update(targets)
    print(
        f"  {n_links:,} links to Main/ pages; {n_resolved / n_links:.1%} resolve to a page"
    )
    return tropes["page"].map(counts).fillna(0).astype(int)


def main():
    tropes = pd.read_parquet(
        TROPES_PARQUET, columns=["page", "folder_labels", "folder_items", "main_links"]
    )
    pages = pd.read_parquet(MAIN_PAGES_PARQUET, columns=["page", "aliases"])

    counts = [
        media_counts(labels, items)
        for labels, items in zip(tropes["folder_labels"], tropes["folder_items"])
    ]
    medium, share = zip(*(primary_medium(c) for c in counts))
    medium = pd.Series(medium)
    tally = medium.value_counts()
    rare = set(tally[tally < MIN_TROPES_PER_MEDIUM].index) - {SPREAD, FEW, UNKNOWN}
    medium = medium.where(~medium.isin(rare), OTHER)

    features = pd.DataFrame(
        {
            "page": tropes["page"],
            "primary_medium": medium,
            "primary_medium_share": share,
            "media_examples": [sum(c.values()) for c in counts],
            "media_summary": [media_summary(c) for c in counts],
            "inbound_links": inbound_links(tropes, pages),
        }
    )
    assert len(features) == len(tropes)
    print("primary medium:", features["primary_medium"].value_counts().to_dict())
    if rare:
        print(f"  folded into {OTHER!r}: {sorted(rare)}")
    q = features["inbound_links"].quantile([0.5, 0.9, 0.99]).astype(int).to_dict()
    top = features.nlargest(5, "inbound_links")
    print(
        f"inbound links: median {q[0.5]}, p90 {q[0.9]}, p99 {q[0.99]}; most linked: {dict(zip(top.page, top.inbound_links))}"
    )
    write_parquet_safely(features, FEATURES_PARQUET)
    print(f"wrote {FEATURES_PARQUET} ({len(features)} rows)")


if __name__ == "__main__":
    main()
