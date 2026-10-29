"""Choose the trope pages from the parsed Main/ pages.

On TV Tropes a trope page is one with examples: example folders, example
subpages, or an "Examples" heading. Pages that only define a term ("No examples,
please") are kept too. Some index pages also arrange their entries in folders, so
three more rules take those back out. Each rule prints how many pages it drops
and a few of their titles. The rules are structural only; nothing here judges
quality.

    uv run select_tropes.py
"""

import re

import pandas as pd

from common import MAIN_PAGES_PARQUET, TROPES_PARQUET, write_parquet_safely

# The note that marks a page as defining a term rather than collecting examples.
# It is usually an h2 ("No examples, please. This only defines the term.") but is
# sometimes a sentence in the description ("Examples are not used for this
# trope", "Please don't list examples of this Omnipresent Trope").
DEFINITION_ONLY = re.compile(
    r"\bno examples\b|\bput examples\b|\bexamples are not (?:used|needed|necessary)\b"
    r"|\b(?:do not|don't) (?:list|add|put) (?:any )?examples\b"
    r"|\b(?:only|merely|just) defines? the term\b|\bomnipresent trope\b",
    re.IGNORECASE,
)
INDEX_TITLE = re.compile(r" (?:Index|Tropes)$")
# Lists of works by period: "Films of the 1930s", "Films of 1995–1999".
PERIOD_LIST_TITLE = re.compile(r"\bof (?:the )?(?:\d{4}s|\d{4}–\d{4})\b")
# Index pages link almost only to other Main/ pages; a trope's examples link to
# works. Big tropes whose examples live on subpages can also pass this, which is
# why both index rules that use it also require the page to have no Laconic.
TROPE_LINK_SHARE = 0.85


def main():
    df = pd.read_parquet(MAIN_PAGES_PARQUET)
    title = df["title"].fillna("")
    index_targets = {
        ref.split("/", 1)[1]
        for refs in df["indexes"]
        for ref in refs
        if ref.startswith("Main/")
    }

    has_examples = (
        (df["folder_labels"].map(len) > 0)
        | (df["example_subpages"].map(len) > 0)
        | df["headings"].map(
            lambda hs: any(h.lower().startswith("example") for h in hs)
        )
    )
    definition_only = (
        df["headings"].map(" | ".join) + "\n" + df["description"].fillna("")
    ).str.contains(DEFINITION_ONLY)
    has_laconic = df["subpages"].map(lambda s: "Laconic" in s)
    links = (df["links_to_main"] + df["links_elsewhere"]).clip(lower=1)
    mostly_trope_links = df["links_to_main"] / links >= TROPE_LINK_SHARE

    rules = [
        (
            "no examples, and not a definition-only page",
            ~(has_examples | definition_only),
        ),
        (
            "titled index ('... Index', '... Tropes') with no Laconic or mostly trope links",
            title.str.contains(INDEX_TITLE) & (~has_laconic | mostly_trope_links),
        ),
        (
            "listed as an index by other pages, no Laconic, mostly trope links",
            df["page"].isin(index_targets) & ~has_laconic & mostly_trope_links,
        ),
        ("list of works by period", title.str.contains(PERIOD_LIST_TITLE)),
    ]

    print(f"{len(df)} canonical Main/ pages")
    keep = pd.Series(True, index=df.index)
    for reason, drop in rules:
        dropped = keep & drop
        sample = title[dropped].sample(min(6, dropped.sum()), random_state=0)
        print(f"  {dropped.sum():6d} dropped: {reason}")
        print(f"         e.g. {' | '.join(sample)}")
        if reason.startswith("no examples"):
            print(
                f"         ({(dropped & df['may_refer_to']).sum()} of them say 'may refer to')"
            )
        keep &= ~drop

    tropes = df[keep].reset_index(drop=True)
    print(f"{len(df)} in -> {len(tropes)} tropes ({len(df) - len(tropes)} dropped)")
    write_parquet_safely(tropes, TROPES_PARQUET)
    print(f"wrote {TROPES_PARQUET} ({len(tropes)} rows)")


if __name__ == "__main__":
    main()
