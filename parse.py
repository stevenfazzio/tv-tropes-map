"""Parse the dump's Main/ namespace into one row per canonical page.

Main/ holds far more than tropes: alternate names that redirect to another page,
index pages, disambiguation pages, and "we don't have an article" stubs. This
stage folds redirects into their target (as aliases) and drops URLs that aren't
Main/ pages at all. It does not decide what counts as a trope. It records the
signals that decision needs (headings, example folders, links, indexes), so the
corpus filter is a separate step whose rules can be seen and counted.

    uv run parse.py
    uv run parse.py --limit 2000   # first 2,000 pages, for testing
"""

import argparse
import itertools
import os
import re
from multiprocessing import Pool

import pandas as pd
from lxml import html as lxml_html

from common import MAIN_PAGES_PARQUET, iter_dump_pages, write_parquet_safely

PAGE_URL = re.compile(r"/pmwiki/pmwiki\.php/([^/?#]+)/([^/?#]+)")
CANONICAL = re.compile(
    r'<link rel="canonical" href="https://tvtropes\.org/pmwiki/pmwiki\.php/([^/"]+)/([^/"]+)"'
)
ARTICLE_ID = re.compile(r'<input type="hidden" id="article_id" value="(\d+)"')
MISSING_TEXT = "We don't have an article named"
# Blocks among the description paragraphs that aren't description: image
# captions, the opening quote, and "see also" boxes.
NON_DESCRIPTION_CLASSES = ("acaption", "quoteright", "indent", "floatbox")
INLINE_TAGS = {
    "a",
    "abbr",
    "b",
    "big",
    "code",
    "em",
    "i",
    "s",
    "small",
    "span",
    "strong",
    "sub",
    "sup",
    "u",
}


def _cp1252(byte: int) -> str:
    try:
        return bytes([byte]).decode("cp1252")
    except UnicodeDecodeError:  # the five bytes cp1252 leaves undefined
        return ""


# Some page text reached the dump as Windows-1252 bytes decoded as Latin-1, so its
# dashes and curly quotes arrive as C1 control characters (U+0080-U+009F). Those
# never occur in real text, so mapping them back is safe.
C1_TO_CP1252 = str.maketrans({chr(b): _cp1252(b) for b in range(0x80, 0xA0)})


def clean(text: str) -> str:
    return " ".join(text.translate(C1_TO_CP1252).split())


def page_ref(href: str | None) -> tuple[str, str] | None:
    m = PAGE_URL.search(href or "")
    return (m.group(1), m.group(2)) if m else None


def block_text(el) -> str:
    if el.tag in ("ul", "ol"):
        return "\n".join("- " + clean(li.text_content()) for li in el.xpath("./li"))
    return clean(el.text_content())


def folder_examples(label) -> int:
    folder = label.getnext()
    while folder is not None and not isinstance(folder.tag, str):  # skip comments
        folder = folder.getnext()
    if folder is None or "folder" not in folder.get("class", "").split():
        return 0
    return len(folder.xpath("./ul/li"))


def split_description(article) -> tuple[list[str], list]:
    """Split an article into its description blocks and the elements after them.

    The description ends at the first heading, rule, or example folder that follows
    some description text. One that comes first (a rule under the opening quote, a
    note in an h2) doesn't end it. Some descriptions are bare text and inline links
    with no <p> around them, so loose text is gathered into blocks too.
    """
    blocks, loose = [], [article.text or ""]

    def flush():
        if text := clean("".join(loose)):
            blocks.append(text)
        loose.clear()

    children = list(article)
    for i, el in enumerate(children):
        if not isinstance(el.tag, str):  # a comment; its tail is still text
            loose.append(el.tail or "")
            continue
        cls = el.get("class", "")
        if el.tag in ("h2", "hr") or "folderlabel" in cls:
            flush()
            if blocks:
                return blocks, [e for e in children[i:] if isinstance(e.tag, str)]
        elif el.tag in INLINE_TAGS:
            loose.append(el.text_content())
        else:
            flush()
            if el.tag in ("p", "ul", "ol") and not any(
                c in cls for c in NON_DESCRIPTION_CLASSES
            ):
                if text := block_text(el):
                    blocks.append(text)
        loose.append(el.tail or "")
    flush()
    return blocks, []


def parse_page(item: tuple[str, str]) -> dict:
    url, html = item
    canonical = CANONICAL.search(html)
    article_id = ARTICLE_ID.search(html)
    row = {
        "url_name": url.rstrip("/").rsplit("/", 1)[-1],
        "canonical_ns": canonical.group(1) if canonical else None,
        "canonical_name": canonical.group(2) if canonical else None,
        "article_id": int(article_id.group(1)) if article_id else None,
        "html_chars": len(html),
    }
    doc = lxml_html.fromstring(html)
    articles = doc.xpath('//div[@id="main-article"]')
    if not articles or clean(articles[0].text_content()).startswith(MISSING_TEXT):
        return row | {"missing": True}
    article = articles[0]

    h1 = doc.xpath('//h1[contains(@class, "entry-title")]')
    if h1:
        for aka in h1[0].xpath('.//div[contains(@class, "aka-title")]'):
            aka.drop_tree()

    # What follows the description is examples (or, on index pages, the index).
    description, after = split_description(article)

    folder_labels, folder_items = [], []
    for label in article.xpath('.//div[contains(@class, "folderlabel")]'):
        text = clean(label.text_content())
        if text != "open/close all folders":
            folder_labels.append(text)
            folder_items.append(folder_examples(label))

    name = row["canonical_name"]
    # Every Main/ page the article links to, description and examples alike, by
    # wiki name; that name is often an alternate one, which features.py resolves
    # before counting each trope's inbound links.
    main_links = {
        r[1]
        for href in article.xpath(".//a/@href")
        if (r := page_ref(href)) and r[0] == "Main"
    }
    links_to_main, links_elsewhere, example_subpages = 0, 0, set()
    for el in after:
        for href in el.xpath(".//a/@href"):
            ref = page_ref(href)
            if ref is None:
                continue
            if ref[0] == "Main":
                links_to_main += 1
            elif ref[0] == name:  # e.g. ChekhovsGun/AnimeAndManga
                example_subpages.add(ref[1])
            else:
                links_elsewhere += 1

    # The footer's Previous / Index / Next rows; the middle link of each row is an
    # index the page belongs to.
    indexes = []
    for ul in doc.xpath(
        '//div[contains(@class, "section-links")]//div[@class="links"]/ul'
    ):
        lis = ul.xpath("./li")
        if len(lis) == 3:
            refs = [page_ref(a.get("href")) for a in lis[1].xpath(".//a")]
            indexes += ["/".join(r) for r in refs if r]

    subpages = {
        r[0]
        for a in doc.xpath('//ul[contains(@class, "subpage-links")]//a')
        if (r := page_ref(a.get("href")))
    }

    return row | {
        "missing": False,
        "title": clean(h1[0].text_content()) if h1 else None,
        "description": "\n\n".join(description),
        "headings": [clean(h.text_content()) for h in article.xpath(".//h2")],
        "main_links": sorted(main_links - {name}),
        "folder_labels": folder_labels,
        "folder_items": folder_items,
        "loose_items": sum(
            len(el.xpath("./li")) for el in after if el.tag in ("ul", "ol")
        ),
        "example_subpages": sorted(example_subpages),
        "links_to_main": links_to_main,
        "links_elsewhere": links_elsewhere,
        "indexes": indexes,
        "subpages": sorted(subpages - {"Main"}),
        "may_refer_to": "may refer to" in clean(article.text_content())[:400],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pages = iter_dump_pages("Main")
    if args.limit is not None:
        pages = itertools.islice(pages, args.limit)
    rows = []
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        for i, row in enumerate(pool.imap(parse_page, pages, chunksize=32), start=1):
            rows.append(row)
            if i % 10_000 == 0:
                print(f"  {i} pages parsed", flush=True)
    df = pd.DataFrame(rows)
    print(f"{len(df)} Main/ URLs in the dump")

    missing = df["missing"]
    print(f"  {missing.sum()} dropped: no article at that URL")
    df = df[~missing]
    outside = df["canonical_ns"] != "Main"
    print(
        f"  {outside.sum()} dropped: redirect out of Main/ {df.loc[outside, 'canonical_ns'].value_counts().head(8).to_dict()}"
    )
    df = df[~outside].copy()
    # Missing-page rows lacked these, which left them float.
    int_cols = ["article_id", "loose_items", "links_to_main", "links_elsewhere"]
    df[int_cols] = df[int_cols].astype("Int64")

    # Several URLs can name one page. Keep the row fetched under the page's own
    # name where there is one, and record the other names as aliases.
    df["is_self"] = df["url_name"] == df["canonical_name"]
    df = df.sort_values(
        ["canonical_name", "is_self", "url_name"], ascending=[True, False, True]
    )
    aliases = df[~df["is_self"]].groupby("canonical_name")["url_name"].agg(sorted)
    pages_df = df.drop_duplicates("canonical_name").reset_index(drop=True)
    pages_df["aliases"] = [aliases.get(n, []) for n in pages_df["canonical_name"]]
    print(
        f"  {len(df)} Main/ URLs -> {len(pages_df)} canonical pages "
        f"({len(df) - len(pages_df)} alternate names folded in as aliases; "
        f"{(~pages_df['is_self']).sum()} pages seen only under an alias)"
    )

    pages_df = pages_df.rename(columns={"canonical_name": "page"})
    pages_df["url"] = "https://tvtropes.org/pmwiki/pmwiki.php/Main/" + pages_df["page"]
    pages_df = pages_df.drop(columns=["url_name", "canonical_ns", "missing", "is_self"])
    # A --limit run is a test; it must never replace the full output.
    output = MAIN_PAGES_PARQUET
    if args.limit is not None:
        output = output.with_name(f"{output.stem}.limit{args.limit}.parquet")
    write_parquet_safely(pages_df, output)
    print(f"wrote {output} ({len(pages_df)} rows)")


if __name__ == "__main__":
    main()
