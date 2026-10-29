"""Embed each trope's title + description with Qwen3-Embedding-8B.

Two variants:

    plain       the text as is
    instructed  prefixed with a clustering instruction, worded after the ones Qwen
                used for its own clustering benchmarks

The map uses `instructed` (MAP_VARIANT). Both variants were embedded, clustered,
named and rendered side by side: instructed's neighbours followed what a trope
means rather than the words in its name (Chekhov's Gun next to Foreshadowing, not
just the other "Chekhov's ..." pages), and its map grouped storytelling devices
together instead of filing them under production. Plain stays available for
comparison.

Writes data/embeddings_<variant>.npz (pages, vectors), row-aligned with
tropes.parquet. Vectors come through embedder.py's disk cache, so re-running is
free for any trope whose text hasn't changed.

    uv run embed.py --limit 64      # first 64 tropes: report time and cost, write nothing
    uv run embed.py                 # the map's variant
    uv run embed.py --variant plain --variant instructed
"""

import argparse
import time

import numpy as np
import pandas as pd

from common import DATA, TROPES_PARQUET, write_npz_atomic
from embedder import MODEL, OpenRouterEmbedder

INSTRUCTIONS = {
    "plain": None,
    "instructed": "Identify the topic or theme of the given TV Tropes trope based on its name and description",
}
MAP_VARIANT = "instructed"
# Long descriptions tend to run on into lists (match types, albums, gun models)
# after the definition, which the opening carries. 8,000 characters is about
# 2,000 tokens; around 1% of tropes are longer.
MAX_CHARS = 8_000


def embeddings_path(variant: str):
    return DATA / f"embeddings_{variant}.npz"


def document_text(row) -> str:
    text = f"{row.title}\n\n{row.description}".strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rsplit(None, 1)[0]  # don't end on half a word
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(INSTRUCTIONS), action="append")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tropes = pd.read_parquet(TROPES_PARQUET, columns=["page", "title", "description"])
    texts = [document_text(r) for r in tropes.itertuples()]
    empty = (tropes["description"].str.len() == 0).sum()
    truncated = sum(
        len(f"{r.title}\n\n{r.description}".strip()) > MAX_CHARS
        for r in tropes.itertuples()
    )
    print(
        f"{len(tropes)} tropes; {truncated} cut to {MAX_CHARS:,} chars; {empty} with no description (title only)"
    )
    if args.limit is not None:
        texts = texts[: args.limit]

    for variant in args.variant or [MAP_VARIANT]:
        embedder = OpenRouterEmbedder(instruction=INSTRUCTIONS[variant])
        start = time.time()
        vectors = embedder.encode(texts, show_progress_bar=True)
        elapsed = time.time() - start
        norms = np.linalg.norm(vectors, axis=1)
        print(
            f"{variant}: {vectors.shape} in {elapsed:.0f}s via {embedder.provider}; "
            f"{embedder.tokens:,} new tokens, ${embedder.cost_usd:.4f}; norms {norms.min():.3f}-{norms.max():.3f}"
        )
        assert np.isfinite(vectors).all()
        if args.limit is not None:
            continue  # a test run never writes the output

        def check(loaded):
            assert loaded["vectors"].shape == vectors.shape
            assert list(loaded["pages"]) == list(tropes["page"])

        path = embeddings_path(variant)
        write_npz_atomic(
            path,
            check=check,
            pages=tropes["page"].to_numpy(dtype=str),
            vectors=vectors,
            model=np.array(MODEL),
            provider=np.array(embedder.provider),
            instruction=np.array(INSTRUCTIONS[variant] or ""),
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
