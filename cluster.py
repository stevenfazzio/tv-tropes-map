"""Reduce one embedding variant to 2-d, then find and name regions with Toponymy.

Runs on the map's embedding variant unless --variant names another (see embed.py
for why it's `instructed`). Clustering runs on the same 2-d coordinates that get
plotted, so named regions match what a viewer sees. That makes UMAP's min_dist a
clustering parameter: it stays low so regions come out dense enough for density
clustering.

    uv run cluster.py --explore                        # free: layer counts at a few settings
    uv run cluster.py --estimate --min-cluster-size N  # free: count and size the naming prompts
    uv run cluster.py --min-cluster-size 30            # paid: fit + Opus naming (~$8)

Writes data/labels_<variant>.parquet with x, y and label_layer_0..k, where layer 0
is the FINEST layer (Toponymy's in-memory order, which DataMapPlot also expects),
and data/topic_names_<variant>.json.
"""

import argparse
import json
import re

import numpy as np
import pandas as pd
import umap
from toponymy import Toponymy, ToponymyClusterer

from common import (
    DATA,
    TROPES_PARQUET,
    write_bytes_atomic,
    write_npz_atomic,
    write_parquet_safely,
)
from embed import INSTRUCTIONS, MAP_VARIANT, document_text, embeddings_path
from embedder import OpenRouterEmbedder
from namer import AsyncClaudeNamer, DryRunNamer

UMAP_PARAMS = dict(
    n_neighbors=15, min_dist=0.05, metric="cosine", n_components=2, random_state=42
)
# Appended to every naming prompt. On a map where every point is a trope, "Tropes"
# in a label is a dead word that costs label space; it carries meaning only for
# the regions about tropes as such (Subverted Trope, Invoked Trope, Trope Namers).
NAMING_INSTRUCTIONS = (
    "Every group here is made of tropes from the TV Tropes wiki, so words like "
    '"tropes", "TV Tropes" and "in fiction" add nothing to a name: leave them out '
    "and name what the tropes are about. Use the word trope only when a group is "
    "about tropes themselves, such as how tropes are subverted, invoked or discussed."
)
UNWANTED_IN_NAMES = re.compile(r"\btropes?\b|\bfiction(al)?\b", re.IGNORECASE)


def coords_path(variant: str):
    return DATA / f"umap_{variant}.npz"


def load_embeddings(variant: str, tropes: pd.DataFrame) -> np.ndarray:
    data = np.load(embeddings_path(variant))
    assert list(data["pages"]) == list(tropes["page"]), (
        "embeddings out of sync with tropes"
    )
    return data["vectors"]


def get_coords(variant: str, pages: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    path = coords_path(variant)
    params_key = json.dumps(UMAP_PARAMS, sort_keys=True)
    if path.exists():
        cached = np.load(path)
        if list(cached["pages"]) == list(pages) and str(cached["params"]) == params_key:
            return cached["coords"]
    # random_state forces UMAP onto one thread; on 28.7k x 4096 that takes minutes.
    print(f"running UMAP {UMAP_PARAMS}")
    coords = umap.UMAP(**UMAP_PARAMS).fit_transform(vectors).astype(np.float32)
    write_npz_atomic(path, pages=pages, coords=coords, params=params_key)
    return coords


def make_clusterer(min_cluster_size: int) -> ToponymyClusterer:
    return ToponymyClusterer(min_clusters=4, base_min_cluster_size=min_cluster_size)


def explore(coords: np.ndarray, vectors: np.ndarray, sizes: list[int]) -> None:
    for size in sizes:
        clusterer = make_clusterer(size)
        clusterer.fit(clusterable_vectors=coords, embedding_vectors=vectors)
        counts = []
        for layer in clusterer.cluster_layers_:  # finest first
            labels = layer.cluster_labels
            counts.append(
                f"{labels.max() + 1} ({(labels == -1).mean():.0%} unlabelled)"
            )
        print(
            f"base_min_cluster_size={size}: {len(counts)} layers, finest->coarsest: "
            + " | ".join(counts)
        )


def make_topic_model(
    variant: str, namer, min_cluster_size: int, detail: tuple[float, float]
) -> Toponymy:
    return Toponymy(
        llm_wrapper=namer,
        # Same model, provider and instruction as the document vectors: Toponymy
        # compares keyphrase and name embeddings against them.
        text_embedding_model=OpenRouterEmbedder(instruction=INSTRUCTIONS[variant]),
        clusterer=make_clusterer(min_cluster_size),
        object_description="tropes",
        corpus_description="the TV Tropes wiki",
        # Spread evenly across layers and rounded onto Toponymy's seven name-length
        # tiers: 0.4-0.8 asks for 4-8 words at the finest layer with four layers.
        lowest_detail_level=detail[0],
        highest_detail_level=detail[1],
    )


def estimate(variant, tropes, coords, vectors, min_cluster_size, detail) -> None:
    namer = DryRunNamer()
    topic_model = make_topic_model(variant, namer, min_cluster_size, detail)
    texts = [document_text(r) for r in tropes.itertuples()]
    topic_model.fit(texts, embedding_vectors=vectors, clusterable_vectors=coords)
    for kind in ["topic", "disambiguation"]:
        chars = [len(s or "") + len(u) for k, s, u in namer.prompts if k == kind]
        if chars:
            print(
                f"{kind}: {len(chars)} prompts, {sum(chars) / 4e6:.2f}M input tokens (at 4 chars/token), "
                f"median {int(np.median(chars)):,} chars, max {max(chars):,}"
            )
    print("layer sizes:", [len(names) for names in topic_model.topic_names_])
    out = DATA / f"dry_run_prompts_{variant}_{min_cluster_size}.json"
    write_bytes_atomic(json.dumps(namer.prompts).encode(), out)
    print(f"wrote {out}")


def fit_and_name(variant, tropes, coords, vectors, min_cluster_size, detail) -> None:
    namer = AsyncClaudeNamer(llm_specific_instructions=NAMING_INSTRUCTIONS)
    topic_model = make_topic_model(variant, namer, min_cluster_size, detail)
    texts = [document_text(r) for r in tropes.itertuples()]
    # Toponymy.fit takes (objects, high-D embeddings, low-D map): the reverse of
    # the clusterer's argument order, so pass by keyword.
    topic_model.fit(texts, embedding_vectors=vectors, clusterable_vectors=coords)

    names = topic_model.topic_names_
    assert len(names[0]) >= len(names[-1]), "expected layer 0 to be the finest"
    out = pd.DataFrame({"page": tropes["page"], "x": coords[:, 0], "y": coords[:, 1]})
    for i, vec in enumerate(topic_model.topic_name_vectors_):
        out[f"label_layer_{i}"] = vec
    labels_path = DATA / f"labels_{variant}.parquet"
    names_path = DATA / f"topic_names_{variant}.json"
    write_parquet_safely(out, labels_path)
    write_bytes_atomic(json.dumps(names, indent=2).encode(), names_path)

    for i, layer_names in enumerate(names):
        share = (out[f"label_layer_{i}"] == "Unlabelled").mean()
        words = np.median([len(n.split()) for n in layer_names]) if layer_names else 0
        print(
            f"layer {i}: {len(layer_names)} names, median {words:.0f} words, {share:.0%} unlabelled"
        )
    # NAMING_INSTRUCTIONS allows "trope" only for regions about tropes as such, so
    # list every name that still has it (or "fiction") for a check by eye.
    flagged = [
        (i, n)
        for i, layer_names in enumerate(names)
        for n in layer_names
        if UNWANTED_IN_NAMES.search(n)
    ]
    print(f"{len(flagged)} names mention trope(s) or fiction:")
    for i, n in flagged:
        print(f"  layer {i}: {n}")
    u = namer.usage
    print(
        f"naming: {u['calls']} calls ({u['cached']} more from cache), "
        f"{u['input_tokens']:,} in / {u['output_tokens']:,} out tokens = ${namer.cost_usd():.2f}"
    )
    print(f"wrote {labels_path} and {names_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(INSTRUCTIONS), default=MAP_VARIANT)
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[15, 30, 60])
    parser.add_argument("--min-cluster-size", type=int, default=None)
    parser.add_argument("--detail", type=float, nargs=2, default=[0.4, 0.8])
    args = parser.parse_args()

    tropes = pd.read_parquet(TROPES_PARQUET, columns=["page", "title", "description"])
    vectors = load_embeddings(args.variant, tropes)
    coords = get_coords(args.variant, tropes["page"].to_numpy(dtype=str), vectors)

    if args.explore:
        explore(coords, vectors, args.sizes)
    elif args.min_cluster_size is None:
        parser.error(
            "pass --explore, or --min-cluster-size N (with --estimate for a dry run)"
        )
    elif args.estimate:
        estimate(
            args.variant,
            tropes,
            coords,
            vectors,
            args.min_cluster_size,
            tuple(args.detail),
        )
    else:
        fit_and_name(
            args.variant,
            tropes,
            coords,
            vectors,
            args.min_cluster_size,
            tuple(args.detail),
        )


if __name__ == "__main__":
    main()
