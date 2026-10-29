"""Name the same captured Toponymy prompts with Opus, Sonnet and Haiku, and compare.

The prompts come from a dry run (cluster.py --estimate), so every model answers
the text the real fit sends. Finest-layer prompts are identical to the real
ones; coarser-layer prompts list keyword placeholders where real sub-region names
would go, but every model sees the same placeholders. Three checks, since each
has a blind spot:

  - format: the reply parses, and the name fits the length the prompt asked for
  - retrieval: embed the name with the map's embedder, rank every region centroid
    in its layer by similarity, and see where the name's own region lands. This
    rewards distinctive names, and also long keyword lists, so read it next to
    the length check.
  - a blind page, data/namer_test/compare.html: each region's keywords and sample
    tropes, with the three names in shuffled order and the models hidden.

Answers go through namer.py's disk cache, so the winning model's finest-layer
names are reused by the real run.

    uv run experiments/compare_namer_models.py
"""

import asyncio
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toponymy.llm_wrappers import llm_output_to_result  # noqa: E402
from toponymy.templates import GET_TOPIC_NAME_REGEX, SUMMARY_KINDS  # noqa: E402

from cluster import coords_path, load_embeddings, make_clusterer  # noqa: E402
from common import DATA, TROPES_PARQUET, write_bytes_atomic  # noqa: E402
from embed import INSTRUCTIONS  # noqa: E402
from embedder import OpenRouterEmbedder  # noqa: E402
from namer import AsyncClaudeNamer  # noqa: E402

VARIANT = "instructed"
MIN_CLUSTER_SIZE = 30
MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
SAMPLE_PER_LAYER = [40, 6, 3, 1]  # finest first
OUT = DATA / "namer_test"
PAGE_TEMPLATE = Path(__file__).with_name("compare_namer_models.html")


def parse_name(text: str) -> str | None:
    try:
        return str(
            llm_output_to_result(text, GET_TOPIC_NAME_REGEX)["topic_name"]
        ).strip()
    except Exception:
        return None


def describe(kind: str, system: str, user: str) -> dict:
    keywords = re.search(r"Keywords for this group include: ([^\n]+)", user)
    subtopics = re.search(
        r"Major subtopics of this group are:\n((?:\s+\*[^\n]*\n)+)", user
    )
    item = {
        "kind": kind,
        "keywords": keywords.group(1).split(", ") if keywords else [],
        "titles": re.findall(r'^\s*\* "([^\n]+)$', user, re.M),
        "subtopics": [s.strip(" *") for s in subtopics.group(1).splitlines()]
        if subtopics
        else [],
    }
    if kind == "topic":
        tier = re.search(r"assign a (.+?) name to this group", system).group(1)
        lo, hi = re.search(r"\((\d+) (?:to|or) (\d+) word\)", tier).groups()
        item |= {"tier": tier, "words": [int(lo), int(hi)]}
    return item


def main():
    prompts = json.load(
        open(DATA / f"dry_run_prompts_{VARIANT}_{MIN_CLUSTER_SIZE}.json")
    )
    unique = list(dict.fromkeys((k, s or "", u) for k, s, u in prompts))
    items = [describe(k, s, u) | {"system": s, "user": u} for k, s, u in unique]
    topic_items = [it for it in items if it["kind"] == "topic"]
    tiers = sorted(
        {it["tier"] for it in topic_items}, key=SUMMARY_KINDS.index
    )  # finest first
    for it in topic_items:
        it["layer"] = tiers.index(it["tier"])

    rng = random.Random(0)
    sample = []
    for layer, n in enumerate(SAMPLE_PER_LAYER):
        pool = [it for it in topic_items if it["layer"] == layer]
        sample += rng.sample(pool, min(n, len(pool)))
    sample += [it for it in items if it["kind"] == "disambiguation"]
    print(
        f"{len(sample)} prompts: "
        + ", ".join(
            f"layer {i}: {sum(it.get('layer') == i for it in sample)}"
            for i in range(len(tiers))
        )
    )

    costs = {}
    for model in MODELS:
        namer = AsyncClaudeNamer(model=model)

        async def run():
            return await asyncio.gather(
                *(namer._create(it["system"], it["user"]) for it in sample),
                return_exceptions=True,
            )

        for it, answer in zip(sample, asyncio.run(run())):
            raw = answer if isinstance(answer, str) else f"ERROR: {answer!r}"
            it.setdefault("answers", {})[model] = {"raw": raw}
        costs[model] = {**namer.usage, "cost_usd": round(namer.cost_usd(), 4)}
        print(
            f"{model}: {namer.usage['calls']} calls (+{namer.usage['cached']} cached), ${namer.cost_usd():.3f}"
        )

    # Which region each prompt names: the majority layer label of its sample tropes.
    tropes = pd.read_parquet(TROPES_PARQUET, columns=["page", "title"])
    vectors = load_embeddings(VARIANT, tropes)
    coords = np.load(coords_path(VARIANT))["coords"]
    clusterer = make_clusterer(MIN_CLUSTER_SIZE)
    clusterer.fit(clusterable_vectors=coords, embedding_vectors=vectors)
    title_pos = {t: i for i, t in reversed(list(enumerate(tropes["title"])))}

    names = []
    for it in sample:
        if it["kind"] != "topic":
            continue
        labels = clusterer.cluster_layers_[it["layer"]].cluster_labels
        members = [labels[title_pos[t]] for t in it["titles"] if t in title_pos]
        members = [m for m in members if m >= 0]
        it["region"] = int(np.bincount(members).argmax()) if members else None
        for model, a in it["answers"].items():
            a["name"] = parse_name(a["raw"])
            if a["name"]:
                a["n_words"] = len(a["name"].split())
                a["in_range"] = it["words"][0] <= a["n_words"] <= it["words"][1]
                names.append(a["name"])

    name_vecs = dict(
        zip(names, OpenRouterEmbedder(instruction=INSTRUCTIONS[VARIANT]).encode(names))
    )
    for it in sample:
        if it["kind"] != "topic" or it["region"] is None:
            continue
        centroids = clusterer.cluster_layers_[it["layer"]].centroid_vectors
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        for a in it["answers"].values():
            if a.get("name"):
                sims = centroids @ name_vecs[a["name"]]
                a["rank"] = int((sims > sims[it["region"]]).sum()) + 1
                a["n_regions"] = len(centroids)

    print(
        "\nmodel               parsed  in-range  median words  finest: rank 1  top 5  median rank   coarser: rank 1"
    )
    summary = {}
    for model in MODELS:
        answers = [it["answers"][model] for it in sample if it["kind"] == "topic"]
        finest = [it["answers"][model] for it in sample if it.get("layer") == 0]
        coarse = [it["answers"][model] for it in sample if it.get("layer", 0) > 0]
        ranks = [a["rank"] for a in finest if "rank" in a]
        s = {
            "parsed": float(np.mean([a.get("name") is not None for a in answers])),
            "in_range": float(np.mean([a.get("in_range", False) for a in answers])),
            "median_words": float(
                np.median([a["n_words"] for a in answers if "n_words" in a])
            ),
            "finest_rank1": float(np.mean([r == 1 for r in ranks])),
            "finest_top5": float(np.mean([r <= 5 for r in ranks])),
            "finest_median_rank": float(np.median(ranks)),
            "coarse_rank1": f"{sum(a.get('rank') == 1 for a in coarse)}/{len(coarse)}",
            **costs[model],
        }
        summary[model] = s
        print(
            f"{model:18s} {s['parsed']:7.0%} {s['in_range']:9.0%} {s['median_words']:13.0f} "
            f"{s['finest_rank1']:15.0%} {s['finest_top5']:6.0%} {s['finest_median_rank']:12.0f} {s['coarse_rank1']:>17s}"
        )

    # Each card shows the models in its own shuffled order, so position gives nothing away.
    for i, it in enumerate(sample):
        it["order"] = random.Random(i).sample(MODELS, len(MODELS))
        del it["system"], it["user"]
    data = {
        "variant": VARIANT,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "models": MODELS,
        "summary": summary,
        "items": sample,
    }
    write_bytes_atomic(json.dumps(data, indent=1).encode(), OUT / "results.json")
    page = PAGE_TEMPLATE.read_text().replace(
        "/*DATA*/null", json.dumps(data).replace("</", "<\\/")
    )
    write_bytes_atomic(page.encode(), OUT / "compare.html")
    print(f"\nwrote {OUT / 'results.json'} and {OUT / 'compare.html'}")


if __name__ == "__main__":
    main()
