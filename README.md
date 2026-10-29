# A Map of TV Tropes

An interactive map of the 28,726 tropes on the [TV Tropes](https://tvtropes.org/)
wiki, placed by the meaning of each trope's description and named at four zoom
levels.

**Status:** work in progress. The map is currently built from a June 2023
snapshot of the wiki and isn't published yet; the published version will come
from a fresh crawl, parsed by the same code.

This is an unofficial project and is not affiliated with TV Tropes. Clicking a
point opens the trope on TV Tropes.

## How it's built

| Stage | Script | What it does |
|---|---|---|
| Fetch | `fetch_dump.py` | Downloads only the `Main/` namespace (2.2 GB) of the [TvTroper](https://huggingface.co/datasets/RyokoExtra/TvTroper) dump, using HTTP range requests into its 20.7 GB zip, and checks it against the zip's CRC. Nothing is fetched from tvtropes.org |
| Parse | `parse.py` | One row per page (32,981 from 46,654 URLs): description, example folders, indexes, subpages; redirects are folded in as alternate names |
| Select | `select_tropes.py` | Keeps pages with examples or a definition-only note, then drops indexes and period lists of works, printing a drop count for each rule: 28,726 tropes |
| Embed | `embed.py` | Qwen3-Embedding-8B, through [OpenRouter](https://openrouter.ai/) pinned to one provider, embeds title + description with a clustering instruction |
| Cluster | `cluster.py` | UMAP to 2-d, then [Toponymy](https://github.com/TutteInstitute/toponymy) finds regions and Claude Opus 5.5 names them |
| Render | `render.py` | [DataMapPlot](https://github.com/TutteInstitute/datamapplot) builds `data/maps/map_instructed.html` |

Regions are found on the same 2-d layout that is plotted, so every named region
matches something visible on the map.

Some choices were settled by side-by-side tests rather than defaults:

- **Embedding instruction.** Both a plain and an instructed embedding were
  clustered, named and rendered. With the instruction, neighbours follow what a
  trope means rather than the words in its name (Chekhov's Gun sits next to
  Foreshadowing, not only the other "Chekhov's …" pages).
- **Naming model.** `experiments/compare_namer_models.py` names the same 52
  captured Toponymy prompts with Opus, Sonnet and Haiku and builds a blind
  comparison page. Opus won the blind read; automatic checks couldn't separate
  the three.
- **No "Tropes" in region names.** On a map where every point is a trope, the
  word adds nothing, so the naming prompt asks for it to be left out, except for
  regions about tropes themselves (Subverted Trope, Invoked Trope).

## Reproducing

Requires [uv](https://docs.astral.sh/uv/), `OPENROUTER_API_KEY` and
`ANTHROPIC_API_KEY`.

```bash
uv run fetch_dump.py                      # 2.2 GB download
uv run parse.py                           # ~45 s
uv run select_tropes.py
uv run embed.py                           # ~6 min, ~$0.12 of embeddings
uv run cluster.py --explore               # free: region counts at a few settings
uv run cluster.py --min-cluster-size 30   # ~$8 of Claude Opus 5.5 naming
uv run render.py
```

Intermediate data lives in `data/` and is not committed. Embeddings and LLM
responses are cached on disk, so re-running a stage does not repeat paid calls.

## Licenses

The code is released under the [MIT License](LICENSE).

TV Tropes content is licensed [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/).
This repository contains none of it, but what the pipeline produces from it (the
rendered maps, which quote the opening of each description, and the region
names, which are written from the descriptions) is shared under that same
license: non-commercially, with credit to TV Tropes. The TvTroper dump is
labelled Apache-2.0 on Hugging Face, but its content is TV Tropes text and keeps
TV Tropes' license.

## Related work

- Gala et al., [Analyzing Gender Bias within Narrative Tropes](https://aclanthology.org/2020.nlpcss-1.23/)
  (2020), whose TVTROPES dataset covers 30K tropes and their examples in film,
  TV and literature.
- [tvtropes-state-space](https://github.com/sh1ftmaker/tvtropes-state-space), an
  earlier interactive map of tropes from that dataset, laid out by genre
  affinity.
