# A Map of TV Tropes

An interactive map of the 28,726 tropes on the [TV Tropes](https://tvtropes.org/)
wiki, placed by the meaning of each trope's description and named at four zoom
levels.

**[Open the map](https://stevenfazzio.com/tv-tropes-map/)**

**Status:** work in progress. The map is built from a June 2023 snapshot of the
wiki; a fresh crawl, parsed by the same code, will replace it.

This is an unofficial project and is not affiliated with TV Tropes. Clicking a
point opens the trope on TV Tropes.

## How it's built

| Stage | Script | What it does |
|---|---|---|
| Fetch | `fetch_dump.py` | Downloads only the `Main/` namespace (2.2 GB) of the [TvTroper](https://huggingface.co/datasets/RyokoExtra/TvTroper) dump, using HTTP range requests into its 20.7 GB zip, and checks it against the zip's CRC. Nothing is fetched from tvtropes.org |
| Parse | `parse.py` | One row per page (32,981 from 46,654 URLs): description, example folders, indexes, subpages; redirects are folded in as alternate names |
| Select | `select_tropes.py` | Keeps pages with examples or a definition-only note, then drops indexes and period lists of works, printing a drop count for each rule: 28,726 tropes |
| Features | `features.py` | Each trope's primary medium (the one holding at least half its examples, after merging folder-name variants) and how many other tropes link to it |
| Embed | `embed.py` | Qwen3-Embedding-8B, through [OpenRouter](https://openrouter.ai/) pinned to one provider, embeds title + description with a clustering instruction |
| Cluster | `cluster.py` | UMAP to 2-d, then [Toponymy](https://github.com/TutteInstitute/toponymy) finds regions and Claude Opus 5.5 names them |
| Render | `render.py` | [DataMapPlot](https://github.com/TutteInstitute/datamapplot) builds `docs/index.html`, which GitHub Pages serves, with its data in zip files beside it and colormaps for primary medium and for prominence (how many other tropes link to each one). `social_preview.py` screenshots it for the link card |

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
- **Primary medium rule.** `experiments/compare_medium_rules.py` puts two rules
  on one map as colormaps: the medium holding at least half of a trope's
  examples, and a medium with at least 3× its share of all examples on the wiki
  (and at least 25% of the trope's). The second colours 29% of tropes rather
  than 8% and gives small media a chance, but it sets a different bar for each
  medium and scatters the colour (a coloured trope's neighbours share its colour
  at 3.5× chance, against 6.3×), so the map keeps the 50% rule.

## Reproducing

Requires [uv](https://docs.astral.sh/uv/), `OPENROUTER_API_KEY` and
`ANTHROPIC_API_KEY`.

```bash
uv run fetch_dump.py                      # 2.2 GB download
uv run parse.py                           # ~45 s
uv run select_tropes.py
uv run features.py
uv run embed.py                           # ~6 min, ~$0.12 of embeddings
uv run cluster.py --explore               # free: region counts at a few settings
uv run cluster.py --min-cluster-size 30   # ~$8 of Claude Opus 5.5 naming
uv run render.py
uv run social_preview.py                  # link-card image; drives your installed Chrome
```

Intermediate data lives in `data/` and is not committed; the published map in
`docs/` is. Embeddings and LLM responses are cached on disk, so re-running a
stage does not repeat paid calls.

## Licenses

The code is released under the [MIT License](LICENSE).

TV Tropes content is licensed [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/).
The published map in `docs/` quotes the opening of each trope's description, and
its region names are written from the descriptions, so it is shared under that
same license: non-commercially, with credit to TV Tropes. The same goes for
anything else the pipeline produces from TV Tropes text. The TvTroper dump is
labelled Apache-2.0 on Hugging Face, but its content is TV Tropes text and keeps
TV Tropes' license.

## Related work

- Gala et al., [Analyzing Gender Bias within Narrative Tropes](https://aclanthology.org/2020.nlpcss-1.23/)
  (2020), whose TVTROPES dataset covers 30K tropes and their examples in film,
  TV and literature.
- [tvtropes-state-space](https://github.com/sh1ftmaker/tvtropes-state-space), an
  earlier interactive map of tropes from that dataset, laid out by genre
  affinity.
