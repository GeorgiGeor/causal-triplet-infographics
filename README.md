# From News Text to Causal Infographics via Semantic Triplets

Does inserting an explicit **semantic-triplet (subject–predicate–object) stage** between an
event's articles and its generated causal-network infographic make the pipeline better — or
just more **inspectable**? This repo holds the paper, data, code, and an interactive
read-only results viewer for a small study (10 Slovene news events) built on the
MedVrsticami / BetweenTheLines pipeline.

**TL;DR.** Decomposing into triplets does *not* produce a better image — the direct
end-to-end render is denser, modestly more faithful, and clearer. The value is
**transparency**: with the graph made explicit we can show extraction is ~70% clean while
the image model mis-draws ~19% of the edges of an otherwise-correct graph (mostly
*wrong connections*) — two error sources the black box conflates.

## Live results viewer
👉 **[Browse the events, images, triplets, and scores](https://georgigeor.github.io/causal-triplet-infographics/)**
(read-only; per-event: direct vs triplet→image vs a programmatic render, the extracted
triplets, the prompts, the source articles, and every rubric score).

## Layout
| Path | Contents |
|------|----------|
| `paper/` | The 4-page paper (`paper.tex`, `refs.bib`, `figs/`, `paper.pdf`). |
| `evaluation/` | `seminar_scores.json` (all rubric scores), `per_event.csv`, and `aggregate_scores.py` (run it on the JSON to reproduce every number/table in the paper). |
| `src/` | The pipeline: triplet extraction + prompts, the two image generators, the programmatic HTML render, and the comparison-site builder. |
| `docs/` | The published read-only site (GitHub Pages): the frozen comparison view + per-event assets. |

## Reproduce the numbers
```bash
python evaluation/aggregate_scores.py evaluation/seminar_scores.json --csv /tmp/per_event.csv
```

## Method, briefly
- **Extraction** (`gemini-3-flash-preview`): article cluster → connected SPO graph with
  stable entity IDs, confidence, source-language lock, event reification, and
  attribution-in-predicate for contested claims.
- **Two evaluated renders** (`gemini-3-pro-image-preview` / *Nano Banana Pro*, identical
  settings): **direct** (text→image) vs **triplet→image** (the same model fed the explicit
  graph). A deterministic `graphviz` render is shown for illustration but not evaluated.
- **Two-layer rubric** (single rater): triplet correctness (per-triplet severity verdicts +
  coverage) and rendered-image faithfulness (faithfulness, clarity, per-edge render
  fidelity, density).

## Note on data
The bundled article text is public Slovene news coverage (via Event Registry), included for
research reproducibility of the evaluation.
