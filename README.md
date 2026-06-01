# From News Text to Causal Infographics via Semantic Triplets

Does inserting an explicit **semantic-triplet (subject–predicate–object) stage** between an
event's articles and its generated causal-network infographic make the pipeline better — or
just more **inspectable**? This repo holds the results of a small study (10 Slovene news
events) built on the MedVrsticami / BetweenTheLines pipeline.

**TL;DR.** Decomposing into triplets does *not* produce a better image — the direct
end-to-end render is denser, modestly more faithful, and clearer. The value is
**transparency**: with the graph made explicit we can show extraction is ~70 % clean while
the image model mis-draws ~19 % of the edges of an otherwise-correct graph (mostly
*wrong connections*) — two error sources the black box conflates.

## 👉 Live results viewer
**[Browse the events, images, triplets, and scores](https://georgigeor.github.io/causal-triplet-infographics/)**
— read-only. Per event: direct vs triplet→image vs a programmatic render, the extracted
triplets, the prompts, the source articles, and every rubric score the rater gave.

## Contents
| Path | What it is |
|------|-----------|
| `paper.pdf` | The 4-page paper. |
| `results/seminar_scores.json` | Every rubric score (per-triplet verdicts, per-edge render fidelity, faithfulness/clarity, coverage, node counts). |
| `results/per_event.csv` | Per-event tidy summary of the same scores. |
| `comparison/`, `output/`, `index.html` | The rendered read-only site (also served via GitHub Pages above): per-event images, triplets, prompts, articles, and scores. |

## What was measured
- **Extraction** (`gemini-3-flash-preview`): article cluster → connected SPO graph with
  stable entity IDs, confidence, source-language lock, event reification, and
  attribution-in-predicate for contested claims.
- **Two evaluated renders** (`gemini-3-pro-image-preview` / *Nano Banana Pro*, identical
  settings): **direct** (text→image) vs **triplet→image** (the same model fed the explicit
  graph). A deterministic render is shown for illustration but not evaluated.
- **Two-layer rubric** (single rater): triplet correctness (per-triplet severity verdicts +
  coverage) and rendered-image faithfulness (faithfulness, clarity, per-edge render
  fidelity, density).

## Note on data
The bundled article text is public Slovene news coverage (via Event Registry), included for
reproducibility of the evaluation.
