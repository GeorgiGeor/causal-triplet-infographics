"""
Aggregate the manually-entered scores exported from the comparison view
(``seminar_scores.json``) into paper-ready numbers + LaTeX snippets.

This supersedes the legacy ``results.py`` (which read the now-defunct
``rubric.csv``). The scoring UI in ``comparison/build_comparison.py`` autosaves
to the browser and exports a single JSON file with this shape:

    { "_meta": {...},
      "scores": { "<event>": {
         "triplets":      {"coverage": "1-5", "comment": str,
                           "per_triplet": [{idx, subject_id, predicate,
                                            object_id, verdict, note}, ...]},
         "direct":        {"faithfulness":"1-5","clarity":"1-5","comment":str,
                           "node_count":str},
         "triplet_image": {"faithfulness":"1-5","clarity":"1-5","comment":str,
                           "edge_fidelity_matched":str,"unsupported_additions":str,
                           "omissions":str,"edge_total":int,"node_total":int}
      }, ... } }

Two evaluation layers, kept distinct on purpose:
  * Subjective head-to-head on the FINAL image (same basis for both pipelines):
    faithfulness + clarity, direct vs triplet->image.
  * Objective render-fidelity for the triplet->image STEP only (checkable
    against the known graph): edge fidelity, unsupported additions, omissions.
  * Triplet layer: per-triplet verdicts (correctness rate + failure codebook)
    and event-level coverage.
  * Density: direct node count (counted by eye) vs triplet->image node count
    (known) -- the expressiveness-gap finding.

Usage:
    python aggregate_scores.py [path/to/seminar_scores.json] [--csv per_event.csv]

Stdlib only.
"""

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_INPUT = Path(__file__).parent / "seminar_scores.json"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Models & settings used to produce the artifacts, recorded from
# config/analysis/settings.py + the experiment script constants. Kept as a
# documented constant (the run is frozen) so the aggregator stays stdlib-only.
MODELS = {
    "Extraction (text→triplets)": {
        "script": "extract_triplets.py → services/llm_service.py",
        "model": "gemini-3-flash-preview",
        "settings": "temperature 0.2, max_tokens 3000, response_format=json_object, "
                    "thinking_level=MINIMAL, fallback gpt-4; article bodies capped at 3000 chars",
    },
    "Direct image (text→image)": {
        "script": "gen_direct.py",
        "model": "gemini-3-pro-image-preview",
        "settings": "image_size 2K (2752×1536), aspect_ratio 16:9, "
                    "modalities [IMAGE,TEXT], safety OFF, ≤5 retries, "
                    "fallback gemini-3.1-flash-image-preview",
    },
    "Triplet→image (triplets→image)": {
        "script": "gen_triplet_image.py",
        "model": "gemini-3-pro-image-preview",
        "settings": "identical to the direct condition (only the data source differs: "
                    "supplied entities+triplets vs raw article text)",
    },
    "Triplet→HTML (triplets→graph)": {
        "script": "gen_triplet_html.py",
        "model": "— (no model)",
        "settings": "deterministic graphviz `dot` layout; renders the triplets verbatim",
    },
}

# Must match the grouped verdict taxonomy in build_comparison.py.
MINOR_CATEGORIES = ["misleading", "awkward_wording", "imprecise"]
ERROR_CATEGORIES = ["wrong_direction", "wrong_predicate", "wrong_endpoints",
                    "unsupported", "redundant"]
# Tier order for display: correct → minor → error → other.
VERDICT_CATEGORIES = ["correct"] + MINOR_CATEGORIES + ERROR_CATEGORIES + ["other"]

# Per-fed-edge render verdicts (how the image drew each supplied triplet).
RENDER_CATEGORIES = ["yes", "wrong_direction", "wrong_connection", "omitted", "other"]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def num(v) -> Optional[float]:
    """Parse a score cell to float; '', None, 'NA' -> None."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.upper() == "NA":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def stats_for(values: List[float]):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def fmt(mean, std) -> str:
    if mean is None:
        return "—"
    return f"{mean:.2f} ± {std:.2f}"


def parse_additions(ti: dict) -> List[dict]:
    """Normalise the additions list to S/P/O dicts (tolerates legacy strings)."""
    out = []
    for a in ti.get("additions") or []:
        if isinstance(a, dict):
            spo = {k: (a.get(k) or "").strip() for k in ("subject", "predicate", "object")}
            if any(spo.values()):
                out.append(spo)
        elif str(a).strip():
            out.append({"subject": str(a).strip(), "predicate": "", "object": ""})
    return out


def addition_str(a: dict) -> str:
    return f"{a['subject']} →{a['predicate']}→ {a['object']}"


def load_scores(path: Path) -> Dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    scores = data.get("scores", data)  # tolerate a bare {event: {...}} too
    # Drop events with no data at all (unscored placeholders).
    return {k: v for k, v in scores.items() if isinstance(v, dict) and v}


# ---------------------------------------------------------------------------
# Layer 1 — subjective head-to-head on the final image
# ---------------------------------------------------------------------------

def image_metric(scores, cond, metric) -> List[float]:
    out = []
    for ev in scores.values():
        v = num((ev.get(cond) or {}).get(metric))
        if v is not None:
            out.append(v)
    return out


def paired(scores, metric):
    """Return (deltas, n_triplet_ge_direct, n_pairs) over events scored on both."""
    deltas, ge = [], 0
    for ev in scores.values():
        d = num((ev.get("direct") or {}).get(metric))
        t = num((ev.get("triplet_image") or {}).get(metric))
        if d is not None and t is not None:
            deltas.append(t - d)
            if t >= d:
                ge += 1
    return deltas, ge, len(deltas)


def print_head_to_head(scores):
    print("\n=== Image metrics — direct vs triplet→image (final image, vs articles) ===")
    print(f"{'metric':<22}{'direct':<16}{'triplet→image':<16}{'Δ (t−d) mean':<16}{'t≥d'}")
    print("-" * 78)
    for m in ("faithfulness", "clarity"):
        dvals = image_metric(scores, "direct", m)
        tvals = image_metric(scores, "triplet_image", m)
        deltas, ge, n = paired(scores, m)
        dmean, dstd = stats_for(dvals)
        tmean, tstd = stats_for(tvals)
        delta_mean = f"{statistics.mean(deltas):+.2f}" if deltas else "—"
        print(f"{m:<22}{fmt(dmean, dstd):<16}{fmt(tmean, tstd):<16}{delta_mean:<16}{ge}/{n}")


# ---------------------------------------------------------------------------
# Layer 2 — density (the expressiveness-gap finding)
# ---------------------------------------------------------------------------

def print_density(scores):
    print("\n=== Density (nodes) — direct (counted) vs triplet→image (known) ===")
    direct_nodes, triplet_nodes, per_event = [], [], []
    for code, ev in scores.items():
        d = num((ev.get("direct") or {}).get("node_count"))
        t = (ev.get("triplet_image") or {}).get("node_total")
        t = float(t) if isinstance(t, (int, float)) else num(t)
        if d is not None:
            direct_nodes.append(d)
        if t is not None:
            triplet_nodes.append(t)
        if d is not None and t is not None:
            per_event.append((code, d, t))
    dmean, dstd = stats_for(direct_nodes)
    tmean, tstd = stats_for(triplet_nodes)
    print(f"  direct nodes:        {fmt(dmean, dstd)}  (n={len(direct_nodes)})")
    print(f"  triplet→image nodes: {fmt(tmean, tstd)}  (n={len(triplet_nodes)})")
    if dmean and tmean:
        print(f"  ratio direct/triplet: {dmean / tmean:.2f}×")
    return per_event


# ---------------------------------------------------------------------------
# Layer 3 — triplet layer (coverage + per-triplet verdicts)
# ---------------------------------------------------------------------------

def collect_verdicts(scores):
    tally = Counter()
    notes = []  # (event, spo, verdict, note)
    total_rated = 0
    for code, ev in scores.items():
        for t in (ev.get("triplets") or {}).get("per_triplet", []) or []:
            verdict = (t.get("verdict") or "").strip()
            if not verdict:
                continue  # left unrated
            total_rated += 1
            tally[verdict] += 1
            note = (t.get("note") or "").strip()
            if note or verdict != "correct":
                spo = f"{t.get('subject_id')} →{t.get('predicate')}→ {t.get('object_id')}"
                notes.append((code, spo, verdict, note))
    return tally, total_rated, notes


def print_triplet_layer(scores):
    print("\n=== Triplet layer ===")
    coverage = [num((ev.get("triplets") or {}).get("coverage")) for ev in scores.values()]
    coverage = [c for c in coverage if c is not None]
    cmean, cstd = stats_for(coverage)
    print(f"  coverage (1–5): {fmt(cmean, cstd)}  (n={len(coverage)})")

    tally, total_rated, notes = collect_verdicts(scores)
    correct = tally.get("correct", 0)
    minor = sum(tally.get(c, 0) for c in MINOR_CATEGORIES)
    error = sum(tally.get(c, 0) for c in ERROR_CATEGORIES)
    other = tally.get("other", 0)

    def pct(n):
        return (n / total_rated * 100) if total_rated else 0.0

    print(f"  triplets rated: {total_rated}")
    print(f"  tiers: correct {correct} ({pct(correct):.1f}%) | "
          f"minor {minor} ({pct(minor):.1f}%) | "
          f"error {error} ({pct(error):.1f}%) | other {other} ({pct(other):.1f}%)")
    print("  failure codebook:")
    for tier, cats in (("correct", ["correct"]), ("minor", MINOR_CATEGORIES),
                       ("error", ERROR_CATEGORIES), ("other", ["other"])):
        for cat in cats:
            n = tally.get(cat, 0)
            print(f"    [{tier:<7}] {cat:<18} {n:>3}  ({pct(n):4.1f}%)")
    if notes:
        print("\n  flagged triplets (errors + notes):")
        for code, spo, verdict, note in notes:
            tail = f" — {note}" if note else ""
            print(f"    [{code}] {verdict}: {spo}{tail}")
    return tally, total_rated, correct


# ---------------------------------------------------------------------------
# Layer 4 — triplet→image render fidelity (objective, vs the known graph)
# ---------------------------------------------------------------------------

def collect_render(scores):
    """Tally per-fed-edge render verdicts + additions across events.
    Returns (tally, total_rated, total_fed, per_event_pct, additions_count, add_texts)."""
    tally = Counter()
    total_rated = total_fed = additions_count = 0
    per_event_pct, add_texts = [], []
    for code, ev in scores.items():
        ti = ev.get("triplet_image") or {}
        et = ti.get("edge_total")
        et = int(et) if isinstance(et, (int, float)) else (int(num(et)) if num(et) is not None else 0)
        total_fed += et
        yes = rated = 0
        for e in ti.get("per_edge", []) or []:
            r = (e.get("render") or "").strip()
            if not r:
                continue
            rated += 1
            tally[r] += 1
            if r == "yes":
                yes += 1
        if rated:
            total_rated += rated
            per_event_pct.append(yes / rated * 100)
        adds = parse_additions(ti)
        additions_count += len(adds)
        for a in adds:
            add_texts.append((code, addition_str(a)))
    return tally, total_rated, total_fed, per_event_pct, additions_count, add_texts


def print_render_fidelity(scores):
    print("\n=== Triplet→image render fidelity (per fed edge, vs the supplied graph) ===")
    tally, total_rated, total_fed, per_event_pct, additions_count, add_texts = collect_render(scores)

    def pct(n):
        return (n / total_rated * 100) if total_rated else 0.0

    yes = tally.get("yes", 0)
    print(f"  fed edges (Σ edge_total): {total_fed}; edges rated: {total_rated}")
    print(f"  edge fidelity (correctly drawn): {yes}/{total_rated} = {pct(yes):.1f}%")
    pemean, pestd = stats_for(per_event_pct)
    print(f"  per-event fidelity: {fmt(pemean, pestd)} %")
    for cat in RENDER_CATEGORIES:
        n = tally.get(cat, 0)
        print(f"    {cat:<16} {n:>3}  ({pct(n):4.1f}%)")
    print(f"  unsupported additions: {additions_count} total")
    for code, a in add_texts:
        print(f"    [{code}] {a}")


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------

def print_completeness(scores, expected_events: Optional[List[str]]):
    print("\n=== Completeness ===")
    print(f"  events with any scores: {len(scores)}")
    if expected_events:
        missing = [e for e in expected_events if e not in scores]
        if missing:
            print(f"  MISSING events ({len(missing)}): {', '.join(missing)}")
    # field-level gaps (subjective Likerts + density)
    gaps = []
    for code, ev in scores.items():
        for cond, fields in (("direct", ("faithfulness", "clarity", "node_count")),
                             ("triplet_image", ("faithfulness", "clarity"))):
            for f in fields:
                if num((ev.get(cond) or {}).get(f)) is None:
                    gaps.append(f"{code}:{cond}.{f}")
        if num((ev.get("triplets") or {}).get("coverage")) is None:
            gaps.append(f"{code}:triplets.coverage")
    if gaps:
        print(f"  unfilled Likert/density fields ({len(gaps)}):")
        for g in gaps:
            print(f"    - {g}")
    else:
        print("  all expected Likert/density fields filled.")
    # per-edge render coverage (how many fed edges got a render verdict)
    for code, ev in scores.items():
        ti = ev.get("triplet_image") or {}
        et = ti.get("edge_total")
        et = int(et) if isinstance(et, (int, float)) else (int(num(et)) if num(et) is not None else 0)
        rated = sum(1 for e in (ti.get("per_edge") or []) if (e.get("render") or "").strip())
        if et and rated < et:
            print(f"    render verdicts: {code} {rated}/{et} fed edges rated")


# ---------------------------------------------------------------------------
# LaTeX snippets
# ---------------------------------------------------------------------------

def latex_tables(scores) -> str:
    def cell(vals):
        return fmt(*stats_for(vals)).replace("±", r"$\pm$")

    f_d = cell(image_metric(scores, "direct", "faithfulness"))
    f_t = cell(image_metric(scores, "triplet_image", "faithfulness"))
    c_d = cell(image_metric(scores, "direct", "clarity"))
    c_t = cell(image_metric(scores, "triplet_image", "clarity"))

    dn = [num((ev.get("direct") or {}).get("node_count")) for ev in scores.values()]
    dn = [x for x in dn if x is not None]
    tn = []
    for ev in scores.values():
        t = (ev.get("triplet_image") or {}).get("node_total")
        t = float(t) if isinstance(t, (int, float)) else num(t)
        if t is not None:
            tn.append(t)
    nd = cell(dn)
    nt = cell(tn)

    t1 = "\n".join([
        r"\begin{table}[t]\centering",
        r"\caption{Final-image quality and density, direct vs.\ triplet$\rightarrow$image "
        r"(10 events, single rater). Faithfulness/clarity 1--5 (higher better).}",
        r"\label{tab:image}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & Direct & Triplet$\rightarrow$image \\",
        r"\midrule",
        rf"Faithfulness (1--5) & {f_d} & {f_t} \\",
        rf"Relation clarity (1--5) & {c_d} & {c_t} \\",
        rf"Nodes drawn & {nd} & {nt} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tally, total_rated, _notes = collect_verdicts(scores)

    def share(n):
        return f"{(n / total_rated * 100) if total_rated else 0:.1f}\\%"

    correct = tally.get("correct", 0)
    minor = sum(tally.get(c, 0) for c in MINOR_CATEGORIES)
    error = sum(tally.get(c, 0) for c in ERROR_CATEGORIES)

    rows = [rf"\textbf{{Correct}} & {correct} & {share(correct)} \\", r"\midrule",
            rf"\textbf{{Minor}} & {minor} & {share(minor)} \\"]
    rows += [rf"\quad {c.replace('_', ' ')} & {tally.get(c, 0)} & {share(tally.get(c, 0))} \\"
             for c in MINOR_CATEGORIES]
    rows += [r"\midrule", rf"\textbf{{Error}} & {error} & {share(error)} \\"]
    rows += [rf"\quad {c.replace('_', ' ')} & {tally.get(c, 0)} & {share(tally.get(c, 0))} \\"
             for c in ERROR_CATEGORIES]
    rows += [r"\midrule", rf"other & {tally.get('other', 0)} & {share(tally.get('other', 0))} \\"]

    t2 = "\n".join([
        r"\begin{table}[t]\centering",
        rf"\caption{{Per-triplet verdicts ({total_rated} triplets rated), grouped by "
        r"severity tier.}",
        r"\label{tab:codebook}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Verdict & Count & Share \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    # Table 3 — triplet→image render fidelity (per fed edge) + additions.
    rtally, r_rated, r_fed, _pe, add_count, _at = collect_render(scores)

    def rshare(n):
        return f"{(n / r_rated * 100) if r_rated else 0:.1f}\\%"

    rrows = [rf"{c.replace('_', ' ')} & {rtally.get(c, 0)} & {rshare(rtally.get(c, 0))} \\"
             for c in RENDER_CATEGORIES]
    t3 = "\n".join([
        r"\begin{table}[t]\centering",
        rf"\caption{{Triplet$\rightarrow$image render fidelity: how the model drew each of "
        rf"the {r_rated} fed edges, plus {add_count} unsupported additions.}}",
        r"\label{tab:render}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Rendered as & Count & Share \\",
        r"\midrule",
        *rrows,
        r"\midrule",
        rf"\textit{{unsupported additions}} & {add_count} & -- \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return t1 + "\n\n" + t2 + "\n\n" + t3


# ---------------------------------------------------------------------------
# Optional tidy CSV
# ---------------------------------------------------------------------------

def dump_csv(scores, path: Path):
    cols = ["event", "coverage",
            "direct_faithfulness", "direct_clarity", "direct_node_count",
            "ti_faithfulness", "ti_clarity", "ti_edge_total", "ti_node_total",
            "ti_edges_rated", "ti_correct_drawn", "ti_wrong_direction",
            "ti_wrong_connection", "ti_omitted", "ti_render_other", "ti_additions",
            "n_triplets_rated", "n_correct"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for code, ev in scores.items():
            tr = ev.get("triplets") or {}
            d = ev.get("direct") or {}
            ti = ev.get("triplet_image") or {}
            pts = tr.get("per_triplet", []) or []
            rated = [p for p in pts if (p.get("verdict") or "").strip()]
            ncorrect = sum(1 for p in rated if p.get("verdict") == "correct")
            edges = [e for e in (ti.get("per_edge") or []) if (e.get("render") or "").strip()]
            rc = Counter(e["render"] for e in edges)
            adds = parse_additions(ti)
            w.writerow([
                code, tr.get("coverage", ""),
                d.get("faithfulness", ""), d.get("clarity", ""), d.get("node_count", ""),
                ti.get("faithfulness", ""), ti.get("clarity", ""),
                ti.get("edge_total", ""), ti.get("node_total", ""),
                len(edges), rc.get("yes", 0), rc.get("wrong_direction", 0),
                rc.get("wrong_connection", 0), rc.get("omitted", 0), rc.get("other", 0),
                len(adds), len(rated), ncorrect,
            ])
    print(f"\nWrote tidy per-event CSV → {path}")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dataset stats (from raw_articles.json) + models/settings manifest
# ---------------------------------------------------------------------------

def event_article_stats(code: str) -> Optional[dict]:
    p = OUTPUT_DIR / code / "raw_articles.json"
    if not p.exists():
        return None
    arts = json.loads(p.read_text(encoding="utf-8"))
    bodies = [(a.get("body") or "") for a in arts]
    chars = [len(b) for b in bodies]
    words = [len(b.split()) for b in bodies]
    langs = sorted({(a.get("language") or "?") for a in arts})
    return {
        "n": len(arts),
        "mean_chars": (statistics.mean(chars) if chars else 0),
        "total_chars": sum(chars),
        "mean_words": (statistics.mean(words) if words else 0),
        "langs": langs,
    }


def print_dataset(expected_events: Optional[List[str]]):
    print("\n=== Dataset (input articles, from raw_articles.json) ===")
    codes = expected_events or sorted(p.name for p in OUTPUT_DIR.iterdir() if p.is_dir())
    rows, n_tot, ch_means = [], [], []
    print(f"  {'event':16}{'#art':>5}{'mean chars':>12}{'mean words':>12}  langs")
    for code in codes:
        s = event_article_stats(code)
        if not s:
            continue
        rows.append((code, s))
        n_tot.append(s["n"])
        ch_means.append(s["mean_chars"])
        print(f"  {code:16}{s['n']:>5}{round(s['mean_chars']):>12}{round(s['mean_words']):>12}"
              f"  {','.join(s['langs'])}")
    if rows:
        nm, nsd = stats_for(n_tot)
        cm, csd = stats_for(ch_means)
        print(f"  {'—':16}{'':>5}")
        print(f"  aggregate: {len(rows)} events, articles/event {fmt(nm, nsd)}, "
              f"mean body chars {fmt(cm, csd)}")
    return rows


def print_models():
    print("\n=== Models & settings (frozen for this run) ===")
    for stage, m in MODELS.items():
        print(f"  {stage}")
        print(f"    model:    {m['model']}")
        print(f"    settings: {m['settings']}")
        print(f"    source:   {m['script']}")


def latex_dataset_models(dataset_rows) -> str:
    drows = [rf"{code} & {s['n']} & {round(s['mean_chars'])} \\" for code, s in dataset_rows]
    if dataset_rows:
        nm, _ = stats_for([s["n"] for _, s in dataset_rows])
        cm, _ = stats_for([s["mean_chars"] for _, s in dataset_rows])
        drows += [r"\midrule", rf"\textit{{mean}} & {nm:.1f} & {round(cm)} \\"]
    t_data = "\n".join([
        r"\begin{table}[t]\centering",
        r"\caption{Per-event input: number of articles and mean article length "
        r"(characters). All articles Slovenian-language.}",
        r"\label{tab:dataset}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Event & \#articles & Mean chars \\",
        r"\midrule",
        *drows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    mrows = [rf"{stage} & \texttt{{{m['model']}}} \\" for stage, m in MODELS.items()]
    t_models = "\n".join([
        r"\begin{table}[t]\centering",
        r"\caption{Models per pipeline stage. Both image conditions use identical "
        r"settings; only the input (raw text vs.\ supplied triplets) differs.}",
        r"\label{tab:models}",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Stage & Model \\",
        r"\midrule",
        *mrows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return t_data + "\n\n" + t_models


def load_expected_events() -> Optional[List[str]]:
    p = Path(__file__).parent.parent / "events.txt"
    if not p.exists():
        return None
    return [ln.strip() for ln in p.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]


def main():
    ap = argparse.ArgumentParser(description="Aggregate seminar scores JSON")
    ap.add_argument("input", nargs="?", default=str(DEFAULT_INPUT),
                    help="path to seminar_scores.json (default: evaluation/seminar_scores.json)")
    ap.add_argument("--csv", help="also write a tidy per-event CSV to this path")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"No scores file at {path}. Export it from the comparison view first.")

    scores = load_scores(path)
    if not scores:
        raise SystemExit("Scores file has no scored events yet.")

    expected = load_expected_events()
    print_completeness(scores, expected)
    dataset_rows = print_dataset(expected)
    print_models()
    print_head_to_head(scores)
    print_density(scores)
    print_triplet_layer(scores)
    print_render_fidelity(scores)

    print("\n" + "=" * 78)
    print("LaTeX snippets (paste into the paper):")
    print("=" * 78)
    print(latex_tables(scores))
    print("\n" + latex_dataset_models(dataset_rows))

    if args.csv:
        dump_csv(scores, Path(args.csv))


if __name__ == "__main__":
    main()
