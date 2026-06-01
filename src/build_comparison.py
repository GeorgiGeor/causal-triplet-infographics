"""
Build the static comparison site for the seminar experiment.

Reads experiments/seminar_triplets/output/{event_code}/ for each event in
events.txt and emits:
  comparison/index.html
  comparison/{event_code}.html

Per-event page layout:
  - Header: event code + first article title (best-effort title proxy).
  - Three image columns side-by-side: direct.webp, triplet_image.webp,
    triplet_html.png. Each has a caption with the condition name.
  - <details> sections below for: rendered prompts, source articles, and
    the extracted entities + triplets table.

This is the seminar's evaluation harness and the source of paper figures.
Open `comparison/index.html` in a browser to score.
"""

import html as html_lib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

EXP_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = EXP_DIR / "output"
EVENTS_FILE = EXP_DIR / "events.txt"
COMP_DIR = Path(__file__).resolve().parent

# When set (via --frozen), the per-event scores baked into a read-only published
# build: {event_code: {triplets/direct/triplet_image ...}}. None = interactive.
FROZEN_SCORES: Dict[str, dict] | None = None

sys.path.insert(0, str(EXP_DIR))
from seminar_prompts import (  # noqa: E402
    DEFAULT_MIN_CONFIDENCE,
    is_below_confidence,
)


def load_events() -> List[str]:
    return [
        line.strip()
        for line in EVENTS_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def load_articles(event_code: str) -> list:
    p = OUTPUT_DIR / event_code / "raw_articles.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def load_triplets(event_code: str) -> dict:
    p = OUTPUT_DIR / event_code / "triplets.json"
    if not p.exists():
        return {"entities": [], "triplets": []}
    return json.loads(p.read_text(encoding="utf-8"))


def load_prompt(event_code: str, name: str) -> str:
    p = OUTPUT_DIR / event_code / f"{name}.txt"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def asset_rel_path(event_code: str, filename: str) -> str:
    """Path from comparison/{event_code}.html to ../output/{event_code}/<file>."""
    return f"../output/{event_code}/{filename}"


def _rendered_triplet_count(data: dict) -> int:
    """Triplets actually drawn (after the confidence filter)."""
    return sum(1 for t in (data.get("triplets") or []) if not is_below_confidence(t))


def _rendered_node_count(data: dict) -> int:
    """Distinct entity nodes drawn in the triplet→image render (post-filter)."""
    used: set = set()
    for t in (data.get("triplets") or []):
        if not is_below_confidence(t):
            used.add(t.get("subject_id"))
            used.add(t.get("object_id"))
    return len(used)


CSS = """
* { box-sizing: border-box; }
body { margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; color: #0f172a; background: #f8fafc; }
.container { max-width: 1600px; margin: 0 auto; padding: 32px 24px 80px; }
h1 { font-size: 22px; font-weight: 700; margin: 0 0 4px; }
.subtitle { color: #64748b; font-size: 14px; margin: 0 0 24px; }
.nav { display: flex; gap: 12px; font-size: 13px; margin-bottom: 24px; }
.nav a { color: #2563eb; text-decoration: none; }
.nav a:hover { text-decoration: underline; }
.event-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 18px; }
.event-card { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; text-decoration: none; color: inherit; transition: transform 0.1s, box-shadow 0.1s; }
.event-card:hover { box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08); transform: translateY(-1px); }
.event-card .code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: #64748b; }
.event-card .title { font-size: 14px; font-weight: 600; margin: 4px 0 10px; line-height: 1.3; }
.event-card .meta { font-size: 11px; color: #94a3b8; display: flex; gap: 10px; }
.event-card .thumb { width: 100%; aspect-ratio: 16 / 9; object-fit: cover; border-radius: 4px; background: #f1f5f9; margin-bottom: 8px; }

.columns { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 28px; }
.col { background: white; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }
.col-head { padding: 10px 14px; background: #f1f5f9; border-bottom: 1px solid #e2e8f0; font-size: 12px; font-weight: 600; color: #475569; letter-spacing: 0.04em; text-transform: uppercase; }
.col-head.direct { background: #dbeafe; color: #1d4ed8; }
.col-head.triplet-image { background: #dcfce7; color: #15803d; }
.col-head.triplet-html { background: #fef3c7; color: #b45309; }
.col img, .col .placeholder { width: 100%; aspect-ratio: 16 / 9; object-fit: contain; background: white; display: block; }
.col .placeholder { display: flex; align-items: center; justify-content: center; color: #94a3b8; font-size: 13px; }

details { background: white; border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 12px; }
details summary { padding: 12px 16px; cursor: pointer; font-weight: 600; font-size: 14px; user-select: none; }
details summary:hover { background: #f8fafc; }
details[open] summary { border-bottom: 1px solid #e2e8f0; }
.details-body { padding: 16px; max-height: 560px; overflow: auto; }

.prompt-tabs { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.prompt-block h4 { font-size: 12px; font-weight: 600; color: #475569; letter-spacing: 0.04em; text-transform: uppercase; margin: 0 0 6px; }
.prompt-block pre { background: #0f172a; color: #e2e8f0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; line-height: 1.5; padding: 12px; border-radius: 6px; max-height: 480px; overflow: auto; white-space: pre-wrap; word-break: break-word; }

.article-list { display: flex; flex-direction: column; gap: 16px; }
.article { border-left: 3px solid #cbd5e1; padding: 4px 14px; }
.article .pub { font-size: 12px; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.04em; }
.article .atitle { font-size: 14px; font-weight: 600; margin: 4px 0 6px; }
.article .body { font-size: 12px; color: #334155; line-height: 1.55; white-space: pre-wrap; }

table { border-collapse: collapse; width: 100%; font-size: 12px; }
th, td { padding: 8px 10px; border-bottom: 1px solid #e2e8f0; text-align: left; vertical-align: top; }
th { font-weight: 600; color: #475569; background: #f8fafc; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
td.predicate { font-weight: 600; color: #1d4ed8; }
.confidence { display: inline-block; padding: 1px 8px; font-size: 10px; border-radius: 999px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
.confidence.high { background: #dcfce7; color: #166534; }
.confidence.medium { background: #fef3c7; color: #92400e; }
.confidence.low { background: #fee2e2; color: #991b1b; }
.central-badge { display: inline-block; padding: 1px 6px; font-size: 10px; border-radius: 999px; background: #c2410c; color: white; font-weight: 600; margin-left: 6px; }
.evidence { color: #64748b; font-style: italic; font-size: 11px; }
tr.dropped { background: #f8fafc; opacity: 0.6; }
tr.dropped td.predicate { text-decoration: line-through; }
.render-excluded { display: inline-block; margin-left: 6px; padding: 1px 6px; font-size: 9px; border-radius: 999px; background: #e2e8f0; color: #475569; font-weight: 600; white-space: nowrap; }
.render-note { font-size: 11px; color: #64748b; margin: 0 0 8px; }

.event-nav { display: flex; justify-content: space-between; align-items: center; margin-top: 32px; padding-top: 24px; border-top: 1px solid #e2e8f0; font-size: 13px; }
.event-nav a { color: #2563eb; text-decoration: none; }
.event-nav a:hover { text-decoration: underline; }

/* --- Scoring UI --- */
.score-card { border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px 20px; margin: 24px 0; background: #fcfdff; }
.score-card > h3 { margin: 0 0 4px; font-size: 15px; color: #0f172a; }
.score-card .sc-sub { margin: 0 0 16px; font-size: 12px; color: #64748b; }
.score-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
@media (max-width: 900px) { .score-grid { grid-template-columns: 1fr; } }
.score-block { border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; background: white; }
.score-block h4 { margin: 0 0 12px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
.score-block.triplets h4 { color: #7c3aed; }
.score-block.direct h4 { color: #b91c1c; }
.score-block.triplet_image h4 { color: #15803d; }
.metric { margin-bottom: 12px; }
.metric-label { font-size: 12px; color: #334155; margin-bottom: 5px; font-weight: 600; }
.lk-hint { font-weight: 400; color: #94a3b8; margin-left: 6px; }
.likert { display: flex; gap: 4px; }
.lk { flex: 1; }
.lk input { position: absolute; opacity: 0; pointer-events: none; }
.lk span { display: block; text-align: center; padding: 6px 0; font-size: 13px; border: 1px solid #cbd5e1; border-radius: 6px; cursor: pointer; color: #475569; user-select: none; }
.lk input:checked + span { background: #2563eb; color: white; border-color: #2563eb; font-weight: 700; }
.lk:hover span { border-color: #2563eb; }
.num { width: 64px; padding: 5px 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px; }
.suffix { color: #64748b; font-size: 12px; margin-left: 6px; }
.metric.comment textarea { width: 100%; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 8px; font-size: 12px; font-family: inherit; resize: vertical; }
.objective-note { font-size: 11px; color: #64748b; margin: 0 0 10px; padding: 6px 10px; background: #f1f5f9; border-radius: 6px; }
.score-bar { position: sticky; top: 0; z-index: 20; display: flex; align-items: center; gap: 12px; padding: 10px 14px; margin: -8px 0 8px; background: #0f172a; color: #e2e8f0; border-radius: 10px; font-size: 12px; }
.score-bar button { font: inherit; padding: 5px 12px; border-radius: 6px; border: 0; cursor: pointer; font-weight: 600; }
.score-bar .exp { background: #22c55e; color: #052e16; }
.score-bar .imp { background: #334155; color: #e2e8f0; }
.score-bar .saved { margin-left: auto; color: #4ade80; opacity: 0; transition: opacity 0.3s; }
.score-bar .saved.show { opacity: 1; }
.score-bar input[type=file] { display: none; }
.event-card.scored { box-shadow: inset 4px 0 0 #22c55e; }
.event-card.scored .code::after { content: " ✓"; color: #16a34a; font-weight: 700; }
.verdict-cell { white-space: nowrap; }
.tr-verdict { font-size: 11px; padding: 3px 4px; border: 1px solid #cbd5e1; border-radius: 5px; max-width: 150px; }
.tr-note { display: block; margin-top: 4px; width: 150px; font-size: 11px; padding: 3px 5px; border: 1px solid #e2e8f0; border-radius: 5px; }
.score-bar .src { background: #1d4ed8; color: white; }

/* --- Source-text reading panel --- */
:root { --sp-width: min(620px, 46vw); }
.source-panel { position: fixed; top: 0; right: 0; height: 100vh; width: var(--sp-width); background: white; border-left: 1px solid #e2e8f0; box-shadow: -6px 0 28px rgba(15,23,42,0.14); transform: translateX(100%); transition: transform 0.2s ease; z-index: 40; display: flex; flex-direction: column; }
body.source-open .source-panel { transform: translateX(0); }
.source-panel .sp-head { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; border-bottom: 1px solid #e2e8f0; font-weight: 600; font-size: 13px; background: #f8fafc; }
.source-panel .sp-close { font: inherit; border: 0; background: #e2e8f0; color: #475569; border-radius: 6px; padding: 4px 10px; cursor: pointer; }
.source-panel .sp-body { padding: 16px; overflow-y: auto; flex: 1; }
body.source-open .container { margin-right: var(--sp-width); max-width: none; transition: margin-right 0.2s ease; }
@media (max-width: 1000px) { :root { --sp-width: 100vw; } body.source-open .container { margin-right: 0; } }
.sp-search { display: flex; gap: 6px; align-items: center; padding: 8px 16px; border-bottom: 1px solid #e2e8f0; background: white; }
.sp-search input { flex: 1; font-size: 12px; padding: 6px 8px; border: 1px solid #cbd5e1; border-radius: 6px; }
.sp-search #sp-search-count { font-size: 11px; color: #64748b; white-space: nowrap; min-width: 56px; text-align: right; }
.sp-search button { font-size: 12px; padding: 5px 9px; border: 1px solid #cbd5e1; border-radius: 6px; background: white; cursor: pointer; }
mark.sp-hit { background: #fde68a; color: inherit; border-radius: 2px; }
mark.sp-hit.cur { background: #f59e0b; color: white; }

/* clickable renders */
.col img.zoomable { cursor: zoom-in; }

/* focus mode: big image + its rating block, left of the source panel */
.focus-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: 38; background: rgba(248,250,252,0.99); display: none; flex-direction: column; padding: 18px 22px; overflow-y: auto; }
body.focus-open .focus-overlay { display: flex; }
body.focus-open.source-open .focus-overlay { right: var(--sp-width); }
.focus-overlay .fo-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.focus-overlay .fo-title { font-weight: 700; font-size: 16px; }
.focus-overlay .fo-close { font: inherit; font-weight: 600; border: 0; background: #0f172a; color: white; border-radius: 6px; padding: 6px 12px; cursor: pointer; }
.focus-overlay .fo-img { width: 100%; max-height: 52vh; object-fit: contain; background: white; border: 1px solid #e2e8f0; border-radius: 8px; }
.focus-overlay .fo-cols { display: flex; gap: 18px; align-items: flex-start; margin-top: 14px; flex-wrap: wrap; }
.focus-overlay .fo-score { flex: 1 1 340px; min-width: 300px; }
.focus-overlay .fo-score .score-block { max-width: 520px; }
.focus-overlay .fo-triplets { flex: 1 1 360px; min-width: 300px; display: none; background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; max-height: 60vh; overflow-y: auto; }
body.focus-cond-triplet_image .focus-overlay .fo-triplets { display: block; }
.fo-triplets h4 { margin: 0 0 10px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: #15803d; }
.fo-triplets .ftl-list { margin: 0; padding-left: 18px; display: flex; flex-direction: column; gap: 8px; }
.fo-triplets .ftl-list li { font-size: 12px; line-height: 1.4; }
.fo-triplets .ftl-list li.dropped { opacity: 0.55; }
.fo-triplets .spo b { color: #0f172a; } .fo-triplets .spo i { color: #1d4ed8; font-style: normal; font-weight: 600; }
.fo-triplets .re-control { display: flex; gap: 6px; margin-top: 4px; }
.fo-triplets .re-verdict { font-size: 11px; padding: 3px 4px; border: 1px solid #cbd5e1; border-radius: 5px; }
.fo-triplets .re-note { flex: 1; font-size: 11px; padding: 3px 6px; border: 1px solid #e2e8f0; border-radius: 5px; }
.ftl-additions { margin-top: 16px; border-top: 1px dashed #cbd5e1; padding-top: 12px; }
.ftl-additions .addition-row { display: flex; gap: 4px; margin-bottom: 6px; }
.ftl-additions .addition-input { flex: 1; min-width: 0; font-size: 11px; padding: 4px 6px; border: 1px solid #cbd5e1; border-radius: 5px; }
.ftl-additions .addition-rm { border: 1px solid #fecaca; background: #fef2f2; color: #b91c1c; border-radius: 6px; cursor: pointer; padding: 0 8px; }
.ftl-additions .add-btn { font-size: 12px; font-weight: 600; padding: 6px 12px; border: 1px dashed #16a34a; background: #f0fdf4; color: #15803d; border-radius: 6px; cursor: pointer; }
@media (max-width: 1000px) { body.focus-open.source-open .focus-overlay { right: 0; } }

/* --- Frozen (read-only published) view: render locked controls as static text/badges --- */
body.frozen .score-bar .exp,
body.frozen .score-bar .imp,
body.frozen .score-bar .saved,
body.frozen .add-btn,
body.frozen .addition-rm { display: none; }
body.frozen .score-bar::after { content: "read-only"; margin-left: auto; font-size: 11px; color: #94a3b8; }
body.frozen #progress { display: none; }
/* likert: show only the chosen value, as a static chip */
body.frozen .likert .lk input:not(:checked) + span { display: none; }
body.frozen .likert .lk input:checked + span { background: #2563eb; color: #fff; border-color: #2563eb; }
body.frozen .likert:not(:has(input:checked))::before { content: "—"; color: #94a3b8; }
/* dropdowns → plain text */
body.frozen select { appearance: none; -webkit-appearance: none; border: 0; background: none; padding: 0; font: inherit; font-weight: 600; color: #0f172a; opacity: 1; }
body.frozen select:disabled { color: #0f172a; }
/* number/text/textarea → borderless static */
body.frozen .num, body.frozen .addition-input, body.frozen .tr-note, body.frozen .re-note,
body.frozen .metric.comment textarea { border: 0; background: none; padding: 0; resize: none; color: #334155; font-weight: 600; opacity: 1; }
body.frozen .metric.comment textarea:disabled { color: #334155; }
body.frozen input:disabled, body.frozen textarea:disabled, body.frozen select:disabled { -webkit-text-fill-color: currentColor; }
"""


def render_articles_block(articles: list) -> str:
    parts = []
    for a in articles:
        pub = html_lib.escape(a.get("publisher_name") or "Unknown")
        title = html_lib.escape(a.get("title") or "")
        body = html_lib.escape(a.get("body") or "")
        parts.append(
            '<div class="article">'
            f'<div class="pub">{pub}</div>'
            f'<div class="atitle">{title}</div>'
            f'<div class="body">{body}</div>'
            "</div>"
        )
    return f'<div class="article-list">{"".join(parts)}</div>'


# Grouped verdict taxonomy. Values MUST match aggregate_scores.py.
VERDICT_TOP = [("", "— rate —"), ("correct", "✓ correct")]
VERDICT_MINOR = [
    ("misleading", "misleading"),
    ("awkward_wording", "awkward / unclear wording"),
    ("imprecise", "imprecise / oversimplified"),
]
VERDICT_ERROR = [
    ("wrong_direction", "wrong direction"),
    ("wrong_predicate", "wrong predicate"),
    ("wrong_endpoints", "wrong subject/object"),
    ("unsupported", "unsupported / hallucinated"),
    ("redundant", "redundant"),
]
VERDICT_BOTTOM = [("other", "other (see note)")]


def _verdict_control(idx: int) -> str:
    def opt(v, lbl):
        return f'<option value="{v}">{html_lib.escape(lbl)}</option>'
    parts = [opt(v, l) for v, l in VERDICT_TOP]
    parts.append('<optgroup label="Minor (imperfect but acceptable)">')
    parts += [opt(v, l) for v, l in VERDICT_MINOR]
    parts.append('</optgroup><optgroup label="Error (wrong)">')
    parts += [opt(v, l) for v, l in VERDICT_ERROR]
    parts.append("</optgroup>")
    parts += [opt(v, l) for v, l in VERDICT_BOTTOM]
    return (
        f'<select class="tr-verdict" data-tidx="{idx}">{"".join(parts)}</select>'
        f'<input class="tr-note" data-tidx="{idx}" type="text" placeholder="note / what was wrong…">'
    )


def render_triplets_table(data: dict) -> str:
    entities = data.get("entities") or []
    triplets = data.get("triplets") or []
    label_by_id = {e["id"]: e["label"] for e in entities}

    ent_rows = []
    for e in entities:
        badge = '<span class="central-badge">central</span>' if e.get("central") else ""
        ent_rows.append(
            f"<tr><td><code>{html_lib.escape(e['id'])}</code></td>"
            f"<td>{html_lib.escape(e['label'])}{badge}</td></tr>"
        )
    ent_table = (
        '<table><thead><tr><th>id</th><th>label</th></tr></thead>'
        f'<tbody>{"".join(ent_rows)}</tbody></table>'
    )

    trip_rows = []
    n_excluded = 0
    for i, t in enumerate(triplets):
        s_label = html_lib.escape(label_by_id.get(t.get("subject_id"), t.get("subject_id", "?")))
        o_label = html_lib.escape(label_by_id.get(t.get("object_id"), t.get("object_id", "?")))
        pred = html_lib.escape(t.get("predicate", ""))
        evidence = html_lib.escape(t.get("evidence", ""))
        conf = (t.get("confidence") or "").lower()
        # Mark rows excluded from the rendered image/HTML (kept in the table
        # for completeness, per the experiment's transparency goal).
        excluded = is_below_confidence(t)
        row_cls = ' class="dropped"' if excluded else ""
        excl_badge = (
            '<span class="render-excluded" title="below confidence threshold — '
            'not drawn in the image or HTML graph">⊘ ni v prikazu</span>'
            if excluded
            else ""
        )
        if excluded:
            n_excluded += 1
        trip_rows.append(
            f'<tr{row_cls} data-tidx="{i}"'
            f' data-s="{html_lib.escape(t.get("subject_id", ""))}"'
            f' data-p="{html_lib.escape(t.get("predicate", ""))}"'
            f' data-o="{html_lib.escape(t.get("object_id", ""))}">'
            f"<td>{s_label}</td>"
            f'<td class="predicate">{pred}</td>'
            f"<td>{o_label}{excl_badge}</td>"
            f'<td><span class="confidence {conf}">{html_lib.escape(conf)}</span></td>'
            f'<td class="evidence">{evidence}</td>'
            f'<td class="verdict-cell">{_verdict_control(i)}</td></tr>'
        )
    trip_table = (
        '<table class="triplets-table"><thead><tr>'
        "<th>subject</th><th>predicate</th><th>object</th><th>conf</th><th>evidence</th>"
        "<th>rating</th>"
        "</tr></thead>"
        f'<tbody>{"".join(trip_rows)}</tbody></table>'
    )

    note = ""
    if n_excluded:
        note = (
            f"<p class='render-note'>⊘ {n_excluded} triplet(s) below the "
            f"<code>{DEFAULT_MIN_CONFIDENCE}</code> confidence threshold are listed here "
            f"but excluded from the rendered image and HTML graph.</p>"
        )

    return (
        f"<h4 style='margin:0 0 10px; font-size:13px; color:#475569;'>Entities ({len(entities)})</h4>"
        f"{ent_table}"
        f"<h4 style='margin:18px 0 10px; font-size:13px; color:#475569;'>Triplets ({len(triplets)})</h4>"
        f"{note}{trip_table}"
    )


# How each fed triplet was DRAWN by the image model (vs the supplied graph).
# Distinct from the triplet's correctness-vs-articles verdict in the main table.
RENDER_OPTIONS = [
    ("", "— drawn? —"),
    ("yes", "✓ correctly drawn"),
    ("wrong_direction", "wrong direction"),
    ("wrong_connection", "wrong connection"),
    ("omitted", "omitted"),
    ("other", "other (note)"),
]


def _render_verdict_control(idx: int) -> str:
    opts = "".join(
        f'<option value="{v}">{html_lib.escape(l)}</option>' for v, l in RENDER_OPTIONS
    )
    return (
        f'<select class="re-verdict" data-tidx="{idx}">{opts}</select>'
        f'<input class="re-note" data-tidx="{idx}" type="text" placeholder="note…">'
    )


def render_focus_triplet_list(data: dict) -> str:
    """Interactive fed-triplet list for triplet→image focus mode. Each FED edge
    (non-dropped) gets a render verdict (yes/wrong direction/wrong connection/
    omitted/other); a +-button below collects free-text unsupported additions."""
    entities = data.get("entities") or []
    triplets = data.get("triplets") or []
    label_by_id = {e["id"]: e["label"] for e in entities}
    rendered = sum(1 for t in triplets if not is_below_confidence(t))
    items = []
    for i, t in enumerate(triplets):
        s = html_lib.escape(label_by_id.get(t.get("subject_id"), t.get("subject_id", "?")))
        o = html_lib.escape(label_by_id.get(t.get("object_id"), t.get("object_id", "?")))
        p = html_lib.escape(t.get("predicate", ""))
        ev = html_lib.escape(t.get("evidence", ""))
        dropped = is_below_confidence(t)
        if dropped:
            items.append(
                f'<li class="dropped" title="{ev}"><span class="spo"><b>{s}</b> '
                f'<i>{p}</i> <b>{o}</b></span> <span class="render-excluded">⊘ not fed</span></li>'
            )
        else:
            items.append(
                f'<li data-tidx="{i}" data-s="{html_lib.escape(t.get("subject_id",""))}"'
                f' data-p="{html_lib.escape(t.get("predicate",""))}"'
                f' data-o="{html_lib.escape(t.get("object_id",""))}" title="{ev}">'
                f'<span class="spo"><b>{s}</b> <i>{p}</i> <b>{o}</b></span>'
                f'<span class="re-control">{_render_verdict_control(i)}</span></li>'
            )
    return (
        f'<h4>Triplets fed to the image ({rendered} fed / {len(triplets)} total) — mark how each was drawn</h4>'
        f'<ol class="ftl-list">{"".join(items)}</ol>'
        '<div class="ftl-additions">'
        '<h4>Unsupported additions — extra subject·predicate·object in the image, not in the graph</h4>'
        '<div id="additions-list"></div>'
        '<button type="button" class="add-btn" onclick="addAddition()">+ add unsupported triplet</button>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# Scoring UI (rater enters scores; autosaved to localStorage; export to JSON)
# ---------------------------------------------------------------------------

def _likert(block: str, metric: str, label: str, hint: str = "") -> str:
    name = f"sc__{block}__{metric}"
    opts = "".join(
        f'<label class="lk"><input type="radio" name="{name}" value="{v}"><span>{v}</span></label>'
        for v in range(1, 6)
    )
    hint_html = f'<span class="lk-hint">{html_lib.escape(hint)}</span>' if hint else ""
    return (
        f'<div class="metric"><div class="metric-label">{html_lib.escape(label)}{hint_html}</div>'
        f'<div class="likert">{opts}</div></div>'
    )


def _number(block: str, metric: str, label: str, suffix: str = "", maxv=None) -> str:
    name = f"sc__{block}__{metric}"
    mx = f' max="{maxv}"' if maxv is not None else ""
    suf = f'<span class="suffix">{html_lib.escape(suffix)}</span>' if suffix else ""
    return (
        f'<div class="metric"><div class="metric-label">{html_lib.escape(label)}</div>'
        f'<input class="num" type="number" min="0"{mx} id="{name}" name="{name}">{suf}</div>'
    )


def _comment(block: str) -> str:
    name = f"sc__{block}__comment"
    return (
        '<div class="metric comment">'
        f'<textarea id="{name}" name="{name}" rows="2" placeholder="Comments…"></textarea></div>'
    )


def render_scoring_section(event_code: str, triplets_data: dict) -> str:
    """Rater form: triplets (correctness/coverage), each image (faithfulness/
    clarity), triplet→image objective counts. Autosaved client-side."""
    triplets = triplets_data.get("triplets") or []
    rendered = sum(1 for t in triplets if not is_below_confidence(t))

    triplets_block = (
        '<div class="score-block triplets"><h4>Triplets (graph vs articles)</h4>'
        + '<p class="objective-note">Per-triplet correctness is rated in the '
        '<b>Extracted triplets</b> table below (one verdict per row). '
        'Coverage is the only event-level triplet score:</p>'
        + _likert("triplets", "coverage", "Coverage (missing relations?)", "1–5")
        + _comment("triplets")
        + "</div>"
    )
    direct_block = (
        '<div class="score-block direct"><h4>Direct image</h4>'
        + _likert("direct", "faithfulness", "Faithfulness to articles", "1–5")
        + _likert("direct", "clarity", "Relation clarity", "1–5")
        + '<p class="objective-note">Density (the triplet→image node count is known automatically):</p>'
        + _number("direct", "node_count", "Nodes in image (count by eye)")
        + _comment("direct")
        + "</div>"
    )
    triplet_image_block = (
        '<div class="score-block triplet_image"><h4>Triplet → image</h4>'
        + _likert("triplet_image", "faithfulness", "Faithfulness to articles", "1–5")
        + _likert("triplet_image", "clarity", "Relation clarity", "1–5")
        + '<p class="objective-note">Per-edge render fidelity &amp; unsupported additions '
        "are rated in the fed-triplet list (click the image to open it).</p>"
        + _comment("triplet_image")
        + "</div>"
    )

    return (
        '<div class="score-card" id="score-card" data-event="'
        + html_lib.escape(event_code)
        + '" data-rendered="'
        + str(rendered)
        + '">'
        "<h3>Evaluation</h3>"
        '<p class="sc-sub">Faithfulness/clarity rate the final image (same basis for both '
        "pipelines). Edge fidelity &amp; unsupported additions are objective counts for the "
        "triplet→image render only. Autosaves in this browser; use Export for a JSON copy.</p>"
        '<div class="score-grid">'
        f"{triplets_block}{direct_block}{triplet_image_block}"
        "</div></div>"
    )


def render_score_bar(source_toggle: bool = False) -> str:
    src_btn = (
        '<button class="src" onclick="toggleSource()">📄 Source text</button>'
        if source_toggle
        else ""
    )
    return (
        '<div class="score-bar">'
        '<strong>Scores</strong>'
        '<button class="exp" onclick="exportScores()">⬇ Export JSON</button>'
        '<label class="imp" style="cursor:pointer;padding:5px 12px;border-radius:6px;">'
        '⬆ Import<input type="file" accept="application/json" onchange="importScores(event)"></label>'
        f"{src_btn}"
        '<span class="saved" id="saved-ind">✓ saved</span>'
        "</div>"
    )


SCORING_JS = r"""
<script>
(function(){
  const EVENT = "__EVENT_CODE__";
  const RENDERED = __RENDERED__;
  const NODE_TOTAL = __NODE_TOTAL__;
  const FROZEN = __FROZEN__;   // when set, a baked-in scores object → read-only view
  const KEY = "seminarScores";
  const FIELDS = {
    triplets: ["coverage","comment"],
    direct: ["faithfulness","clarity","comment","node_count"],
    triplet_image: ["faithfulness","clarity","comment"]
  };
  function loadAll(){ try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e){ return {}; } }
  function fieldName(b,m){ return "sc__"+b+"__"+m; }
  function readField(name){
    const r = document.querySelector('input[type=radio][name="'+CSS.escape(name)+'"]:checked');
    if (r) return r.value;
    const el = document.getElementById(name);
    return el ? el.value : "";
  }
  function setField(name, val){
    if (val === undefined || val === null || val === "") return;
    const radios = document.querySelectorAll('input[type=radio][name="'+CSS.escape(name)+'"]');
    if (radios.length){ radios.forEach(r => { if (r.value === String(val)) r.checked = true; }); return; }
    const el = document.getElementById(name);
    if (el) el.value = val;
  }
  function collectPerTriplet(){
    const rows = [];
    document.querySelectorAll('tr[data-tidx]').forEach(tr => {
      const idx = tr.getAttribute('data-tidx');
      const verdict = tr.querySelector('.tr-verdict');
      const note = tr.querySelector('.tr-note');
      const v = verdict ? verdict.value : "";
      const n = note ? note.value : "";
      if (v || n) rows.push({ idx: Number(idx), subject_id: tr.getAttribute('data-s'),
        predicate: tr.getAttribute('data-p'), object_id: tr.getAttribute('data-o'),
        verdict: v, note: n });
    });
    return rows;
  }
  function restorePerTriplet(rows){
    if (!rows) return;
    rows.forEach(r => {
      const tr = document.querySelector('tr[data-tidx="'+r.idx+'"]');
      if (!tr) return;
      const verdict = tr.querySelector('.tr-verdict');
      const note = tr.querySelector('.tr-note');
      if (verdict && r.verdict) verdict.value = r.verdict;
      if (note && r.note) note.value = r.note;
    });
  }
  // Per-edge render fidelity (how the image drew each FED triplet) + free-text
  // unsupported additions, both collected from the focus fed-triplet list.
  function collectPerEdge(){
    const rows = [];
    document.querySelectorAll('.ftl-list li[data-tidx]').forEach(li => {
      const sel = li.querySelector('.re-verdict');
      const note = li.querySelector('.re-note');
      const v = sel ? sel.value : "";
      const n = note ? note.value : "";
      if (v || n) rows.push({ idx: Number(li.getAttribute('data-tidx')),
        subject_id: li.getAttribute('data-s'), predicate: li.getAttribute('data-p'),
        object_id: li.getAttribute('data-o'), render: v, note: n });
    });
    return rows;
  }
  function restorePerEdge(rows){
    if (!rows) return;
    rows.forEach(r => {
      const li = document.querySelector('.ftl-list li[data-tidx="'+r.idx+'"]');
      if (!li) return;
      const sel = li.querySelector('.re-verdict');
      const note = li.querySelector('.re-note');
      if (sel && r.render) sel.value = r.render;
      if (note && r.note) note.value = r.note;
    });
  }
  // Each unsupported addition is itself a subject·predicate·object triplet
  // (an extra relation the image drew that isn't in the fed graph). A lone
  // invented node can be entered as subject only.
  function additionRow(value){
    value = value || {};
    const div = document.createElement('div');
    div.className = 'addition-row';
    function mk(key, ph){
      const i = document.createElement('input');
      i.type = 'text'; i.className = 'addition-input'; i.dataset.key = key; i.placeholder = ph;
      if (value[key]) i.value = value[key];
      return i;
    }
    const s = mk('subject', 'subject'), p = mk('predicate', 'predicate'), o = mk('object', 'object');
    const rm = document.createElement('button');
    rm.type = 'button'; rm.className = 'addition-rm'; rm.textContent = '✕';
    rm.onclick = function(){ div.remove(); save(); };
    div.appendChild(s); div.appendChild(p); div.appendChild(o); div.appendChild(rm);
    return div;
  }
  window.addAddition = function(value){
    const list = document.getElementById('additions-list');
    if (!list) return;
    const row = additionRow(typeof value === 'object' ? value : undefined);
    list.appendChild(row);
    if (!value) row.querySelector('input').focus();
  };
  function collectAdditions(){
    const rows = [];
    document.querySelectorAll('#additions-list .addition-row').forEach(r => {
      const get = k => { const el = r.querySelector('.addition-input[data-key="'+k+'"]'); return el ? el.value.trim() : ''; };
      const s = get('subject'), p = get('predicate'), o = get('object');
      if (s || p || o) rows.push({ subject: s, predicate: p, object: o });
    });
    return rows;
  }
  function restoreAdditions(arr){
    const list = document.getElementById('additions-list');
    if (!list) return;
    list.innerHTML = "";
    (arr || []).forEach(v => {
      if (typeof v === 'string') v = { subject: v, predicate: '', object: '' };
      list.appendChild(additionRow(v));
    });
  }
  function collect(){
    const out = {};
    for (const b in FIELDS){ out[b] = {}; FIELDS[b].forEach(m => { out[b][m] = readField(fieldName(b,m)); }); }
    out.triplets.per_triplet = collectPerTriplet();
    out.triplet_image.per_edge = collectPerEdge();
    out.triplet_image.additions = collectAdditions();
    out.triplet_image.edge_total = RENDERED;
    out.triplet_image.node_total = NODE_TOTAL;
    return out;
  }
  let savedTimer = null;
  function save(){
    const all = loadAll();
    all[EVENT] = collect();
    localStorage.setItem(KEY, JSON.stringify(all));
    const ind = document.getElementById("saved-ind");
    if (ind){ ind.classList.add("show"); clearTimeout(savedTimer); savedTimer = setTimeout(()=>ind.classList.remove("show"), 1200); }
  }
  function restore(){
    const e = FROZEN || loadAll()[EVENT]; if (!e) return;
    for (const b in FIELDS){ if (!e[b]) continue; FIELDS[b].forEach(m => setField(fieldName(b,m), e[b][m])); }
    if (e.triplets) restorePerTriplet(e.triplets.per_triplet);
    if (e.triplet_image){ restorePerEdge(e.triplet_image.per_edge); restoreAdditions(e.triplet_image.additions); }
  }
  function freeze(){
    // Read-only: lock every control and drop edit affordances.
    document.body.classList.add("frozen");
    document.querySelectorAll("#score-card, .triplets-table, .focus-overlay, .ftl-additions")
      .forEach(scope => scope.querySelectorAll("input, select, textarea, button")
        .forEach(el => { el.disabled = true; el.setAttribute("tabindex", "-1"); }));
  }
  window.exportScores = function(){
    const ALL_EVENTS = __ALL_EVENTS__;
    const stored = loadAll();
    const scores = {};
    ALL_EVENTS.forEach(c => { scores[c] = stored[c] || {}; });
    const payload = { _meta: { exported: new Date().toISOString(), rubric: "v1" }, scores: scores };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {type:"application/json"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "seminar_scores.json";
    a.click();
    URL.revokeObjectURL(a.href);
  };
  window.importScores = function(ev){
    const f = ev.target.files[0]; if (!f) return;
    const r = new FileReader();
    r.onload = function(){
      try {
        const parsed = JSON.parse(r.result);
        const scores = parsed.scores || parsed;
        localStorage.setItem(KEY, JSON.stringify(scores));
        location.reload();
      } catch(e){ alert("Invalid JSON: " + e.message); }
    };
    r.readAsText(f);
  };
  // Source-text reading panel (persists open/closed across event navigation)
  const SRC_KEY = "sourcePanelOpen";
  window.toggleSource = function(){
    const open = document.body.classList.toggle("source-open");
    try { localStorage.setItem(SRC_KEY, open ? "1" : "0"); } catch(e){}
  };
  window.closeSource = function(){
    document.body.classList.remove("source-open");
    try { localStorage.setItem(SRC_KEY, "0"); } catch(e){}
  };

  // Focus mode: click an image → big image top-left + its rating block below,
  // source panel open on the right. The score-block is MOVED (not copied) so
  // the inputs stay single-source and keep autosaving.
  let focusReturn = null;
  const FOCUS_TITLES = { direct: "Direct — text → image", triplet_image: "Triplet → image" };
  window.openFocus = function(cond){
    const img = document.querySelector('.col img[data-cond="'+cond+'"]');
    if (!img) return;
    document.getElementById("fo-title").textContent = FOCUS_TITLES[cond] || cond;
    document.getElementById("fo-img").src = img.getAttribute("src");
    const block = document.querySelector(".score-block." + cond);
    const holder = document.getElementById("fo-score");
    holder.innerHTML = "";
    if (block){
      focusReturn = { node: block, parent: block.parentNode, next: block.nextSibling };
      holder.appendChild(block);
    } else { focusReturn = null; }
    document.body.classList.remove("focus-cond-direct", "focus-cond-triplet_image");
    document.body.classList.add("focus-open", "source-open", "focus-cond-" + cond);
    try { localStorage.setItem(SRC_KEY, "1"); } catch(e){}
  };
  window.closeFocus = function(){
    if (focusReturn){ focusReturn.parent.insertBefore(focusReturn.node, focusReturn.next); focusReturn = null; }
    document.body.classList.remove("focus-open", "focus-cond-direct", "focus-cond-triplet_image");
  };

  // Source-panel in-place search (highlight + cycle through matches).
  let spHits = [], spIdx = -1;
  function spBody(){ return document.querySelector(".source-panel .sp-body"); }
  function spClear(body){
    body.querySelectorAll("mark.sp-hit").forEach(m => {
      m.replaceWith(document.createTextNode(m.textContent));
    });
    body.normalize();
  }
  window.spSearch = function(){
    const body = spBody(); if (!body) return;
    const q = document.getElementById("sp-search-input").value;
    spClear(body); spHits = []; spIdx = -1;
    const countEl = document.getElementById("sp-search-count");
    if (!q){ countEl.textContent = ""; return; }
    const ql = q.toLowerCase();
    const walker = document.createTreeWalker(body, NodeFilter.SHOW_TEXT, null);
    const nodes = []; let nn; while ((nn = walker.nextNode())) nodes.push(nn);
    nodes.forEach(node => {
      const text = node.nodeValue, lower = text.toLowerCase();
      if (!lower.includes(ql)) return;
      const frag = document.createDocumentFragment(); let i = 0, idx;
      while ((idx = lower.indexOf(ql, i)) !== -1){
        if (idx > i) frag.appendChild(document.createTextNode(text.slice(i, idx)));
        const mark = document.createElement("mark"); mark.className = "sp-hit";
        mark.textContent = text.slice(idx, idx + q.length);
        frag.appendChild(mark); spHits.push(mark); i = idx + q.length;
      }
      if (i < text.length) frag.appendChild(document.createTextNode(text.slice(i)));
      node.parentNode.replaceChild(frag, node);
    });
    if (spHits.length) spGo(0);
    else countEl.textContent = "no matches";
  };
  function spGo(i){
    if (!spHits.length) return;
    if (spIdx >= 0 && spHits[spIdx]) spHits[spIdx].classList.remove("cur");
    spIdx = (i + spHits.length) % spHits.length;
    const m = spHits[spIdx]; m.classList.add("cur"); m.scrollIntoView({ block: "center" });
    document.getElementById("sp-search-count").textContent = (spIdx + 1) + " / " + spHits.length;
  }
  window.spSearchNext = function(){ spGo(spIdx + 1); };
  window.spSearchPrev = function(){ spGo(spIdx - 1); };

  document.addEventListener("DOMContentLoaded", function(){
    restore();
    if (localStorage.getItem(SRC_KEY) === "1") document.body.classList.add("source-open");
    document.querySelectorAll('.col img[data-cond]').forEach(img => {
      img.addEventListener("click", () => window.openFocus(img.getAttribute("data-cond")));
    });
    const si = document.getElementById("sp-search-input");
    if (si){
      let t; si.addEventListener("input", () => { clearTimeout(t); t = setTimeout(window.spSearch, 120); });
      si.addEventListener("keydown", e => { if (e.key === "Enter"){ e.preventDefault(); e.shiftKey ? spSearchPrev() : spSearchNext(); } });
    }
    document.addEventListener("keydown", function(e){
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "f"){
        e.preventDefault();
        document.body.classList.add("source-open");
        try { localStorage.setItem(SRC_KEY, "1"); } catch(err){}
        const inp = document.getElementById("sp-search-input");
        if (inp){ inp.focus(); inp.select(); }
      } else if (e.key === "Escape"){
        if (document.body.classList.contains("focus-open")) window.closeFocus();
      }
    });
    if (FROZEN){ freeze(); return; }
    // Listen at document level so the Evaluation card, the per-triplet controls,
    // and the moved score-block (in focus overlay) all trigger autosave.
    document.addEventListener("input", function(e){ if (e.target.closest("#score-card, .triplets-table, .focus-overlay")) save(); });
    document.addEventListener("change", function(e){ if (e.target.closest("#score-card, .triplets-table, .focus-overlay")) save(); });
  });
})();
</script>
"""


def render_event_page(
    event_code: str,
    idx: int,
    total: int,
    events: List[str],
    articles: list,
    triplets_data: dict,
) -> str:
    title_proxy = articles[0].get("title") if articles else event_code

    direct_path = OUTPUT_DIR / event_code / "direct.webp"
    triplet_image_path = OUTPUT_DIR / event_code / "triplet_image.webp"
    triplet_html_png = OUTPUT_DIR / event_code / "triplet_html.png"
    triplet_html_doc = OUTPUT_DIR / event_code / "triplet_html.html"

    def img_or_placeholder(path: Path, filename: str, label: str, cond: str = "") -> str:
        if path.exists():
            attr = f' class="zoomable" data-cond="{cond}" title="Click to focus + rate"' if cond else ""
            return f'<img{attr} src="{asset_rel_path(event_code, filename)}" alt="{label} for {event_code}">'
        return f'<div class="placeholder">{label} not generated yet</div>'

    direct_block = img_or_placeholder(direct_path, "direct.webp", "Direct baseline", "direct")
    triplet_image_block = img_or_placeholder(
        triplet_image_path, "triplet_image.webp", "Triplet→image", "triplet_image"
    )
    if triplet_html_png.exists():
        triplet_html_block = (
            f'<a href="{asset_rel_path(event_code, "triplet_html.html")}" target="_blank">'
            f'<img src="{asset_rel_path(event_code, "triplet_html.png")}" alt="HTML render for {event_code}">'
            "</a>"
        )
    elif triplet_html_doc.exists():
        triplet_html_block = (
            f'<a class="placeholder" href="{asset_rel_path(event_code, "triplet_html.html")}" '
            'target="_blank">Open HTML render (no screenshot)</a>'
        )
    else:
        triplet_html_block = '<div class="placeholder">HTML render not generated</div>'

    extraction_prompt = load_prompt(event_code, "extraction_prompt")
    direct_prompt = load_prompt(event_code, "direct_prompt")
    triplet_prompt = load_prompt(event_code, "triplet_image_prompt")

    prev_link = (
        f'<a href="{events[idx - 1]}.html">&larr; {events[idx - 1]}</a>'
        if idx > 0 else "<span></span>"
    )
    next_link = (
        f'<a href="{events[idx + 1]}.html">{events[idx + 1]} &rarr;</a>'
        if idx + 1 < total else "<span></span>"
    )

    frozen_ev = (FROZEN_SCORES or {}).get(event_code)
    frozen_json = json.dumps(frozen_ev) if FROZEN_SCORES else "null"
    body_class = ' class="frozen"' if FROZEN_SCORES else ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html_lib.escape(event_code)} &middot; comparison</title>
<style>{CSS}</style>
</head>
<body{body_class}>
<div class="container">
  <div class="nav">
    <a href="index.html">&larr; All events</a>
    <span>&middot;</span>
    <span>{idx + 1} / {total}</span>
  </div>
  {render_score_bar(source_toggle=True)}
  <h1>{html_lib.escape(event_code)}</h1>
  <p class="subtitle">{html_lib.escape(title_proxy or "")}</p>

  <div class="columns">
    <div class="col">
      <div class="col-head direct">Direct &mdash; text &rarr; image</div>
      {direct_block}
    </div>
    <div class="col">
      <div class="col-head triplet-image">Triplet &rarr; image</div>
      {triplet_image_block}
    </div>
    <div class="col">
      <div class="col-head triplet-html">Triplet &rarr; HTML/CSS</div>
      {triplet_html_block}
    </div>
  </div>

  {render_scoring_section(event_code, triplets_data)}

  <details>
    <summary>Prompts</summary>
    <div class="details-body">
      <div class="prompt-block">
        <h4>Triplet extraction prompt &mdash; text &rarr; triplets (stage 1, both triplet conditions)</h4>
        <pre>{html_lib.escape(extraction_prompt) or "(not generated)"}</pre>
      </div>
      <div class="prompt-tabs">
        <div class="prompt-block">
          <h4>Direct prompt &mdash; fed to Gemini 3 Pro Image Preview</h4>
          <pre>{html_lib.escape(direct_prompt)}</pre>
        </div>
        <div class="prompt-block">
          <h4>Triplet&rarr;image prompt &mdash; fed to Gemini 3 Pro Image Preview</h4>
          <pre>{html_lib.escape(triplet_prompt)}</pre>
        </div>
      </div>
    </div>
  </details>

  <details open>
    <summary>Extracted triplets</summary>
    <div class="details-body">
      {render_triplets_table(triplets_data)}
    </div>
  </details>

  <div class="event-nav">
    {prev_link}
    {next_link}
  </div>
</div>
<div class="focus-overlay" id="focus-overlay">
  <div class="fo-head"><span class="fo-title" id="fo-title"></span>
    <button class="fo-close" onclick="closeFocus()" title="Close (Esc)">✕ Close</button></div>
  <img class="fo-img" id="fo-img" alt="focused render">
  <div class="fo-cols">
    <div class="fo-score" id="fo-score"></div>
    <div class="fo-triplets">{render_focus_triplet_list(triplets_data)}</div>
  </div>
</div>
<aside class="source-panel" aria-label="Source articles">
  <div class="sp-head"><span>Source articles ({len(articles)})</span>
    <button class="sp-close" onclick="closeSource()" title="Close">✕</button></div>
  <div class="sp-search">
    <input type="text" id="sp-search-input" placeholder="Search source… (⌘/Ctrl+F)" autocomplete="off">
    <span id="sp-search-count"></span>
    <button onclick="spSearchPrev()" title="Previous">↑</button>
    <button onclick="spSearchNext()" title="Next">↓</button>
  </div>
  <div class="sp-body">{render_articles_block(articles)}</div>
</aside>
{SCORING_JS.replace("__EVENT_CODE__", event_code).replace("__RENDERED__", str(_rendered_triplet_count(triplets_data))).replace("__NODE_TOTAL__", str(_rendered_node_count(triplets_data))).replace("__ALL_EVENTS__", json.dumps(events)).replace("__FROZEN__", frozen_json)}
</body>
</html>"""


INDEX_JS = r"""
<script>
(function(){
  const KEY = "seminarScores";
  function loadAll(){ try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e){ return {}; } }
  window.exportScores = function(){
    const ALL_EVENTS = __ALL_EVENTS__;
    const stored = loadAll();
    const scores = {};
    ALL_EVENTS.forEach(c => { scores[c] = stored[c] || {}; });
    const payload = { _meta: { exported: new Date().toISOString(), rubric: "v1" }, scores: scores };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {type:"application/json"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "seminar_scores.json"; a.click();
    URL.revokeObjectURL(a.href);
  };
  window.importScores = function(ev){
    const f = ev.target.files[0]; if (!f) return;
    const r = new FileReader();
    r.onload = function(){
      try { const p = JSON.parse(r.result); localStorage.setItem(KEY, JSON.stringify(p.scores || p)); location.reload(); }
      catch(e){ alert("Invalid JSON: " + e.message); }
    };
    r.readAsText(f);
  };
  document.addEventListener("DOMContentLoaded", function(){
    const all = loadAll();
    const cards = document.querySelectorAll(".event-card");
    let n = 0;
    cards.forEach(c => { if (all[c.getAttribute("data-code")]) { c.classList.add("scored"); n++; } });
    const el = document.getElementById("progress");
    if (el) el.textContent = n + " / " + cards.length + " events have scores saved in this browser";
  });
})();
</script>
"""


def render_index(events: List[str], titles_by_code: Dict[str, str]) -> str:
    cards = []
    for code in events:
        triplet_img = OUTPUT_DIR / code / "triplet_image.webp"
        direct_img = OUTPUT_DIR / code / "direct.webp"
        thumb_path = triplet_img if triplet_img.exists() else direct_img
        thumb_html = (
            f'<img class="thumb" src="../output/{code}/{thumb_path.name}" alt="">'
            if thumb_path.exists() else '<div class="thumb"></div>'
        )
        title = titles_by_code.get(code, code)
        articles = load_articles(code)
        triplets_data = load_triplets(code)
        cards.append(
            f'<a class="event-card" data-code="{html_lib.escape(code)}" href="{code}.html">'
            f"{thumb_html}"
            f'<div class="code">{html_lib.escape(code)}</div>'
            f'<div class="title">{html_lib.escape(title)}</div>'
            '<div class="meta">'
            f"<span>{len(articles)} articles</span>"
            f"<span>{len(triplets_data.get('entities', []))} entities</span>"
            f"<span>{len(triplets_data.get('triplets', []))} triplets</span>"
            "</div>"
            "</a>"
        )
    body_class = ' class="frozen"' if FROZEN_SCORES else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Seminar comparison &middot; all events</title>
<style>{CSS}</style>
</head>
<body{body_class}>
<div class="container">
  <h1>Seminar triplet comparison</h1>
  <p class="subtitle">10 events &middot; three conditions per event (direct, triplet&rarr;image, triplet&rarr;HTML/CSS)</p>
  {render_score_bar()}
  <p class="subtitle" id="progress" style="margin-top:8px;"></p>
  <div class="event-grid">
    {"".join(cards)}
  </div>
</div>
{INDEX_JS.replace("__ALL_EVENTS__", json.dumps(events))}
</body>
</html>"""


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build the comparison site")
    ap.add_argument("--frozen", help="path to seminar_scores.json → read-only baked-in build")
    ap.add_argument("--out", help="output directory for the HTML (default: comparison/)")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else COMP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.frozen:
        global FROZEN_SCORES
        data = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
        FROZEN_SCORES = data.get("scores", data)

    events = load_events()
    titles_by_code = {}
    for code in events:
        articles = load_articles(code)
        if articles:
            titles_by_code[code] = articles[0].get("title") or code

    for idx, code in enumerate(events):
        articles = load_articles(code)
        triplets_data = load_triplets(code)
        page = render_event_page(code, idx, len(events), events, articles, triplets_data)
        (out_dir / f"{code}.html").write_text(page, encoding="utf-8")

    index = render_index(events, titles_by_code)
    (out_dir / "index.html").write_text(index, encoding="utf-8")

    mode = "frozen read-only" if args.frozen else "interactive"
    print(f"OK ({mode}): {out_dir}/index.html + {len(events)} per-event pages")


if __name__ == "__main__":
    main()
