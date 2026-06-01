"""
Interactive HTML/CSS render of extracted triplets — the seminar's
"what you lose without GenAI" condition with the inspectability that the
triplet representation was supposed to enable.

For each event:
  1. Build a DOT graph from the extracted entities + triplets.
  2. Run graphviz `dot -Tjson0 -Grankdir=LR` to compute layered node
     positions and bezier edge paths. Graphviz handles crossing
     minimisation, edge routing, and label placement.
  3. Emit an HTML page with:
     - Left pane: SVG graph using the dot-computed positions. Nodes are
       rectangles, edges are paths with arrowheads + short predicate
       labels.
     - Right pane: sidebar with all entities listed. Clicking a node
       (or a sidebar entry) drills into that entity: shows the human
       label, the involving triplets with subject/predicate/object,
       supporting evidence, and the confidence badge.
     - All UI text in Slovenian to match the production infographic
       output language.
  4. Take a headless-Chrome screenshot for the static PNG that goes into
     the paper. Initial state shows the entity list, so the screenshot
     is information-rich rather than empty.

Output (per event):
  experiments/seminar_triplets/output/{event_code}/
    triplet_html.html   - interactive page (open in browser)
    triplet_html.png    - default-state screenshot for paper figures
"""

import argparse
import html as html_lib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seminar_prompts import (  # noqa: E402
    DEFAULT_MIN_CONFIDENCE,
    filter_triplets_by_confidence,
)

OUTPUT_DIR = Path(__file__).parent / "output"
EVENTS_FILE = Path(__file__).parent / "events.txt"

NODE_W = 180
NODE_H = 56
NODE_PAD_X = 14
SCREENSHOT_W = 1600
SCREENSHOT_H = 950

# Slovenian UI strings — everything user-facing.
UI = {
    "title": "Vzročna mreža",
    "hint": "Klikni vozlišče ali povezavo za podrobnosti.",
    "entities_header": "Entitete",
    "central_badge": "centralna",
    "involves_header": "Sodeluje v {n} razmerjih",
    "select_hint": "Izberi entiteto na sliki ali na seznamu, da prikažeš njene povezave.",
    "predicate": "predikat",
    "subject": "subjekt",
    "object": "objekt",
    "evidence": "dokaz",
    "confidence": "zaupanje",
    "high": "visoko",
    "medium": "srednje",
    "low": "nizko",
    "back": "Nazaj na vse entitete",
}


# ---------------------------------------------------------------------------
# Graphviz layout
# ---------------------------------------------------------------------------

def _escape_dot(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def compute_layout(entities: List[dict], triplets: List[dict]) -> Tuple[dict, dict, float, float]:
    """Run graphviz dot and return (node_pos, edge_paths, canvas_w, canvas_h).

    node_pos: {entity_id: (x, y)} in SVG coordinates (Y already flipped).
    edge_paths: {(subject_id, object_id, triplet_index): "M x,y C cp1 cp2 endx,endy"}
    """
    # graphviz wants point units; we set a reasonable node size.
    width_in = NODE_W / 72
    height_in = NODE_H / 72
    lines = ["digraph G {"]
    lines.append('  graph [rankdir=LR, nodesep=0.35, ranksep=1.1, splines=true, margin=0.2];')
    lines.append(
        f'  node [shape=box, fontname="Helvetica", fontsize=11, '
        f'width={width_in:.3f}, height={height_in:.3f}, fixedsize=true];'
    )
    lines.append('  edge [fontname="Helvetica", fontsize=9];')

    for e in entities:
        lines.append(f'  "{_escape_dot(e["id"])}" [label="{_escape_dot(e["label"])}"];')
    for i, t in enumerate(triplets):
        s, o = t.get("subject_id"), t.get("object_id")
        if not s or not o:
            continue
        pred = _escape_dot(t.get("predicate", ""))
        # Use comment to carry the triplet index so we can map dot edges back.
        lines.append(f'  "{_escape_dot(s)}" -> "{_escape_dot(o)}" [label="{pred}", id="t{i}"];')
    lines.append("}")
    dot_source = "\n".join(lines)

    proc = subprocess.run(
        ["dot", "-Tjson0"],
        input=dot_source,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"dot failed: {proc.stderr}")
    layout = json.loads(proc.stdout)

    bbox = layout.get("bb", "0,0,1000,600").split(",")
    cx0, cy0, cx1, cy1 = (float(v) for v in bbox)
    canvas_w = cx1 - cx0
    canvas_h = cy1 - cy0

    def flip_y(y: float) -> float:
        return canvas_h - (y - cy0)

    node_pos: Dict[str, Tuple[float, float]] = {}
    objects = layout.get("objects", []) or []
    for obj in objects:
        name = obj.get("name")
        pos = obj.get("pos")
        if name and pos:
            x_str, y_str = pos.split(",")
            x = float(x_str) - cx0
            y = flip_y(float(y_str))
            node_pos[name] = (x, y)

    edge_paths: Dict[str, str] = {}
    for edge in layout.get("edges", []) or []:
        eid = edge.get("id")  # we set "t<idx>"
        # `pos` is "e,endx,endy spline1x,spline1y spline2x,spline2y ..."
        pos = edge.get("pos", "")
        if not pos or not eid:
            continue
        tokens = pos.split()
        end_point = None
        ctrl_points: List[Tuple[float, float]] = []
        for tok in tokens:
            if tok.startswith("e,"):
                ex, ey = tok[2:].split(",")
                end_point = (float(ex) - cx0, flip_y(float(ey)))
            elif "," in tok:
                px, py = tok.split(",")
                ctrl_points.append((float(px) - cx0, flip_y(float(py))))
        if not ctrl_points:
            continue
        # Build SVG path: start with first ctrl point, then bezier curves through the rest,
        # finally close with the arrowhead endpoint.
        path = [f"M {ctrl_points[0][0]:.2f},{ctrl_points[0][1]:.2f}"]
        i = 1
        while i + 2 < len(ctrl_points):
            c1, c2, p = ctrl_points[i], ctrl_points[i + 1], ctrl_points[i + 2]
            path.append(f"C {c1[0]:.2f},{c1[1]:.2f} {c2[0]:.2f},{c2[1]:.2f} {p[0]:.2f},{p[1]:.2f}")
            i += 3
        # Drop the synthetic ending line; the dot bezier already reaches the node border.
        if end_point:
            path.append(f"L {end_point[0]:.2f},{end_point[1]:.2f}")
        edge_paths[eid] = " ".join(path)

    return node_pos, edge_paths, canvas_w, canvas_h


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; color: #0f172a; background: #f8fafc; }
.shell { display: grid; grid-template-columns: 1fr 320px; min-height: 100vh; }
.graph-pane { padding: 24px 12px 24px 24px; }
header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
header h1 { margin: 0; font-size: 13px; font-weight: 600; color: #475569; letter-spacing: 0.07em; text-transform: uppercase; }
header .code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; color: #94a3b8; }
.hint { color: #64748b; font-size: 12px; margin: 0 0 12px; }

.svg-wrap { background: white; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }
svg { display: block; width: 100%; height: auto; }

.node rect { fill: #1e3a5f; stroke: #0f172a; stroke-width: 0.5; cursor: pointer; transition: fill 0.15s; }
.node.central rect { fill: #c2410c; }
.node.selected rect { stroke: #fbbf24; stroke-width: 3; }
.node.dimmed { opacity: 0.25; }
.node-label { fill: #fff; font-size: 11px; font-weight: 600; text-anchor: middle; alignment-baseline: middle; pointer-events: none; }

.edge path { fill: none; stroke: #94a3b8; stroke-width: 1.4; transition: stroke 0.15s, stroke-width 0.15s; cursor: pointer; }
.edge.highlight path { stroke: #2563eb; stroke-width: 2.5; }
.edge.dimmed path { opacity: 0.15; }
.edge-hit { fill: none; stroke: transparent; stroke-width: 12; cursor: pointer; }
.edge-label-wrap rect { fill: white; }
.edge-label { fill: #475569; font-size: 9px; font-weight: 600; text-anchor: middle; letter-spacing: 0.04em; pointer-events: none; }
.edge.highlight .edge-label { fill: #1d4ed8; }

.sidebar { background: white; border-left: 1px solid #e2e8f0; padding: 24px 20px; overflow: auto; max-height: 100vh; position: sticky; top: 0; }
.sidebar h2 { font-size: 11px; font-weight: 600; color: #475569; letter-spacing: 0.07em; text-transform: uppercase; margin: 0 0 12px; }
.entity-list { list-style: none; padding: 0; margin: 0 0 20px; display: flex; flex-direction: column; gap: 4px; }
.entity-list li { padding: 8px 10px; border-radius: 6px; cursor: pointer; font-size: 13px; border: 1px solid transparent; transition: background 0.15s; display: flex; justify-content: space-between; gap: 6px; align-items: baseline; }
.entity-list li:hover { background: #f1f5f9; }
.entity-list li.selected { background: #eff6ff; border-color: #93c5fd; }
.entity-list li.central { font-weight: 600; }
.entity-list .badge { background: #c2410c; color: white; font-size: 10px; padding: 1px 6px; border-radius: 999px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }
.entity-list .count { color: #94a3b8; font-size: 11px; }

.empty-state { color: #94a3b8; font-size: 12px; line-height: 1.5; }
.detail .back { background: none; border: none; color: #2563eb; font-size: 12px; cursor: pointer; padding: 0; margin-bottom: 8px; }
.detail .back:hover { text-decoration: underline; }
.detail h3 { margin: 0 0 4px; font-size: 16px; }
.detail .meta { color: #64748b; font-size: 12px; margin: 0 0 16px; }

.relations { display: flex; flex-direction: column; gap: 10px; }
.relation { padding: 10px 12px; background: #f8fafc; border-radius: 6px; border-left: 3px solid #cbd5e1; font-size: 12px; }
.relation.highlight { border-left-color: #2563eb; background: #eff6ff; }
.relation .triple { font-size: 13px; }
.relation .triple .predicate { color: #1d4ed8; font-weight: 600; padding: 0 4px; }
.relation .triple .self { font-weight: 600; color: #0f172a; }
.relation .ev { color: #475569; margin-top: 4px; font-style: italic; line-height: 1.4; }
.confidence { display: inline-block; padding: 1px 8px; font-size: 10px; border-radius: 999px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 6px; }
.confidence.high { background: #dcfce7; color: #166534; }
.confidence.medium { background: #fef3c7; color: #92400e; }
.confidence.low { background: #fee2e2; color: #991b1b; }
"""


def _slovenian_confidence(c: str) -> str:
    return {"high": UI["high"], "medium": UI["medium"], "low": UI["low"]}.get(
        (c or "").lower(), html_lib.escape(c or "")
    )


def _wrap_label(label: str, max_chars: int = 22) -> List[str]:
    if len(label) <= max_chars:
        return [label]
    mid = len(label) // 2
    split = label.rfind(" ", 0, mid + 6)
    if split <= 0:
        return [label[:max_chars] + "…"]
    return [label[:split], label[split + 1:]]


def render_node_svg(entity: dict, x: float, y: float) -> str:
    eid = html_lib.escape(entity["id"])
    label = entity.get("label", entity["id"])
    central_cls = " central" if entity.get("central") else ""
    rect_x = x - NODE_W / 2
    rect_y = y - NODE_H / 2
    lines = _wrap_label(label)
    if len(lines) == 1:
        text_svg = f'<text class="node-label" x="{x:.1f}" y="{y + 4:.1f}">{html_lib.escape(lines[0])}</text>'
    else:
        text_svg = (
            f'<text class="node-label" x="{x:.1f}" y="{y - 4:.1f}">{html_lib.escape(lines[0])}</text>'
            f'<text class="node-label" x="{x:.1f}" y="{y + 12:.1f}">{html_lib.escape(lines[1])}</text>'
        )
    return (
        f'<g class="node{central_cls}" data-entity-id="{eid}" id="node-{eid}">'
        f'<rect x="{rect_x:.1f}" y="{rect_y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="6" ry="6"/>'
        f"{text_svg}"
        "</g>"
    )


def render_edge_svg(triplet: dict, triplet_idx: int, path: str, node_pos: Dict[str, Tuple[float, float]]) -> str:
    s, o = triplet.get("subject_id"), triplet.get("object_id")
    predicate = (triplet.get("predicate") or "").upper()
    eid = f"t{triplet_idx}"
    # Label placement: midpoint between subject and object (approximate)
    if s in node_pos and o in node_pos:
        sx, sy = node_pos[s]
        ox, oy = node_pos[o]
        mx = (sx + ox) / 2
        my = (sy + oy) / 2 - 4
    else:
        mx, my = 0, 0
    label_len = len(predicate) * 5.5
    return (
        f'<g class="edge" data-triplet-idx="{triplet_idx}" id="edge-{eid}">'
        f'<path d="{path}" marker-end="url(#arrowhead)"/>'
        f'<path class="edge-hit" d="{path}"/>'
        f'<g class="edge-label-wrap">'
        f'<rect x="{mx - label_len / 2:.1f}" y="{my - 7:.1f}" width="{label_len:.1f}" height="12" rx="3"/>'
        f'<text class="edge-label" x="{mx:.1f}" y="{my + 2:.1f}">{html_lib.escape(predicate)}</text>'
        "</g>"
        "</g>"
    )


def render_sidebar(entities: List[dict], involve_count: Dict[str, int]) -> str:
    items = []
    for e in entities:
        eid = html_lib.escape(e["id"])
        label = html_lib.escape(e.get("label", e["id"]))
        central_cls = " central" if e.get("central") else ""
        badge = f'<span class="badge">{UI["central_badge"]}</span>' if e.get("central") else ""
        count = involve_count.get(e["id"], 0)
        count_span = f'<span class="count">{count}</span>'
        items.append(
            f'<li class="entity-item{central_cls}" data-entity-id="{eid}">'
            f'<span>{label} {badge}</span>'
            f"{count_span}"
            "</li>"
        )
    return (
        f'<aside class="sidebar">'
        f'<h2>{UI["entities_header"]} ({len(entities)})</h2>'
        f'<ul class="entity-list" id="entity-list">{"".join(items)}</ul>'
        f'<div id="detail" class="empty-state">{html_lib.escape(UI["select_hint"])}</div>'
        "</aside>"
    )


def render_javascript(entities: List[dict], triplets: List[dict]) -> str:
    """Inline JS reads data from a JSON blob in the page and handles clicks."""
    data = {
        "entities": entities,
        "triplets": triplets,
        "ui": UI,
    }
    data_json = json.dumps(data, ensure_ascii=False)
    # The JS is intentionally vanilla — no framework, no build step.
    return f"""
<script>
const DATA = {data_json};

const entityById = Object.fromEntries(DATA.entities.map(e => [e.id, e]));
const tripletsByEntity = {{}};
DATA.entities.forEach(e => tripletsByEntity[e.id] = []);
DATA.triplets.forEach((t, idx) => {{
  if (tripletsByEntity[t.subject_id]) tripletsByEntity[t.subject_id].push({{...t, idx, role: 'subject'}});
  if (tripletsByEntity[t.object_id])  tripletsByEntity[t.object_id].push({{...t, idx, role: 'object'}});
}});

function escapeHtml(s) {{
  return (s || '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
}}

function clearHighlights() {{
  document.querySelectorAll('.node').forEach(n => n.classList.remove('selected','dimmed'));
  document.querySelectorAll('.edge').forEach(e => e.classList.remove('highlight','dimmed'));
  document.querySelectorAll('.entity-item').forEach(li => li.classList.remove('selected'));
}}

function selectEntity(entityId) {{
  clearHighlights();
  const sel = entityById[entityId];
  if (!sel) return;

  // Highlight the node
  const node = document.getElementById('node-' + entityId);
  if (node) node.classList.add('selected');
  // Dim others
  document.querySelectorAll('.node').forEach(n => {{
    if (n.dataset.entityId !== entityId) n.classList.add('dimmed');
  }});

  // Highlight involved edges
  const rels = tripletsByEntity[entityId] || [];
  const involvedEdgeIdxs = new Set(rels.map(r => r.idx));
  document.querySelectorAll('.edge').forEach(e => {{
    if (involvedEdgeIdxs.has(parseInt(e.dataset.tripletIdx))) e.classList.add('highlight');
    else e.classList.add('dimmed');
  }});
  // Un-dim the nodes at the ends of involved edges
  rels.forEach(r => {{
    const otherId = r.role === 'subject' ? r.object_id : r.subject_id;
    const otherNode = document.getElementById('node-' + otherId);
    if (otherNode) otherNode.classList.remove('dimmed');
  }});

  // Sidebar selection
  document.querySelectorAll('.entity-item').forEach(li => {{
    if (li.dataset.entityId === entityId) li.classList.add('selected');
  }});

  // Render detail panel
  const detail = document.getElementById('detail');
  detail.classList.remove('empty-state');
  detail.classList.add('detail');
  const centralBadge = sel.central ? ` <span class="badge" style="background:#c2410c;color:white;font-size:10px;padding:1px 6px;border-radius:999px;font-weight:600;text-transform:uppercase;">${{DATA.ui.central_badge}}</span>` : '';
  let html = '';
  html += `<button class="back" onclick="resetSelection()">← ${{DATA.ui.back}}</button>`;
  html += `<h3>${{escapeHtml(sel.label)}}${{centralBadge}}</h3>`;
  html += `<p class="meta">${{DATA.ui.involves_header.replace('{{n}}', rels.length)}}</p>`;
  html += '<div class="relations">';
  rels.forEach(r => {{
    const subjLabel = (entityById[r.subject_id] || {{}}).label || r.subject_id;
    const objLabel  = (entityById[r.object_id]  || {{}}).label || r.object_id;
    const selfId = entityId;
    const sCls = r.subject_id === selfId ? ' class="self"' : '';
    const oCls = r.object_id  === selfId ? ' class="self"' : '';
    const confKey = (r.confidence || '').toLowerCase();
    const confLabel = ({{high: DATA.ui.high, medium: DATA.ui.medium, low: DATA.ui.low}})[confKey] || r.confidence || '';
    html += '<div class="relation highlight">';
    html += `<div class="triple"><span${{sCls}}>${{escapeHtml(subjLabel)}}</span> <span class="predicate">${{escapeHtml(r.predicate || '')}}</span> <span${{oCls}}>${{escapeHtml(objLabel)}}</span></div>`;
    if (r.evidence) html += `<div class="ev">„${{escapeHtml(r.evidence)}}"</div>`;
    if (confKey)   html += `<span class="confidence ${{confKey}}">${{DATA.ui.confidence}}: ${{escapeHtml(confLabel)}}</span>`;
    html += '</div>';
  }});
  html += '</div>';
  detail.innerHTML = html;
}}

function selectTriplet(idx) {{
  const t = DATA.triplets[idx];
  if (!t) return;
  // Select the subject entity but ALSO highlight only this edge
  selectEntity(t.subject_id);
  // Re-clear other involving edges and only keep the clicked one highlighted
  document.querySelectorAll('.edge').forEach(e => {{
    e.classList.remove('highlight');
    if (parseInt(e.dataset.tripletIdx) === idx) {{
      e.classList.add('highlight');
      e.classList.remove('dimmed');
    }} else {{
      e.classList.add('dimmed');
    }}
  }});
}}

function resetSelection() {{
  clearHighlights();
  const detail = document.getElementById('detail');
  detail.classList.add('empty-state');
  detail.classList.remove('detail');
  detail.innerHTML = `<p>${{escapeHtml(DATA.ui.select_hint)}}</p>`;
}}

document.querySelectorAll('.node').forEach(n => {{
  n.addEventListener('click', e => {{
    e.stopPropagation();
    selectEntity(n.dataset.entityId);
  }});
}});
document.querySelectorAll('.edge').forEach(g => {{
  g.addEventListener('click', e => {{
    e.stopPropagation();
    selectTriplet(parseInt(g.dataset.tripletIdx));
  }});
}});
document.querySelectorAll('.entity-item').forEach(li => {{
  li.addEventListener('click', () => selectEntity(li.dataset.entityId));
}});
document.querySelector('svg').addEventListener('click', () => resetSelection());
</script>
"""


def render_html(event_code: str, entities: List[dict], triplets: List[dict]) -> str:
    node_pos, edge_paths, canvas_w, canvas_h = compute_layout(entities, triplets)
    canvas_h_padded = max(canvas_h + 40, 400)

    nodes_svg = "".join(
        render_node_svg(ent, *node_pos.get(ent["id"], (canvas_w / 2, canvas_h / 2)))
        for ent in entities
    )
    edges_svg_parts = []
    for i, t in enumerate(triplets):
        eid = f"t{i}"
        path = edge_paths.get(eid)
        if not path:
            continue
        edges_svg_parts.append(render_edge_svg(t, i, path, node_pos))
    edges_svg = "".join(edges_svg_parts)

    involve_count: Dict[str, int] = {e["id"]: 0 for e in entities}
    for t in triplets:
        for k in ("subject_id", "object_id"):
            ref = t.get(k)
            if ref in involve_count:
                involve_count[ref] += 1
    sidebar = render_sidebar(entities, involve_count)
    script = render_javascript(entities, triplets)

    return f"""<!doctype html>
<html lang="sl">
<head>
<meta charset="utf-8">
<title>{html_lib.escape(UI['title'])} · {html_lib.escape(event_code)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="shell">
  <main class="graph-pane">
    <header>
      <h1>{html_lib.escape(UI['title'])}</h1>
      <span class="code">{html_lib.escape(event_code)}</span>
    </header>
    <p class="hint">{html_lib.escape(UI['hint'])}</p>
    <div class="svg-wrap">
      <svg viewBox="0 0 {canvas_w:.1f} {canvas_h_padded:.1f}" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <marker id="arrowhead" markerWidth="9" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 9 3, 0 6" fill="#94a3b8"/>
          </marker>
        </defs>
        {edges_svg}
        {nodes_svg}
      </svg>
    </div>
  </main>
  {sidebar}
</div>
{script}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Screenshot
# ---------------------------------------------------------------------------

def find_headless_chrome() -> str | None:
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chrome"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


def screenshot(html_path: Path, png_path: Path) -> bool:
    chrome = find_headless_chrome()
    if not chrome:
        logger.warning(f"No headless Chrome found; PNG skipped for {html_path.name}")
        return False
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--screenshot={png_path}",
        f"--window-size={SCREENSHOT_W},{SCREENSHOT_H}",
        "--hide-scrollbars",
        f"file://{html_path.resolve()}",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        return png_path.exists()
    except Exception as e:
        logger.warning(f"Headless Chrome failed for {html_path.name}: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_events() -> List[str]:
    return [
        line.strip()
        for line in EVENTS_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def render_for_event(event_code: str, min_confidence: str = DEFAULT_MIN_CONFIDENCE) -> bool:
    out_dir = OUTPUT_DIR / event_code
    triplets_path = out_dir / "triplets.json"
    if not triplets_path.exists():
        logger.error(f"[{event_code}] no triplets.json; run extract_triplets.py first")
        return False

    data = json.loads(triplets_path.read_text(encoding="utf-8"))
    entities = data.get("entities") or []
    triplets = data.get("triplets") or []
    if not entities or not triplets:
        logger.error(f"[{event_code}] empty triplets/entities")
        return False

    # Identical confidence filter to gen_triplet_image.py so both triplet
    # renders depict the same graph. Full set remains in triplets.json.
    entities, triplets, dropped = filter_triplets_by_confidence(
        entities, triplets, min_confidence
    )
    if dropped:
        logger.info(
            f"[{event_code}] excluding {len(dropped)} triplet(s) below "
            f"'{min_confidence}' confidence from HTML graph"
        )
    if not entities or not triplets:
        logger.error(f"[{event_code}] nothing left after confidence filter; skipping")
        return False

    try:
        html_str = render_html(event_code, entities, triplets)
    except Exception as e:
        logger.error(f"[{event_code}] render failed: {e}")
        return False

    html_path = out_dir / "triplet_html.html"
    html_path.write_text(html_str, encoding="utf-8")

    png_path = out_dir / "triplet_html.png"
    if screenshot(html_path, png_path):
        logger.info(f"[{event_code}] OK: triplet_html.{{html,png}}")
    else:
        logger.info(f"[{event_code}] OK: triplet_html.html (no PNG)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Render triplets as interactive HTML graphs")
    parser.add_argument("--event", help="Run for a single event code")
    parser.add_argument(
        "--min-confidence",
        default=DEFAULT_MIN_CONFIDENCE,
        choices=["low", "medium", "high"],
        help="Minimum triplet confidence to render (default: %(default)s; 'low' renders all)",
    )
    args = parser.parse_args()

    events = [args.event] if args.event else load_events()
    logger.info(f"Processing {len(events)} event(s) (min_confidence={args.min_confidence})")
    for code in events:
        render_for_event(code, min_confidence=args.min_confidence)


if __name__ == "__main__":
    main()
