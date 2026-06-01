"""
Prompts for the seminar triplet experiment.

Two templates:

  1. TRIPLET_EXTRACTION_PROMPT_TEMPLATE — text-only LLM prompt that turns
     enriched article text into a JSON object containing an entities list
     and a triplets list. Targets calibrated from the production
     infographic inventory: ~10-15 triplets over ~6-10 entities, all in
     one connected component.

  2. TRIPLET_TO_IMAGE_PROMPT_TEMPLATE — image-gen prompt that feeds the
     extracted entities + triplets to Gemini 3 Pro Image Preview. Strips
     the "extract a knowledge graph from text" step from the existing
     CAUSAL_NETWORK_INFOGRAPHIC_PROMPT and replaces it with explicit
     entity and edge lists. Visual-style block is preserved verbatim so
     the only difference vs. the baseline is the data source.

Format() placeholders are doubled (`{{ }}`) inside JSON examples so the
template can be filled in with str.format() without escaping the JSON.
"""

# ---------------------------------------------------------------------------
# 1. Triplet extraction (text in -> entities + triplets JSON out)
# ---------------------------------------------------------------------------

TRIPLET_EXTRACTION_SYSTEM = (
    "You are a careful data journalist who extracts the central causal and "
    "explanatory relations from news coverage. You always ground every "
    "relation in the supplied article text and never introduce background "
    "knowledge. You always return strict JSON."
)

TRIPLET_EXTRACTION_PROMPT_TEMPLATE = """Read the news articles below and produce two outputs as a single strict JSON object.

The supplied articles are written in **{source_language}**. Your entire output — every `label`, every `evidence` field — MUST be in {source_language}. Do not translate proper nouns, institution names, predicates, or quoted spans into any other language. The `id` field is the only field that stays in lowercase ASCII snake_case.

(1) An `entities` list of 6-10 central actors, institutions, countries, policies, events, conditions, or **mechanism states** (e.g. "barriers down", "red light", "fourth-day deadline") that appear across the articles. Each entity has:
    - `id` (snake_case stable identifier, unique within this output, ASCII)
    - `label` (human-readable name in {source_language}, shown in the rendered graph)
    - `central` (boolean — exactly ONE entity must have central=true; this is the focal subject of the news story)

(2) A `triplets` list of 10-15 directed causal or explanatory relations between those entities. Each triplet has:
    - `subject_id` (must match an entity id)
    - `predicate` (1-3 word specific verb in {source_language} — e.g. raises, triggers, blames, criticizes, excludes, enables, requires, resigns from, proposes, motivates, leads to, results in, reinforces, undermines, vetoes)
    - `object_id` (must match an entity id)
    - `evidence` (short DIRECT QUOTE from the article in {source_language} supporting the relation — do NOT translate or paraphrase to a different language)
    - `confidence` ("low" | "medium" | "high")

Hard constraints:
- **Language lock**: every `label`, every `predicate`, every `evidence` must be in {source_language}. No translation.
- **No direction redundancy**: do NOT include two triplets expressing the same fact in opposite directions (e.g. `train → collides_with → van` AND `van → results_in → collision` AND `train → results_in → collision` is three triplets for one fact — keep only ONE). Pick the single most informative orientation.
- **Predicate-object type consistency**: communicative predicates (*announces, reports, confirms, criticizes, warns, blames, threatens*) take a STATEMENT or EVENT as object, never a person. If the natural object is a person, either rewrite the predicate (`kills`, `succeeds`, `criticizes_for_X`) or introduce a separate node for the statement/event (e.g. `mohamed_odeh_death`) and use it as the object.
- **Event reification — never make a person the direct cause or effect of another event**: when a causal relation originates from or results in an EVENT, ACTION, or STATE that involves a person (a death, killing, attack, ruling, resignation, an event that triggers a protest), create a DEDICATED entity node for that event/state and attach the causal edge to it — do NOT attach the causal verb to the person. A person's *death* causes consequences, the person does not. E.g. NOT `ales_sutar → povzročil → protest` but introduce `smrt_aleša_šutarja` and write `smrt_aleša_šutarja → sprožila → protest`.
- **Attribution / hedging for contested claims**: when a relation is contested, alleged, or attributed to a specific party (the defense, a single witness, one news source) rather than established fact, do NOT state it flat. Fold the attribution/uncertainty INTO the predicate string using {source_language} markers — for Slovenian use "naj bi …", "domnevno …", or "po navedbah <vir> …". This is especially important for accusations and claims that assign blame. E.g. NOT `ales_sutar → izzval → samire_siljic` but `ales_sutar → naj bi izzval (po navedbah obrambe) → samire_siljic`.
- **Causal mechanism over commentary**: for incident, accident, or policy stories, AT LEAST 30% of triplets must describe the CAUSAL MECHANISM — conditions, contributing factors, sequences that led to the central event. Pure reaction triplets (`expresses_condolences`, `announces`, `praises`) are capped at 3 per event total.
- **Specific predicates**: banned generic verbs: "affects", "relates to", "is associated with", "involves", "concerns", "deals with".
- The directed graph induced by the triplets MUST be connected when treated as undirected. No isolated entities. No disconnected chains.
- Every non-central entity must participate in at least 2 triplets. The central entity may participate in more.
- Every triplet must be grounded in the supplied articles. Do NOT introduce background knowledge or facts not stated or strongly implied in the text.
- Return ONLY the JSON object. No prose, no markdown fences, no commentary.

All examples below use a FICTIONAL, illustrative scenario (a made-up country "Vendia", its minister, a dam) purely to show the rules. They are NOT from the supplied articles — never copy these entities, names, or wordings into your output; extract only what the supplied articles support.

DIRECTION EXAMPLE — correct vs wrong:
  Fictional article: "Minister Adler confirmed that twelve people died in the dam failure."
  WRONG: {{"subject_id":"dam_failure","predicate":"causes","object_id":"minister_adler", ...}}     ← the failure does not cause the minister
  WRONG: {{"subject_id":"minister_adler","predicate":"confirms","object_id":"minister_adler", ...}}  ← invalid; self-loop
  RIGHT: {{"subject_id":"minister_adler","predicate":"confirms casualties of","object_id":"dam_failure", ...}}  ← minister confirms (re: the failure)

EVENT-REIFICATION EXAMPLE — put the event, not the person, on the causal edge:
  Fictional article: "The mayor's resignation triggered early elections in Vendia."
  WRONG: {{"subject_id":"mayor","predicate":"triggers","object_id":"early_elections", ...}}  ← the person did not trigger the elections
  RIGHT: add an entity {{"id":"mayor_resignation","label":"<the resignation, in source language>"}} then
         {{"subject_id":"mayor_resignation","predicate":"triggers","object_id":"early_elections", ...}}  ← the resignation triggered them

ATTRIBUTION EXAMPLE — hedge contested/attributed claims in the predicate:
  Fictional article: "The operator's lawyer argued that heavy rainfall, not negligence, caused the dam failure."
  WRONG: {{"subject_id":"rainfall","predicate":"causes","object_id":"dam_failure", ...}}  ← states one party's contested claim as established fact
  RIGHT: {{"subject_id":"rainfall","predicate":"allegedly caused (per the operator)","object_id":"dam_failure", ...}}  ← marked as a contested, attributed claim
  (In Slovenian, render such hedges as "naj bi …" / "domnevno …" / "po navedbah <vir> …".)

Output schema:
{{
  "entities": [
    {{"id": "eu", "label": "Evropska unija", "central": false}},
    {{"id": "carbon_tax", "label": "Ogljični davek", "central": true}}
  ],
  "triplets": [
    {{
      "subject_id": "eu",
      "predicate": "uvaja",
      "object_id": "carbon_tax",
      "evidence": "EU je uradno sprejela mehanizem za ogljični davek dne...",
      "confidence": "high"
    }}
  ]
}}

ARTICLES (in {source_language}):

{enriched_articles}
"""


def get_triplet_extraction_prompt(
    enriched_articles: str, source_language: str = "Slovenian"
) -> str:
    """Render the triplet extraction prompt with the supplied article text.

    `source_language` is the language of the input articles; it is also the
    language the model is required to use for `label`, `predicate`, and
    `evidence` fields (preserves the inspectable intermediate).
    """
    return TRIPLET_EXTRACTION_PROMPT_TEMPLATE.format(
        enriched_articles=enriched_articles,
        source_language=source_language,
    )


# ---------------------------------------------------------------------------
# 2. Triplet -> infographic image (Gemini 3 Pro Image Preview)
# ---------------------------------------------------------------------------

TRIPLET_TO_IMAGE_PROMPT_TEMPLATE = """**Role:** Expert Data Journalist and Information Designer.
**Task:** Generate a high-definition, publication-ready causal network infographic from the entities and relations supplied below. You are NOT extracting a knowledge graph from raw text — the graph has already been extracted for you. Render it.

### 1. Visual Style & Layout (Strict Adherence)
* **Aesthetic:** Tier-1 Publication Style (similar to The Economist or Bloomberg visual data). Clean, very extensive in information content, professional, information rich, and trustworthy.
* **Medium:** Flat vector illustration with high contrast.
* **Background:** Pure white (#FFFFFF) for maximum legibility.
* **Typography:** Modern Sans-Serif (e.g., Roboto, Helvetica style). **CRITICAL:** All text must be perfectly legible, sharp, and horizontally aligned where possible.
* **Structure:** A directed network graph (node-link diagram) flowing logically (e.g., left-to-right or top-to-bottom).
* **Color Palette:**
  * **Entities (People/Orgs/Places):** Navy Blue or Deep Slate.
  * **Claims/Events:** Muted Grey or subtle Teal.
  * **Causal Links (Arrows):** Distinct, thin black or dark grey lines with clear arrowheads.
  * **Highlights:** Apply Burnt Orange ONLY to the entity marked `central: true` in the supplied list.
* **Node Icons:** Each node MUST include a small, relevant icon or symbol:
  - Countries: National flag or recognizable emblem
  - Organizations: Logo or institutional symbol (e.g., EU stars)
  - Financial concepts: Currency symbols, money bags, coins
  - Military/Defense: Weapons, shields, aircraft silhouettes
  - Outcomes/Goals: Checkmarks, targets, arrows
  - Events/Actions: Contextually appropriate symbols
  Icons should be placed ABOVE the text label (centered), sized proportionally (≈20-25% of node height).

### 2. Content Constraints (CRITICAL)
You MUST use ONLY the entities and relations supplied below. You must NOT:
- introduce any node that is not in the supplied entities list,
- invent any directed edge that is not in the supplied relations list,
- add background knowledge or implied context,
- **translate, paraphrase, or rewrite** any supplied label or predicate. Render each node label and each edge predicate EXACTLY as supplied below, in the supplied language. If a label is "Osebni zdravniki", the node text must read "Osebni zdravniki", not "Personal Doctors" or "Family physician". Same for predicates.

Render each entity as one node, using its `label` as the visible text (verbatim). Render each relation as one directed arrow whose label is the supplied predicate (verbatim, though you may apply TitleCase or uppercase styling for visual consistency).

### 3. Output Instruction
Generate the content in professional {language} language. Generate the infographic image now. The image must be high-resolution ({image_resolution}), containing *only* the infographic with the supplied causal network (DO NOT generate any additional text).

### 4. Entities (use exactly these as nodes)

{entities_block}

### 5. Causal Relations (use exactly these as directed edges)

{relations_block}
"""


def _format_entities_block(entities: list[dict]) -> str:
    """Render the entities list as a bullet block for the image prompt."""
    lines = []
    for ent in entities:
        marker = " [CENTRAL — highlight in burnt orange]" if ent.get("central") else ""
        lines.append(f"- {ent['label']}{marker}")
    return "\n".join(lines)


def _format_relations_block(entities: list[dict], triplets: list[dict]) -> str:
    """Render triplets as `Subject Label -> predicate -> Object Label` lines."""
    label_by_id = {e["id"]: e["label"] for e in entities}
    lines = []
    for t in triplets:
        subj = label_by_id.get(t["subject_id"], t["subject_id"])
        obj = label_by_id.get(t["object_id"], t["object_id"])
        lines.append(f"- {subj} → {t['predicate']} → {obj}")
    return "\n".join(lines)


def get_triplet_to_image_prompt(
    entities: list[dict],
    triplets: list[dict],
    language: str = "Slovenian",
    image_resolution: str = "4K",
) -> str:
    """Render the triplet→image prompt from extracted entities + triplets."""
    return TRIPLET_TO_IMAGE_PROMPT_TEMPLATE.format(
        language=language,
        image_resolution=image_resolution,
        entities_block=_format_entities_block(entities),
        relations_block=_format_relations_block(entities, triplets),
    )


# ---------------------------------------------------------------------------
# 3. Confidence filtering (shared by both renderers AND the comparison view so
#    triplet→image and triplet→HTML always render the SAME filtered graph)
# ---------------------------------------------------------------------------

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
# Keep medium+high, drop low. The triplets.json keeps everything; filtering is
# applied only at render time.
DEFAULT_MIN_CONFIDENCE = "medium"


def confidence_rank(value: str) -> int:
    """Rank a confidence string. Unknown/missing is treated as high (kept)."""
    return CONFIDENCE_RANK.get((value or "high").lower(), 2)


def is_below_confidence(triplet: dict, min_confidence: str = DEFAULT_MIN_CONFIDENCE) -> bool:
    """True if this triplet would be excluded at the given threshold."""
    threshold = CONFIDENCE_RANK.get((min_confidence or "").lower(), 1)
    return confidence_rank(triplet.get("confidence")) < threshold


def filter_triplets_by_confidence(
    entities: list[dict],
    triplets: list[dict],
    min_confidence: str = DEFAULT_MIN_CONFIDENCE,
):
    """Drop triplets below ``min_confidence`` and prune now-orphaned entities.

    Returns ``(kept_entities, kept_triplets, dropped_triplets)``. Entities that
    no longer appear in any surviving triplet are removed so the rendered graph
    has no floating nodes. Input lists are not mutated.
    """
    kept, dropped = [], []
    for t in triplets:
        (dropped if is_below_confidence(t, min_confidence) else kept).append(t)
    used: set = set()
    for t in kept:
        used.add(t.get("subject_id"))
        used.add(t.get("object_id"))
    kept_entities = [e for e in entities if e.get("id") in used]
    return kept_entities, kept, dropped
