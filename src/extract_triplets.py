"""
Extract SPO causal triplets for every event listed in events.txt.

Reads articles from output/{event_code}/raw_articles.json (which is populated
once by fetch_articles_from_prod.py from the production database). Calls the
LLM via LLMService — no Flask app context, no DB writes. Validates the JSON
response. Retries once with the validator output if validation fails.

Output (per event):
  experiments/seminar_triplets/output/{event_code}/
    articles.txt            - the enriched article text fed to the LLM
    triplets.json           - the extracted entities + triplets
    validation_errors.txt   - present only if validation failed after retry
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger  # noqa: E402

from config import config  # noqa: E402
from services.llm_service import LLMService  # noqa: E402

from seminar_prompts import (  # noqa: E402
    TRIPLET_EXTRACTION_SYSTEM,
    get_triplet_extraction_prompt,
)

OUTPUT_DIR = Path(__file__).parent / "output"
EVENTS_FILE = Path(__file__).parent / "events.txt"
MAX_TOKENS = 3000

TRIPLET_MIN, TRIPLET_MAX = 10, 15
ENTITY_MIN, ENTITY_MAX = 6, 10

LANGUAGE_BY_PREFIX = {
    "slv": "Slovenian",
    "eng": "English",
    "hrv": "Croatian",
    "srp": "Serbian",
    "fra": "French",
    "deu": "German",
}


def detect_source_language(event_code: str, articles: List[Dict[str, Any]]) -> str:
    """Pick the language to lock the extraction prompt to.

    Priority: language field on first article (canonical 3-letter code) → event
    code prefix → fallback to Slovenian.
    """
    if articles:
        lang_field = (articles[0].get("language") or "").lower().strip()
        if lang_field in {"sl", "slv"}:
            return "Slovenian"
        if lang_field in {"en", "eng"}:
            return "English"
        if lang_field in LANGUAGE_BY_PREFIX:
            return LANGUAGE_BY_PREFIX[lang_field]
    prefix = event_code.split("-", 1)[0].lower()
    return LANGUAGE_BY_PREFIX.get(prefix, "Slovenian")


def load_events() -> List[str]:
    return [
        line.strip()
        for line in EVENTS_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def load_articles(event_code: str) -> List[Dict[str, Any]]:
    path = OUTPUT_DIR / event_code / "raw_articles.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def format_enriched_articles(articles: List[Dict[str, Any]]) -> str:
    """Mirror orchestrator._prepare_article_formats() 'enriched_articles' shape."""
    parts: List[str] = []
    for i, art in enumerate(articles, 1):
        source = art.get("publisher_name") or "Unknown"
        title = art.get("title") or ""
        body = (art.get("body") or "")[:3000]
        parts.append(f"Article {i} - {source}:")
        if title:
            parts.append(f"Title: {title}")
        parts.append(body)
        parts.append("")
    return "\n".join(parts)


def validate_triplets(data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    entities = data.get("entities") or []
    triplets = data.get("triplets") or []

    if not entities:
        errors.append("no entities returned")
        return errors
    if not triplets:
        errors.append("no triplets returned")
        return errors

    if not (ENTITY_MIN - 1 <= len(entities) <= ENTITY_MAX + 2):
        errors.append(f"entity count {len(entities)} outside target {ENTITY_MIN}-{ENTITY_MAX}")
    if not (TRIPLET_MIN - 1 <= len(triplets) <= TRIPLET_MAX + 3):
        errors.append(f"triplet count {len(triplets)} outside target {TRIPLET_MIN}-{TRIPLET_MAX}")

    entity_ids = {e.get("id") for e in entities if e.get("id")}
    central = [e for e in entities if e.get("central")]
    if len(central) != 1:
        errors.append(f"central entity count is {len(central)}, must be exactly 1")

    for i, t in enumerate(triplets):
        if t.get("subject_id") not in entity_ids:
            errors.append(f"triplet {i}: subject_id {t.get('subject_id')!r} not in entities")
        if t.get("object_id") not in entity_ids:
            errors.append(f"triplet {i}: object_id {t.get('object_id')!r} not in entities")
        if not t.get("predicate"):
            errors.append(f"triplet {i}: empty predicate")

    use_count: Dict[str, int] = {eid: 0 for eid in entity_ids}
    for t in triplets:
        for key in ("subject_id", "object_id"):
            ref = t.get(key)
            if ref in use_count:
                use_count[ref] += 1
    for ent in entities:
        eid = ent.get("id")
        if eid and not ent.get("central") and use_count.get(eid, 0) < 2:
            errors.append(f"entity {eid!r} appears in only {use_count.get(eid, 0)} triplet(s); need ≥ 2")

    if entity_ids:
        adj: Dict[str, set] = {eid: set() for eid in entity_ids}
        for t in triplets:
            s, o = t.get("subject_id"), t.get("object_id")
            if s in adj and o in adj:
                adj[s].add(o)
                adj[o].add(s)
        start = (central[0]["id"] if central else next(iter(entity_ids), None))
        if start:
            visited = {start}
            queue = [start]
            while queue:
                node = queue.pop()
                for nbr in adj.get(node, set()):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            unreached = entity_ids - visited
            if unreached:
                errors.append(f"graph disconnected; unreachable from central: {sorted(unreached)}")

    return errors


def backup_existing_triplets(out_dir: Path) -> None:
    """If a triplets.json already exists, move it to triplets.v1.json for diffing."""
    current = out_dir / "triplets.json"
    if not current.exists():
        return
    backup = out_dir / "triplets.v1.json"
    if backup.exists():
        return
    backup.write_text(current.read_text(encoding="utf-8"), encoding="utf-8")


def call_llm(prompt: str, retry_feedback: str = "") -> Dict[str, Any]:
    model_config = config.get_model_config("extraction")
    user_prompt = prompt
    if retry_feedback:
        user_prompt = (
            f"{prompt}\n\n"
            f"Your previous response failed validation with these issues:\n{retry_feedback}\n\n"
            "Fix all the listed issues and return new JSON matching the schema."
        )

    response_content, _usage, _ms = LLMService.create_completion(
        model=model_config["model"],
        messages=[
            {"role": "system", "content": TRIPLET_EXTRACTION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=model_config.get("temperature", 0.2),
        response_format={"type": "json_object"},
        fallback_model=model_config.get("fallback_model"),
    )
    return json.loads(response_content)


def extract_for_event(event_code: str, prompts_only: bool = False) -> bool:
    out_dir = OUTPUT_DIR / event_code
    out_dir.mkdir(parents=True, exist_ok=True)

    articles = load_articles(event_code)
    if not articles:
        logger.error(f"[{event_code}] no raw_articles.json; run fetch_articles_from_prod.py first")
        return False
    if len(articles) < 3:
        logger.warning(f"[{event_code}] only {len(articles)} articles, skipping")
        return False

    enriched = format_enriched_articles(articles)
    (out_dir / "articles.txt").write_text(enriched, encoding="utf-8")

    source_language = detect_source_language(event_code, articles)
    prompt = get_triplet_extraction_prompt(enriched, source_language=source_language)
    # Persist the exact extraction prompt sent (mirrors *_prompt.txt for the
    # image stages) so the comparison view can show all three pipeline prompts.
    (out_dir / "extraction_prompt.txt").write_text(prompt, encoding="utf-8")
    if prompts_only:
        logger.info(f"[{event_code}] wrote extraction_prompt.txt (prompts-only; LLM skipped)")
        return True
    backup_existing_triplets(out_dir)
    logger.info(
        f"[{event_code}] extracting triplets from {len(articles)} articles "
        f"(source language: {source_language})..."
    )

    try:
        data = call_llm(prompt)
    except Exception as e:
        logger.error(f"[{event_code}] LLM call failed: {e}")
        return False

    errors = validate_triplets(data)
    if errors:
        logger.warning(f"[{event_code}] validation failed ({len(errors)} issues); retrying once")
        try:
            data = call_llm(prompt, retry_feedback="\n  - ".join([""] + errors))
        except Exception as e:
            logger.error(f"[{event_code}] retry LLM call failed: {e}")
            (out_dir / "validation_errors.txt").write_text("\n".join(errors), encoding="utf-8")
            return False
        errors = validate_triplets(data)

    (out_dir / "triplets.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "validation_errors.txt").unlink(missing_ok=True)

    if errors:
        (out_dir / "validation_errors.txt").write_text("\n".join(errors), encoding="utf-8")
        logger.warning(
            f"[{event_code}] persisted with {len(errors)} unresolved validation errors"
        )
    else:
        logger.info(
            f"[{event_code}] OK: {len(data['triplets'])} triplets over "
            f"{len(data['entities'])} entities"
        )

    return True


def main():
    parser = argparse.ArgumentParser(description="Extract SPO triplets for seminar events")
    parser.add_argument(
        "--event",
        help="Run for a single event code instead of the full events.txt list",
    )
    parser.add_argument(
        "--prompts-only",
        action="store_true",
        help="Only (re)write extraction_prompt.txt from existing articles; skip the LLM "
             "and leave triplets.json untouched.",
    )
    args = parser.parse_args()

    events = [args.event] if args.event else load_events()
    logger.info(f"Processing {len(events)} event(s)" + (" (prompts-only)" if args.prompts_only else ""))
    for code in events:
        extract_for_event(code, prompts_only=args.prompts_only)


if __name__ == "__main__":
    main()
