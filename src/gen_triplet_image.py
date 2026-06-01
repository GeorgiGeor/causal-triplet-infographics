"""
Generate the triplet-mediated infographic for every event in events.txt.

For each event:
  1. Read output/{event_code}/triplets.json (must already exist; run
     extract_triplets.py first).
  2. Build the triplet→image prompt: same visual-style block as
     `CAUSAL_NETWORK_INFOGRAPHIC_PROMPT` but with the "extract a knowledge
     graph from text" step replaced by explicit ENTITIES and CAUSAL RELATIONS
     blocks fed from the extracted triplets.
  3. Call Gemini 3 Pro Image Preview directly (same model + safety filters
     + aspect ratio + resolution as the direct baseline) so the only
     difference between conditions is the prompt content.
  4. Save the WebP and the rendered prompt.

Output (per event):
  experiments/seminar_triplets/output/{event_code}/
    triplet_image.webp           - GenAI infographic built from triplets
    triplet_image_prompt.txt     - the rendered prompt that Gemini saw
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger  # noqa: E402

from config import config  # noqa: E402
from services.causal_network_infographic_service import LANGUAGE_NAMES  # noqa: E402
from services.image_utils import save_image_as_webp  # noqa: E402

from seminar_prompts import (  # noqa: E402
    DEFAULT_MIN_CONFIDENCE,
    filter_triplets_by_confidence,
    get_triplet_to_image_prompt,
)

OUTPUT_DIR = Path(__file__).parent / "output"
EVENTS_FILE = Path(__file__).parent / "events.txt"
UI_LANG = "sl"
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 2

SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)


def load_events() -> List[str]:
    return [
        line.strip()
        for line in EVENTS_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _inspect_response(response) -> tuple[bytes | None, str]:
    """Return (image_bytes, diagnostic_string)."""
    if not response.candidates:
        pf = getattr(response, "prompt_feedback", None)
        return None, f"no candidates; block_reason={getattr(pf, 'block_reason', None)}"
    cand = response.candidates[0]
    finish = getattr(cand, "finish_reason", None)
    content = getattr(cand, "content", None)
    parts = getattr(content, "parts", None) if content else None
    if not parts:
        return None, f"finish={finish}; no parts in content"
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data, f"finish={finish}; image bytes={len(inline.data)}"
    text_parts = [getattr(p, "text", "")[:200] for p in parts if getattr(p, "text", None)]
    return None, f"finish={finish}; parts had no inline_data; text={text_parts}"


def gen_image_for_event(event_code: str, min_confidence: str = DEFAULT_MIN_CONFIDENCE) -> bool:
    out_dir = OUTPUT_DIR / event_code
    triplets_path = out_dir / "triplets.json"
    if not triplets_path.exists():
        logger.error(f"[{event_code}] no triplets.json; run extract_triplets.py first")
        return False

    data = json.loads(triplets_path.read_text(encoding="utf-8"))
    entities = data.get("entities") or []
    triplets = data.get("triplets") or []
    if not entities or not triplets:
        logger.error(f"[{event_code}] triplets.json has empty entities or triplets")
        return False

    # Drop low-confidence triplets (and any entity left orphaned) before
    # rendering. The full set stays in triplets.json; this only affects what the
    # image shows. gen_triplet_html.py applies the IDENTICAL filter so both
    # triplet renders depict the same graph.
    entities, triplets, dropped = filter_triplets_by_confidence(
        entities, triplets, min_confidence
    )
    if dropped:
        logger.info(
            f"[{event_code}] excluding {len(dropped)} triplet(s) below "
            f"'{min_confidence}' confidence from image: "
            + "; ".join(f"{t.get('subject_id')}→{t.get('object_id')}" for t in dropped)
        )
    if not entities or not triplets:
        logger.error(f"[{event_code}] nothing left after confidence filter; skipping")
        return False

    from config.analysis.settings import get_image_model_config
    img_config = get_image_model_config("causal_network")
    language_name = LANGUAGE_NAMES.get(UI_LANG, "Slovenian")
    prompt = get_triplet_to_image_prompt(
        entities=entities,
        triplets=triplets,
        language=language_name,
        image_resolution=img_config["image_size"],
    )
    (out_dir / "triplet_image_prompt.txt").write_text(prompt, encoding="utf-8")

    from google import genai
    if not config.base.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set")
        return False
    client = genai.Client(api_key=config.base.GEMINI_API_KEY)

    from google.genai import types  # local import: mirrors the production service

    logger.info(
        f"[{event_code}] calling Gemini ({img_config['model']}, "
        f"{img_config['image_size']}) with {len(entities)} entities, {len(triplets)} triplets"
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=img_config["model"],
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    image_config=types.ImageConfig(
                        aspect_ratio=img_config["aspect_ratio"],
                        image_size=img_config["image_size"],
                    ),
                    safety_settings=[
                        types.SafetySetting(category=c, threshold="OFF")
                        for c in SAFETY_CATEGORIES
                    ],
                ),
            )
        except Exception as e:
            logger.error(f"[{event_code}] attempt {attempt+1} raised: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
            continue

        image_bytes, diag = _inspect_response(response)
        if not image_bytes:
            logger.error(f"[{event_code}] triplet attempt {attempt+1}: {diag}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
            continue

        dst_no_ext = out_dir / "triplet_image"
        written = save_image_as_webp(image_bytes, str(dst_no_ext), quality=85)
        logger.info(f"[{event_code}] OK: saved {written}")
        return True

    logger.error(f"[{event_code}] all {MAX_RETRIES} attempts failed")
    return False


def main():
    parser = argparse.ArgumentParser(description="Generate triplet-mediated infographics")
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
        gen_image_for_event(code, min_confidence=args.min_confidence)


if __name__ == "__main__":
    main()
