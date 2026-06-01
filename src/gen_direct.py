"""
Generate the direct text-to-image baseline for every event in events.txt.

This is the seminar's "current production pipeline" condition. We reuse the
production `CAUSAL_NETWORK_INFOGRAPHIC_PROMPT` template and the same Gemini
3 Pro Image Preview model + safety settings + aspect ratio + resolution as
`services/causal_network_infographic_service.py`. The only deviation:

  - articles are read from output/{event_code}/raw_articles.json (populated
    once by fetch_articles_from_prod.py from production), instead of going
    through the analysis orchestrator.
  - we call Gemini directly (not through CausalNetworkInfographicService)
    so we bypass that service's caching + DB usage tracking. The cache
    would short-circuit fresh runs, and the DB tracking writes need
    schema parity with production that we may not have locally.

Output (per event):
  experiments/seminar_triplets/output/{event_code}/
    direct.webp           - new infographic generated from the article text
    direct_prompt.txt     - the rendered prompt that Gemini saw
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from loguru import logger  # noqa: E402

from config import config  # noqa: E402
from services.causal_network_infographic_service import (  # noqa: E402
    CAUSAL_NETWORK_INFOGRAPHIC_PROMPT,
    LANGUAGE_NAMES,
)
from services.image_utils import save_image_as_webp  # noqa: E402

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


def load_articles(event_code: str) -> List[Dict[str, Any]]:
    path = OUTPUT_DIR / event_code / "raw_articles.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def join_articles(articles: List[Dict[str, Any]]) -> str:
    """Mirror orchestrator._prepare_article_formats() 'joined_articles' shape."""
    bodies = [(a.get("body") or "") for a in articles]
    return "\n\n".join(b for b in bodies if b)


def get_image_config() -> dict:
    from config.analysis.settings import get_image_model_config
    return get_image_model_config("causal_network")


def _get_gemini_client():
    from google import genai
    if not config.base.GEMINI_API_KEY:
        return None
    return genai.Client(api_key=config.base.GEMINI_API_KEY)


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
        text_parts = []
        if content and getattr(content, "parts", None):
            for p in content.parts:
                txt = getattr(p, "text", None)
                if txt:
                    text_parts.append(txt[:200])
        return None, f"finish={finish}; no image parts; text={text_parts}"
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data, f"finish={finish}; image bytes={len(inline.data)}"
    text_parts = [getattr(p, "text", "")[:200] for p in parts if getattr(p, "text", None)]
    return None, f"finish={finish}; parts had no inline_data; text={text_parts}"


def gen_for_event(event_code: str) -> bool:
    out_dir = OUTPUT_DIR / event_code
    out_dir.mkdir(parents=True, exist_ok=True)

    articles = load_articles(event_code)
    if not articles:
        logger.error(f"[{event_code}] no raw_articles.json; run fetch_articles_from_prod.py first")
        return False

    articles_text = join_articles(articles)
    img_config = get_image_config()
    language_name = LANGUAGE_NAMES.get(UI_LANG, "Slovenian")
    prompt = CAUSAL_NETWORK_INFOGRAPHIC_PROMPT.format(
        articles=articles_text,
        language=language_name,
        image_resolution=img_config["image_size"],
    )
    (out_dir / "direct_prompt.txt").write_text(prompt, encoding="utf-8")

    client = _get_gemini_client()
    if not client:
        logger.error("Gemini client unavailable (no GEMINI_API_KEY)")
        return False

    from google.genai import types  # local import: matches production service

    logger.info(
        f"[{event_code}] calling Gemini ({img_config['model']}, {img_config['image_size']}) "
        f"with {len(articles)} articles ({len(articles_text)} chars)..."
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
            logger.error(f"[{event_code}] direct attempt {attempt+1} raised: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
            continue

        image_bytes, diag = _inspect_response(response)
        if not image_bytes:
            logger.error(f"[{event_code}] direct attempt {attempt+1}: {diag}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
            continue

        dst_no_ext = out_dir / "direct"
        written = save_image_as_webp(image_bytes, str(dst_no_ext), quality=85)
        logger.info(f"[{event_code}] OK: saved {written}")
        return True

    logger.error(f"[{event_code}] all {MAX_RETRIES} attempts failed")
    return False


def main():
    parser = argparse.ArgumentParser(description="Generate direct baseline infographics")
    parser.add_argument("--event", help="Run for a single event code")
    args = parser.parse_args()

    events = [args.event] if args.event else load_events()
    logger.info(f"Processing {len(events)} event(s)")
    for code in events:
        gen_for_event(code)


if __name__ == "__main__":
    main()
