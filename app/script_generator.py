import base64
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_TARGET_WORDS = 80
DEFAULT_MAX_WORDS = 95

OUTLINE_PROMPT = """
You are preparing a narration outline for a PDF report that has been split into ordered slide images. 
Use ONLY the visual content from these slides. Do not add external facts or opinions.

TASK:
- Produce a JSON object with these keys:
  {
    "throughline": "one short paragraph describing the overall flow",
    "glossary": ["term and meaning pairs if consistent across slides"],
    "slides": [
      {"index": 1, "intent": "one-line summary of the slide's intent"},
      {"index": 2, "intent": "..."}
    ]
  }

RULES:
- Slide intents must be one sentence each and grounded only in what appears on that slide.
- Keep the throughline concise and factual.
- Glossary is optional (5-10 items) and only if terms clearly repeat; otherwise return an empty list.
- Return ONLY the JSON object with no additional text.
""".strip()

FACTS_PROMPT = """
You are extracting concise, factual notes from THIS slide image.
Follow the JSON schema exactly and use only the visible content.

SCHEMA (all fields required):
{
  "slide_index": number,
  "title": string,
  "top_points": [string, string, string],
  "numbers_to_mention": [string],
  "what_to_avoid": [string],
  "transition_hint": string
}

RULES:
- top_points must capture 2-3 concise insights, not row-by-row table narration.
- numbers_to_mention should include only standout figures or percentages.
- what_to_avoid lists distracting details to skip.
- transition_hint should mention how this slide connects to its neighbors using provided intents only.
- Return ONLY a JSON object that matches the schema.
""".strip()

SCRIPT_PROMPT = """
Write a Hinglish (Latin script) narration for this slide using ONLY the provided facts and outline context.
Tone: professional, calm, engaging. No slang or emojis.

STRUCTURE (exactly 4 labeled parts):
Hook:
Key points:
Takeaway:
Transition:

LENGTH RULES:
- Target {target_words} words; never exceed {max_words} words.
- Summarize tables instead of reading rows.

CONTENT RULES:
- Stick to the facts; no new data or opinions.
- Key points must be 2-3 short sentences.
- Transition must reference the next slide intent to build continuity using the provided hint.
- Language must stay in Latin script (no Devanagari).
""".strip()


class SlideFacts(TypedDict):
    slide_index: int
    title: str
    top_points: List[str]
    numbers_to_mention: List[str]
    what_to_avoid: List[str]
    transition_hint: str


def _build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=api_key)


def _encode_image(image: bytes) -> str:
    return base64.b64encode(image).decode("utf-8")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _word_count(text: str) -> int:
    return len(text.split())


def _enforce_word_limit(text: str, max_words: int) -> tuple[str, bool]:
    words = text.split()
    if len(words) <= max_words:
        return text.strip(), False
    truncated = " ".join(words[:max_words]).strip()
    return truncated, True


def script_has_required_sections(text: str) -> bool:
    lowered = text.lower()
    return all(section in lowered for section in ["hook:", "key points:", "takeaway:", "transition:"])


def transition_mentions_intent(text: str, next_intent: str) -> bool:
    transition_line = None
    for line in text.splitlines():
        if line.strip().lower().startswith("transition:"):
            transition_line = line.lower()
            break
    return bool(transition_line and next_intent.lower() in transition_line)


def _parse_json_response(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON response")
        return None


def _generate_outline(client: OpenAI, images: List[bytes], model: str, job_dir: Path) -> Dict[str, Any]:
    image_payload = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_image(img)}"}}
        for img in images
    ]
    logger.info("Generating outline for %s slides", len(images))
    result = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": OUTLINE_PROMPT}] + image_payload,
            }
        ],
    )
    outline = _parse_json_response(result.choices[0].message.content or "") or {}
    outline_path = job_dir / "outline.json"
    _ensure_dir(job_dir)
    outline_path.write_text(json.dumps(outline, indent=2), encoding="utf-8")
    return outline


def _validate_facts(raw: Any) -> Optional[SlideFacts]:
    try:
        slide_index = int(raw.get("slide_index"))
        title = str(raw.get("title", "")).strip()
        top_points = raw.get("top_points") or []
        numbers = raw.get("numbers_to_mention") or []
        avoid = raw.get("what_to_avoid") or []
        transition_hint = str(raw.get("transition_hint", "")).strip()
    except Exception:
        return None
    if not title or not transition_hint:
        return None
    if not isinstance(top_points, list) or len(top_points) < 2:
        return None
    if not isinstance(numbers, list) or not isinstance(avoid, list):
        return None
    return {
        "slide_index": slide_index,
        "title": title,
        "top_points": [str(p).strip() for p in top_points[:3]],
        "numbers_to_mention": [str(n).strip() for n in numbers],
        "what_to_avoid": [str(a).strip() for a in avoid],
        "transition_hint": transition_hint,
    }


def _generate_facts(
    client: OpenAI,
    image: bytes,
    model: str,
    index: int,
    prev_intent: str,
    next_intent: str,
) -> SlideFacts:
    prompt = FACTS_PROMPT + f"\nPrevious slide intent: {prev_intent or 'None'}\nNext slide intent: {next_intent or 'None'}\n"
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_image(image)}"}},
    ]
    for attempt in range(2):
        result = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
        )
        parsed = _parse_json_response(result.choices[0].message.content or "")
        validated = _validate_facts(parsed)
        if validated:
            validated["slide_index"] = index
            return validated
        logger.warning("Facts schema invalid for slide %s on attempt %s", index, attempt + 1)
    raise ValueError(f"Failed to generate valid facts for slide {index}")


def _generate_script_from_facts(
    client: OpenAI,
    facts: SlideFacts,
    model: str,
    outline: Dict[str, Any],
    target_words: int,
    max_words: int,
) -> str:
    slides = outline.get("slides", []) if isinstance(outline, dict) else []
    next_intent = ""
    for slide in slides:
        if slide.get("index") == facts["slide_index"] + 1:
            next_intent = str(slide.get("intent", ""))
            break
    formatted_prompt = SCRIPT_PROMPT.format(target_words=target_words, max_words=max_words)
    user_text = (
        f"Throughline: {outline.get('throughline', '')}\n"
        f"Current slide intent: {next((s.get('intent') for s in slides if s.get('index') == facts['slide_index']), '')}\n"
        f"Next slide intent: {next_intent}\n"
        f"Glossary: {outline.get('glossary', [])}\n"
        f"Facts JSON: {json.dumps(facts, ensure_ascii=False)}"
    )
    result = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": user_text},
        ],
    )
    return (result.choices[0].message.content or "").strip()


def generate_scripts_from_images(
    images: List[bytes],
    model: str,
    target_words: int = DEFAULT_TARGET_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
    job_id: Optional[str] = None,
) -> List[str]:
    client = _build_client()
    scripts: List[str] = []
    total_pages = len(images)
    job_identifier = job_id or uuid.uuid4().hex
    job_dir = Path("outputs") / job_identifier
    scripts_dir = job_dir / "scripts"
    _ensure_dir(scripts_dir)

    outline = _generate_outline(client, images, model, job_dir)
    slide_intents = {slide.get("index"): slide.get("intent", "") for slide in outline.get("slides", []) if isinstance(slide, dict)}

    for index, image in enumerate(images, start=1):
        logger.info("Generating content for slide %s/%s (job %s)", index, total_pages, job_identifier)
        prev_intent = slide_intents.get(index - 1, "")
        next_intent = slide_intents.get(index + 1, "")
        facts = _generate_facts(client, image, model, index, prev_intent, next_intent)
        script = _generate_script_from_facts(client, facts, model, outline, target_words, max_words)
        script, truncated = _enforce_word_limit(script, max_words)
        scripts.append(script)

        script_path = scripts_dir / f"slide_{index}.txt"
        script_path.write_text(script, encoding="utf-8")
        meta_path = scripts_dir / f"slide_{index}_meta.json"
        meta_payload = {
            "word_count": _word_count(script),
            "truncation_applied": truncated,
            "slide_intent_used": slide_intents.get(index, ""),
        }
        meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    return scripts
