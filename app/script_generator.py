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
Write Hinglish (Latin script) narration using ONLY the provided facts and outline context.
Professional, calm, engaging. SIMPLE English (CLB 8–9): short sentences, common words.
Do NOT use complex words like: distinctly, preceding, reinforce, psychological, persistent, backdrop, framework, actionable plan, etc.

OUTPUT FORMAT (STRICT):
- Return ONLY narration text. No titles. No headings. No labels.
- Do NOT include: "Slide", "Hook:", "Key points:", "Takeaway:", "Transition:", "Part 1", "Ab Part", "Key takeaway:".
- Use 5–7 short sentences total.
- {transition_rule}

LENGTH:
- Target {target_words} words; MAX {max_words} words.

SLIDE 1 RULE:
- If this is slide 1 (cover/title slide), DO NOT explain the slide.
- Write a welcome + what viewers will get in this report + set today’s tone.
- Keep it punchy: 45–70 words unless the slide clearly contains real data.

CONTENT:
- Stick to visible slide facts only. No opinions/predictions.
- Never narrate tables row-by-row; summarize top 2–3 insights only.
- Language must stay in Latin script (no Devanagari).
""".strip()

SCRIPT_PROMPT_SLIDE_1 = """
Write Hinglish (Latin script) narration using ONLY the provided facts and outline context.
Keep sentences crisp and high-energy.

OUTPUT FORMAT (STRICT):
- Return ONLY narration text. No titles. No headings. No labels.
- Do NOT include: "Slide", "Hook:", "Key points:", "Takeaway:", "Transition:", "Part 1", "Ab Part", "Key takeaway:".
- Use 3–4 short sentences total: 2–3 punchy sentences plus a transition sentence.
- Total length MUST stay between 30 and 45 words.
- Must include one strong hook line (a question or bold statement).
- Avoid words: framework, frameworks, briefing, goal, tone, clear framework.
- Last sentence MUST start exactly with: "Next, we’ll look at " and should naturally match the next slide intent.

LENGTH:
- Target {target_words} words; MAX {max_words} words.

CONTENT:
- Stick to visible slide facts only. No opinions/predictions.
- Never narrate tables row-by-row; summarize top 2–3 insights only.
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


BANNED_PHRASES = [
    "slide",
    "hook:",
    "key points:",
    "takeaway:",
    "transition:",
    "key takeaway:",
    "part 1",
    "part 2",
    "ab part",
    "ab hum part",
    "start kar rahe",
]


def script_has_no_banned_labels(text: str) -> bool:
    lowered = " ".join(text.lower().split())
    if lowered.startswith("slide ") or " slide " in lowered:
        return False
    banned_checks = BANNED_PHRASES + ["slide 1", "slide 2", "slide 3"]
    return not any(phrase in lowered for phrase in banned_checks)


CTA_KEYWORDS = ["like", "subscribe", "follow", "share"]
RISK_KEYWORDS = ["risk", "capital", "stop", "position size"]


def find_transition_sentence(text: str) -> str:
    normalized = text.replace("\n", " ")
    candidates = [s.strip() for s in normalized.split(".") if s.strip()]
    for sentence in candidates:
        lowered = sentence.lower()
        if lowered.startswith("next, we’ll look at") or lowered.startswith("next, we'll look at"):
            return sentence
    return ""


def script_is_plain_narration(text: str) -> bool:
    if not script_has_no_banned_labels(text):
        return False
    return bool(find_transition_sentence(text))


def contains_cta_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in CTA_KEYWORDS)


def contains_risk_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in RISK_KEYWORDS)


def validate_script_rules(text: str, is_last: bool, next_intent: str) -> tuple[bool, List[str]]:
    errors: List[str] = []
    lowered = text.lower()
    transition_sentence = find_transition_sentence(text)

    if is_last:
        if "next, we'll look at" in lowered or "next, we’ll look at" in lowered:
            errors.append("last slide has next transition")
        if not contains_cta_keyword(text):
            errors.append("missing cta")
        if not contains_risk_keyword(text):
            errors.append("missing risk reminder")
    else:
        if not transition_sentence:
            errors.append("missing transition")
        if transition_sentence and next_intent and next_intent.lower() not in transition_sentence.lower():
            errors.append("transition missing intent")
        if contains_cta_keyword(text):
            errors.append("cta too early")

    return not errors, errors


def slide_one_has_hook(text: str) -> bool:
    words = text.split()
    if len(words) > 45:
        return False
    normalized = " ".join(words)
    if "?" in normalized:
        return True
    hook_phrases = [
        "aaj ka plan",
        "soch rahe ho",
        "ready to jump",
        "kya aap",
    ]
    lowered = normalized.lower()
    return any(lowered.startswith(phrase) for phrase in hook_phrases)


def transition_mentions_intent(text: str, next_intent: str) -> bool:
    t = find_transition_sentence(text).lower()
    return bool(t and next_intent and next_intent.lower() in t)


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
    is_first: bool,
    is_last: bool,
) -> str:
    slides = outline.get("slides", []) if isinstance(outline, dict) else []
    next_intent = ""
    for slide in slides:
        if slide.get("index") == facts["slide_index"] + 1:
            next_intent = str(slide.get("intent", ""))
            break
    transition_rule = (
        "Last sentence must be a closing line (like/subscribe + risk reminder). Do NOT use ‘Next, we’ll look at …’."
        if is_last
        else "Last sentence MUST start exactly with: \"Next, we’ll look at \" and should naturally match the next slide intent."
    )
    prompt_template = SCRIPT_PROMPT_SLIDE_1 if is_first else SCRIPT_PROMPT
    formatted_prompt = prompt_template.format(
        target_words=target_words, max_words=max_words, transition_rule=transition_rule
    )
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
        effective_target = target_words
        effective_max = max_words
        is_first = facts["slide_index"] == 1
        is_last = facts["slide_index"] == total_pages
        if is_first:
            effective_target = 40
            effective_max = 45
        script = _generate_script_from_facts(
            client, facts, model, outline, effective_target, effective_max, is_first, is_last
        )

        needs_regen = False
        if not script_has_no_banned_labels(script):
            needs_regen = True
        valid_content, _ = validate_script_rules(script, is_last=is_last, next_intent=next_intent)
        if not valid_content:
            needs_regen = True
        if needs_regen:
            logger.info("Repairing script for slide %s due to format issues", index)
            repair_prompt = (
                script
                + "\n\nRewrite the script. Remove any labels/headers. Keep simple English. "
                + (
                    "End with a short CTA line (like/subscribe) plus risk reminder. Do not use ‘Next, we’ll look at …’."
                    if is_last
                    else "Keep last sentence starting with ‘Next, we’ll look at …’."
                )
            )
            result = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (SCRIPT_PROMPT_SLIDE_1 if is_first else SCRIPT_PROMPT).format(
                            target_words=effective_target,
                            max_words=effective_max,
                            transition_rule=(
                                "Last sentence must be a closing line (like/subscribe + risk reminder). Do NOT use ‘Next, we’ll look at …’."
                                if is_last
                                else "Last sentence MUST start exactly with: \"Next, we’ll look at \" and should naturally match the next slide intent."
                            ),
                        ),
                    },
                    {"role": "user", "content": repair_prompt},
                ],
            )
            script = (result.choices[0].message.content or "").strip()

        script, truncated = _enforce_word_limit(script, effective_max)
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
