import base64
import json
import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_TARGET_WORDS = 90
DEFAULT_MAX_WORDS = 120
SLIDE_ONE_MIN_WORDS = 40
SLIDE_ONE_MAX_WORDS = 70

CONNECTOR_BANK = [
    "Let’s start with",
    "Here’s the snapshot",
    "Quick recap",
    "Moving to",
    "Next up",
    "Let’s zoom in",
    "The key takeaway",
    "On sectors",
    "On flows",
    "Under the hood",
    "For the next session",
]

BASE_SYSTEM_PROMPT = """
You are a professional video voiceover writer for Indian retail traders.
Use ONLY the content visible on the slide image. Do not add external facts, data, or predictions.
Tone: professional, slightly dramatic, confident. Keep language simple and natural.
Do not read the slide verbatim; narrate the ideas in a smooth voiceover flow.
Output is English overall, but ANY Hindi words/phrases MUST be in Devanagari script.
Never use romanized Hindi or Hinglish in Latin script.
Avoid starting any slide with "Now," unless explicitly instructed.
Return ONLY the narration text. No headings, labels, or bullet points.
""".strip()


def _build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=api_key)


def _encode_image(image: bytes) -> str:
    return base64.b64encode(image).decode("utf-8")


def _word_count(text: str) -> int:
    return len(text.split())


def _enforce_word_limit(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def _slide_instruction(
    slide_index: int, total_slides: int, opener: Optional[str]
) -> str:
    bridge_hint = (
        "If the next topic is unclear, use this exact bridge line: "
        "\"अब अगला section देखते हैं।\""
    )
    if slide_index == 1:
        return (
            "Slide 1 requirements:\n"
            f"- Keep it minimal: {SLIDE_ONE_MIN_WORDS}-{SLIDE_ONE_MAX_WORDS} words.\n"
            "- Start with the exact greeting: \"नमस्ते!\" (or \"हेलो!\" in Devanagari).\n"
            "- Include this exact line: \"चलिए, शुरू करते हैं।\"\n"
            "- Keep it host-like and simple; no dramatic metaphors or extra facts.\n"
            "- End with ONE short bridge sentence that tees up the next section.\n"
            f"- {bridge_hint}\n"
        )
    if slide_index == total_slides:
        return (
            "Last slide requirements:\n"
            f"- Start with this opener exactly: \"{opener}\".\n"
            "- Include a 1–2 line recap strictly from the slide data.\n"
            "- End with a strong CTA: Like / Subscribe / Comment.\n"
            "- Do NOT include any bridge to the next slide.\n"
            "- Do NOT ask a viewer question (it will be generated separately).\n"
        )
    return (
        "Slide requirements:\n"
        f"- Start with this opener exactly: \"{opener}\".\n"
        "- End with ONE short bridge sentence that tees up the next topic.\n"
        f"- {bridge_hint}\n"
    )


def _select_opener(last_opener: Optional[str], cursor: int) -> tuple[str, int]:
    if not CONNECTOR_BANK:
        raise RuntimeError("Connector bank is empty")
    opener = CONNECTOR_BANK[cursor % len(CONNECTOR_BANK)]
    if last_opener and opener == last_opener:
        cursor += 1
        opener = CONNECTOR_BANK[cursor % len(CONNECTOR_BANK)]
    return opener, cursor + 1


def normalize_hindi_to_devanagari(text: str, client: OpenAI, model_name: str) -> str:
    if os.environ.get("HINDI_DEVANAGARI", "1") == "0":
        return text
    prompt = (
        "Convert ONLY romanized Hindi words/phrases into Devanagari. "
        "Do NOT translate English. Do NOT change numbers, tickers, or symbols. "
        "Return ONLY the corrected text."
    )
    result = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
    )
    return (result.choices[0].message.content or "").strip()


def generate_scripts_from_images(
    images: List[bytes],
    model_name: str,
    target_words: int = DEFAULT_TARGET_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
    job_id: Optional[str] = None,
) -> List[str]:
    client = _build_client()
    scripts: List[str] = []
    total_slides = len(images)
    job_identifier = job_id or uuid.uuid4().hex
    scripts_dir = Path("outputs") / job_identifier / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    last_opener: Optional[str] = None
    opener_cursor = 0

    for index, image in enumerate(images, start=1):
        logger.info("Generating script for slide %s/%s", index, total_slides)
        opener = None
        if index >= 2:
            opener, opener_cursor = _select_opener(last_opener, opener_cursor)
            last_opener = opener

        slide_target_words = target_words
        slide_max_words = max_words
        if index == 1:
            slide_target_words = max(
                SLIDE_ONE_MIN_WORDS, min(target_words, SLIDE_ONE_MAX_WORDS)
            )
            slide_max_words = min(max_words, SLIDE_ONE_MAX_WORDS)

        user_prompt = (
            f"Slide {index} of {total_slides}.\n"
            f"Target length: {slide_target_words} words. Max: {slide_max_words} words.\n"
            "Write a cohesive video voiceover script for THIS slide only.\n"
            + _slide_instruction(index, total_slides, opener)
        )
        content = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_encode_image(image)}"},
            },
        ]
        result = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": BASE_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        )
        script = (result.choices[0].message.content or "").strip()
        script = _enforce_word_limit(script, slide_max_words)
        script = normalize_hindi_to_devanagari(script, client, model_name)
        scripts.append(script)

        script_path = scripts_dir / f"slide_{index}.txt"
        script_path.write_text(script, encoding="utf-8")
        meta_path = scripts_dir / f"slide_{index}_meta.json"
        meta_payload = {
            "word_count": _word_count(script),
            "target_words": slide_target_words,
            "max_words": slide_max_words,
        }
        meta_path.write_text(
            json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return scripts


def generate_viewer_question(full_script: str, model_name: str) -> str:
    client = _build_client()
    prompt = (
        "You are creating a single short viewer question for a YouTube voiceover. "
        "Use ONLY the information already present in the script below. "
        "Return ONLY one concise question. "
        "Do NOT add any extra text or labels."
    )
    result = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": full_script},
        ],
    )
    question = (result.choices[0].message.content or "").strip()
    question = normalize_hindi_to_devanagari(question, client, model_name)
    if question and not question.endswith("?"):
        question = f"{question}?"
    return question
