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

BASE_SYSTEM_PROMPT = """
You are a professional video voiceover writer for Indian retail traders.
Use ONLY the content visible on the slide image. Do not add external facts, data, or predictions.
Tone: professional, slightly dramatic, confident. Keep language simple and natural.
Do not read the slide verbatim; narrate the ideas in a smooth voiceover flow.
Output is English overall, but ANY Hindi words/phrases MUST be in Devanagari script.
Never use romanized Hindi or Hinglish in Latin script.
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


def _slide_instruction(slide_index: int, total_slides: int) -> str:
    if slide_index == 1:
        return (
            "Slide 1 requirements:\n"
            "- Start with a greeting in Devanagari (e.g., \"नमस्ते\" or \"हेलो\").\n"
            "- Add a short hook line (question or bold statement).\n"
            "- End with ONE transition sentence that starts with \"Next, we’ll look at\" and also includes a "
            "\"चलिए, अब deep dive शुरू करते हैं।\" style phrase in Devanagari.\n"
            "- No greetings after the first word.\n"
        )
    if slide_index == total_slides:
        return (
            "Last slide requirements:\n"
            "- Start with a natural connector (Now, Next, Moving on, etc.).\n"
            "- End with a short closing CTA line like: \"If you found this useful, follow Index Theory… कल फिर मिलते हैं।\"\n"
            "- Do not include any \"Next, we’ll look at…\" transition.\n"
        )
    return (
        "Slide requirements:\n"
        "- Start with a natural connector (Now, Next, Moving on, etc.).\n"
        "- End with ONE forward transition sentence that starts with \"Next, we’ll look at\".\n"
        "- If the next topic is unclear, keep it generic (e.g., \"Next, we’ll look at the next slide.\").\n"
    )


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

    for index, image in enumerate(images, start=1):
        logger.info("Generating script for slide %s/%s", index, total_slides)
        user_prompt = (
            f"Slide {index} of {total_slides}.\n"
            f"Target length: {target_words} words. Max: {max_words} words.\n"
            "Write a cohesive video voiceover script for THIS slide only.\n"
            + _slide_instruction(index, total_slides)
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
        script = _enforce_word_limit(script, max_words)
        script = normalize_hindi_to_devanagari(script, client, model_name)
        scripts.append(script)

        script_path = scripts_dir / f"slide_{index}.txt"
        script_path.write_text(script, encoding="utf-8")
        meta_path = scripts_dir / f"slide_{index}_meta.json"
        meta_payload = {
            "word_count": _word_count(script),
            "target_words": target_words,
            "max_words": max_words,
        }
        meta_path.write_text(
            json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return scripts
