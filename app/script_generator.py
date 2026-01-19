import base64
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_TARGET_WORDS = 70
DEFAULT_MAX_WORDS = 90
SLIDE_ONE_MIN_WORDS = 18
SLIDE_ONE_MAX_WORDS = 40
DEFAULT_MODEL_NAME = "gpt-5.2"
NUMERIC_DENSE_TOKEN_THRESHOLD = 25
NUMERIC_DENSE_RATIO_THRESHOLD = 0.25
NUMERIC_DENSE_CHAR_RATIO_THRESHOLD = 0.16
NUMERIC_DENSE_MAX_NUMBERS = 6
NUMERIC_DENSE_TARGET_WORDS = 45
NUMERIC_DENSE_MAX_WORDS = 65

BASE_SYSTEM_PROMPT = """
You are a professional video voiceover writer for Indian retail traders.
Use ONLY the content visible on the slide image. Do not add external facts, data, or predictions.
Do not infer macro effects or explanations unless explicitly written on the slide.
No added opinions or forecasts. Keep it retail-friendly and professional.
Tone: professional, confident, crisp. Keep language simple and natural.
Write 2-4 short spoken sentences per slide.
Paraphrase the slide; do not read bullet lists verbatim.
Avoid phrases like "this slide shows", "as per the slide", or "in this slide".
No transitions, bridges, or references to other slides.
Return ONLY the narration text. No headings, labels, or bullet points.
""".strip()

BASE_SYSTEM_PROMPT_YOUTUBE = """
You are a professional YouTube voiceover writer for Indian retail traders.
Use ONLY the content visible on the slide image. Do not add external facts, data, or predictions.
Do not infer causes or explanations unless explicitly written on the slide.
Write 2-4 short spoken sentences with a natural, human rhythm.
Paraphrase the slide; do not read bullet lists verbatim.
Avoid phrases like "this slide shows", "as per the slide", or "in this slide".
No transitions, bridges, or references to other slides.
Return ONLY the narration text. No headings, labels, or bullet points.
""".strip()

BANNED_REPETITIVE_PHRASES = [
    "Let’s start with",
    "Let's start with",
    "Quick recap",
    "Here’s the snapshot",
    "Here's the snapshot",
]

def _build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=api_key)


def _get_model_name() -> str:
    return os.environ.get("MODEL_NAME") or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL_NAME


def _get_voice_style() -> str:
    style = os.environ.get("VOICE_STYLE", "formal").strip().lower()
    if style not in {"formal", "youtube"}:
        return "formal"
    return style


def _system_prompt() -> str:
    base = (
        BASE_SYSTEM_PROMPT_YOUTUBE
        if _get_voice_style() == "youtube"
        else BASE_SYSTEM_PROMPT
    )
    return base.strip()


def _encode_image(image: bytes) -> str:
    return base64.b64encode(image).decode("utf-8")


def _word_count(text: str) -> int:
    return len(text.split())


def _enforce_word_limit(text: str, max_words: int) -> tuple[str, bool]:
    words = text.split()
    if len(words) <= max_words:
        return text.strip(), False
    return " ".join(words[:max_words]).strip(), True


def _strip_leading_now(text: str) -> str:
    return re.sub(r"^\s*Now,?\s+", "", text, flags=re.IGNORECASE)


def _digit_sequences(text: str) -> List[str]:
    return re.findall(r"\d[\d,.]*", text)


def _get_numeric_dense_thresholds() -> tuple[int, float, float]:
    raw_token = os.environ.get(
        "NUMERIC_DENSE_TOKEN_THRESHOLD", str(NUMERIC_DENSE_TOKEN_THRESHOLD)
    )
    raw_ratio = os.environ.get(
        "NUMERIC_DENSE_RATIO_THRESHOLD", str(NUMERIC_DENSE_RATIO_THRESHOLD)
    )
    raw_char_ratio = os.environ.get(
        "NUMERIC_DENSE_CHAR_RATIO_THRESHOLD", str(NUMERIC_DENSE_CHAR_RATIO_THRESHOLD)
    )
    try:
        token_threshold = max(0, int(raw_token))
    except ValueError:
        token_threshold = NUMERIC_DENSE_TOKEN_THRESHOLD
    try:
        ratio_threshold = max(0.0, float(raw_ratio))
    except ValueError:
        ratio_threshold = NUMERIC_DENSE_RATIO_THRESHOLD
    try:
        char_ratio_threshold = max(0.0, float(raw_char_ratio))
    except ValueError:
        char_ratio_threshold = NUMERIC_DENSE_CHAR_RATIO_THRESHOLD
    return token_threshold, ratio_threshold, char_ratio_threshold


def _get_numeric_dense_limits() -> tuple[int, int, int]:
    raw_max_numbers = os.environ.get(
        "NUMERIC_DENSE_MAX_NUMBERS", str(NUMERIC_DENSE_MAX_NUMBERS)
    )
    raw_target_words = os.environ.get(
        "NUMERIC_DENSE_TARGET_WORDS", str(NUMERIC_DENSE_TARGET_WORDS)
    )
    raw_max_words = os.environ.get(
        "NUMERIC_DENSE_MAX_WORDS", str(NUMERIC_DENSE_MAX_WORDS)
    )
    try:
        max_numbers = max(1, int(raw_max_numbers))
    except ValueError:
        max_numbers = NUMERIC_DENSE_MAX_NUMBERS
    try:
        target_words = max(10, int(raw_target_words))
    except ValueError:
        target_words = NUMERIC_DENSE_TARGET_WORDS
    try:
        max_words = max(target_words, int(raw_max_words))
    except ValueError:
        max_words = NUMERIC_DENSE_MAX_WORDS
    return max_numbers, target_words, max_words


def _numeric_token_stats(text: str) -> tuple[int, float, float]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9,./%+-]*", text)
    total_tokens = len(tokens)
    numeric_tokens = [token for token in tokens if any(ch.isdigit() for ch in token)]
    numeric_token_count = len(numeric_tokens)
    token_ratio = (numeric_token_count / total_tokens) if total_tokens else 0.0
    total_chars = len(text)
    digit_chars = sum(ch.isdigit() for ch in text)
    char_ratio = (digit_chars / total_chars) if total_chars else 0.0
    return numeric_token_count, token_ratio, char_ratio


def is_numeric_dense_text(text: str) -> bool:
    token_threshold, ratio_threshold, char_ratio_threshold = _get_numeric_dense_thresholds()
    numeric_token_count, token_ratio, char_ratio = _numeric_token_stats(text)
    return (
        numeric_token_count >= token_threshold
        or token_ratio >= ratio_threshold
        or char_ratio >= char_ratio_threshold
    )


def _count_numbers(text: str) -> int:
    return len(_digit_sequences(text))


def _prune_numeric_sentences(text: str, max_numbers: int) -> str:
    if _count_numbers(text) <= max_numbers:
        return text
    sentences = [sentence for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence]
    if not sentences:
        return text
    counts = [len(_digit_sequences(sentence)) for sentence in sentences]
    total = sum(counts)
    if total <= max_numbers:
        return text
    removal_order = sorted(range(len(sentences)), key=lambda idx: counts[idx], reverse=True)
    keep = [True] * len(sentences)
    for idx in removal_order:
        if total <= max_numbers:
            break
        if counts[idx] == 0:
            continue
        keep[idx] = False
        total -= counts[idx]
    remaining = [sentence for i, sentence in enumerate(sentences) if keep[i]]
    if not remaining:
        return sentences[0].strip()
    return " ".join(remaining).strip()


def _rewrite_numeric_dense_script(
    script: str,
    *,
    max_numbers: int,
    page_text: str,
    client: OpenAI,
    model_name: str,
) -> str:
    instruction = (
        "Rewrite the narration to reduce numeric clutter. "
        f"Mention at most {max_numbers} numbers total. "
        "Keep any numbers you do mention EXACT. "
        "Do NOT add new numbers or facts. "
        "Do NOT read tables or long lists; summarize at a high level. "
        "Return ONLY the narration text."
    )
    result = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": script},
        ],
    )
    rewritten = (result.choices[0].message.content or "").strip()
    allowed_numbers = set(_digit_sequences(page_text))
    output_numbers = set(_digit_sequences(rewritten))
    if allowed_numbers and not output_numbers.issubset(allowed_numbers):
        logger.warning("Numeric dense rewrite introduced numbers not in page text.")
        return script
    rewritten = _prune_numeric_sentences(rewritten, max_numbers)
    return rewritten


def _remove_repeated_phrases(text: str, phrases: List[str]) -> str:
    updated = text
    for phrase in phrases:
        pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
        matches = list(pattern.finditer(updated))
        if len(matches) <= 1:
            continue
        start = matches[0].start()
        end = matches[0].end()
        first = updated[start:end]
        remainder = updated[end:]
        remainder = pattern.sub("", remainder)
        updated = f"{updated[:start]}{first}{remainder}"
    return re.sub(r"\s{2,}", " ", updated).strip()


def _generate_slide_body(
    image: bytes,
    client: OpenAI,
    model_name: str,
    instruction: str,
    target_words: int,
    max_words: int,
) -> str:
    user_prompt = (
        f"Target length: {target_words} words. Max: {max_words} words.\n"
        "Write narration for THIS slide only.\n"
        f"{instruction}"
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
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": content},
        ],
    )
    script = (result.choices[0].message.content or "").strip()
    script = _strip_leading_now(script)
    trimmed, _ = _enforce_word_limit(script, max_words)
    return trimmed


def humanize_full_script(full_script: str, client: OpenAI, model_name: str) -> str:
    prompt = (
        "Rewrite this into ONE continuous YouTube talk-track. "
        "Use ONLY the facts already present. Do NOT add any new info. "
        "Keep all numbers, tickers, and percentages EXACT. "
        "Avoid repetitive transitions. Return ONLY the narration."
    )
    result = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": full_script},
        ],
    )
    output = (result.choices[0].message.content or "").strip()
    output = _remove_repeated_phrases(output, BANNED_REPETITIVE_PHRASES)
    input_digits = set(_digit_sequences(full_script))
    output_digits = set(_digit_sequences(output))
    if not input_digits.issubset(output_digits):
        logger.warning("Digit guard failed in humanize pass; using original script.")
        return full_script
    return output


def create_scripts_job_dir(
    job_identifier: Optional[str] = None,
) -> Tuple[Path, Path]:
    job_identifier = job_identifier or uuid.uuid4().hex
    scripts_dir = Path("outputs") / job_identifier / "scripts"
    original_dir = scripts_dir / "original"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)
    return scripts_dir, original_dir


def generate_script_for_slide(
    image: bytes,
    *,
    client: Optional[OpenAI] = None,
    model_name: str,
    slide_index: int,
    total_slides: int,
    target_words: int,
    max_words: int,
    scripts_dir: Optional[Path] = None,
    is_numeric_dense: bool = False,
    page_text: str = "",
) -> str:
    active_client = client or _build_client()
    voice_style = _get_voice_style()
    max_numbers, dense_target_words, dense_max_words = _get_numeric_dense_limits()

    if slide_index == 1 and voice_style != "youtube":
        slide_target_words = max(
            SLIDE_ONE_MIN_WORDS, min(target_words, SLIDE_ONE_MAX_WORDS)
        )
        slide_max_words = min(max_words, SLIDE_ONE_MAX_WORDS)
        welcome_line = (
            "Hi, this is Jatin Dhiman. Welcome to Index Theory’s Post Market Report."
        )
        fixed_words = _word_count(welcome_line)
        body_target = max(10, slide_target_words - fixed_words)
        body_max = max(12, slide_max_words - fixed_words)
        body_instruction = (
            "Provide exactly ONE dramatic hook sentence (6-10 words) derived only from "
            "this slide. Do NOT mention any date, brand, welcome, or CTA. "
            "Do NOT add any opener or transition."
        )
        body = _generate_slide_body(
            image, active_client, model_name, body_instruction, body_target, body_max
        )
        theme_line = re.split(r"[.!?।]", body, maxsplit=1)[0].strip()
        begin_line = "Lets begin."
        slide_lines = [welcome_line, theme_line, begin_line]
        script = "\n".join(line for line in slide_lines if line)
        script, _ = _enforce_word_limit(script, SLIDE_ONE_MAX_WORDS)
    else:
        slide_target_words = dense_target_words if is_numeric_dense else target_words
        slide_max_words = dense_max_words if is_numeric_dense else max_words
        if slide_index == total_slides:
            body_instruction = (
                "Provide 2-4 short sentences strictly from this slide. "
                "Do NOT include any greeting, opener, transition, CTA, or viewer question."
            )
        else:
            body_instruction = (
                "Write 2-4 short sentences from only the slide content. "
                "Do NOT include any greeting, opener, transition, or CTA."
            )
        if is_numeric_dense:
            body_instruction = (
                f"{body_instruction} Summarize at a high level; do NOT read tables. "
                f"Mention at most {max_numbers} numbers total, keeping those numbers exact."
            )
        body = _generate_slide_body(
            image,
            active_client,
            model_name,
            body_instruction,
            slide_target_words,
            slide_max_words,
        )
        script = body.strip()
        if slide_index == total_slides and voice_style != "youtube":
            cta_lines = [
                "Like, Subscribe, and share your view in the comments.",
                "Hit the bell for updates.",
            ]
            script = f"{script}\n" + "\n".join(cta_lines)
        if is_numeric_dense and _count_numbers(script) > max_numbers:
            script = _rewrite_numeric_dense_script(
                script,
                max_numbers=max_numbers,
                page_text=page_text,
                client=active_client,
                model_name=model_name,
            )
            if _count_numbers(script) > max_numbers:
                script = _prune_numeric_sentences(script, max_numbers)

    if scripts_dir is not None:
        scripts_dir = Path(scripts_dir)
        original_dir = scripts_dir / "original"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        original_dir.mkdir(parents=True, exist_ok=True)

        script_path = scripts_dir / f"slide_{slide_index}.txt"
        script_path.write_text(script, encoding="utf-8")
        original_script_path = original_dir / f"slide_{slide_index}.txt"
        original_script_path.write_text(script, encoding="utf-8")
        meta_path = scripts_dir / f"slide_{slide_index}_meta.json"
        meta_payload = {
            "word_count": _word_count(script),
            "target_words": slide_target_words,
            "max_words": slide_max_words,
            "is_numeric_dense": is_numeric_dense,
        }
        meta_path.write_text(
            json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return script


_BANNED_LABELS = [
    "slide",
    "hook",
    "key points",
    "takeaway",
    "transition",
]


def script_has_no_banned_labels(script: str) -> bool:
    for line in script.splitlines():
        cleaned = line.strip().lower()
        if not cleaned:
            continue
        for label in _BANNED_LABELS:
            if cleaned.startswith(f"{label}"):
                return False
    return True


def script_is_plain_narration(script: str) -> bool:
    return script_has_no_banned_labels(script)


def find_transition_sentence(script: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", script.strip())
    for sentence in sentences:
        cleaned = sentence.strip()
        lower = cleaned.lower()
        if lower.startswith("next, we'll look at") or lower.startswith("next, we’ll look at"):
            return cleaned
    return ""


def transition_mentions_intent(script: str, intent: str) -> bool:
    transition = find_transition_sentence(script)
    if not transition:
        return False
    return all(word in transition.lower() for word in intent.lower().split())


def slide_one_has_hook(script: str) -> bool:
    return "?" in script


def validate_script_rules(
    script: str,
    *,
    is_last: bool,
    next_intent: str,
    is_first: bool,
    is_low_context: bool,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not script_is_plain_narration(script):
        errors.append("contains banned label")

    word_count = _word_count(script)
    if is_first and word_count > 40:
        errors.append("slide one word cap exceeded")
    if is_low_context and word_count > 45:
        errors.append("low context word cap exceeded")

    lowered = script.lower()
    if not is_first and ("welcome" in lowered or "namaste" in lowered):
        errors.append("banned greeting")

    if is_last:
        if find_transition_sentence(script):
            errors.append("last slide has next transition")
        if "like" not in lowered or "subscribe" not in lowered:
            errors.append("missing cta")
        if not any(token in lowered for token in ["risk", "protect", "capital", "over-trading"]):
            errors.append("missing risk note")
    else:
        if not find_transition_sentence(script):
            errors.append("missing transition")
        if any(token in lowered for token in ["like", "subscribe"]):
            errors.append("cta not allowed")
        if next_intent and not transition_mentions_intent(script, next_intent):
            errors.append("transition intent missing")

    return not errors, errors


def generate_scripts_from_images(
    images: List[bytes],
    model_name: str,
    target_words: int = DEFAULT_TARGET_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
    page_texts: Optional[List[str]] = None,
) -> Tuple[List[str], Path]:
    client = _build_client()
    scripts: List[str] = []
    total_slides = len(images)
    scripts_dir, _ = create_scripts_job_dir()
    page_texts = page_texts or []

    for index, image in enumerate(images, start=1):
        logger.info("Generating script for slide %s/%s", index, total_slides)
        script = generate_script_for_slide(
            image,
            client=client,
            model_name=model_name,
            slide_index=index,
            total_slides=total_slides,
            target_words=target_words,
            max_words=max_words,
            scripts_dir=scripts_dir,
            is_numeric_dense=is_numeric_dense_text(page_texts[index - 1])
            if index - 1 < len(page_texts)
            else False,
            page_text=page_texts[index - 1] if index - 1 < len(page_texts) else "",
        )
        scripts.append(script)

    return scripts, scripts_dir


def generate_viewer_question(full_script: str) -> str:
    client = _build_client()
    model_name = _get_model_name()
    prompt = (
        "You are creating a single short viewer question for a YouTube voiceover. "
        "Use ONLY the information already present in the script below. "
        "Return ONLY one VERY SHORT question. "
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
    if question and not question.endswith("?"):
        question = f"{question}?"
    return question
