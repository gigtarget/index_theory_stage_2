import base64
import hashlib
import json
import logging
import os
import random
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_TARGET_WORDS = 95
DEFAULT_MAX_WORDS = 130
SLIDE_ONE_MIN_WORDS = 40
SLIDE_ONE_MAX_WORDS = 80
DEFAULT_MODEL_NAME = "gpt-5.2"

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

TOPIC_BRIDGES: Dict[str, List[str]] = {
    "indices": [
        "अब indices का snapshot देखते हैं।",
        "चलो indices का snapshot देखते हैं।",
    ],
    "flows": [
        "अब flows—FII vs DII—देखते हैं।",
        "अब FII बनाम DII के flows देखते हैं।",
    ],
    "sectors": [
        "अब sectors पर चलते हैं।",
        "अब sector-wise तस्वीर देखते हैं।",
    ],
    "gainers": [
        "अब top gainers देखते हैं।",
        "अब top movers देखते हैं।",
    ],
    "losers": [
        "अब top losers देखते हैं।",
        "अब laggards पर नज़र डालते हैं।",
    ],
    "breadth": [
        "अब breadth और VIX से tone confirm करते हैं।",
        "अब breadth और VIX से टोन confirm करते हैं।",
    ],
    "levels": [
        "अब कल का playbook—key levels—देखते हैं।",
        "अब key levels और pivots पर चलते हैं।",
    ],
}

FALLBACK_BRIDGES = [
    "चलिए, आगे बढ़ते हैं।",
    "आगे चलते हैं।",
]

BASE_SYSTEM_PROMPT = """
You are a professional video voiceover writer for Indian retail traders.
Use ONLY the content visible on the slide image. Do not add external facts, data, or predictions.
Do not infer macro effects or explanations unless explicitly written on the slide.
No added opinions or forecasts. Keep it retail-friendly and professional.
Tone: professional, slightly dramatic, confident. Keep language simple and natural.
Do not read the slide verbatim; narrate the ideas in a smooth voiceover flow.
Return ONLY the narration text. No headings, labels, or bullet points.
""".strip()

BASE_SYSTEM_PROMPT_YOUTUBE = """
You are a professional YouTube voiceover writer for Indian retail traders.
Use ONLY the content visible on the slide image. Do not add external facts, data, or predictions.
Do not infer causes or explanations unless explicitly written on the slide.
Write in short, speakable lines with a natural, human rhythm.
Avoid robotic phrasing, headings, labels, or bullet points.
Do not read the slide verbatim; narrate the ideas in a smooth flow.
Return ONLY the narration text.
""".strip()

YOUTUBE_OPENERS = [
    "",
    "",
    "",
    "Alright.",
    "Okay.",
    "Quick look.",
]

YOUTUBE_BRIDGES: Dict[str, List[str]] = {
    "indices": ["", "Now to the index picture.", "Moving to indices."],
    "flows": ["", "Now to flows.", "Let’s check flows."],
    "sectors": ["", "Now to sectors.", "Sector-wise next."],
    "gainers": ["", "Now to top movers."],
    "losers": ["", "Now to laggards."],
    "breadth": ["", "Now to breadth and VIX."],
    "levels": ["", "Now to key levels."],
    "fallback": ["", "Moving ahead."],
}

BANNED_REPETITIVE_PHRASES = [
    "Let’s start with",
    "Let's start with",
    "Quick recap",
    "Here’s the snapshot",
    "Here's the snapshot",
]

DEFAULT_OPENER_PROB = 0.10
DEFAULT_BRIDGE_PROB = 0.20


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


def _get_probability(env_name: str, default_value: float) -> float:
    raw = os.environ.get(env_name)
    if raw is None:
        return default_value
    try:
        return float(raw)
    except ValueError:
        return default_value


def _hindi_instruction() -> str:
    if os.environ.get("HINDI_DEVANAGARI", "1") == "0":
        return (
            "Output is English overall. Hindi words/phrases can be in Latin (Hinglish) "
            "or Devanagari script."
        )
    return (
        "Output is English overall, but ANY Hindi words/phrases MUST be in Devanagari script. "
        "Never use romanized Hindi or Hinglish in Latin script."
    )


def _system_prompt() -> str:
    base = (
        BASE_SYSTEM_PROMPT_YOUTUBE
        if _get_voice_style() == "youtube"
        else BASE_SYSTEM_PROMPT
    )
    return f"{base}\n{_hindi_instruction()}".strip()


def _encode_image(image: bytes) -> str:
    return base64.b64encode(image).decode("utf-8")


def _word_count(text: str) -> int:
    return len(text.split())


def _enforce_word_limit(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def _strip_leading_now(text: str) -> str:
    return re.sub(r"^\s*Now,?\s+", "", text, flags=re.IGNORECASE)


def _select_opener(last_opener: Optional[str], cursor: int) -> tuple[str, int]:
    if not CONNECTOR_BANK:
        raise RuntimeError("Connector bank is empty")
    opener = CONNECTOR_BANK[cursor % len(CONNECTOR_BANK)]
    if last_opener and opener == last_opener:
        cursor += 1
        opener = CONNECTOR_BANK[cursor % len(CONNECTOR_BANK)]
    return opener, cursor + 1


def _select_bridge(
    topic: str, last_bridge: Optional[str], cursors: Dict[str, int]
) -> str:
    options = TOPIC_BRIDGES.get(topic, FALLBACK_BRIDGES)
    cursor = cursors[topic]
    bridge = options[cursor % len(options)]
    if last_bridge and bridge == last_bridge and len(options) > 1:
        cursor += 1
        bridge = options[cursor % len(options)]
    cursors[topic] = cursor + 1
    if last_bridge and bridge == last_bridge and options == FALLBACK_BRIDGES:
        alternate = FALLBACK_BRIDGES[(cursor + 1) % len(FALLBACK_BRIDGES)]
        bridge = alternate
        cursors[topic] = cursor + 2
    return bridge


def _stable_job_seed(images: List[bytes]) -> str:
    hasher = hashlib.sha256()
    for image in images:
        hasher.update(image)
    return hasher.hexdigest()


def _maybe_pick(
    rng: random.Random, items: List[str], prob: float, last: Optional[str]
) -> str:
    if rng.random() > prob:
        return ""
    if not items:
        return ""
    options = [item for item in items if item != last]
    if not options:
        options = items
    return rng.choice(options)


def _digit_sequences(text: str) -> List[str]:
    return re.findall(r"\d[\d,.]*", text)


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


def _extract_slide_keywords(image: bytes, client: OpenAI, model_name: str) -> List[str]:
    prompt = (
        "Extract 3-6 short keywords or phrases from this slide only. "
        "Use only words visible on the slide. "
        "Return a comma-separated list, no extra text."
    )
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_encode_image(image)}"},
        },
    ]
    result = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You extract keywords only."},
            {"role": "user", "content": content},
        ],
    )
    raw = (result.choices[0].message.content or "").strip()
    keywords = re.split(r"[,\n;]+", raw)
    return [kw.strip() for kw in keywords if kw.strip()]


def _topic_from_keywords(keywords: List[str]) -> str:
    text = " ".join(keywords).lower()
    if any(term in text for term in ["index", "indices", "snapshot", "nifty", "sensex"]):
        return "indices"
    if any(term in text for term in ["flow", "flows", "fii", "dii", "fpi"]):
        return "flows"
    if any(term in text for term in ["sector", "sectors", "sectoral"]):
        return "sectors"
    if any(term in text for term in ["gainers", "top gainers", "movers", "top movers"]):
        return "gainers"
    if any(term in text for term in ["losers", "laggards", "top losers", "decliners"]):
        return "losers"
    if any(term in text for term in ["breadth", "vix", "advance", "decline"]):
        return "breadth"
    if any(term in text for term in ["levels", "pivot", "support", "resistance", "range"]):
        return "levels"
    return "fallback"


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
    return _enforce_word_limit(script, max_words)


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
    output = normalize_hindi_to_devanagari(output, client, model_name)
    input_digits = set(_digit_sequences(full_script))
    output_digits = set(_digit_sequences(output))
    if not input_digits.issubset(output_digits):
        logger.warning("Digit guard failed in humanize pass; using original script.")
        return full_script
    return output


def generate_scripts_from_images(
    images: List[bytes],
    model_name: str,
    target_words: int = DEFAULT_TARGET_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
) -> List[str]:
    client = _build_client()
    scripts: List[str] = []
    total_slides = len(images)
    job_identifier = uuid.uuid4().hex
    voice_style = _get_voice_style()
    rng_seed = os.environ.get("JOB_ID") or _stable_job_seed(images)
    rng = random.Random(rng_seed)
    opener_prob = _get_probability("OPENER_PROB", DEFAULT_OPENER_PROB)
    bridge_prob = _get_probability("BRIDGE_PROB", DEFAULT_BRIDGE_PROB)
    scripts_dir = Path("outputs") / job_identifier / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    last_opener: Optional[str] = None
    opener_cursor = 0
    last_bridge: Optional[str] = None
    bridge_cursors: Dict[str, int] = defaultdict(int)

    keywords_by_slide = []
    for index, image in enumerate(images, start=1):
        logger.info("Extracting keywords for slide %s/%s", index, total_slides)
        keywords_by_slide.append(_extract_slide_keywords(image, client, model_name))

    for index, image in enumerate(images, start=1):
        logger.info("Generating script for slide %s/%s", index, total_slides)
        opener = None
        if index >= 2 and voice_style != "youtube":
            opener, opener_cursor = _select_opener(last_opener, opener_cursor)
            last_opener = opener
        elif index >= 2 and voice_style == "youtube":
            opener = _maybe_pick(rng, YOUTUBE_OPENERS, opener_prob, last_opener)
            if opener:
                last_opener = opener

        next_keywords = keywords_by_slide[index] if index < total_slides else []
        topic = _topic_from_keywords(next_keywords)
        bridge_line = ""
        if index < total_slides and voice_style != "youtube":
            bridge_line = _select_bridge(topic, last_bridge, bridge_cursors)
            last_bridge = bridge_line
        elif index < total_slides and voice_style == "youtube":
            bridge_options = YOUTUBE_BRIDGES.get(topic, YOUTUBE_BRIDGES["fallback"])
            bridge_line = _maybe_pick(rng, bridge_options, bridge_prob, last_bridge)
            if bridge_line:
                last_bridge = bridge_line

        if index == 1 and voice_style != "youtube":
            slide_target_words = max(
                SLIDE_ONE_MIN_WORDS, min(target_words, SLIDE_ONE_MAX_WORDS)
            )
            slide_max_words = min(max_words, SLIDE_ONE_MAX_WORDS)
            welcome_line = "नमस्ते! Welcome to Index Theory’s Post Market Report."
            outline_line = (
                "We’ll cover indices, flows, sectors, breadth, and key levels—only from today’s slides."
            )
            kickoff_line = f"{bridge_line} चलिए, शुरू करते हैं।".strip()
            fixed_words = _word_count(f"{welcome_line} {outline_line} {kickoff_line}")
            body_target = max(10, slide_target_words - fixed_words)
            body_max = max(12, slide_max_words - fixed_words)
            body_instruction = (
                "Provide 1-2 short sentences summarizing only this slide. "
                "Do not add greetings, outlines, transitions, or CTA."
            )
            body = _generate_slide_body(
                image, client, model_name, body_instruction, body_target, body_max
            )
            slide_lines = [welcome_line, outline_line, body, kickoff_line]
            script = "\n".join(line for line in slide_lines if line)
            script = _enforce_word_limit(script, SLIDE_ONE_MAX_WORDS)
        else:
            slide_target_words = target_words
            slide_max_words = max_words
            if index == total_slides and voice_style != "youtube":
                body_instruction = (
                    "Provide a 1-2 line recap strictly from this slide. "
                    "Do NOT include any greeting, opener, transition, CTA, or viewer question."
                )
            elif index == total_slides and voice_style == "youtube":
                body_instruction = (
                    "Provide a short recap strictly from this slide. "
                    "Do NOT include any greeting, opener, transition, CTA, or viewer question."
                )
            else:
                body_instruction = (
                    "Do NOT include any greeting, opener, transition, or CTA. "
                    "Write smooth narration from only the slide content."
                )
            body = _generate_slide_body(
                image, client, model_name, body_instruction, slide_target_words, slide_max_words
            )
            opener_text = f"{opener} " if opener else ""
            script = f"{opener_text}{body}".strip()
            if index < total_slides and bridge_line:
                script = f"{script} {bridge_line}".strip()
            if index == total_slides and voice_style != "youtube":
                cta_lines = [
                    "Like, Subscribe, and share your view in the comments.",
                    "Hit the bell for updates.",
                ]
                script = f"{script}\n" + "\n".join(cta_lines)

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


def generate_viewer_question(full_script: str) -> str:
    client = _build_client()
    model_name = _get_model_name()
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
