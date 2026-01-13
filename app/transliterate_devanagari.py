import logging
import re
import time
from typing import Iterable

logger = logging.getLogger(__name__)

MAX_SEGMENT_CHARS = 2000
MAX_RETRIES = 3

SYSTEM_PROMPT = (
    "You convert Latin-script Hinglish/English narration into Devanagari script ONLY "
    "(transliteration), keeping the SAME words and meaning. Never translate. "
    "Keep punctuation, numbers, whitespace unchanged. Leave protected tokens unchanged."
)

USER_PROMPT_TEMPLATE = (
    "Return ONLY the transliterated text.\n"
    "Rules:\n"
    "- No translation.\n"
    "- Keep digits as-is.\n"
    "- Keep punctuation and spacing identical.\n"
    "Input:\n"
    "<<<{segment}>>>"
)

PROTECTED_PATTERN = re.compile(
    r"("
    r"https?://[^\s]+"
    r"|www\.[^\s]+"
    r"|\b[\w.+-]+@[\w.-]+\.\w+\b"
    r"|@\w+"
    r"|#\w+"
    r"|`[^`]*`"
    r"|\b[A-Z]{2,8}\b"
    r"|\bIndex Theory(?:['’]s)?\b"
    r"|\bBank Nifty(?:['’]s)?\b"
    r"|\bNifty(?:['’]s)?\b"
    r"|\bSensex(?:['’]s)?\b"
    r"|\bFII\b"
    r"|\bDII\b"
    r")"
)


def _iter_parts(text: str) -> Iterable[tuple[str, bool]]:
    cursor = 0
    for match in PROTECTED_PATTERN.finditer(text):
        start, end = match.span()
        if start > cursor:
            yield text[cursor:start], False
        yield text[start:end], True
        cursor = end
    if cursor < len(text):
        yield text[cursor:], False


def _split_sentences(text: str) -> list[str]:
    if not text:
        return [""]
    parts: list[str] = []
    cursor = 0
    for match in re.finditer(r".*?(?:[.!?]+(?:\s+|$)|\n+)", text, flags=re.S):
        parts.append(match.group(0))
        cursor = match.end()
    if cursor < len(text):
        parts.append(text[cursor:])
    return parts


def _chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    buffer = ""
    for sentence in _split_sentences(text):
        if not sentence:
            continue
        if len(buffer) + len(sentence) <= max_chars:
            buffer += sentence
            continue
        if buffer:
            chunks.append(buffer)
            buffer = ""
        if len(sentence) > max_chars:
            chunks.extend(
                [
                    sentence[i : i + max_chars]
                    for i in range(0, len(sentence), max_chars)
                ]
            )
        else:
            buffer = sentence
    if buffer:
        chunks.append(buffer)
    return chunks


def _call_llm(segment: str, client, model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(segment=segment)},
        ],
    )
    return response.choices[0].message.content or ""


def _transliterate_segment(segment: str, client, model: str) -> str:
    if not segment:
        return segment
    attempts = 0
    delay = 0.5
    while True:
        try:
            return _call_llm(segment, client, model)
        except Exception as exc:  # pragma: no cover - network/API safety
            attempts += 1
            if attempts >= MAX_RETRIES:
                logger.warning("LLM transliteration failed after retries: %s", exc)
                return segment
            logger.warning("LLM transliteration failed, retrying: %s", exc)
            time.sleep(delay)
            delay *= 2


def llm_transliterate_to_devanagari(text: str, client, model: str) -> str:
    if not text:
        return ""

    output_parts: list[str] = []
    for part, is_protected in _iter_parts(text):
        if is_protected:
            output_parts.append(part)
            continue
        for chunk in _chunk_text(part, MAX_SEGMENT_CHARS):
            output_parts.append(_transliterate_segment(chunk, client, model))
    return "".join(output_parts)
