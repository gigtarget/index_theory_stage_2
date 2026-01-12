import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Awaitable, Callable, Iterable, List, Optional

from openai import OpenAI

from app.script_generator import DEFAULT_MODEL_NAME, _build_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a financial market commentator. Rewrite text into natural Hinglish narration for voiceover. Follow rules strictly.
Return plain text only. Do not call tools. Do not output JSON.
""".strip()

USER_PROMPT_TEMPLATE = """
Convert the following market flow text into natural Hinglish narration suitable for a voiceover.

Rules:
- Hinglish, conversational but professional.
- Must flow like a human explanation, not like reading a PPT/report.
- Keep cause–effect reasoning (why it matters, what it signals).
- No bold, no headings, no bullet points, no numbering.
- Keep numbers exactly as provided (do not convert).
- Do not copy sentences verbatim; rephrase everything.
- Do not add new data or assumptions.
- Avoid repetitive phrasing and robotic structure.
- Output length should be similar to input.
- Return ONLY the rewritten narration text.

Input:
<<<SCRIPT_BLOCK>>>
""".strip()

DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
DIGIT_GUARD_RETRY_TEMPERATURE = 0.2
MAX_REWRITE_ATTEMPTS = 3


class HinglishRewriteError(Exception):
    """Base Hinglish rewrite error."""


class DigitGuardError(HinglishRewriteError):
    """Digits were not preserved in the output."""


class UnchangedOutputError(HinglishRewriteError):
    """Output matched the input, indicating a failed rewrite."""


def _get_env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _get_model_name() -> str:
    return (
        os.environ.get("HINGLISH_MODEL")
        or os.environ.get("MODEL_NAME")
        or os.environ.get("OPENAI_MODEL")
        or DEFAULT_MODEL_NAME
    )


def _get_temperature() -> float:
    raw = os.environ.get("HINGLISH_TEMPERATURE")
    if not raw:
        return DEFAULT_TEMPERATURE
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid HINGLISH_TEMPERATURE=%s; using default.", raw)
        return DEFAULT_TEMPERATURE


def _max_tokens_for_text(text: str) -> int:
    word_count = max(1, len(text.split()))
    return max(64, int(word_count * 2.5))


def _digit_sequences(text: str) -> List[str]:
    return re.findall(r"-?\d[\d,]*\.?\d*", text)


def _canonical_digits(text: str) -> set[str]:
    return {token.replace(",", "") for token in _digit_sequences(text)}


def _normalize_output(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _response_format_unsupported(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "response_format" in message
        and (
            "unsupported" in message
            or "unknown" in message
            or "not allowed" in message
            or "unrecognized" in message
            or "invalid" in message
        )
    )


def _build_prompt(block: str, extra_instruction: str | None = None) -> str:
    prompt = USER_PROMPT_TEMPLATE
    if extra_instruction:
        prompt = prompt.replace("Rules:", f"Rules:\n- {extra_instruction}")
    return prompt.replace("<<<SCRIPT_BLOCK>>>", block)


async def _maybe_await(
    callback: Optional[Callable[..., Awaitable[None] | None]], *args: object
) -> None:
    if not callback:
        return
    result = callback(*args)
    if asyncio.iscoroutine(result):
        await result


def rewrite_block_to_hinglish(
    text: str,
    *,
    client: Optional[OpenAI] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: float = DEFAULT_TOP_P,
    extra_instruction: str | None = None,
) -> str:
    active_client = client or _build_client()
    active_model = model_name or _get_model_name()
    active_temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
    prompt = _build_prompt(text, extra_instruction=extra_instruction)

    # FIX: newer OpenAI models reject `max_tokens` and require `max_completion_tokens`
    try:
        result = active_client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=active_temperature,
            top_p=top_p,
            max_completion_tokens=_max_tokens_for_text(text),
            response_format={"type": "text"},
        )
    except Exception as exc:
        if not _response_format_unsupported(exc):
            raise
        logger.info(
            "response_format not supported for model %s; retrying without it.",
            active_model,
        )
        result = active_client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=active_temperature,
            top_p=top_p,
            max_completion_tokens=_max_tokens_for_text(text),
        )

    msg = result.choices[0].message
    output = (getattr(msg, "content", None) or "").strip()
    output = _normalize_output(output)
    if not output:
        finish_reason = getattr(result.choices[0], "finish_reason", None)
        logger.warning(
            "Empty Hinglish rewrite output. model=%s finish_reason=%s message=%r choice=%r",
            active_model,
            finish_reason,
            msg,
            result.choices[0],
        )
        raise ValueError("Empty Hinglish rewrite output.")
    if _normalize_output(text) == output:
        raise UnchangedOutputError("Hinglish rewrite output matched input.")
    input_digits = _canonical_digits(text)
    output_digits = _canonical_digits(output)
    if not input_digits.issubset(output_digits):
        raise DigitGuardError("Digit guard failed in Hinglish rewrite.")
    return output


async def rewrite_all_blocks(
    blocks: Iterable[str],
    *,
    client: Optional[OpenAI] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    on_start: Optional[Callable[[int], Awaitable[None] | None]] = None,
    on_slide_start: Optional[Callable[[int, int], Awaitable[None] | None]] = None,
    on_slide_fallback: Optional[Callable[[int, int, str], Awaitable[None] | None]] = None,
    on_done: Optional[Callable[[int], Awaitable[None] | None]] = None,
) -> list[str]:
    if not _get_env_flag("ENABLE_HINGLISH_REWRITE", True):
        return list(blocks)

    blocks_list = list(blocks)
    total = len(blocks_list)
    active_client = client or _build_client()
    active_model = model_name or _get_model_name()
    active_temperature = _get_temperature() if temperature is None else temperature

    rewritten: list[str] = []
    fallback_count = 0

    await _maybe_await(on_start, total)
    logger.info("Starting Hinglish rewrite for %s slides.", total)

    for index, block in enumerate(blocks_list, start=1):
        await _maybe_await(on_slide_start, index, total)

        last_error: Optional[Exception] = None
        last_reason = "unknown"
        retry_temperature = active_temperature
        extra_instruction: str | None = None

        for attempt in range(MAX_REWRITE_ATTEMPTS):
            try:
                output = await asyncio.to_thread(
                    rewrite_block_to_hinglish,
                    block,
                    client=active_client,
                    model_name=active_model,
                    temperature=retry_temperature,
                    extra_instruction=extra_instruction,
                )
                rewritten.append(output)
                last_error = None
                logger.info("Hinglish rewrite succeeded for slide %s.", index)
                break

            except DigitGuardError as exc:
                last_error = exc
                last_reason = "digit guard"
                retry_temperature = min(retry_temperature, DIGIT_GUARD_RETRY_TEMPERATURE)
                extra_instruction = "Preserve numbers EXACTLY as written."
                logger.warning(
                    "Hinglish rewrite digit guard failed for slide %s (attempt %s/%s): %s",
                    index,
                    attempt + 1,
                    MAX_REWRITE_ATTEMPTS,
                    exc,
                )

            except UnchangedOutputError as exc:
                last_error = exc
                last_reason = "unchanged output"
                logger.warning(
                    "Hinglish rewrite returned unchanged output for slide %s (attempt %s/%s): %s",
                    index,
                    attempt + 1,
                    MAX_REWRITE_ATTEMPTS,
                    exc,
                )

            except Exception as exc:
                last_error = exc
                last_reason = "openai error"
                logger.warning(
                    "Hinglish rewrite failed for slide %s (attempt %s/%s): %s",
                    index,
                    attempt + 1,
                    MAX_REWRITE_ATTEMPTS,
                    exc,
                )

        if last_error:
            fallback_count += 1
            logger.warning(
                "Falling back to original script for slide %s after retries (reason: %s).",
                index,
                last_reason,
            )
            await _maybe_await(on_slide_fallback, index, total, last_reason)
            rewritten.append(block)

    logger.info(
        "Completed Hinglish rewrite. slides=%s fallbacks=%s",
        total,
        fallback_count,
    )
    await _maybe_await(on_done, total)
    return rewritten


def rewrite_files(input_dir: Path, output_dir: Path) -> list[str]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_files = [path for path in input_dir.glob("*.txt") if path.is_file()]
    script_files.sort(key=lambda path: _extract_index(path.name))
    blocks = [path.read_text(encoding="utf-8") for path in script_files]
    rewritten = asyncio.run(rewrite_all_blocks(blocks))
    for path, block in zip(script_files, rewritten, strict=False):
        out_path = output_dir / path.name
        out_path.write_text(block, encoding="utf-8")
    return rewritten


def write_blocks(blocks: Iterable[str], output_dir: Path, prefix: str = "slide_") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, block in enumerate(blocks, start=1):
        out_path = output_dir / f"{prefix}{index}.txt"
        out_path.write_text(block, encoding="utf-8")


def _extract_index(filename: str) -> int:
    match = re.search(r"(\d+)", filename)
    if not match:
        return 0
    return int(match.group(1))
