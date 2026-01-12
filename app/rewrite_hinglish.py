import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Awaitable, Callable, Iterable, List, Optional

from openai import OpenAI

from app.num_to_words import convert_numerals_to_words, has_disallowed_digits
from app.script_generator import _build_client

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
DEFAULT_HINGLISH_MODEL = "gpt-4.1-mini"
DEFAULT_FALLBACK_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_COMPLETION_TOKENS = 2048
DEFAULT_RETRY_MAX_COMPLETION_TOKENS = 4096
DEFAULT_MAX_RETRIES = 3


class HinglishRewriteError(Exception):
    """Base Hinglish rewrite error."""

    def __init__(
        self,
        message: str,
        *,
        finish_reason: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.finish_reason = finish_reason
        self.model = model


class DigitGuardError(HinglishRewriteError):
    """Digits were not preserved in the output."""


class UnchangedOutputError(HinglishRewriteError):
    """Output matched the input, indicating a failed rewrite."""


class RetryableOutputError(HinglishRewriteError):
    """Retryable output failure, such as empty or truncated output."""


class DigitPostProcessError(HinglishRewriteError):
    """Digits remained after numeral-to-words post-processing."""


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
        or DEFAULT_HINGLISH_MODEL
    )


def _get_fallback_model_name() -> str:
    return os.environ.get("HINGLISH_FALLBACK_MODEL") or DEFAULT_FALLBACK_MODEL


def _get_temperature() -> float:
    raw = os.environ.get("HINGLISH_TEMPERATURE")
    if not raw:
        return DEFAULT_TEMPERATURE
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid HINGLISH_TEMPERATURE=%s; using default.", raw)
        return DEFAULT_TEMPERATURE


def _get_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%s; using default %s.", name, raw, default)
        return default


def _get_max_completion_tokens() -> int:
    return _get_int_env("HINGLISH_MAX_COMPLETION_TOKENS", DEFAULT_MAX_COMPLETION_TOKENS)


def _get_retry_max_completion_tokens() -> int:
    return _get_int_env(
        "HINGLISH_RETRY_MAX_COMPLETION_TOKENS",
        DEFAULT_RETRY_MAX_COMPLETION_TOKENS,
    )


def _get_max_retries() -> int:
    return max(0, _get_int_env("HINGLISH_MAX_RETRIES", DEFAULT_MAX_RETRIES))


def _digit_sequences(text: str) -> List[str]:
    return re.findall(r"-?\d[\d,]*\.?\d*", text)


def _canonical_digits(text: str) -> set[str]:
    return {token.replace(",", "") for token in _digit_sequences(text)}


def _normalize_output(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


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
    max_completion_tokens: Optional[int] = None,
    enforce_digit_guard: bool = True,
) -> str:
    active_client = client or _build_client()
    active_model = model_name or _get_model_name()
    active_temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
    prompt = _build_prompt(text, extra_instruction=extra_instruction)
    active_max_completion_tokens = (
        _get_max_completion_tokens() if max_completion_tokens is None else max_completion_tokens
    )

    result = active_client.chat.completions.create(
        model=active_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=active_temperature,
        top_p=top_p,
        max_completion_tokens=active_max_completion_tokens,
    )

    msg = result.choices[0].message
    output = (getattr(msg, "content", None) or "").strip()
    output = _normalize_output(output)
    finish_reason = getattr(result.choices[0], "finish_reason", None)
    if not output or (finish_reason or "").lower() == "length":
        raise RetryableOutputError(
            "Hinglish rewrite returned empty or truncated output.",
            finish_reason=finish_reason,
            model=active_model,
        )
    if _normalize_output(text) == output:
        raise UnchangedOutputError(
            "Hinglish rewrite output matched input.",
            finish_reason=finish_reason,
            model=active_model,
        )
    if enforce_digit_guard:
        input_digits = _canonical_digits(text)
        output_digits = _canonical_digits(output)
        if not input_digits.issubset(output_digits):
            raise DigitGuardError(
                "Digit guard failed in Hinglish rewrite.",
                finish_reason=finish_reason,
                model=active_model,
            )
    return output


def _postprocess_hinglish_output(text: str) -> str:
    if not _get_env_flag("NUM_TO_WORDS_ENABLED", True):
        return text
    return convert_numerals_to_words(text)


def _has_disallowed_digits(text: str) -> bool:
    if not _get_env_flag("NUM_TO_WORDS_ENABLED", True):
        return False
    return has_disallowed_digits(text)


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
    slide_indices: Optional[Iterable[int]] = None,
) -> list[str]:
    if not _get_env_flag("ENABLE_HINGLISH_REWRITE", True):
        return list(blocks)

    blocks_list = list(blocks)
    total = len(blocks_list)
    active_client = client or _build_client()
    active_model = model_name or _get_model_name()
    fallback_model = _get_fallback_model_name()
    active_temperature = _get_temperature() if temperature is None else temperature
    max_completion_tokens = _get_max_completion_tokens()
    retry_max_completion_tokens = _get_retry_max_completion_tokens()
    max_retries = _get_max_retries()

    rewritten: list[str] = []
    fallback_count = 0
    slide_index_list = list(slide_indices) if slide_indices is not None else None
    if slide_index_list is not None and len(slide_index_list) != total:
        logger.warning(
            "slide_indices length %s does not match blocks length %s; ignoring overrides.",
            len(slide_index_list),
            total,
        )
        slide_index_list = None

    await _maybe_await(on_start, total)
    logger.info("Starting Hinglish rewrite for %s slides.", total)

    for index, block in enumerate(blocks_list, start=1):
        slide_index = slide_index_list[index - 1] if slide_index_list else index
        await _maybe_await(on_slide_start, slide_index, total)

        last_error: Optional[Exception] = None
        last_reason = "unknown"
        last_finish_reason: Optional[str] = None
        last_model: Optional[str] = None
        retry_temperature = active_temperature
        extra_instruction: str | None = None
        skip_model_fallback = False

        for attempt in range(max_retries + 1):
            try:
                attempt_max_tokens = (
                    max_completion_tokens if attempt == 0 else retry_max_completion_tokens
                )
                output = await asyncio.to_thread(
                    rewrite_block_to_hinglish,
                    block,
                    client=active_client,
                    model_name=active_model,
                    temperature=retry_temperature,
                    extra_instruction=extra_instruction,
                    max_completion_tokens=attempt_max_tokens,
                )
                post_processed = _postprocess_hinglish_output(output)
                if _has_disallowed_digits(post_processed):
                    logger.warning(
                        "Digits remain after post-processing for slide %s; retrying once.",
                        slide_index,
                    )
                    try:
                        retry_output = await asyncio.to_thread(
                            rewrite_block_to_hinglish,
                            block,
                            client=active_client,
                            model_name=fallback_model,
                            temperature=retry_temperature,
                            extra_instruction="Avoid numerals; spell out numbers in words.",
                            max_completion_tokens=retry_max_completion_tokens,
                            enforce_digit_guard=False,
                        )
                        retry_processed = _postprocess_hinglish_output(retry_output)
                    except Exception as exc:
                        last_error = exc
                        last_reason = "digits"
                        last_finish_reason = None
                        last_model = fallback_model
                        skip_model_fallback = True
                        logger.warning(
                            "Hinglish digit post-processing retry failed for slide %s: %s",
                            slide_index,
                            exc,
                        )
                        break

                    if _has_disallowed_digits(retry_processed):
                        last_error = DigitPostProcessError(
                            "Digits remain after post-processing retry.",
                            finish_reason=None,
                            model=fallback_model,
                        )
                        last_reason = "digits"
                        last_finish_reason = None
                        last_model = fallback_model
                        skip_model_fallback = True
                        logger.warning(
                            "Hinglish digit post-processing failed for slide %s after retry.",
                            slide_index,
                        )
                        break

                    rewritten.append(retry_processed)
                    last_error = None
                    logger.info(
                        "Hinglish rewrite succeeded for slide %s after digit retry.",
                        slide_index,
                    )
                    break

                rewritten.append(post_processed)
                last_error = None
                logger.info("Hinglish rewrite succeeded for slide %s.", slide_index)
                break

            except DigitGuardError as exc:
                last_error = exc
                last_reason = "digit guard"
                last_finish_reason = exc.finish_reason
                last_model = exc.model or active_model
                retry_temperature = min(retry_temperature, DIGIT_GUARD_RETRY_TEMPERATURE)
                extra_instruction = "Preserve numbers EXACTLY as written."
                logger.warning(
                    "Hinglish rewrite digit guard failed for slide %s (attempt %s/%s): %s",
                    slide_index,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )

            except UnchangedOutputError as exc:
                last_error = exc
                last_reason = "unchanged output"
                last_finish_reason = exc.finish_reason
                last_model = exc.model or active_model
                logger.warning(
                    "Hinglish rewrite returned unchanged output for slide %s (attempt %s/%s): %s",
                    slide_index,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )

            except RetryableOutputError as exc:
                last_error = exc
                last_reason = "empty output"
                last_finish_reason = exc.finish_reason
                last_model = exc.model or active_model
                logger.warning(
                    "Hinglish rewrite returned empty/truncated output for slide %s (attempt %s/%s): %s",
                    slide_index,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )

            except Exception as exc:
                last_error = exc
                last_reason = "openai error"
                last_finish_reason = None
                last_model = active_model
                logger.warning(
                    "Hinglish rewrite failed for slide %s (attempt %s/%s): %s",
                    slide_index,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )

        if last_error and not skip_model_fallback:
            try:
                output = await asyncio.to_thread(
                    rewrite_block_to_hinglish,
                    block,
                    client=active_client,
                    model_name=fallback_model,
                    temperature=retry_temperature,
                    extra_instruction=extra_instruction,
                    max_completion_tokens=retry_max_completion_tokens,
                )
                post_processed = _postprocess_hinglish_output(output)
                if _has_disallowed_digits(post_processed):
                    raise DigitPostProcessError(
                        "Digits remain after post-processing fallback output.",
                        finish_reason=None,
                        model=fallback_model,
                    )
                rewritten.append(post_processed)
                last_error = None
                logger.info(
                    "Hinglish rewrite succeeded for slide %s using fallback model %s.",
                    slide_index,
                    fallback_model,
                )
            except HinglishRewriteError as exc:
                last_error = exc
                last_reason = "digits" if isinstance(exc, DigitPostProcessError) else "model failed"
                last_finish_reason = exc.finish_reason
                last_model = exc.model or fallback_model
            except Exception as exc:
                last_error = exc
                last_reason = "model failed"
                last_finish_reason = None
                last_model = fallback_model

        if last_error:
            fallback_count += 1
            logger.warning(
                "Falling back to original script for slide %s after retries (reason: %s, model=%s, finish_reason=%s).",
                slide_index,
                last_reason,
                last_model,
                last_finish_reason,
            )
            await _maybe_await(on_slide_fallback, slide_index, total, last_reason)
            fallback_block = (
                block if last_reason == "digits" else _postprocess_hinglish_output(block)
            )
            if last_reason != "digits" and _has_disallowed_digits(fallback_block):
                logger.warning(
                    "Digits remain in fallback script for slide %s after post-processing.",
                    slide_index,
                )
            rewritten.append(fallback_block)

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
