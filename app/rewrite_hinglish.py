import logging
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

from openai import OpenAI

from app.script_generator import DEFAULT_MODEL_NAME, _build_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a financial market commentator. Rewrite text into natural Hinglish narration for voiceover. Follow rules strictly.
""".strip()

USER_PROMPT_TEMPLATE = """
Convert the following market flow text into natural Hinglish narration suitable for a voiceover.

Rules:
- Hinglish, conversational but professional.
- Must flow like a human explanation, not like reading a PPT/report.
- Keep cause–effect reasoning (why it matters, what it signals).
- No bold, no headings, no bullet points, no numbering.
- Keep numbers exactly as provided (do not convert).
- Do not add new data or assumptions.
- Avoid repetitive phrasing and robotic structure.
- Output length should be similar to input.
- Return ONLY the rewritten narration text.

Input:
<<<SCRIPT_BLOCK>>>
""".strip()

DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9


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
    return re.findall(r"\d[\d,.]*", text)


def _normalize_output(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _build_prompt(block: str) -> str:
    return USER_PROMPT_TEMPLATE.replace("<<<SCRIPT_BLOCK>>>", block)


def rewrite_block_to_hinglish(
    text: str,
    *,
    client: Optional[OpenAI] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: float = DEFAULT_TOP_P,
) -> str:
    active_client = client or _build_client()
    active_model = model_name or _get_model_name()
    active_temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
    prompt = _build_prompt(text)
    result = active_client.chat.completions.create(
        model=active_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=active_temperature,
        top_p=top_p,
        max_tokens=_max_tokens_for_text(text),
    )
    output = (result.choices[0].message.content or "").strip()
    output = _normalize_output(output)
    if not output:
        raise ValueError("Empty Hinglish rewrite output.")
    input_digits = set(_digit_sequences(text))
    output_digits = set(_digit_sequences(output))
    if not input_digits.issubset(output_digits):
        raise ValueError("Digit guard failed in Hinglish rewrite.")
    return output


def rewrite_all_blocks(blocks: Iterable[str]) -> list[str]:
    if not _get_env_flag("ENABLE_HINGLISH_REWRITE", True):
        return list(blocks)
    client = _build_client()
    model_name = _get_model_name()
    temperature = _get_temperature()
    rewritten: list[str] = []
    for index, block in enumerate(blocks, start=1):
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                rewritten.append(
                    rewrite_block_to_hinglish(
                        block,
                        client=client,
                        model_name=model_name,
                        temperature=temperature,
                    )
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Hinglish rewrite failed for block %s (attempt %s): %s",
                    index,
                    attempt + 1,
                    exc,
                )
        if last_error:
            logger.warning(
                "Falling back to original script for block %s after retries.",
                index,
            )
            rewritten.append(block)
    return rewritten


def rewrite_files(input_dir: Path, output_dir: Path) -> list[str]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_files = [path for path in input_dir.glob("*.txt") if path.is_file()]
    script_files.sort(key=lambda path: _extract_index(path.name))
    blocks = [path.read_text(encoding="utf-8") for path in script_files]
    rewritten = rewrite_all_blocks(blocks)
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
