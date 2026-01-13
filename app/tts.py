import logging
from pathlib import Path
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

MAX_TTS_CHARS = 3500


def _build_client() -> OpenAI:
    return OpenAI()


def prepare_tts_payload(text: str, instructions: Optional[str]) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    if len(cleaned) > MAX_TTS_CHARS:
        cleaned = cleaned[:MAX_TTS_CHARS]
    if instructions and instructions.strip():
        cleaned = f"{instructions.strip()} {cleaned}"
    return cleaned


def synthesize_tts_to_file(
    text: str,
    out_path: str,
    *,
    model: str,
    voice: str,
    response_format: str,
    speed: float,
    instructions: Optional[str],
) -> str:
    cleaned = prepare_tts_payload(text, instructions)
    if not cleaned:
        logger.info("Skipping TTS synthesis because text is empty.")
        return ""

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = _build_client()
    try:
        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=cleaned,
            response_format=response_format,
            speed=speed,
        ) as response:
            response.stream_to_file(output_path)
    except Exception as exc:  # pragma: no cover - safety net around SDK
        logger.exception("TTS synthesis failed: %s", exc)
        raise RuntimeError("TTS synthesis failed") from exc

    return str(output_path)
