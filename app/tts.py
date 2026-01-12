import logging
from pathlib import Path
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

MAX_TTS_CHARS = 3500
_logged_instructions_unsupported = False


def _build_client() -> OpenAI:
    return OpenAI()


def _sanitize_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    if len(cleaned) > MAX_TTS_CHARS:
        cleaned = cleaned[:MAX_TTS_CHARS]
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
    cleaned = _sanitize_text(text)
    if not cleaned:
        logger.info("Skipping TTS synthesis because text is empty.")
        return ""

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = _build_client()
    try:
        kwargs = {
            "model": model,
            "voice": voice,
            "input": cleaned,
            "response_format": response_format,
            "speed": speed,
        }
        if instructions and instructions.strip():
            kwargs["instructions"] = instructions.strip()
        try:
            with client.audio.speech.with_streaming_response.create(**kwargs) as response:
                response.stream_to_file(output_path)
        except TypeError as exc:
            if (
                "instructions" in str(exc)
                and "unexpected keyword argument" in str(exc)
                and "instructions" in kwargs
            ):
                global _logged_instructions_unsupported
                if not _logged_instructions_unsupported:
                    logger.warning(
                        "TTS: 'instructions' not supported by installed openai SDK; "
                        "retrying without instructions"
                    )
                    _logged_instructions_unsupported = True
                kwargs.pop("instructions", None)
                with client.audio.speech.with_streaming_response.create(**kwargs) as response:
                    response.stream_to_file(output_path)
            else:
                raise
    except Exception as exc:  # pragma: no cover - safety net around SDK
        logger.exception("TTS synthesis failed: %s", exc)
        raise RuntimeError("TTS synthesis failed") from exc

    return str(output_path)
