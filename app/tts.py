import logging
from pathlib import Path
from typing import Optional

import httpx
from openai import OpenAI

from app.tts_kokoro import synthesize_kokoro_to_file
from app.tts_text import hinglish_to_devanagari

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


def _kokoro_fal_tts_to_file(
    text: str,
    out_path: str,
    *,
    voice: str,
    speed: float,
    endpoint: str,
    fal_key: str,
) -> str:
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "prompt": text,
        "voice": voice,
        "speed": speed,
    }
    headers = {
        "Authorization": f"Key {fal_key}",
        "Content-Type": "application/json",
    }
    url = f"https://fal.run/{endpoint}"

    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        response.raise_for_status()
        result = response.json()
        audio_url = result["audio"]["url"]
        audio_response = httpx.get(audio_url, timeout=60.0)
        audio_response.raise_for_status()
        output_path.write_bytes(audio_response.content)
    except Exception as exc:  # pragma: no cover - safety net around HTTP
        logger.exception("Kokoro TTS synthesis failed: %s", exc)
        raise RuntimeError("Kokoro TTS synthesis failed") from exc

    return str(output_path)


def synthesize_tts_to_file(
    text: str,
    out_path: str,
    *,
    model: str,
    text_model: Optional[str] = None,
    voice: str,
    response_format: str,
    speed: float,
    instructions: Optional[str],
    tts_text_mode: str = "hinglish",
    provider: str = "openai",
    kokoro_lang: str = "h",
    kokoro_voice: str = "hm_omega",
    kokoro_speed: float = 1.0,
    kokoro_endpoint: str = "fal-ai/kokoro/hindi",
    fal_key: Optional[str] = None,
) -> str:
    cleaned = _sanitize_text(text)
    if not cleaned:
        logger.info("Skipping TTS synthesis because text is empty.")
        return ""

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if provider == "fal_kokoro":
        if not fal_key:
            logger.error("Kokoro TTS requested but FAL_KEY is missing.")
            raise RuntimeError("Kokoro TTS not configured")
        return _kokoro_fal_tts_to_file(
            cleaned,
            str(output_path),
            voice=kokoro_voice,
            speed=kokoro_speed,
            endpoint=kokoro_endpoint,
            fal_key=fal_key,
        )

    if provider == "kokoro_local":
        kokoro_input = cleaned
        if tts_text_mode == "devanagari" and text_model:
            try:
                kokoro_input = hinglish_to_devanagari(
                    cleaned,
                    model_name=text_model,
                    client=_build_client(),
                )
            except Exception as exc:  # pragma: no cover - safety net
                logger.exception("Failed to convert Hinglish to Devanagari: %s", exc)
                kokoro_input = cleaned
        return synthesize_kokoro_to_file(
            kokoro_input,
            str(output_path),
            lang_code=kokoro_lang,
            voice=kokoro_voice,
            speed=kokoro_speed,
        )

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
