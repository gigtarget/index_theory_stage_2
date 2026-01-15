import json
import logging
import os
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

MAX_TTS_CHARS = 3500
DEFAULT_VOICE_ID = "VbDz3QQGkAGePVWfkfwE"
DEFAULT_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
DEFAULT_USAGE_STATE_PATH = "/tmp/elevenlabs_usage.json"
_KEYS_LOGGED = False


def prepare_tts_payload(text: str, instructions: Optional[str]) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    if len(cleaned) > MAX_TTS_CHARS:
        cleaned = cleaned[:MAX_TTS_CHARS]
    return cleaned


def _load_usage_state(state_path: Path) -> dict[str, int]:
    if not state_path.exists():
        return {}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read ElevenLabs usage state: %s", exc)
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: int(value) for key, value in data.items() if isinstance(value, int)}


def _save_usage_state(state_path: Path, usage: dict[str, int]) -> None:
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(usage, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to persist ElevenLabs usage state: %s", exc)


def _sanitize_key(raw: str) -> str:
    return raw.strip().strip('"').strip("'").replace("\r", "").replace("\n", "")


def _parse_api_keys() -> list[str]:
    global _KEYS_LOGGED
    raw_keys = os.environ.get("ELEVENLABS_API_KEYS", "")
    raw_single = os.environ.get("ELEVENLABS_API_KEY", "")
    raw_entries = raw_keys.split(",") if raw_keys else ([raw_single] if raw_single else [])
    keys = [_sanitize_key(key) for key in raw_entries]
    keys = [key for key in keys if key]
    raw_combined = ",".join(raw_entries)
    has_whitespace = any(" " in entry for entry in raw_entries)
    has_newlines = "\n" in raw_combined or "\r" in raw_combined
    if raw_entries and (has_whitespace or has_newlines):
        logger.warning("ELEVENLABS_API_KEYS contains whitespace/newlines; sanitized.")
    if not keys:
        single_key = _sanitize_key(raw_single)
        if single_key:
            keys = [single_key]
    if keys and not _KEYS_LOGGED:
        logger.info("ElevenLabs keys loaded: %s", [_mask_key(key) for key in keys])
        logger.info("ElevenLabs key lengths: %s", [len(key) for key in keys])
        _KEYS_LOGGED = True
    if keys:
        return keys
    raise RuntimeError("ElevenLabs API keys not configured")


def _get_voice_id(voice: str) -> str:
    env_voice = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
    if voice and voice.strip():
        return voice.strip()
    return env_voice or DEFAULT_VOICE_ID


def _get_model_id(model: str) -> str:
    env_model = os.environ.get("ELEVENLABS_MODEL_ID", "").strip()
    candidate = model.strip() if model else ""
    return env_model or candidate or DEFAULT_MODEL_ID


def _map_output_format(response_format: str) -> str:
    env_format = os.environ.get("ELEVENLABS_OUTPUT_FORMAT", "").strip()
    if env_format:
        return env_format
    fmt = response_format.strip().lower() if response_format else ""
    if not fmt or fmt == "mp3":
        return DEFAULT_OUTPUT_FORMAT
    if fmt == "wav":
        return "wav_44100"
    return fmt


def _get_max_chars_per_key() -> int | None:
    raw = os.environ.get("ELEVENLABS_MAX_CHARS_PER_KEY", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid ELEVENLABS_MAX_CHARS_PER_KEY=%s", raw)
        return None
    return value if value > 0 else None


def _get_usage_state_path() -> Path:
    raw_path = os.environ.get("ELEVENLABS_USAGE_STATE_PATH", "").strip()
    return Path(raw_path or DEFAULT_USAGE_STATE_PATH)


def _mask_key(key: str) -> str:
    if len(key) <= 4:
        return "****"
    return f"****{key[-4:]}"


def _should_rotate_from_response(response: httpx.Response) -> bool:
    if response.status_code in {401, 403, 429, 402}:
        return True
    body = ""
    try:
        body = response.text.lower()
    except Exception:
        body = ""
    tokens = ("quota", "insufficient", "limit", "credits")
    return any(token in body for token in tokens)


def validate_key(api_key: str) -> bool:
    headers = {"xi-api-key": api_key}
    try:
        response = httpx.get("https://api.elevenlabs.io/v1/user", headers=headers, timeout=10.0)
    except httpx.RequestError as exc:
        logger.warning("ElevenLabs key validation failed due to network error: %s", exc)
        return False
    if response.status_code == 200:
        logger.info("ElevenLabs key validated ok.")
        return True
    logger.warning("ElevenLabs key validation failed with status %s.", response.status_code)
    return False


def _request_tts(
    *,
    api_key: str,
    voice_id: str,
    model_id: str,
    output_format: str,
    text: str,
) -> httpx.Response:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "output_format": output_format,
    }
    return httpx.post(url, headers=headers, json=payload, timeout=30.0)


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

    keys = _parse_api_keys()
    voice_id = _get_voice_id(voice)
    model_id = _get_model_id(model)
    output_format = _map_output_format(response_format)
    max_chars = _get_max_chars_per_key()
    state_path = _get_usage_state_path()
    usage_state = _load_usage_state(state_path)

    last_error: Exception | None = None

    for index, api_key in enumerate(keys, start=1):
        api_key = _sanitize_key(api_key)
        masked = _mask_key(api_key)
        used_chars = usage_state.get(api_key, 0)
        if max_chars is not None and used_chars + len(cleaned) > max_chars:
            logger.warning(
                "ElevenLabs key %s (index %s) skipped: usage %s exceeds max %s.",
                masked,
                index,
                used_chars,
                max_chars,
            )
            continue
        if not validate_key(api_key):
            continue
        try:
            response = _request_tts(
                api_key=api_key,
                voice_id=voice_id,
                model_id=model_id,
                output_format=output_format,
                text=cleaned,
            )
        except httpx.RequestError as exc:
            last_error = exc
            logger.warning(
                "ElevenLabs key %s (index %s) failed with network error: %s",
                masked,
                index,
                exc,
            )
            continue

        if response.status_code >= 400:
            if _should_rotate_from_response(response):
                last_error = RuntimeError(
                    f"ElevenLabs key {masked} (index {index}) failed with status "
                    f"{response.status_code}"
                )
                logger.warning(
                    "ElevenLabs key %s (index %s) rejected with status %s; rotating.",
                    masked,
                    index,
                    response.status_code,
                )
                continue
            error_detail = response.text
            logger.error(
                "ElevenLabs key %s (index %s) failed with status %s: %s",
                masked,
                index,
                response.status_code,
                error_detail,
            )
            raise RuntimeError(
                f"ElevenLabs TTS failed with status {response.status_code}: {error_detail}"
            )

        output_path.write_bytes(response.content)
        char_header = response.headers.get("x-character-count")
        request_id = response.headers.get("request-id")
        logger.info(
            "ElevenLabs TTS succeeded with key %s (index %s). request-id=%s",
            masked,
            index,
            request_id,
        )
        if char_header is not None:
            try:
                usage_state[api_key] = usage_state.get(api_key, 0) + int(char_header)
            except ValueError:
                logger.warning("Invalid x-character-count header: %s", char_header)
        _save_usage_state(state_path, usage_state)
        return str(output_path)

    logger.exception("ElevenLabs TTS failed after trying %s keys.", len(keys))
    raise RuntimeError("ElevenLabs TTS failed after exhausting API keys") from last_error
