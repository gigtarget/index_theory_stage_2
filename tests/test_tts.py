import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app import tts


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        headers: dict[str, str] | None = None,
        content: bytes = b"audio",
        text: str = "",
        json_data: dict | None = None,
    ):
        self.status_code = status_code
        self.headers = headers or {}
        self.content = content
        self.text = text
        self._json_data = json_data

    def json(self) -> dict:
        if self._json_data is None:
            raise ValueError("No JSON")
        return self._json_data


def test_synthesize_tts_skips_empty(monkeypatch, tmp_path):
    def _fail_post(*args, **kwargs):
        raise AssertionError("HTTP call should not be made for empty text")

    monkeypatch.setattr(tts.httpx, "post", _fail_post)

    result = tts.synthesize_tts_to_file(
        "   ",
        str(tmp_path / "empty.mp3"),
        model="model",
        voice="voice",
        response_format="mp3",
        speed=1.0,
        instructions=None,
    )
    assert result == ""


def test_synthesize_tts_truncates_long_text(monkeypatch, tmp_path):
    recorder: dict = {}

    def _fake_post(url, headers, json, timeout):
        recorder["payload"] = json
        return _FakeResponse(
            status_code=200,
            headers={"x-character-count": str(len(json["text"]))},
            content=b"audio",
        )

    monkeypatch.setattr(tts.httpx, "post", _fake_post)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "key-1")

    long_text = "a" * (tts.MAX_TTS_CHARS + 200)
    output_path = tmp_path / "out.mp3"
    result = tts.synthesize_tts_to_file(
        long_text,
        str(output_path),
        model="",
        voice="",
        response_format="mp3",
        speed=1.0,
        instructions=None,
    )

    assert result == str(output_path)
    assert output_path.exists()
    assert len(recorder["payload"]["text"]) == tts.MAX_TTS_CHARS


def test_synthesize_tts_rotates_keys(monkeypatch, tmp_path):
    used_keys: list[str] = []

    def _fake_post(url, headers, json, timeout):
        used_keys.append(headers["xi-api-key"])
        if headers["xi-api-key"] == "key-1":
            return _FakeResponse(status_code=401, text="unauthorized")
        return _FakeResponse(
            status_code=200,
            headers={"x-character-count": "12"},
            content=b"audio",
        )

    monkeypatch.setattr(tts.httpx, "post", _fake_post)
    monkeypatch.setenv("ELEVENLABS_API_KEYS", "key-1,key-2")

    output_path = tmp_path / "out.mp3"
    result = tts.synthesize_tts_to_file(
        "hello",
        str(output_path),
        model="model",
        voice="voice",
        response_format="mp3",
        speed=1.0,
        instructions=None,
    )

    assert result == str(output_path)
    assert output_path.read_bytes() == b"audio"
    assert used_keys == ["key-1", "key-2"]


def test_synthesize_tts_usage_tracking(monkeypatch, tmp_path):
    state_path = tmp_path / "usage.json"

    def _fake_post(url, headers, json, timeout):
        return _FakeResponse(
            status_code=200,
            headers={"x-character-count": "15", "request-id": "req-123"},
            content=b"audio",
        )

    monkeypatch.setattr(tts.httpx, "post", _fake_post)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "key-usage")
    monkeypatch.setenv("ELEVENLABS_USAGE_STATE_PATH", str(state_path))

    output_path = tmp_path / "out.mp3"
    result = tts.synthesize_tts_to_file(
        "hello",
        str(output_path),
        model="model",
        voice="voice",
        response_format="mp3",
        speed=1.0,
        instructions=None,
    )

    assert result == str(output_path)
    usage = json.loads(state_path.read_text(encoding="utf-8"))
    assert usage["key-usage"] == 15
