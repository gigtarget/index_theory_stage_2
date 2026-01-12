import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app import tts


class _FakeResponse:
    def __init__(self, recorder: dict):
        self._recorder = recorder

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def stream_to_file(self, path: Path) -> None:
        path.write_text("audio", encoding="utf-8")
        self._recorder["streamed_to"] = str(path)


class _FakeStreaming:
    def __init__(self, recorder: dict):
        self._recorder = recorder

    def create(self, **kwargs):
        self._recorder["kwargs"] = kwargs
        return _FakeResponse(self._recorder)


class _FakeSpeech:
    def __init__(self, recorder: dict):
        self.with_streaming_response = _FakeStreaming(recorder)


class _FakeAudio:
    def __init__(self, recorder: dict):
        self.speech = _FakeSpeech(recorder)


class _FakeClient:
    def __init__(self, recorder: dict):
        self.audio = _FakeAudio(recorder)


def test_synthesize_tts_skips_empty(monkeypatch, tmp_path):
    def _fail_build_client():
        raise AssertionError("OpenAI client should not be constructed for empty text")

    monkeypatch.setattr(tts, "_build_client", _fail_build_client)

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

    monkeypatch.setattr(tts, "_build_client", lambda: _FakeClient(recorder))

    long_text = "a" * (tts.MAX_TTS_CHARS + 200)
    output_path = tmp_path / "out.mp3"
    result = tts.synthesize_tts_to_file(
        long_text,
        str(output_path),
        model="model",
        voice="voice",
        response_format="mp3",
        speed=1.0,
        instructions=None,
    )

    assert result == str(output_path)
    assert output_path.exists()
    assert len(recorder["kwargs"]["input"]) == tts.MAX_TTS_CHARS

