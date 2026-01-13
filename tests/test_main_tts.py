import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app import main


class _DummyMessage:
    def __init__(self):
        self.message_id = 1


class _FakeBot:
    def __init__(self):
        self.audio_calls = []

    async def send_audio(self, chat_id, audio, filename, caption):
        self.audio_calls.append(
            {"chat_id": chat_id, "filename": filename, "caption": caption}
        )

    async def delete_message(self, chat_id, message_id):
        return None


async def _noop_send_message(context, chat_id, text):
    return _DummyMessage()


async def _noop_send_long(context, chat_id, text, limit=3500):
    return None


def test_generate_and_send_scripts_calls_tts_per_slide(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OUTPUT_MODE", "slides")
    monkeypatch.setenv("TTS_ENABLED", "true")
    monkeypatch.setenv("TTS_KEEP_FILES", "true")
    monkeypatch.setenv("VOICE_STYLE", "youtube")

    monkeypatch.setattr(main, "_send_message", _noop_send_message)
    monkeypatch.setattr(main, "_send_long", _noop_send_long)
    monkeypatch.setattr(main, "generate_script_for_slide", lambda *args, **kwargs: "script")

    monkeypatch.setattr(main, "format_allcaps_words", lambda text: text)
    monkeypatch.setattr(main, "_build_client", lambda: object())

    def _create_scripts_job_dir():
        scripts_dir = tmp_path / "scripts"
        original_dir = scripts_dir / "original"
        original_dir.mkdir(parents=True, exist_ok=True)
        return scripts_dir, original_dir

    monkeypatch.setattr(main, "create_scripts_job_dir", _create_scripts_job_dir)

    tts_calls = []

    def _fake_synthesize(text, out_path, **kwargs):
        tts_calls.append({"text": text, "out_path": out_path})
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("audio", encoding="utf-8")
        return out_path

    monkeypatch.setattr(main, "synthesize_tts_to_file", _fake_synthesize)

    async def _fast_sleep(_):
        return None

    monkeypatch.setattr(main.asyncio, "sleep", _fast_sleep)

    context = SimpleNamespace(bot=_FakeBot())

    asyncio.run(main._generate_and_send_scripts(context, 123, [b"one", b"two"]))

    assert len(tts_calls) == 2
    assert len(context.bot.audio_calls) == 2
    assert context.bot.audio_calls[0]["caption"] == "Audio | Slide 1"
    assert context.bot.audio_calls[1]["caption"] == "Audio | Slide 2"
