# Slide Voiceover Bot

Telegram polling worker that converts uploaded PDF reports into per-slide voiceover scripts using OpenAI vision models. Designed for Railway deployment.

## Features
- Telegram polling worker that accepts PDF documents.
- Splits PDFs into page-aligned PNG images with PyMuPDF.
- Generates one video-style voiceover script per page via OpenAI vision using strict prompt rules.
- Enforces English narration with Hindi words in Devanagari (no romanized Hindi).
- Writes outputs in `outputs/<job_id>/scripts/` (text + metadata).
- Creates a parallel script folder at `outputs/<job_id>/scripts/original/`.
- Sends scripts back to the user as sequential Telegram messages with preserved slide order.
- Splits long Telegram messages safely for full-script output mode.
- Generates a separate viewer question from the full script after all slides are processed.

## Setup
1. Create a `.env` file (or configure Railway variables) with:

```
TELEGRAM_BOT_TOKEN="<telegram bot token>"
OPENAI_API_KEY="<openai api key>"
ELEVENLABS_API_KEYS="key1,key2,key3"
ELEVENLABS_VOICE_ID="VbDz3QQGkAGePVWfkfwE"
ELEVENLABS_MODEL_ID="eleven_multilingual_v2"
MODEL_NAME="gpt-5.2"  # optional (falls back to OPENAI_MODEL or default)
HINDI_DEVANAGARI="1"  # optional (set to 0 to skip normalization)
VOICE_STYLE="formal"  # optional (youtube|formal)
OUTPUT_MODE="slides"  # optional (slides|full|both)
OPENER_PROB="0.10"    # optional (youtube mode only)
BRIDGE_PROB="0.20"    # optional (youtube mode only)
HUMANIZE_FULL_SCRIPT="1"  # optional (youtube mode only)
TARGET_WORDS="80"  # optional
MAX_WORDS="95"     # optional
NODE_ENV="production"
DATABASE_URL="<optional postgres url>"
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run locally:

```
python -m app.main
```

Upload a PDF document to the bot to receive one text message per slide following the configured script prompt.
After all slides are sent, the bot posts a separate "Viewer question" message.

### Environment variables
- `TELEGRAM_BOT_TOKEN` (required): Telegram bot token.
- `OPENAI_API_KEY` (required): OpenAI API key for the vision model.
- `MODEL_NAME` (optional): OpenAI model name (falls back to `OPENAI_MODEL`).
- `HINDI_DEVANAGARI` (optional): set to `0` to skip Devanagari normalization.
- `VOICE_STYLE` (optional): `formal` (default) or `youtube` for a human, short-line narration style.
- `OUTPUT_MODE` (optional): `slides` (default), `full` (single continuous script), or `both`.
- `OPENER_PROB` (optional): probability for optional openers in YouTube mode (default `0.10`).
- `BRIDGE_PROB` (optional): probability for optional topic bridges in YouTube mode (default `0.20`).
- `HUMANIZE_FULL_SCRIPT` (optional): `1` to run a second-pass YouTube humanizer for the full script (default `1` in YouTube mode).
- `ELEVENLABS_API_KEYS` (required for TTS): comma-separated ElevenLabs API keys; the app rotates keys on auth/rate/quota/network errors.
- `ELEVENLABS_API_KEY` (optional): single ElevenLabs API key (used if `ELEVENLABS_API_KEYS` is unset).
- `ELEVENLABS_VOICE_ID` (optional): ElevenLabs voice ID (default `VbDz3QQGkAGePVWfkfwE`).
- `ELEVENLABS_MODEL_ID` (optional): ElevenLabs model ID (default `eleven_multilingual_v2`).
- `ELEVENLABS_OUTPUT_FORMAT` (optional): ElevenLabs output format (default `mp3_44100_128`; app maps `mp3` to this).
- `ELEVENLABS_MAX_CHARS_PER_KEY` (optional): rotate keys once usage would exceed this limit.
- `ELEVENLABS_USAGE_STATE_PATH` (optional): path to persist per-key character usage (default `/tmp/elevenlabs_usage.json`).

When TTS is enabled, the bot uses ElevenLabs for audio synthesis only. If a key returns 401/403, 429, 402, quota/limit errors, or a network timeout, the next key is tried automatically. The default TTS format is `mp3`; the ElevenLabs integration maps this to `mp3_44100_128` unless you override `ELEVENLABS_OUTPUT_FORMAT`.

### Recommended Railway settings for YouTube mode
```
VOICE_STYLE=youtube
OUTPUT_MODE=full
HUMANIZE_FULL_SCRIPT=1
OPENER_PROB=0.10
BRIDGE_PROB=0.20
HINDI_DEVANAGARI=0
```

### CLI pipeline (outline + per-slide scripts)

Run the generator locally against a PDF without Telegram:

```
python -m app.cli /path/to/report.pdf --target_words 80 --max_words 95
```

Outputs are saved to `outputs/<job_id>/scripts/slide_<n>.txt` with matching `slide_<n>_meta.json` describing word counts and limits used. Original scripts are mirrored in `outputs/<job_id>/scripts/original/`.
The CLI also prints a final "Viewer question" generated from the combined script.
