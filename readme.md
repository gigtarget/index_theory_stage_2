# Slide Voiceover Bot

Telegram polling worker that converts uploaded PDF reports into per-slide voiceover scripts using OpenAI vision models. Designed for Railway deployment.

## Features
- Telegram polling worker that accepts PDF documents.
- Splits PDFs into page-aligned PNG images with PyMuPDF.
- Generates one video-style voiceover script per page via OpenAI vision using strict prompt rules.
- Enforces English narration with Hindi words in Devanagari (no romanized Hindi).
- Writes outputs in `outputs/<job_id>/scripts/` (text + metadata).
- Creates parallel script folders at `outputs/<job_id>/scripts/original/` and `outputs/<job_id>/scripts/hinglish/`.
- Sends scripts back to the user as sequential Telegram messages with preserved slide order.
- Splits long Telegram messages safely for full-script output mode.
- Generates a separate viewer question from the full script after all slides are processed.
- Optionally runs a second-pass Hinglish rewrite per slide and sends those scripts after the originals.

## Setup
1. Create a `.env` file (or configure Railway variables) with:

```
TELEGRAM_BOT_TOKEN="<telegram bot token>"
OPENAI_API_KEY="<openai api key>"
MODEL_NAME="gpt-5.2"  # optional (falls back to OPENAI_MODEL or default)
HINDI_DEVANAGARI="1"  # optional (set to 0 to skip normalization)
VOICE_STYLE="formal"  # optional (youtube|formal)
OUTPUT_MODE="slides"  # optional (slides|full|both)
OPENER_PROB="0.10"    # optional (youtube mode only)
BRIDGE_PROB="0.20"    # optional (youtube mode only)
HUMANIZE_FULL_SCRIPT="1"  # optional (youtube mode only)
ENABLE_HINGLISH_REWRITE="true"  # optional
HINGLISH_MODEL="gpt-4.1-mini"  # optional
HINGLISH_TEMPERATURE="0.6"  # optional
HINGLISH_FALLBACK_MODEL="gpt-4.1-mini"  # optional
HINGLISH_MAX_COMPLETION_TOKENS="2048"  # optional
HINGLISH_RETRY_MAX_COMPLETION_TOKENS="4096"  # optional
HINGLISH_MAX_RETRIES="3"  # optional
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
- `ENABLE_HINGLISH_REWRITE` (optional): `true` to run a second-pass Hinglish rewrite per slide (default `true`).
- `HINGLISH_MODEL` (optional): model name for Hinglish rewrite (defaults to `MODEL_NAME`/`OPENAI_MODEL` or `gpt-4.1-mini`).
- `HINGLISH_TEMPERATURE` (optional): sampling temperature for Hinglish rewrite (default `0.6`).
- `HINGLISH_FALLBACK_MODEL` (optional): fallback model for Hinglish rewrite (default `gpt-4.1-mini`).
- `HINGLISH_MAX_COMPLETION_TOKENS` (optional): max completion tokens for Hinglish rewrite (default `2048`).
- `HINGLISH_RETRY_MAX_COMPLETION_TOKENS` (optional): retry max completion tokens for Hinglish rewrite (default `4096`).
- `HINGLISH_MAX_RETRIES` (optional): max retries for Hinglish rewrite (default `3`).

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

Outputs are saved to `outputs/<job_id>/scripts/slide_<n>.txt` with matching `slide_<n>_meta.json` describing word counts and limits used. Original scripts are mirrored in `outputs/<job_id>/scripts/original/` and Hinglish rewrites in `outputs/<job_id>/scripts/hinglish/`.
The CLI also prints a final "Viewer question" generated from the combined script.
