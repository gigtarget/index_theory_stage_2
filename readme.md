# Slide Voiceover Bot

Telegram polling worker that converts uploaded PDF reports into per-slide voiceover scripts using OpenAI vision models. Designed for Railway deployment.

## Features
- Telegram polling worker that accepts PDF documents.
- Splits PDFs into page-aligned PNG images with PyMuPDF.
- Generates one video-style voiceover script per page via OpenAI vision using strict prompt rules.
- Enforces English narration with Hindi words in Devanagari (no romanized Hindi).
- Writes outputs in `outputs/<job_id>/scripts/` (text + metadata).
- Sends scripts back to the user as sequential Telegram messages with preserved slide order.

## Setup
1. Create a `.env` file (or configure Railway variables) with:

```
TELEGRAM_BOT_TOKEN="<telegram bot token>"
OPENAI_API_KEY="<openai api key>"
MODEL_NAME="gpt-5.2"  # optional (falls back to OPENAI_MODEL or default)
HINDI_DEVANAGARI="1"  # optional (set to 0 to skip normalization)
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

### Environment variables
- `TELEGRAM_BOT_TOKEN` (required): Telegram bot token.
- `OPENAI_API_KEY` (required): OpenAI API key for the vision model.
- `MODEL_NAME` (optional): OpenAI model name (falls back to `OPENAI_MODEL`).
- `HINDI_DEVANAGARI` (optional): set to `0` to skip Devanagari normalization.

### CLI pipeline (outline + per-slide scripts)

Run the generator locally against a PDF without Telegram:

```
python -m app.cli /path/to/report.pdf --target_words 80 --max_words 95
```

Outputs are saved to `outputs/<job_id>/scripts/slide_<n>.txt` with matching `slide_<n>_meta.json` describing word counts and limits used.
