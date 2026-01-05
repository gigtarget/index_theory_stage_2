# Slide Voiceover Bot

Telegram polling worker that converts uploaded PDF reports into per-slide voiceover scripts using OpenAI vision models. Designed for Railway deployment.

## Features
- Telegram polling worker that accepts PDF documents.
- Splits PDFs into page-aligned PNG images with PyMuPDF.
- Global outline pass that captures a throughline, per-slide intents, and optional glossary.
- Two-stage per-slide generation: extract facts JSON then craft a structured narration with strict word limits.
- Generates one Hinglish voiceover script per page via OpenAI using the supplied prompt rules.
- Writes outputs in `outputs/<job_id>/outline.json` and per-slide `scripts/` files (text + metadata).
- Sends scripts back to the user as sequential Telegram messages with preserved slide order.

## Setup
1. Create a `.env` file (or configure Railway variables) with:

```
TELEGRAM_BOT_TOKEN="<telegram bot token>"
OPENAI_API_KEY="<openai api key>"
OPENAI_MODEL="gpt-5.2"
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

### CLI pipeline (outline + per-slide scripts)

Run the generator locally against a PDF without Telegram:

```
python -m app.cli /path/to/report.pdf --target_words 80 --max_words 95
```

Outputs are saved to `outputs/<job_id>/outline.json` and `outputs/<job_id>/scripts/slide_<n>.txt` with matching `slide_<n>_meta.json` describing word counts and intents used.
