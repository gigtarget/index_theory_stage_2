# Slide Voiceover Bot

Telegram polling worker that converts uploaded PDF reports into per-slide voiceover scripts using Gemini vision models. Designed for Railway deployment.

## Features
- Telegram polling worker that accepts PDF documents.
- Splits PDFs into page-aligned PNG images with PyMuPDF.
- Generates one Hinglish voiceover script per page via Gemini using the supplied prompt rules.
- Sends scripts back to the user as sequential Telegram messages with preserved slide order.

## Setup
1. Create a `.env` file (or configure Railway variables) with:

```
TELEGRAM_BOT_TOKEN="<telegram bot token>"
GEMINI_API_KEY="<gemini api key>"
GEMINI_MODEL="gemini-2.5-flash"
NODE_ENV="production"
DATABASE_URL="<optional postgres url>"
OPENAI_API_KEY="<optional openai api key>"
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
