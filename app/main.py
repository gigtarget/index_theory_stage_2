import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException

from app.pdf_processor import save_temp_pdf, split_pdf_to_images
from app.script_generator import generate_scripts_from_images
from app.telegram_client import TelegramClient

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Slide Voiceover Bot")
logger = logging.getLogger(__name__)


def _build_telegram_client() -> TelegramClient:
    """Helper to construct a Telegram client for each request scope."""
    return TelegramClient()


def _get_model_name() -> str:
    model = os.environ.get("GEMINI_MODEL")
    if not model:
        raise RuntimeError("GEMINI_MODEL is not configured")
    return model


def _validate_document(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    document = message.get("document")
    if not document:
        return None
    logger.debug("Validating document payload: keys=%s", list(document.keys()))
    mime_type = document.get("mime_type", "")
    file_name = document.get("file_name", "")
    if mime_type != "application/pdf" and not file_name.lower().endswith(".pdf"):
        return None
    return document


async def _process_pdf(chat_id: int, file_id: str) -> None:
    telegram = _build_telegram_client()
    try:
        logger.info("Fetching file info for file_id=%s", file_id)
        file_info = await telegram.get_file(file_id)
        logger.debug("File info payload: %s", file_info)
        pdf_bytes = await telegram.download_file(file_info["file_path"])

        with save_temp_pdf(pdf_bytes) as temp_pdf:
            logger.info("Splitting PDF into images: path=%s, size=%s bytes", temp_pdf.name, len(pdf_bytes))
            images = split_pdf_to_images(temp_pdf.name)

        logger.info("Generating scripts from %s slide images", len(images))
        scripts = generate_scripts_from_images(images, _get_model_name())
        for index, script in enumerate(scripts, start=1):
            header = f"Slide {index} Script\n"
            logger.info("Sending generated script for slide %s", index)
            await telegram.send_message(chat_id, header + script)
        logger.info("Completed PDF processing for chat_id=%s", chat_id)
    finally:
        await telegram.close()


@app.post("/telegram/webhook")
async def telegram_webhook(update: Dict[str, Any]):
    logger.info("Received Telegram webhook update: %s", update)

    message = update.get("message") or update.get("edited_message")
    if not message:
        logger.info("No message content found in update")
        return {"ok": True}

    chat_id = message.get("chat", {}).get("id")
    logger.info("Parsed message: chat_id=%s, keys=%s", chat_id, list(message.keys()))
    if chat_id is None:
        logger.warning("Received update without chat id: %s", update)
        raise HTTPException(status_code=400, detail="chat_id missing")

    text = (message.get("text") or "").strip()
    if text.lower() in {"/start", "start"}:
        telegram = _build_telegram_client()
        try:
            await telegram.send_message(
                chat_id,
                "Send me a PDF report. I'll split each page and reply with one slide script per message.",
            )
        finally:
            await telegram.close()
        return {"ok": True}

    document = _validate_document(message)
    if not document:
        logger.info("No valid PDF document found; text length=%s", len(text))
        if text:
            telegram = _build_telegram_client()
            try:
                await telegram.send_message(chat_id, "Please upload a PDF document to process.")
            finally:
                await telegram.close()
        return {"ok": True}

    logger.info("Processing PDF for chat_id=%s with file_id=%s", chat_id, document.get("file_id"))

    try:
        await _process_pdf(chat_id, document["file_id"])
    except Exception as exc:  # pragma: no cover - logged to user
        logger.exception("Failed to process PDF: %s", exc)
        telegram = _build_telegram_client()
        try:
            await telegram.send_message(
                chat_id,
                "मैं अभी आपकी फ़ाइल प्रोसेस नहीं कर पाया। कृपया थोड़ी देर बाद पुनः प्रयास करें।",
            )
        finally:
            await telegram.close()
        # Surface the error to the caller for observability in logs/metrics
        raise HTTPException(status_code=500, detail=str(exc))

    return {"ok": True}


@app.get("/")
async def health() -> Dict[str, str]:
    return {"status": "ok"}
