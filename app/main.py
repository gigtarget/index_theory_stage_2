import asyncio
import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI

from app.pdf_processor import save_temp_pdf, split_pdf_to_images
from app.script_generator import generate_scripts_from_images
from app.telegram_client import TelegramClient

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Slide Voiceover Bot")
logger = logging.getLogger(__name__)


DEFAULT_MODEL_NAME = "gemini-2.5-flash"


def _build_telegram_client() -> TelegramClient:
    """Helper to construct a Telegram client for each request scope."""
    return TelegramClient()


def _get_model_name() -> str:
    return os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL_NAME


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
        logger.info("Starting file download for chat_id=%s file_path=%s", chat_id, file_info.get("file_path"))
        pdf_bytes = await telegram.download_file(file_info["file_path"])
        logger.info("Completed file download for chat_id=%s bytes=%s", chat_id, len(pdf_bytes))

        with save_temp_pdf(pdf_bytes) as temp_pdf:
            logger.info(
                "Splitting PDF into images: path=%s, size=%s bytes", temp_pdf.name, len(pdf_bytes)
            )
            images = split_pdf_to_images(temp_pdf.name)

        logger.info("PDF page count=%s", len(images))
        logger.info("Generating scripts from %s slide images", len(images))
        scripts = generate_scripts_from_images(images, _get_model_name())
        for index, script in enumerate(scripts, start=1):
            header = f"Slide {index} Script\n"
            logger.info("Sending generated script for slide %s", index)
            await telegram.send_message(chat_id, header + script)
        logger.info("Completed PDF processing for chat_id=%s", chat_id)
    finally:
        await telegram.close()


async def _handle_update(update: Dict[str, Any]) -> None:
    update_id = update.get("update_id")
    message = update.get("message") or update.get("edited_message")
    chat_id = message.get("chat", {}).get("id") if message else None
    logger.info("Webhook received update_id=%s chat_id=%s", update_id, chat_id)

    try:
        if not message:
            logger.info("No message content found in update")
            return

        if chat_id is None:
            logger.warning("Received update without chat id: %s", update)
            return

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
            return

        document = _validate_document(message)
        if not document:
            logger.info("No valid PDF document found; text length=%s", len(text))
            if text:
                telegram = _build_telegram_client()
                try:
                    await telegram.send_message(chat_id, "Please upload a PDF document to process.")
                finally:
                    await telegram.close()
            return

        logger.info("Processing PDF for chat_id=%s with file_id=%s", chat_id, document.get("file_id"))

        if not os.environ.get("GEMINI_API_KEY"):
            telegram = _build_telegram_client()
            try:
                await telegram.send_message(chat_id, "GEMINI_API_KEY is not set in Railway Variables.")
            finally:
                await telegram.close()
            return

        await _process_pdf(chat_id, document["file_id"])
    except Exception as exc:  # pragma: no cover - logged to user
        logger.exception("Failed to process update_id=%s: %s", update_id, exc)
        if chat_id:
            telegram = _build_telegram_client()
            try:
                await telegram.send_message(chat_id, f"Error: {exc}. Check Railway logs.")
            finally:
                await telegram.close()


@app.post("/telegram/webhook")
async def telegram_webhook(update: Dict[str, Any]):
    asyncio.create_task(_handle_update(update))
    return {"ok": True}


@app.get("/")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
async def detailed_health() -> Dict[str, Any]:
    return {
        "ok": True,
        "public_base_url_set": bool(os.environ.get("PUBLIC_BASE_URL")),
        "gemini_model": os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL_NAME,
        "has_telegram_token": bool(os.environ.get("TELEGRAM_BOT_TOKEN")),
        "has_gemini_key": bool(os.environ.get("GEMINI_API_KEY")),
    }


@app.get("/telegram/webhook-info")
async def get_webhook_info() -> Dict[str, Any]:
    telegram = _build_telegram_client()
    try:
        info = await telegram.get_webhook_info()
    finally:
        await telegram.close()
    return info


@app.on_event("startup")
async def register_webhook() -> None:
    base_url = os.environ.get("PUBLIC_BASE_URL")
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not base_url or not bot_token:
        logger.info("Skipping webhook registration; PUBLIC_BASE_URL or TELEGRAM_BOT_TOKEN missing")
        return

    telegram = _build_telegram_client()
    try:
        webhook_url = base_url.rstrip("/") + "/telegram/webhook"
        try:
            await telegram.set_webhook(webhook_url)
        except Exception as exc:  # pragma: no cover - startup logging only
            logger.exception("Failed to register Telegram webhook: %s", exc)
    finally:
        await telegram.close()
