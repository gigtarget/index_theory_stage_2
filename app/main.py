import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException

from app.pdf_processor import save_temp_pdf, split_pdf_to_images
from app.script_generator import generate_scripts_from_images
from app.telegram_client import TelegramClient

app = FastAPI(title="Slide Voiceover Bot")


def _get_model_name() -> str:
    model = os.environ.get("GEMINI_MODEL")
    if not model:
        raise RuntimeError("GEMINI_MODEL is not configured")
    return model


def _validate_document(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    document = message.get("document")
    if not document:
        return None
    mime_type = document.get("mime_type", "")
    file_name = document.get("file_name", "")
    if mime_type != "application/pdf" and not file_name.lower().endswith(".pdf"):
        return None
    return document


async def _process_pdf(chat_id: int, file_id: str) -> None:
    telegram = TelegramClient()
    try:
        file_info = await telegram.get_file(file_id)
        pdf_bytes = await telegram.download_file(file_info["file_path"])

        with save_temp_pdf(pdf_bytes) as temp_pdf:
            images = split_pdf_to_images(temp_pdf.name)

        scripts = generate_scripts_from_images(images, _get_model_name())
        for index, script in enumerate(scripts, start=1):
            header = f"Slide {index} Script\n"
            await telegram.send_message(chat_id, header + script)
    finally:
        await telegram.close()


@app.post("/telegram/webhook")
async def telegram_webhook(update: Dict[str, Any]):
    message = update.get("message") or update.get("edited_message")
    if not message:
        return {"ok": True}

    document = _validate_document(message)
    if not document:
        return {"ok": True}

    chat_id = message["chat"]["id"]
    try:
        await _process_pdf(chat_id, document["file_id"])
    except Exception as exc:  # pragma: no cover - logged to user
        raise HTTPException(status_code=500, detail=str(exc))

    return {"ok": True}


@app.get("/")
async def health() -> Dict[str, str]:
    return {"status": "ok"}
