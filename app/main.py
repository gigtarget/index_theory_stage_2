import asyncio
import logging
import os

from telegram import Message, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.pdf_processor import save_temp_pdf, split_pdf_to_images
from app.script_generator import (
    DEFAULT_MAX_WORDS,
    DEFAULT_TARGET_WORDS,
    generate_scripts_from_images,
    generate_viewer_question,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "gpt-5.2"


def _get_model_name() -> str:
    return os.environ.get("MODEL_NAME") or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL_NAME


def _get_word_limits() -> tuple[int, int]:
    try:
        target = int(os.environ.get("TARGET_WORDS", DEFAULT_TARGET_WORDS))
    except ValueError:
        target = DEFAULT_TARGET_WORDS
    try:
        max_words = int(os.environ.get("MAX_WORDS", DEFAULT_MAX_WORDS))
    except ValueError:
        max_words = DEFAULT_MAX_WORDS
    return target, max_words


async def _send_message(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str
) -> Message:
    return await context.bot.send_message(chat_id=chat_id, text=text)


async def _process_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    chat_id = update.effective_chat.id if update.effective_chat else None
    document = message.document if message else None

    if not message or not document or chat_id is None:
        logger.info("Received PDF handler call without message or document")
        return

    logger.info(
        "Processing PDF for chat_id=%s with file_id=%s", chat_id, document.file_id
    )

    if not os.environ.get("OPENAI_API_KEY"):
        await _send_message(context, chat_id, "OPENAI_API_KEY is not set in Railway Variables.")
        return

    status_messages: list[Message] = []

    try:
        status_messages.append(
            await _send_message(context, chat_id, "Starting PDF download...")
        )
        file = await context.bot.get_file(document.file_id)
        pdf_bytes = await file.download_as_bytearray()
        logger.info(
            "Completed file download for chat_id=%s bytes=%s", chat_id, len(pdf_bytes)
        )

        status_messages.append(
            await _send_message(context, chat_id, "Splitting PDF into slide images...")
        )
        temp_pdf_path = None
        try:
            temp_pdf = save_temp_pdf(bytes(pdf_bytes))
            temp_pdf_path = temp_pdf.name
            temp_pdf.close()
            logger.info(
                "Splitting PDF into images: path=%s, size=%s bytes",
                temp_pdf_path,
                len(pdf_bytes),
            )
            images = split_pdf_to_images(temp_pdf_path)
        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.remove(temp_pdf_path)
                except Exception:
                    logger.exception("Failed to remove temp PDF at %s", temp_pdf_path)

        logger.info("PDF page count=%s", len(images))
        status_messages.append(
            await _send_message(
                context,
                chat_id,
                f"Generating scripts for {len(images)} slides using OpenAI...",
            )
        )
        target_words, max_words = _get_word_limits()
        scripts = generate_scripts_from_images(
            images,
            _get_model_name(),
            target_words=target_words,
            max_words=max_words,
        )
        for index, script in enumerate(scripts, start=1):
            logger.info("Sending generated script for slide %s", index)
            await _send_message(context, chat_id, script)
        full_script = "\n".join(scripts)
        viewer_question = generate_viewer_question(full_script, _get_model_name())
        if viewer_question:
            await _send_message(context, chat_id, f"Viewer question: {viewer_question}")
        logger.info("Completed PDF processing for chat_id=%s", chat_id)
    except Exception as exc:  # pragma: no cover - logged to user
        logger.exception("Failed to process PDF for chat_id=%s: %s", chat_id, exc)
        await _send_message(
            context, chat_id, f"Error: {exc}. Check Railway logs."
        )
    finally:
        if status_messages:
            for status_message in status_messages:
                try:
                    await context.bot.delete_message(
                        chat_id=chat_id, message_id=status_message.message_id
                    )
                except Exception:
                    logger.exception(
                        "Failed to delete status message id=%s for chat_id=%s",
                        status_message.message_id,
                        chat_id,
                    )


async def _handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None:
        return
    await _send_message(
        context,
        chat_id,
        "Send me a PDF report. I'll split each page and reply with one slide script per message.",
    )


async def _handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None:
        return
    await _send_message(context, chat_id, "Please upload a PDF document to process.")


async def _handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    asyncio.create_task(_process_pdf(update, context))


def _build_application() -> Application:
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")

    application = ApplicationBuilder().token(bot_token).build()
    application.add_handler(CommandHandler("start", _handle_start))
    application.add_handler(MessageHandler(filters.Document.PDF, _handle_pdf))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_text))
    return application


def main() -> None:
    application = _build_application()
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
