import asyncio
import json
import logging
import os
import socket
import uuid
from io import BytesIO
from pathlib import Path

from telegram import Message, Update
from telegram.error import Conflict
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.pdf_processor import (
    WATERMARK_PADDING_PX,
    save_temp_pdf,
    split_pdf_to_images,
    watermark_images_with_logo,
)
from app.script_generator import (
    DEFAULT_MAX_WORDS,
    DEFAULT_TARGET_WORDS,
    _build_client,
    create_scripts_job_dir,
    generate_script_for_slide,
    generate_viewer_question,
    humanize_full_script,
)
from app.rewrite_hinglish import rewrite_all_blocks
from app.text_postprocess import format_allcaps_words
from app.tts import synthesize_tts_to_file


logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
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


def _get_voice_style() -> str:
    style = os.environ.get("VOICE_STYLE", "formal").strip().lower()
    if style not in {"formal", "youtube"}:
        return "formal"
    return style


def _get_output_mode() -> str:
    mode = os.environ.get("OUTPUT_MODE", "slides").strip().lower()
    if mode not in {"slides", "full", "both"}:
        return "slides"
    return mode


def _get_humanize_full_script() -> bool:
    voice_style = _get_voice_style()
    default_value = "1" if voice_style == "youtube" else "0"
    return os.environ.get("HUMANIZE_FULL_SCRIPT", default_value) == "1"


def _get_tts_enabled() -> bool:
    return os.environ.get("TTS_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def _get_tts_keep_files() -> bool:
    return os.environ.get("TTS_KEEP_FILES", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def _get_tts_model() -> str:
    return os.environ.get("TTS_MODEL", "gpt-4o-mini-tts").strip() or "gpt-4o-mini-tts"


def _get_tts_provider() -> str:
    provider = os.environ.get("TTS_PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "fal_kokoro"}:
        return "openai"
    return provider


def _get_fal_key() -> str:
    return os.environ.get("FAL_KEY", "").strip()


def _get_kokoro_endpoint() -> str:
    return os.environ.get("KOKORO_ENDPOINT", "fal-ai/kokoro/hindi").strip() or "fal-ai/kokoro/hindi"


def _get_kokoro_voice() -> str:
    return os.environ.get("KOKORO_VOICE", "hf_alpha").strip() or "hf_alpha"


def _get_kokoro_speed() -> float:
    try:
        return float(os.environ.get("KOKORO_SPEED", "1.0"))
    except ValueError:
        return 1.0


def _get_tts_voice() -> str:
    return os.environ.get("TTS_VOICE", "cedar").strip() or "cedar"


def _get_tts_format() -> str:
    return os.environ.get("TTS_FORMAT", "mp3").strip().lower() or "mp3"


def _get_tts_speed() -> float:
    try:
        return float(os.environ.get("TTS_SPEED", "1.0"))
    except ValueError:
        return 1.0


def _get_tts_instructions() -> str | None:
    instructions = os.environ.get(
        "TTS_INSTRUCTIONS",
        "Speak in a clear Indian-English market-news tone with natural Hinglish flow. "
        "Keep pauses at commas and full-stops.",
    ).strip()
    return instructions or None


def _get_tts_text_mode() -> str:
    mode = os.environ.get("TTS_TEXT_MODE", "hinglish").strip().lower()
    if mode not in {"hinglish", "devanagari"}:
        return "hinglish"
    return mode


def _convert_hinglish_to_devanagari(text: str, *, client, model_name: str) -> str:
    system_prompt = (
        "You are a Hindi TTS assistant. Convert Hinglish narration into natural spoken Hindi "
        "in Devanagari script. Keep numbers and symbols exactly as provided. Return plain text only."
    )
    user_prompt = f"Rewrite this narration into spoken Hindi (Devanagari only):\n{text}"
    result = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_completion_tokens=2048,
    )
    msg = result.choices[0].message
    return (getattr(msg, "content", None) or "").strip()


def _get_instance_id() -> str:
    hostname = socket.gethostname()
    short_id = uuid.uuid4().hex[:8]
    return f"{hostname}-{short_id}"


async def _send_message(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str
) -> Message:
    return await context.bot.send_message(chat_id=chat_id, text=text)


async def _send_long(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, limit: int = 3500
) -> None:
    paragraphs = text.split("\n")
    buffer: list[str] = []
    buffer_len = 0
    for paragraph in paragraphs:
        chunk = paragraph.strip()
        if len(chunk) > limit:
            if buffer:
                await _send_message(context, chat_id, "\n".join(buffer).strip())
                buffer = []
                buffer_len = 0
            for i in range(0, len(chunk), limit):
                await _send_message(context, chat_id, chunk[i : i + limit])
            continue
        extra = len(chunk) + (1 if buffer else 0)
        if buffer_len + extra > limit and buffer:
            await _send_message(context, chat_id, "\n".join(buffer).strip())
            buffer = []
            buffer_len = 0
        buffer.append(chunk)
        buffer_len += extra
    if buffer:
        await _send_message(context, chat_id, "\n".join(buffer).strip())


async def _generate_and_send_scripts(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, images: list[bytes]
) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        await _send_message(context, chat_id, "OPENAI_API_KEY is not set in Railway Variables.")
        return

    status_messages: list[Message] = []
    try:
        status_messages.append(
            await _send_message(
                context,
                chat_id,
                f"Generating scripts for {len(images)} slides using OpenAI...",
            )
        )
        target_words, max_words = _get_word_limits()
        model_name = _get_model_name()
        voice_style = _get_voice_style()
        output_mode = _get_output_mode()
        total_slides = len(images)
        client = _build_client()
        scripts_dir, original_dir, hinglish_dir = create_scripts_job_dir()
        tts_enabled = _get_tts_enabled()
        tts_keep_files = _get_tts_keep_files()
        tts_provider = _get_tts_provider()
        tts_model = _get_tts_model()
        tts_voice = _get_tts_voice()
        tts_format = _get_tts_format()
        tts_speed = _get_tts_speed()
        tts_instructions = _get_tts_instructions()
        kokoro_endpoint = _get_kokoro_endpoint()
        kokoro_voice = _get_kokoro_voice()
        kokoro_speed = _get_kokoro_speed()
        tts_text_mode = _get_tts_text_mode()
        fal_key = _get_fal_key()
        tts_dir = Path("artifacts") / "tts"
        tts_delay_seconds = 0.75
        scripts: list[str] = []
        hinglish_enabled = os.environ.get("ENABLE_HINGLISH_REWRITE", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        hinglish_scripts: list[str] = []

        if tts_provider == "fal_kokoro":
            tts_format = "wav"
            if not fal_key:
                logger.error("TTS_PROVIDER=fal_kokoro but FAL_KEY is missing; skipping TTS.")
                tts_enabled = False

        for index, image in enumerate(images, start=1):
            logger.info("Generating script for slide %s/%s", index, total_slides)
            if output_mode in ["slides", "both"]:
                await _send_message(
                    context,
                    chat_id,
                    f"===== SLIDE {index}/{total_slides}: ORIGINAL =====",
                )
            script = await asyncio.to_thread(
                generate_script_for_slide,
                image,
                client=client,
                model_name=model_name,
                slide_index=index,
                total_slides=total_slides,
                target_words=target_words,
                max_words=max_words,
                scripts_dir=scripts_dir,
            )

            if index == total_slides and voice_style != "youtube":
                full_script_for_question = "\n".join(scripts + [script])
                viewer_question = await asyncio.to_thread(
                    generate_viewer_question, full_script_for_question
                )
                if viewer_question:
                    question_line = f"Comment below—{viewer_question}"
                    script = f"{script}\n{question_line}"
                    if scripts_dir:
                        script_path = scripts_dir / f"slide_{index}.txt"
                        script_path.write_text(script, encoding="utf-8")
                        original_script_path = original_dir / f"slide_{index}.txt"
                        original_script_path.write_text(script, encoding="utf-8")
                        meta_path = scripts_dir / f"slide_{index}_meta.json"
                        meta_payload = {}
                        if meta_path.exists():
                            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
                        meta_payload["word_count"] = len(script.split())
                        meta_path.write_text(
                            json.dumps(meta_payload, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )

            script = format_allcaps_words(script)
            if scripts_dir:
                script_path = scripts_dir / f"slide_{index}.txt"
                script_path.write_text(script, encoding="utf-8")
                original_script_path = original_dir / f"slide_{index}.txt"
                original_script_path.write_text(script, encoding="utf-8")
                meta_path = scripts_dir / f"slide_{index}_meta.json"
                meta_payload = {}
                if meta_path.exists():
                    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
                meta_payload["word_count"] = len(script.split())
                meta_path.write_text(
                    json.dumps(meta_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            scripts.append(script)

            if output_mode in ["slides", "both"]:
                await _send_long(context, chat_id, script)

            if hinglish_enabled:
                if output_mode in ["slides", "both"]:
                    await _send_message(
                        context,
                        chat_id,
                        f"===== SLIDE {index}/{total_slides}: HINGLISH =====",
                    )
                fallback_state = {"used": False, "reason": ""}

                async def _mark_fallback(slide_index: int, total: int, reason: str) -> None:
                    fallback_state["used"] = True
                    fallback_state["reason"] = reason

                hinglish_block = await rewrite_all_blocks(
                    [script],
                    client=client,
                    on_slide_fallback=_mark_fallback,
                    slide_indices=[index],
                )
                hinglish_script = hinglish_block[0] if hinglish_block else script
                hinglish_script = format_allcaps_words(hinglish_script)
                if fallback_state["used"]:
                    if fallback_state["reason"] == "digits":
                        display_script = f"[HINGLISH FALLBACK USED: digits]\n{hinglish_script}"
                    else:
                        display_script = f"[HINGLISH FALLBACK USED] (model failed)\n{hinglish_script}"
                else:
                    display_script = hinglish_script
                hinglish_scripts.append(display_script)
                hinglish_path = hinglish_dir / f"slide_{index}.txt"
                hinglish_path.write_text(hinglish_script, encoding="utf-8")
                if output_mode in ["slides", "both"]:
                    await _send_long(context, chat_id, display_script)

                if tts_enabled:
                    tts_filename = f"hinglish_slide_{index:02d}.{tts_format}"
                    tts_path = tts_dir / tts_filename
                    tts_input = hinglish_script
                    if tts_text_mode == "devanagari":
                        try:
                            tts_input = await asyncio.to_thread(
                                _convert_hinglish_to_devanagari,
                                hinglish_script,
                                client=client,
                                model_name=model_name,
                            )
                        except Exception as exc:  # pragma: no cover - safe fallback
                            logger.exception(
                                "Failed to convert Hinglish to Devanagari for slide %s: %s",
                                index,
                                exc,
                            )
                            tts_input = hinglish_script
                    try:
                        tts_result = await asyncio.to_thread(
                            synthesize_tts_to_file,
                            tts_input,
                            str(tts_path),
                            model=tts_model,
                            voice=tts_voice,
                            response_format=tts_format,
                            speed=tts_speed,
                            instructions=tts_instructions,
                            provider=tts_provider,
                            kokoro_voice=kokoro_voice,
                            kokoro_speed=kokoro_speed,
                            kokoro_endpoint=kokoro_endpoint,
                            fal_key=fal_key or None,
                        )
                        if tts_result:
                            with open(tts_result, "rb") as audio_file:
                                await context.bot.send_audio(
                                    chat_id=chat_id,
                                    audio=audio_file,
                                    filename=tts_filename,
                                    caption=f"Hinglish Audio | Slide {index}",
                                )
                            await asyncio.sleep(tts_delay_seconds)
                    except Exception as exc:  # pragma: no cover - logged for robustness
                        logger.exception(
                            "Failed to send TTS audio for slide %s: %s", index, exc
                        )
                    finally:
                        if not tts_keep_files and tts_path.exists():
                            try:
                                tts_path.unlink()
                            except Exception:
                                logger.exception(
                                    "Failed to delete TTS file at %s", tts_path
                                )

        if output_mode in ["full", "both"]:
            full_payload = "\n".join(scripts)
            if _get_humanize_full_script() and voice_style == "youtube":
                full_payload = await asyncio.to_thread(
                    humanize_full_script, full_payload, client=client, model_name=model_name
                )
            await _send_long(context, chat_id, full_payload)

        if hinglish_enabled and hinglish_scripts and output_mode in ["full", "both"]:
            await _send_long(context, chat_id, "\n".join(hinglish_scripts))
        logger.info("Completed script generation for chat_id=%s", chat_id)
    except Exception as exc:  # pragma: no cover - logged to user
        logger.exception("Failed to generate scripts for chat_id=%s: %s", chat_id, exc)
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
        logo_path = (
            Path(__file__).resolve().parents[1]
            / "material"
            / "index_theory_small_logo.png"
        )
        watermarked_images = watermark_images_with_logo(
            images,
            str(logo_path),
            padding_px=WATERMARK_PADDING_PX,
        )
        for index, image_bytes in enumerate(watermarked_images, start=1):
            filename = f"slide_{index:02d}.png"
            image_buffer = BytesIO(image_bytes)
            image_buffer.name = filename
            await context.bot.send_document(
                chat_id=chat_id, document=image_buffer, filename=filename
            )
        context.chat_data["pending_images"] = images
        await _send_message(
            context,
            chat_id,
            "Slides sent. Reply CONFIRM to generate scripts, or CANCEL to discard.",
        )
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
    message = update.effective_message
    text = message.text.strip().lower() if message and message.text else ""
    pending_images = context.chat_data.get("pending_images")
    if pending_images:
        if text in {"confirm", "yes", "proceed", "go", "ok"}:
            context.chat_data.pop("pending_images", None)
            asyncio.create_task(_generate_and_send_scripts(context, chat_id, pending_images))
            return
        if text in {"cancel", "stop", "no"}:
            context.chat_data.pop("pending_images", None)
            await _send_message(context, chat_id, "Cancelled. No scripts were generated.")
            return
        await _send_message(
            context,
            chat_id,
            "Reply CONFIRM to generate scripts, or CANCEL to discard.",
        )
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
    instance_id = _get_instance_id()
    logger.info("Starting Telegram bot polling. instance_id=%s", instance_id)
    try:
        application.run_polling(drop_pending_updates=True)
    except Conflict:
        logger.error(
            "Another bot instance is running. Ensure only 1 replica/service is polling."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
