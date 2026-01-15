import asyncio
import json
import logging
import os
import socket
import uuid
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

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
from app.text_postprocess import format_allcaps_words
from app.tts import prepare_tts_payload, synthesize_tts_to_file
from app.video_creator import VideoMergeError, create_slide_video, merge_videos_concat
from app.youtube_uploader import (
    DEFAULT_TAGS,
    build_description,
    decode_b64_secrets_to_tmp,
    upload_video,
)


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


def _get_tts_voice() -> str:
    return os.environ.get("TTS_VOICE", "cedar").strip() or "cedar"


def _get_tts_format() -> str:
    return os.environ.get("TTS_FORMAT", "mp3").strip().lower() or "mp3"


def _get_tts_speed() -> float:
    try:
        return float(os.environ.get("TTS_SPEED", "1.0"))
    except ValueError:
        return 1.0


def _get_enable_slide_videos() -> bool:
    return os.environ.get("ENABLE_SLIDE_VIDEOS", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def _get_video_keep_files() -> bool:
    return os.environ.get("VIDEO_KEEP_FILES", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def _get_video_fps() -> int:
    try:
        return int(os.environ.get("VIDEO_FPS", "30"))
    except ValueError:
        return 30


def _get_tts_instructions() -> str | None:
    instructions = os.environ.get(
        "TTS_INSTRUCTIONS",
        "Speak in a clear Indian-English market-news tone. Keep pauses at commas and full-stops.",
    ).strip()
    return instructions or None


def _get_video_message_filter() -> filters.MessageFilter:
    video_filter = filters.VIDEO
    document_video_filter = getattr(filters.Document, "VIDEO", None)
    if document_video_filter is not None:
        return video_filter | document_video_filter

    mime_filter = getattr(filters.Document, "MimeType", None)
    if mime_filter is None:
        return video_filter

    video_mime_types = [
        "video/mp4",
        "video/quicktime",
        "video/x-matroska",
        "video/webm",
        "video/avi",
        "video/mpeg",
        "video/3gpp",
        "video/3gpp2",
        "video/x-msvideo",
    ]
    document_filter = None
    for mime_type in video_mime_types:
        current_filter = mime_filter(mime_type)
        document_filter = (
            current_filter if document_filter is None else document_filter | current_filter
        )
    if document_filter is None:
        return video_filter
    return video_filter | document_filter



def _format_tts_notification(
    *,
    model: str,
    voice: str,
    response_format: str,
    speed: float,
) -> str:
    model_line = f"OpenAI model: {model}"
    voice_line = f"Voice: {voice} | Lang: n/a"
    speed_line = f"Format: {response_format} | Speed: {speed}"
    return (
        "Generating TTS audio.\n"
        f"{model_line}\n"
        f"{voice_line}\n"
        f"{speed_line}"
    )


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


def _format_report_date(report_date: datetime | None) -> str:
    tz_name = os.environ.get("YT_TIMEZONE", "Asia/Kolkata")
    try:
        tzinfo = ZoneInfo(tz_name)
    except Exception:
        logger.warning("Invalid YT_TIMEZONE=%s, falling back to UTC.", tz_name)
        tzinfo = timezone.utc
    if report_date is None:
        report_date = datetime.now(timezone.utc)
    if report_date.tzinfo is None:
        report_date = report_date.replace(tzinfo=timezone.utc)
    local_date = report_date.astimezone(tzinfo).date()
    return local_date.strftime("%d %b %Y")


async def _maybe_upload_to_youtube(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    final_video_path: Path,
    report_date: datetime | None,
) -> None:
    if os.environ.get("YT_UPLOAD_ON_COMPLETE") != "1":
        logger.info("YT_UPLOAD_ON_COMPLETE not enabled; skipping YouTube upload.")
        return
    logger.info(
        "Preparing YouTube upload for chat_id=%s path=%s",
        chat_id,
        final_video_path,
    )
    if decode_b64_secrets_to_tmp() is None:
        logger.warning("YouTube upload skipped due to missing credentials.")
        return
    delay_raw = os.environ.get("YT_SCHEDULE_DELAY_MINUTES", "60")
    try:
        delay_minutes = int(delay_raw)
    except ValueError:
        logger.warning(
            "Invalid YT_SCHEDULE_DELAY_MINUTES=%s, defaulting to 60.", delay_raw
        )
        delay_minutes = 60
    date_str = _format_report_date(report_date)
    title = f"Post Market Report {date_str} | Jatin Dhiman | Nifty 50 | Bank Nifty"
    description = build_description(date_str)
    tags_env = os.environ.get("YT_KEYWORDS")
    tags = DEFAULT_TAGS
    if tags_env:
        tags = [tag.strip() for tag in tags_env.split(",") if tag.strip()]
    category_id = os.environ.get("YT_CATEGORY", "22")
    publish_at = datetime.now(timezone.utc) + timedelta(minutes=delay_minutes)
    logger.info(
        "YouTube upload settings: delay_minutes=%s category_id=%s tags_count=%s publish_at=%s",
        delay_minutes,
        category_id,
        len(tags),
        publish_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    try:
        video_id = await asyncio.to_thread(
            upload_video,
            str(final_video_path),
            title,
            description,
            tags,
            category_id,
            publish_at,
        )
    except Exception as exc:
        logger.warning("YouTube upload failed: %s", exc)
        await _send_message(
            context,
            chat_id,
            f"YouTube upload failed: {exc}",
        )
        return

    publish_time = publish_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    await _send_message(
        context,
        chat_id,
        "YouTube upload scheduled.\n"
        f"Video ID: {video_id}\n"
        f"Publish time (UTC): {publish_time}",
    )


async def _process_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    chat_id = update.effective_chat.id if update.effective_chat else None
    if not message or chat_id is None:
        logger.info("Received video handler call without message or chat_id.")
        return

    video = message.video
    document = message.document
    file_id = None
    filename = None
    if video is not None:
        file_id = video.file_id
        filename = video.file_name
    elif document is not None:
        file_id = document.file_id
        filename = document.file_name

    if not file_id:
        logger.info("Video handler invoked without video file_id.")
        return

    safe_name = Path(filename or "video.mp4").name
    incoming_dir = Path("artifacts") / "incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    incoming_path = incoming_dir / f"{uuid.uuid4().hex}_{safe_name}"

    try:
        await _send_message(context, chat_id, "Downloading video...")
        file = await context.bot.get_file(file_id)
        video_bytes = await file.download_as_bytearray()
        incoming_path.write_bytes(bytes(video_bytes))
        logger.info(
            "Saved incoming video for chat_id=%s to %s", chat_id, incoming_path
        )
    except Exception as exc:  # pragma: no cover - logged for robustness
        logger.exception("Failed to download video for chat_id=%s: %s", chat_id, exc)
        await _send_message(context, chat_id, f"Error downloading video: {exc}")
        return

    if decode_b64_secrets_to_tmp() is None:
        await _send_message(
            context,
            chat_id,
            "YouTube credentials are missing. Please configure them and try again.",
        )
        return

    delay_raw = os.environ.get("YT_SCHEDULE_DELAY_MINUTES", "60")
    try:
        delay_minutes = int(delay_raw)
    except ValueError:
        logger.warning(
            "Invalid YT_SCHEDULE_DELAY_MINUTES=%s, defaulting to 60.", delay_raw
        )
        delay_minutes = 60

    report_date = message.date if message else None
    date_str = _format_report_date(report_date)
    title = f"Post Market Report {date_str} | Jatin Dhiman | Nifty 50 | Bank Nifty"
    description = build_description(date_str)
    tags_env = os.environ.get("YT_KEYWORDS")
    tags = DEFAULT_TAGS
    if tags_env:
        tags = [tag.strip() for tag in tags_env.split(",") if tag.strip()]
    category_id = os.environ.get("YT_CATEGORY", "22")
    publish_at = datetime.now(timezone.utc) + timedelta(minutes=delay_minutes)

    try:
        video_id = await asyncio.to_thread(
            upload_video,
            str(incoming_path),
            title,
            description,
            tags,
            category_id,
            publish_at,
        )
    except Exception as exc:
        logger.exception("YouTube upload failed for chat_id=%s: %s", chat_id, exc)
        await _send_message(context, chat_id, f"YouTube upload failed: {exc}")
        return

    publish_time = publish_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    await _send_message(
        context,
        chat_id,
        "YouTube upload scheduled.\n"
        f"Video ID: {video_id}\n"
        f"Publish time (UTC): {publish_time}",
    )


async def _send_tts_input_preview(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    *,
    slide_index: int,
    total_slides: int,
    text: str,
) -> None:
    logger.info(
        "Sending TTS input to Telegram for slide %s/%s",
        slide_index,
        total_slides,
    )
    title = f"TTS INPUT (SLIDE {slide_index}/{total_slides})"
    message = "\n".join([title, text])
    await _send_message(context, chat_id, message)


async def _generate_and_send_scripts(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    images: list[bytes],
    *,
    watermarked_images: list[bytes] | None = None,
    report_date: datetime | None = None,
) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        await _send_message(context, chat_id, "OPENAI_API_KEY is not set in Railway Variables.")
        return
    if watermarked_images is not None and len(watermarked_images) != len(images):
        raise ValueError("watermarked_images length does not match images length.")

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
        scripts_dir, original_dir = create_scripts_job_dir()
        tts_enabled = _get_tts_enabled()
        tts_keep_files = _get_tts_keep_files()
        tts_model = _get_tts_model()
        tts_voice = _get_tts_voice()
        tts_format = _get_tts_format()
        tts_speed = _get_tts_speed()
        tts_instructions = _get_tts_instructions()
        tts_dir = Path("artifacts") / "tts"
        tts_delay_seconds = 0.75
        scripts: list[str] = []
        video_enabled = _get_enable_slide_videos()
        video_fps = _get_video_fps()
        video_keep_files = _get_video_keep_files()
        clip_paths: list[Path] = []
        if video_enabled and watermarked_images is None:
            logger.warning("Video generation enabled, but no watermarked images provided.")

        if tts_enabled:
            await _send_message(
                context,
                chat_id,
                _format_tts_notification(
                    model=tts_model,
                    voice=tts_voice,
                    response_format=tts_format,
                    speed=tts_speed,
                ),
            )

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

            if tts_enabled:
                tts_filename = f"slide_{index:02d}.{tts_format}"
                tts_path = tts_dir / tts_filename
                tts_result = ""
                try:
                    tts_input = script
                    tts_payload = prepare_tts_payload(tts_input, tts_instructions)
                    if tts_payload:
                        await _send_tts_input_preview(
                            context,
                            chat_id,
                            slide_index=index,
                            total_slides=total_slides,
                            text=tts_payload,
                        )
                    tts_result = await asyncio.to_thread(
                        synthesize_tts_to_file,
                        tts_payload,
                        str(tts_path),
                        model=tts_model,
                        voice=tts_voice,
                        response_format=tts_format,
                        speed=tts_speed,
                        instructions=tts_instructions,
                    )
                    if tts_result:
                        with open(tts_result, "rb") as audio_file:
                            await context.bot.send_audio(
                                chat_id=chat_id,
                                audio=audio_file,
                                filename=tts_filename,
                                caption=f"Audio | Slide {index}",
                            )
                        await asyncio.sleep(tts_delay_seconds)
                        audio_path = Path(tts_result)
                        if (
                            video_enabled
                            and watermarked_images is not None
                            and index <= len(watermarked_images)
                            and audio_path.exists()
                        ):
                            job_root = scripts_dir.parent.parent
                            videos_dir = job_root / "videos"
                            videos_dir.mkdir(parents=True, exist_ok=True)
                            await _send_message(
                                context,
                                chat_id,
                                f"Creating video clip {index}/{total_slides}...",
                            )
                            clip_path = videos_dir / f"clip_{index:02d}.mp4"
                            await create_slide_video(
                                image_bytes=watermarked_images[index - 1],
                                audio_path=audio_path,
                                out_path=clip_path,
                                chat_id=chat_id,
                                bot=context.bot,
                                fps=video_fps,
                            )
                            clip_paths.append(clip_path)
                except Exception as exc:  # pragma: no cover - logged for robustness
                    logger.exception("Failed to send TTS audio for slide %s: %s", index, exc)
                finally:
                    if not tts_keep_files and tts_path.exists():
                        try:
                            tts_path.unlink()
                        except Exception:
                            logger.exception("Failed to delete TTS file at %s", tts_path)

        if output_mode in ["full", "both"]:
            full_payload = "\n".join(scripts)
            if _get_humanize_full_script() and voice_style == "youtube":
                full_payload = await asyncio.to_thread(
                    humanize_full_script, full_payload, client=client, model_name=model_name
                )
            await _send_long(context, chat_id, full_payload)

        if clip_paths:
            await _send_message(
                context,
                chat_id,
                f"Merging {len(clip_paths)} clips into final video...",
            )
            job_root = scripts_dir.parent.parent
            videos_dir = job_root / "videos"
            merged_path = videos_dir / "final.mp4"
            await merge_videos_concat(
                clip_paths,
                merged_path,
                chat_id=chat_id,
                bot=context.bot,
            )
            with open(merged_path, "rb") as merged_file:
                await context.bot.send_document(
                    chat_id=chat_id,
                    document=merged_file,
                    filename="index_theory_final.mp4",
                    caption="Final merged video",
                )
            await _maybe_upload_to_youtube(context, chat_id, merged_path, report_date)
            if not video_keep_files:
                for clip_path in clip_paths:
                    try:
                        if clip_path.exists():
                            clip_path.unlink()
                    except Exception:
                        logger.exception("Failed to remove clip file at %s", clip_path)
        logger.info("Completed script generation for chat_id=%s", chat_id)
    except Exception as exc:  # pragma: no cover - logged to user
        if isinstance(exc, VideoMergeError):
            logger.exception("Failed to merge video for chat_id=%s: %s", chat_id, exc)
        else:
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
        report_date = message.date if message else None
        asyncio.create_task(
            _generate_and_send_scripts(
                context,
                chat_id,
                images,
                watermarked_images=watermarked_images,
                report_date=report_date,
            )
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
    await _send_message(context, chat_id, "Please upload a PDF document to process.")


async def _handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    asyncio.create_task(_process_pdf(update, context))


async def _handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    asyncio.create_task(_process_video(update, context))


def _build_application() -> Application:
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")

    application = ApplicationBuilder().token(bot_token).build()
    application.add_error_handler(_handle_error)
    application.add_handler(CommandHandler("start", _handle_start))
    application.add_handler(MessageHandler(_get_video_message_filter(), _handle_video))
    application.add_handler(MessageHandler(filters.Document.PDF, _handle_pdf))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_text))
    return application


async def _handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    error = context.error
    logger.exception("Telegram error handler caught exception: %s", error)
    if isinstance(error, Conflict):
        logger.error(
            "Polling conflict detected (getUpdates). Ensure only one instance is running."
        )
        os._exit(1)


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
        os._exit(1)


if __name__ == "__main__":
    main()
