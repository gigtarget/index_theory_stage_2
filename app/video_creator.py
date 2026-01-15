import asyncio
import logging
import os
import shlex
import subprocess
import time
from collections import deque
from pathlib import Path

from telegram import Bot

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_bytes(path: Path, data: bytes) -> None:
    ensure_dir(path.parent)
    path.write_bytes(data)


def _format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def probe_duration_seconds(media_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - surfaced to caller
        command_text = _format_command(cmd)
        logger.error(
            "ffprobe failed to read duration. cmd=%s stderr=%s",
            command_text,
            exc.stderr,
        )
        raise RuntimeError(
            f"ffprobe failed to read duration. cmd={command_text} stderr={exc.stderr}"
        ) from exc
    try:
        return float(result.stdout.strip())
    except ValueError as exc:  # pragma: no cover - surfaced to caller
        command_text = _format_command(cmd)
        logger.error(
            "ffprobe returned unexpected duration. cmd=%s output=%s",
            command_text,
            result.stdout,
        )
        raise RuntimeError(
            f"ffprobe returned unexpected duration. cmd={command_text} output={result.stdout}"
        ) from exc


def probe_has_audio_stream(media_path: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(media_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - surfaced to caller
        command_text = _format_command(cmd)
        logger.error(
            "ffprobe failed to read audio streams. cmd=%s stderr=%s",
            command_text,
            exc.stderr,
        )
        raise RuntimeError(
            f"ffprobe failed to read audio streams. cmd={command_text} stderr={exc.stderr}"
        ) from exc
    return bool(result.stdout.strip())


def _format_duration(ms: int) -> str:
    total_seconds = max(ms, 0) // 1000
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _build_ffmpeg_command(cmd: list[str]) -> list[str]:
    if not cmd or cmd[0] != "ffmpeg":
        raise ValueError("ffmpeg command must start with 'ffmpeg'")
    return [
        cmd[0],
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "error",
        "-progress",
        "pipe:1",
        *cmd[1:],
    ]


def _format_progress_message(
    *,
    title: str,
    out_time_ms: int,
    total_duration_ms: int | None,
    speed: str | None,
) -> str:
    time_str = _format_duration(out_time_ms)
    speed_str = speed or "n/a"
    if total_duration_ms:
        total_str = _format_duration(total_duration_ms)
        percent = min(100, int(out_time_ms / total_duration_ms * 100))
        return f"🎬 {title}: {percent}% ({time_str} / {total_str}) speed {speed_str}"
    return f"🎬 {title}: {time_str} speed {speed_str}"


async def _read_stderr(
    stream: asyncio.StreamReader, *, max_lines: int = 40, max_bytes: int = 4096
) -> deque[str]:
    lines: deque[str] = deque()
    total_bytes = 0
    while True:
        chunk = await stream.readline()
        if not chunk:
            break
        text = chunk.decode(errors="replace").rstrip()
        if not text:
            continue
        lines.append(text)
        total_bytes += len(text.encode())
        while len(lines) > max_lines or total_bytes > max_bytes:
            removed = lines.popleft()
            total_bytes -= len(removed.encode())
    return lines


async def run_ffmpeg_with_telegram_progress(
    cmd: list[str],
    chat_id: int,
    bot: Bot,
    title: str,
    total_duration_ms: int | None,
    throttle_seconds: int = 3,
) -> None:
    full_cmd = _build_ffmpeg_command(cmd)
    initial_message = _format_progress_message(
        title=title,
        out_time_ms=0,
        total_duration_ms=total_duration_ms,
        speed="1.0x",
    )
    status_message = await bot.send_message(chat_id=chat_id, text=initial_message)
    last_edit = time.monotonic()
    last_text = initial_message
    process = await asyncio.create_subprocess_exec(
        *full_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stderr_task = asyncio.create_task(_read_stderr(process.stderr))
    out_time_ms = 0
    speed = None
    try:
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            text = line.decode(errors="replace").strip()
            if not text or "=" not in text:
                continue
            key, value = text.split("=", 1)
            if key == "out_time_ms":
                try:
                    out_time_ms = int(value)
                except ValueError:
                    continue
            elif key == "speed":
                speed = value
            elif key == "progress" and value == "end":
                out_time_ms = total_duration_ms or out_time_ms
            now = time.monotonic()
            if now - last_edit < throttle_seconds:
                continue
            progress_text = _format_progress_message(
                title=title,
                out_time_ms=out_time_ms,
                total_duration_ms=total_duration_ms,
                speed=speed,
            )
            if progress_text != last_text:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message.message_id,
                    text=progress_text,
                )
                last_text = progress_text
                last_edit = now
        return_code = await process.wait()
        stderr_lines = await stderr_task
    except Exception:
        process.kill()
        await process.wait()
        stderr_lines = await stderr_task
        failure_text = "❌ Render failed"
        if stderr_lines:
            failure_text = f"{failure_text}\n" + "\n".join(stderr_lines)
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_message.message_id,
            text=failure_text,
        )
        raise

    if return_code != 0:
        failure_text = "❌ Render failed"
        if stderr_lines:
            failure_text = f"{failure_text}\n" + "\n".join(stderr_lines)
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_message.message_id,
            text=failure_text,
        )
        raise RuntimeError(f"ffmpeg exited with code {return_code}. cmd={_format_command(cmd)}")

    await bot.edit_message_text(
        chat_id=chat_id,
        message_id=status_message.message_id,
        text="✅ Render complete",
    )


async def create_slide_video(
    *,
    image_bytes: bytes,
    audio_path: Path,
    out_path: Path,
    chat_id: int,
    bot: Bot,
    fps: int = 30,
) -> None:
    ensure_dir(out_path.parent)
    image_path = out_path.with_suffix(".png")
    write_bytes(image_path, image_bytes)
    duration = await asyncio.to_thread(probe_duration_seconds, audio_path)
    target = duration + 0.05
    total_duration_ms = int(duration * 1000)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-i",
        str(audio_path),
        "-t",
        f"{target:.3f}",
        "-c:v",
        "libx264",
        "-tune",
        "stillimage",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-af",
        "aresample=async=1:first_pts=0",
        "-shortest",
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    try:
        await run_ffmpeg_with_telegram_progress(
            cmd,
            chat_id,
            bot,
            "Rendering",
            total_duration_ms,
        )
    finally:
        try:
            if image_path.exists():
                image_path.unlink()
        except Exception:
            logger.exception("Failed to remove temp image at %s", image_path)


async def merge_videos_concat(
    video_paths: list[Path],
    out_path: Path,
    *,
    chat_id: int,
    bot: Bot,
) -> None:
    ensure_dir(out_path.parent)
    try:
        fps = int(os.environ.get("VIDEO_FPS", "30"))
    except ValueError:
        fps = 30
    intro_path = Path(__file__).resolve().parents[1] / "material" / "intro_video.mp4"
    if intro_path.exists():
        concat_paths = [intro_path, *video_paths]
    else:
        concat_paths = list(video_paths)
    durations: list[float] = []
    has_audio: list[bool] = []
    if concat_paths:
        for path in concat_paths:
            durations.append(await asyncio.to_thread(probe_duration_seconds, path))
            has_audio.append(await asyncio.to_thread(probe_has_audio_stream, path))
    input_cmd: list[str] = []
    for path in concat_paths:
        input_cmd.extend(["-i", str(path)])
    filter_parts: list[str] = []
    for index in range(len(concat_paths)):
        filter_parts.append(f"[{index}:v:0]setpts=PTS-STARTPTS[v{index}]")
        if has_audio[index]:
            filter_parts.append(
                f"[{index}:a:0]"
                "aformat=sample_rates=48000:channel_layouts=stereo,"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )
        else:
            filter_parts.append(
                "anullsrc=channel_layout=stereo:sample_rate=48000,"
                f"atrim=0:{durations[index]:.3f},asetpts=PTS-STARTPTS[a{index}]"
            )
    filter_inputs = "".join(
        f"[v{index}][a{index}]" for index in range(len(concat_paths))
    )
    filter_parts.append(
        f"{filter_inputs}concat=n={len(concat_paths)}:v=1:a=1[v][a]"
    )
    filter_complex = ";".join(filter_parts)
    cmd = [
        "ffmpeg",
        "-y",
        *input_cmd,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    total_duration = sum(durations)
    total_duration_ms = int(total_duration * 1000) if total_duration else None
    await run_ffmpeg_with_telegram_progress(
        cmd,
        chat_id,
        bot,
        "Concatenating",
        total_duration_ms,
    )
