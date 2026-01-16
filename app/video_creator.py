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


class VideoMergeError(RuntimeError):
    pass


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
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=nw=1:nk=1",
        str(media_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except Exception:  # pragma: no cover - surfaced to caller
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def probe_video_stream_info(media_path: Path) -> dict[str, float | int] | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "default=nw=1:nk=1",
        str(media_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - surfaced to caller
        logger.error(
            "ffprobe failed to read video stream info. cmd=%s stderr=%s",
            _format_command(cmd),
            exc.stderr,
        )
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 3:
        return None
    try:
        width = int(lines[0])
        height = int(lines[1])
    except ValueError:
        return None
    fps_value = None
    rate_parts = lines[2].split("/", 1)
    try:
        if len(rate_parts) == 2:
            numerator = float(rate_parts[0])
            denominator = float(rate_parts[1])
            if denominator:
                fps_value = numerator / denominator
        else:
            fps_value = float(lines[2])
    except ValueError:
        fps_value = None
    info: dict[str, float | int] = {"width": width, "height": height}
    if fps_value is not None:
        info["fps"] = fps_value
    return info


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


def _coerce_even(value: int) -> int:
    if value % 2 == 0:
        return value
    return value - 1 if value > 1 else value + 1


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
            stderr_text = "\n".join(stderr_lines)
            failure_text = f"{failure_text}\n{stderr_text}"
            logger.error(
                "ffmpeg failed with code %s. stderr (tail): %s",
                return_code,
                stderr_text[-2000:],
            )
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
    audio_path: Path | None,
    out_path: Path,
    chat_id: int,
    bot: Bot,
    fps: int = 30,
    fallback_duration: float = 6.0,
) -> None:
    ensure_dir(out_path.parent)
    image_path = out_path.with_suffix(".png")
    write_bytes(image_path, image_bytes)
    has_audio = audio_path is not None and audio_path.exists()
    if has_audio:
        duration = await asyncio.to_thread(probe_duration_seconds, audio_path)
    else:
        duration = fallback_duration
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
    ]
    if has_audio and audio_path is not None:
        cmd.extend(["-i", str(audio_path)])
    cmd.extend(
        [
            "-t",
            f"{target:.3f}",
            "-c:v",
            "libx264",
            "-tune",
            "stillimage",
            "-pix_fmt",
            "yuv420p",
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
        ]
    )
    if has_audio:
        cmd.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-af",
                "aresample=async=1:first_pts=0",
                "-shortest",
            ]
        )
    cmd.append(str(out_path))
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
    try:
        ensure_dir(out_path.parent)
        try:
            fps = int(os.environ.get("VIDEO_FPS", "30"))
        except ValueError:
            fps = 30
        try:
            target_max = int(os.environ.get("VIDEO_TARGET_MAX", "1920"))
        except ValueError:
            target_max = 1920
        intro_path = Path(__file__).resolve().parents[1] / "material" / "intro_video.mp4"
        if intro_path.exists():
            concat_paths = [intro_path, *video_paths]
        else:
            concat_paths = list(video_paths)
        if not concat_paths:
            raise VideoMergeError("No videos to merge.")
        target_width = 1920
        target_height = 1080
        reference_path = concat_paths[0] if concat_paths else None
        if reference_path is not None:
            reference_info = await asyncio.to_thread(probe_video_stream_info, reference_path)
            if reference_info:
                target_width = int(reference_info["width"])
                target_height = int(reference_info["height"])
        if target_max and target_width > target_max:
            scale_ratio = target_max / target_width
            target_width = int(round(target_width * scale_ratio))
            target_height = int(round(target_height * scale_ratio))
        target_width = _coerce_even(target_width)
        target_height = _coerce_even(target_height)
        logger.info("concat target: %sx%s fps=%s", target_width, target_height, fps)
        durations: list[float] = []
        for path in concat_paths:
            durations.append(float(await asyncio.to_thread(probe_duration_seconds, path)))
        total_duration = sum(durations)
        total_duration_ms = int(total_duration * 1000) if total_duration else None

        norm_dir = out_path.parent / "_norm"
        ensure_dir(norm_dir)
        normalized_paths: list[Path] = []
        total_clips = len(concat_paths)
        for index, path in enumerate(concat_paths):
            duration = durations[index]
            norm_path = norm_dir / f"norm_{index:02d}.mp4"
            has_audio = await asyncio.to_thread(probe_has_audio_stream, path)
            stream_info = await asyncio.to_thread(probe_video_stream_info, path)
            if stream_info:
                logger.info(
                    "normalize input %s: path=%s size=%sx%s fps=%s has_audio=%s dur=%.3f",
                    index,
                    path,
                    stream_info["width"],
                    stream_info["height"],
                    f"{stream_info.get('fps', 'n/a'):.3f}"
                    if isinstance(stream_info.get("fps"), float)
                    else stream_info.get("fps", "n/a"),
                    has_audio,
                    duration,
                )
            else:
                logger.info(
                    "normalize input %s: path=%s has_audio=%s dur=%.3f",
                    index,
                    path,
                    has_audio,
                    duration,
                )
            vf_filter = (
                f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,"
                f"fps={fps},format=yuv420p,setsar=1"
            )
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
            ]
            if not has_audio:
                cmd.extend(
                    [
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=channel_layout=stereo:sample_rate=48000",
                    ]
                )
            cmd.extend(
                [
                    "-vf",
                    vf_filter,
                    "-af",
                    "aformat=sample_rates=48000:channel_layouts=stereo",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0" if not has_audio else "0:a:0",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                ]
            )
            if not has_audio:
                cmd.append("-shortest")
            cmd.append(str(norm_path))
            total_duration_ms = int(duration * 1000) if duration else None
            await run_ffmpeg_with_telegram_progress(
                cmd,
                chat_id,
                bot,
                f"Normalizing clip {index + 1}/{total_clips}",
                total_duration_ms,
            )
            normalized_paths.append(norm_path)

        concat_list_path = norm_dir / "concat_list.txt"
        concat_lines = [f"file '{path.name}'" for path in normalized_paths]
        concat_list_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

        concat_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        try:
            await run_ffmpeg_with_telegram_progress(
                concat_cmd,
                chat_id,
                bot,
                "Concatenating clips...",
                total_duration_ms,
            )
        except RuntimeError as exc:
            logger.warning("Stream copy concat failed; retrying with re-encode. error=%s", exc)
            concat_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list_path),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(out_path),
            ]
            await run_ffmpeg_with_telegram_progress(
                concat_cmd,
                chat_id,
                bot,
                "Concatenating clips...",
                total_duration_ms,
            )
    except VideoMergeError:
        raise
    except Exception as exc:
        raise VideoMergeError("Failed to merge video") from exc
