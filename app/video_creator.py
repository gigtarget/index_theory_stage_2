import logging
import os
import shlex
import subprocess
from pathlib import Path

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


def create_slide_video(
    *,
    image_bytes: bytes,
    audio_path: Path,
    out_path: Path,
    fps: int = 30,
) -> None:
    ensure_dir(out_path.parent)
    image_path = out_path.with_suffix(".png")
    write_bytes(image_path, image_bytes)
    duration = probe_duration_seconds(audio_path)
    target = duration + 0.05
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
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - surfaced to caller
        command_text = _format_command(cmd)
        logger.error(
            "ffmpeg failed to create slide video. cmd=%s stderr=%s",
            command_text,
            exc.stderr,
        )
        raise RuntimeError(
            f"ffmpeg failed to create slide video. cmd={command_text} stderr={exc.stderr}"
        ) from exc
    finally:
        try:
            if image_path.exists():
                image_path.unlink()
        except Exception:
            logger.exception("Failed to remove temp image at %s", image_path)


def merge_videos_concat(video_paths: list[Path], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    try:
        fps = int(os.environ.get("VIDEO_FPS", "30"))
    except ValueError:
        fps = 30
    input_cmd: list[str] = []
    for path in video_paths:
        input_cmd.extend(["-i", str(path)])
    filter_inputs = "".join(
        f"[{index}:v:0][{index}:a:0]" for index in range(len(video_paths))
    )
    filter_complex = f"{filter_inputs}concat=n={len(video_paths)}:v=1:a=1[v][a]"
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
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        command_text = _format_command(cmd)
        logger.error(
            "ffmpeg concat filter failed. cmd=%s stderr=%s",
            command_text,
            exc.stderr,
        )
        raise RuntimeError(
            f"ffmpeg failed to merge videos. cmd={command_text} stderr={exc.stderr}"
        ) from exc
