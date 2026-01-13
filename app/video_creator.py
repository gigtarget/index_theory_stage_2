import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_bytes(path: Path, data: bytes) -> None:
    ensure_dir(path.parent)
    path.write_bytes(data)


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
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-i",
        str(audio_path),
        "-c:v",
        "libx264",
        "-tune",
        "stillimage",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - surfaced to caller
        logger.error("ffmpeg failed to create slide video: %s", exc.stderr)
        raise RuntimeError(
            f"ffmpeg failed to create slide video: {exc.stderr}"
        ) from exc
    finally:
        try:
            if image_path.exists():
                image_path.unlink()
        except Exception:
            logger.exception("Failed to remove temp image at %s", image_path)


def merge_videos_concat(video_paths: list[Path], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    list_path = out_path.with_suffix(".txt")
    list_contents = "\n".join(
        f"file '{path.resolve()}'" for path in video_paths
    )
    list_path.write_text(list_contents + "\n", encoding="utf-8")
    concat_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(out_path),
    ]
    try:
        subprocess.run(concat_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        logger.warning("ffmpeg concat copy failed, re-encoding: %s", exc.stderr)
        reencode_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(out_path),
        ]
        try:
            subprocess.run(reencode_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as reencode_exc:  # pragma: no cover
            logger.error("ffmpeg concat re-encode failed: %s", reencode_exc.stderr)
            raise RuntimeError(
                f"ffmpeg failed to merge videos: {reencode_exc.stderr}"
            ) from reencode_exc
    finally:
        try:
            if list_path.exists():
                list_path.unlink()
        except Exception:
            logger.exception("Failed to remove concat list at %s", list_path)
