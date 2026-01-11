import argparse
import logging
import os
import re
from typing import Optional

import fitz
from app.script_generator import (
    DEFAULT_MAX_WORDS,
    DEFAULT_TARGET_WORDS,
    generate_scripts_from_images,
    generate_viewer_question,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate slide scripts from a PDF")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME") or os.environ.get("OPENAI_MODEL", "gpt-5.2"),
    )
    parser.add_argument("--target_words", type=int, default=DEFAULT_TARGET_WORDS)
    parser.add_argument("--max_words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate placeholder scripts without calling OpenAI.",
    )
    return parser.parse_args()


def _require_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set to run the generator")


def _sentence_from_line(line: str) -> str:
    cleaned = re.sub(r"\s+", " ", line.strip())
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def _dry_run_scripts(pdf_path: str) -> list[str]:
    scripts: list[str] = []
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc, start=1):
            lines = [ln.strip("•- \t") for ln in page.get_text("text").splitlines()]
            lines = [ln for ln in lines if ln]
            sentences: list[str] = []
            if index == 1:
                sentences.append("नमस्ते! Welcome to Index Theory’s Post Market Report.")
            for line in lines:
                sentence = _sentence_from_line(line)
                if sentence:
                    sentences.append(sentence)
                if len(sentences) >= 4:
                    break
            if not sentences:
                sentences.append("No readable text found on this slide.")
            scripts.append(" ".join(sentences))
    return scripts


def main(args: Optional[argparse.Namespace] = None) -> None:
    parsed = args or _parse_args()
    if parsed.dry_run:
        scripts = _dry_run_scripts(parsed.pdf)
    else:
        _require_api_key()
        from app.pdf_processor import split_pdf_to_images

        images = split_pdf_to_images(parsed.pdf)
        scripts = generate_scripts_from_images(
            images,
            parsed.model,
            target_words=parsed.target_words,
            max_words=parsed.max_words,
        )
    full_script = "\n".join(scripts)
    voice_style = os.environ.get("VOICE_STYLE", "formal").strip().lower()
    if not parsed.dry_run and voice_style != "youtube":
        viewer_question = generate_viewer_question(full_script)
        if viewer_question and scripts:
            scripts[-1] = f"{scripts[-1]}\nComment below—{viewer_question}"
    for idx, script in enumerate(scripts, start=1):
        print(f"\n--- Slide {idx} ---\n{script}\n")


if __name__ == "__main__":
    main()
