import argparse
import logging
import os
from typing import Optional

from app.pdf_processor import split_pdf_to_images
from app.script_generator import (
    DEFAULT_MAX_WORDS,
    DEFAULT_TARGET_WORDS,
    generate_scripts_from_images,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate slide scripts from a PDF")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.2"))
    parser.add_argument("--job_id", default=None, help="Optional job identifier for outputs/")
    parser.add_argument("--target_words", type=int, default=DEFAULT_TARGET_WORDS)
    parser.add_argument("--max_words", type=int, default=DEFAULT_MAX_WORDS)
    return parser.parse_args()


def _require_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set to run the generator")


def main(args: Optional[argparse.Namespace] = None) -> None:
    parsed = args or _parse_args()
    _require_api_key()

    images = split_pdf_to_images(parsed.pdf)
    scripts = generate_scripts_from_images(
        images,
        parsed.model,
        target_words=parsed.target_words,
        max_words=parsed.max_words,
        job_id=parsed.job_id,
    )
    for idx, script in enumerate(scripts, start=1):
        print(f"\n--- Slide {idx} ---\n{script}\n")


if __name__ == "__main__":
    main()
