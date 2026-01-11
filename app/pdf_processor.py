import tempfile
from io import BytesIO
from typing import List

import fitz
from PIL import Image

WATERMARK_PADDING_PX = 0


def split_pdf_to_images(pdf_path: str) -> List[bytes]:
    """Render each PDF page to a PNG image.

    Returns a list of PNG byte strings in document order.
    """
    images: List[bytes] = []
    with fitz.open(pdf_path) as doc:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            images.append(pixmap.tobytes("png"))
    return images


def save_temp_pdf(content: bytes) -> tempfile.NamedTemporaryFile:
    temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    temp_file.write(content)
    temp_file.flush()
    return temp_file


def add_logo_watermark_png(
    png_bytes: bytes,
    logo_path: str,
    *,
    width_ratio: float = 0.12,
    min_logo_width: int = 80,
    max_logo_width: int = 320,
    padding_px: int | None = WATERMARK_PADDING_PX,
    padding_ratio: float | None = None,
) -> bytes:
    base = Image.open(BytesIO(png_bytes)).convert("RGBA")
    logo = Image.open(logo_path).convert("RGBA")

    slide_width, slide_height = base.size
    target_width = int(slide_width * width_ratio)
    target_width = max(min_logo_width, min(max_logo_width, target_width))
    scale = target_width / logo.width if logo.width else 1
    target_height = max(1, int(logo.height * scale))
    logo = logo.resize((target_width, target_height), Image.LANCZOS)

    if padding_px is not None:
        padding = max(0, padding_px)
    elif padding_ratio is not None:
        padding = max(0, int(min(slide_width, slide_height) * padding_ratio))
    else:
        padding = 0
    x = max(0, slide_width - target_width - padding)
    y = max(0, slide_height - target_height - padding)
    base.paste(logo, (x, y), mask=logo)

    output = BytesIO()
    base.save(output, format="PNG")
    return output.getvalue()


def watermark_images_with_logo(
    images: List[bytes],
    logo_path: str,
    *,
    padding_px: int | None = WATERMARK_PADDING_PX,
    padding_ratio: float | None = None,
) -> List[bytes]:
    return [
        add_logo_watermark_png(
            image,
            logo_path,
            padding_px=padding_px,
            padding_ratio=padding_ratio,
        )
        for image in images
    ]
