import tempfile
from typing import List

import fitz


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
