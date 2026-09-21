import fitz
import threading
import numpy as np
import enum
from pydantic import BaseModel, Field
from PIL import Image

from dots_mocr.log import logger


# PyMuPDF / MuPDF is not thread-safe for concurrent document open/render.
# The API server runs conversions in a multi-worker ThreadPoolExecutor, so
# concurrent requests race inside fitz and intermittently yield a 0-page doc.
# Serialize all fitz critical sections with this shared lock.
FITZ_LOCK = threading.RLock()


class SupportedPdfParseMethod(enum.Enum):
    OCR = 'ocr'
    TXT = 'txt'


class PageInfo(BaseModel):
    """The width and height of page
    """
    w: float = Field(description='the width of page')
    h: float = Field(description='the height of page')


def get_matrix(page, dpi_default=200, max_pixels=11289600):
    rect = page.rect
    if rect.width * rect.height > max_pixels:
        factor = (max_pixels / (rect.width * rect.height)) ** 0.5
    else:
        factor = dpi_default / 72
    mat = fitz.Matrix(factor, factor)
    return mat

def is_page_safe_to_render(page, max_image_pixels=30_000_000):
    """
    Check whether a page contains an oversized image that could cause memory issues.

    Args:
        page (pymupdf.Page): The page object to check.
        max_image_pixels (int): Maximum allowed pixel count (width*height) for a
                                single image. Defaults to 30M pixels, roughly a
                                5000x6000 image (~120MB decompressed as RGBA),
                                which is a fairly safe upper bound.

    Returns:
        bool: True if the page is safe, otherwise False.
        str: A human-readable reason describing the result.
    """
    image_list = page.get_images(full=True)
    if not image_list:
        return True, "Page contains no images."

    for img_index, img in enumerate(image_list):
        xref = img[0]
        if xref == 0:  # Inline image, usually small, but can still be checked.
            continue

        try:
            # Only read image metadata, do not decompress! This is the key point.
            width = img[2]  # Width read directly from metadata.
            height = img[3] # Height read directly from metadata.

            if width * height > max_image_pixels:
                reason = (
                    f"Page contains an oversized embedded image (xref: {xref}, "
                    f"size: {width}x{height}); pixel count exceeds threshold {max_image_pixels}."
                )
                return False, reason

        except Exception as e:
            # If even reading metadata fails, treat the page as unsafe.
            reason = f"Error reading metadata of image xref:{xref}: {e}"
            return False, reason

    return True, "All image sizes on the page are within safe limits."

def fitz_doc_to_image(doc, target_dpi=200, origin_dpi=None) -> dict:
    """Convert fitz.Document to image, Then convert the image to numpy array.

    Args:
        doc (_type_): pymudoc page
        dpi (int, optional): reset the dpi of dpi. Defaults to 200.

    Returns:
        dict:  {'img': numpy array, 'width': width, 'height': height }
    """
    from PIL import Image
    # mat = fitz.Matrix(target_dpi / 72, target_dpi / 72)
    mat = get_matrix(doc, target_dpi)
    pm = doc.get_pixmap(matrix=mat, alpha=False)
    logger.debug(
        "fitz_doc_to_image: target_dpi={} matrix=({:.3f},{:.3f}) pixmap={}x{}",
        target_dpi, mat.a, mat.d, pm.width, pm.height,
    )
    if pm.width == 0 or pm.height == 0:
        logger.warning("fitz_doc_to_image: empty pixmap ({}x{}), skip", pm.width, pm.height)
        return None

    if pm.width > 4500 or pm.height > 4500:
        mat = fitz.Matrix(72 / 72, 72 / 72)  # use fitz default dpi
        pm = doc.get_pixmap(matrix=mat, alpha=False)
        logger.debug(
            "fitz_doc_to_image: oversized page re-rendered at 72 dpi -> {}x{}",
            pm.width, pm.height,
        )

    image = Image.frombytes('RGB', (pm.width, pm.height), pm.samples)
    return image



def render_pdf_pages(
    pdf_file, dpi=200, start_page_id=0, end_page_id=None
) -> tuple[list[tuple[int, Image.Image]], list[tuple[int, str, str]]]:
    """Render the selected PDF pages, keeping their true page numbers.

    Returns ``(pages, skipped)`` where ``pages`` is ``[(page_no, image), ...]`` and
    ``skipped`` is ``[(page_no, code, reason), ...]``. Page numbers are the real
    0-indexed PDF page numbers, so a skipped page does not shift the ones after it
    and the API can report "page 7 was skipped because ...".
    """
    pages: list[tuple[int, Image.Image]] = []
    skipped: list[tuple[int, str, str]] = []
    with FITZ_LOCK, fitz.open(pdf_file) as doc:
        pdf_page_num = doc.page_count
        logger.debug(
            "render_pdf_pages: file={} page_count={} dpi={} range=[{},{}]",
            pdf_file, pdf_page_num, dpi, start_page_id, end_page_id,
        )
        end_page_id = (
            end_page_id
            if end_page_id is not None and end_page_id >= 0
            else pdf_page_num - 1
        )
        if end_page_id > pdf_page_num - 1:
            logger.debug("end_page_id is out of range, use images length")
            end_page_id = pdf_page_num - 1

        for index in range(0, doc.page_count):
            if start_page_id <= index <= end_page_id:
                page = doc[index]
                is_safe, reason = is_page_safe_to_render(page)
                if not is_safe:
                    # Skip only this page rather than discarding the whole document.
                    logger.warning(
                        "pdf page {} of {} is not safe to render, skip: {}",
                        index, pdf_file, reason,
                    )
                    skipped.append((index, "page_skipped", reason))
                    continue
                img = fitz_doc_to_image(page, target_dpi=dpi)
                if img is None:
                    logger.warning("pdf page {} of {} is empty, skip", index, pdf_file)
                    skipped.append(
                        (index, "page_skipped", "page rendered to an empty pixmap")
                    )
                    continue
                logger.debug(
                    "render_pdf_pages: page {} rendered -> {}x{}",
                    index, img.width, img.height,
                )
                pages.append((index, img))
    logger.info(
        "render_pdf_pages: file={} rendered={} skipped={}",
        pdf_file, len(pages), len(skipped),
    )
    return pages, skipped


def load_images_from_pdf(pdf_file, dpi=200, start_page_id=0, end_page_id=None) -> list:
    """Images only, page numbers discarded. Kept for non-API callers."""
    pages, _ = render_pdf_pages(
        pdf_file, dpi=dpi, start_page_id=start_page_id, end_page_id=end_page_id
    )
    return [img for _, img in pages]
