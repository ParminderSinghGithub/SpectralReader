import pdfplumber
from typing import Tuple
from app.core.config import settings
from app.core.exceptions import InvalidDocumentError
from app.core.logger import get_logger

logger = get_logger(__name__)

class PDFDetector:
    """Detects whether a PDF is machine-readable searchable text or a scanned raster image."""

    @staticmethod
    def inspect_pdf(pdf_file) -> Tuple[bool, int, str]:
        """Inspect PDF text layer. Returns (is_scanned: bool, num_pages: int, sample_text: str)."""
        # Fast validation: check if PDF stream is 0 bytes / empty
        if hasattr(pdf_file, "getvalue"):
            val = pdf_file.getvalue()
            if not val or len(val.strip()) == 0:
                raise InvalidDocumentError("Uploaded PDF document is empty.")

        total_chars = 0
        text_rich_pages = 0
        full_text = ""

        try:
            with pdfplumber.open(pdf_file) as pdf:
                num_pages = len(pdf.pages)
                if num_pages == 0:
                    raise InvalidDocumentError("Uploaded PDF document contains 0 pages.")
                for page in pdf.pages:
                    txt = page.extract_text() or ""
                    clean_txt = txt.strip()
                    char_count = len(clean_txt)
                    total_chars += char_count
                    if char_count >= settings.OCR_MIN_TEXT_CHARS_PER_PAGE:
                        text_rich_pages += 1
                    if txt:
                        full_text += txt + "\n"

            # Re-seek file pointer
            if hasattr(pdf_file, "seek"):
                pdf_file.seek(0)

            # Document is scanned if text-rich pages are fewer than half total pages or avg chars per page < threshold
            avg_chars = total_chars / max(1, num_pages)
            is_scanned = (text_rich_pages == 0) or (avg_chars < settings.OCR_MIN_TEXT_CHARS_PER_PAGE)

            logger.info(
                f"PDF Inspection: {num_pages} pages, {total_chars} total chars, "
                f"{text_rich_pages} text-rich pages -> is_scanned={is_scanned}"
            )
            return is_scanned, num_pages, full_text

        except InvalidDocumentError:
            raise
        except Exception as e:
            logger.warning(f"PDF Detector encountered inspection error: {str(e)}. Defaulting to is_scanned=True")
            if hasattr(pdf_file, "seek"):
                pdf_file.seek(0)
            return True, 0, ""
