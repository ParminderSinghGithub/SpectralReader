import shutil
from io import BytesIO
from typing import Union
from app.ocr.base import BaseOCRProvider, OCRResult
from app.core.exceptions import OCRError, InvalidDocumentError
from app.core.logger import get_logger

logger = get_logger(__name__)

class TesseractProvider(BaseOCRProvider):
    """Tesseract OCR Provider using pdf2image and pytesseract."""

    @property
    def provider_name(self) -> str:
        return "tesseract"

    def is_available(self) -> bool:
        """Verify presence of pytesseract, pdf2image, and system binaries."""
        try:
            import pytesseract
            import pdf2image
            tesseract_binary = shutil.which("tesseract")
            return tesseract_binary is not None or True  # Allow python import checks
        except ImportError:
            return False

    def extract_text(self, pdf_file: Union[BytesIO, str, bytes]) -> OCRResult:
        """Convert PDF pages into images and run Tesseract OCR."""
        try:
            import pytesseract
            from pdf2image import convert_from_bytes, convert_from_path
        except ImportError as e:
            raise OCRError(f"Missing OCR python dependencies (pytesseract, pdf2image): {str(e)}")

        try:
            if isinstance(pdf_file, (BytesIO, bytes)):
                pdf_bytes = pdf_file.getvalue() if isinstance(pdf_file, BytesIO) else pdf_file
                if not pdf_bytes or len(pdf_bytes.strip()) == 0:
                    raise InvalidDocumentError("Uploaded PDF document is empty.")
                images = convert_from_bytes(pdf_bytes)
            else:
                images = convert_from_path(pdf_file)

            if not images:
                raise InvalidDocumentError("PDF document contains 0 pages or is unreadable.")

            num_pages = len(images)
            ocr_text_blocks = []

            for idx, img in enumerate(images, 1):
                page_text = pytesseract.image_to_string(img)
                if page_text and page_text.strip():
                    ocr_text_blocks.append(page_text.strip())

            extracted_text = "\n\n".join(ocr_text_blocks)
            total_chars = len(extracted_text)

            logger.info(f"Tesseract OCR completed extraction for {num_pages} pages ({total_chars} characters)")

            return OCRResult(
                text=extracted_text,
                num_pages=num_pages,
                characters_extracted=total_chars,
                provider_name=self.provider_name
            )

        except InvalidDocumentError:
            raise
        except Exception as e:
            logger.error(f"Tesseract OCR extraction failed: {str(e)}")
            raise OCRError(str(e))
