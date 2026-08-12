from io import BytesIO
from unittest.mock import patch, MagicMock
from app.ocr.pdf_detector import PDFDetector
from app.ocr.ocr_service import OCRService

def test_ocr_service_available():
    assert OCRService.is_available("tesseract") in (True, False)

def test_pdf_detector_fallback_on_error():
    fake_pdf = BytesIO(b"invalid_pdf_content")
    is_scanned, num_pages, sample_text = PDFDetector.inspect_pdf(fake_pdf)
    assert is_scanned is True
