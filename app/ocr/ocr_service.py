from typing import Dict, Type
from app.ocr.base import BaseOCRProvider, OCRResult
from app.ocr.providers.tesseract import TesseractProvider
from app.core.config import settings
from app.core.exceptions import OCRError
from app.core.logger import get_logger

logger = get_logger(__name__)

class OCRService:
    """Gateway manager for OCR providers."""
    _providers: Dict[str, Type[BaseOCRProvider]] = {
        "tesseract": TesseractProvider
    }

    @classmethod
    def get_provider(cls, name: str = "tesseract") -> BaseOCRProvider:
        provider_name = name.lower()
        if provider_name not in cls._providers:
            raise OCRError(f"Unsupported OCR provider '{provider_name}'. Supported: {list(cls._providers.keys())}")
        return cls._providers[provider_name]()

    @classmethod
    def extract(cls, pdf_file, provider_name: str = "tesseract") -> OCRResult:
        """Execute OCR extraction for scanned document using configured provider."""
        if not settings.ENABLE_OCR:
            raise OCRError("OCR engine is disabled in environment settings (ENABLE_OCR=False).")
        provider = cls.get_provider(provider_name)
        return provider.extract_text(pdf_file)

    @classmethod
    def is_available(cls, provider_name: str = "tesseract") -> bool:
        try:
            provider = cls.get_provider(provider_name)
            return provider.is_available()
        except Exception:
            return False
