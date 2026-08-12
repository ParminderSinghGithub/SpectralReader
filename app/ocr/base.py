from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class OCRResult:
    text: str
    num_pages: int
    characters_extracted: int
    provider_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseOCRProvider(ABC):
    """Abstract interface for OCR engines."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    def extract_text(self, pdf_file) -> OCRResult:
        """Extract text from scanned PDF stream or path using OCR."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if system dependencies (e.g. tesseract binary) are available."""
        pass
