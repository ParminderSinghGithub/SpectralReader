import pdfplumber
from typing import Tuple
from app.core.logger import get_logger

logger = get_logger(__name__)

class DocumentService:
    @staticmethod
    def extract_preview(pdf_file, max_pages: int = 3) -> str:
        """Extract text from the first few pages of a document for preview."""
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages[:max_pages]:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text

    @staticmethod
    def extract_full_text(pdf_file) -> Tuple[str, int]:
        """Extract full text from a document along with total page count."""
        full_text = ""
        with pdfplumber.open(pdf_file) as pdf:
            num_pages = len(pdf.pages)
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    full_text += extracted + "\n"
        return full_text, num_pages
