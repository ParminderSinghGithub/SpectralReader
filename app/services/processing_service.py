import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

class ProcessingService:
    @staticmethod
    def process_text(text: str) -> List[str]:
        """Clean document text and split by chapter/act/scene boundaries or character splitter fallback."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'-\s+', '', text)
        chapter_splits = re.split(r'\n\s*(CHAPTER|ACT|SCENE)\s+[IVXLCDM]+\s*\n', text)
        if len(chapter_splits) > 1:
            return [chap for chap in chapter_splits if len(chap.strip()) > 100]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", r"(?<=\. )", " ", ""]
        )
        return splitter.split_text(text)
