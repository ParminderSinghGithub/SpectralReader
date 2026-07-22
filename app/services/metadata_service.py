import re
from typing import List
from app.core.logger import get_logger

logger = get_logger(__name__)

class MetadataService:
    @staticmethod
    def extract_character_info(text: str) -> List[str]:
        """Extract character metadata from text based on regex matching and occurrence thresholds."""
        characters = set()
        matches = re.finditer(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})', text)
        for match in matches:
            name = match.group(1)
            if text.count(name) > 2 and len(name) > 3:
                characters.add(name)
        return sorted(characters)
