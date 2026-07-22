import re
from typing import List, Dict, Any
from app.core.logger import get_logger

logger = get_logger(__name__)

class MetadataService:
    @staticmethod
    def extract_character_info(text: str) -> List[str]:
        """Extract character entity metadata from text based on regex matching and occurrence thresholds.
        
        Preserved internally for backward compatibility.
        """
        characters = set()
        matches = re.finditer(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})', text)
        for match in matches:
            name = match.group(1)
            if text.count(name) > 2 and len(name) > 3:
                characters.add(name)
        return sorted(characters)

    @classmethod
    def extract_entities(cls, text: str) -> List[str]:
        """Generic interface to extract key entities from document text."""
        return cls.extract_character_info(text)

    @classmethod
    def extract_metadata(cls, text: str) -> Dict[str, Any]:
        """Generic interface to extract document metadata including detected entities."""
        entities = cls.extract_entities(text)
        return {
            "entities": entities,
            "entity_count": len(entities)
        }
