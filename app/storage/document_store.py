import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from app.core.logger import get_logger

logger = get_logger(__name__)

class DocumentStore:
    """In-memory document storage singleton.
    
    Temporary storage implementation for holding processed document text,
    chunk representations, and extracted metadata within process memory.
    """
    _instance: Optional['DocumentStore'] = None
    _documents: Dict[str, dict] = {}

    @classmethod
    def get_instance(cls) -> 'DocumentStore':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add_document(
        self,
        filename: str,
        full_text: str,
        num_pages: int,
        chunks: List[str],
        entities: List[str]
    ) -> dict:
        doc_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        doc_data = {
            "document_id": doc_id,
            "filename": filename,
            "full_text": full_text,
            "num_pages": num_pages,
            "chunks": chunks,
            "num_chunks": len(chunks),
            "entities": entities,
            "characters": entities,  # Backward compatibility key
            "created_at": created_at
        }
        self._documents[doc_id] = doc_data
        logger.info(f"Added document {doc_id} ('{filename}') to in-memory store.")
        return doc_data

    def get_document(self, document_id: str) -> Optional[dict]:
        return self._documents.get(document_id)

    def list_documents(self) -> List[dict]:
        return list(self._documents.values())

    def delete_document(self, document_id: str) -> bool:
        if document_id in self._documents:
            del self._documents[document_id]
            logger.info(f"Deleted document {document_id} from in-memory store.")
            return True
        return False
