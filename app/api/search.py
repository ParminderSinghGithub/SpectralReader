from fastapi import APIRouter
from app.models.schemas import SearchRequest, SearchResponse
from app.storage.document_store import DocumentStore
from app.services.metadata_service import MetadataService
from app.core.exceptions import DocumentNotFoundError
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])

@router.post("", response_model=SearchResponse)
def search_passages(request: SearchRequest):
    """Search / retrieve relevant passages from document chunks."""
    store = DocumentStore.get_instance()
    doc = store.get_document(request.document_id)
    if not doc:
        logger.warning(f"Search requested for non-existent document ID '{request.document_id}'")
        raise DocumentNotFoundError(request.document_id)

    chunks = doc["chunks"]
    top_k = request.top_k or 3

    matching_passages = []
    for chunk in chunks:
        if any(entity in chunk for entity in MetadataService.extract_entities(chunk)):
            matching_passages.append(chunk)

    if not matching_passages:
        matching_passages = chunks[:top_k]
    else:
        matching_passages = matching_passages[:top_k]

    logger.info(f"Retrieved {len(matching_passages)} passages for search query on document '{request.document_id}'")

    return SearchResponse(
        document_id=request.document_id,
        query=request.query,
        results=matching_passages
    )
