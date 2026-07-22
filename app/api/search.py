from fastapi import APIRouter, HTTPException, status
from app.models.schemas import SearchRequest, SearchResponse
from app.storage.document_store import DocumentStore
from app.services.metadata_service import MetadataService

router = APIRouter(prefix="/search", tags=["Search"])

@router.post("", response_model=SearchResponse)
def search_passages(request: SearchRequest):
    """Search / retrieve relevant passages from document chunks."""
    store = DocumentStore.get_instance()
    doc = store.get_document(request.document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{request.document_id}' not found."
        )

    chunks = doc["chunks"]
    top_k = request.top_k or 3

    # Filter passages using character presence or text matching logic
    matching_passages = []
    for chunk in chunks:
        if any(char in chunk for char in MetadataService.extract_character_info(chunk)):
            matching_passages.append(chunk)

    if not matching_passages:
        matching_passages = chunks[:top_k]
    else:
        matching_passages = matching_passages[:top_k]

    return SearchResponse(
        document_id=request.document_id,
        query=request.query,
        results=matching_passages
    )
