from fastapi import APIRouter
from app.models.schemas import SearchRequest, SearchResponse
from app.storage.document_store import DocumentStore
from app.services.retrieval_service import RetrievalService
from app.core.config import settings
from app.core.exceptions import DocumentNotFoundError
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])

@router.post("", response_model=SearchResponse)
def search_passages(request: SearchRequest):
    """Search / retrieve relevant passages from document chunks using FAISS dense retrieval and CrossEncoder reranking."""
    store = DocumentStore.get_instance()
    doc = store.get_document(request.document_id)
    if not doc:
        logger.warning(f"Search requested for non-existent document ID '{request.document_id}'")
        raise DocumentNotFoundError(request.document_id)

    retrieval_service = RetrievalService.get_instance()

    # 1. Retrieve candidates via FAISS index using configured RETRIEVAL_TOP_K
    candidates = retrieval_service.retrieve_candidates(
        doc_id=request.document_id,
        query=request.query,
        top_k=settings.RETRIEVAL_TOP_K
    )

    # 2. Rerank candidates using CrossEncoder down to requested top_k (or RERANK_TOP_N)
    top_n = request.top_k or settings.RERANK_TOP_N
    reranked_results = retrieval_service.rerank(
        query=request.query,
        candidates=candidates,
        top_n=top_n
    )

    logger.info(f"Retrieved and reranked {len(reranked_results)} passages for query on document '{request.document_id}'")

    return SearchResponse(
        document_id=request.document_id,
        query=request.query,
        results=reranked_results
    )

