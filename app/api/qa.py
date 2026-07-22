import time
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import QARequest, QAResponse
from app.storage.document_store import DocumentStore
from app.services.model_service import ModelService
from app.services.qa_service import QAService
from app.services.metadata_service import MetadataService

router = APIRouter(prefix="/qa", tags=["QA"])

@router.post("", response_model=QAResponse)
def answer_question(request: QARequest):
    """Execute question answering over processed document chunks."""
    store = DocumentStore.get_instance()
    doc = store.get_document(request.document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{request.document_id}' not found."
        )

    models = ModelService.get_model_container()
    if models is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load backend ML models."
        )

    start_time = time.perf_counter()
    answer = QAService.answer_question(
        question=request.question,
        docs=doc["chunks"],
        tokenizer=models.tokenizer,
        model=models.qa_model
    )
    end_time = time.perf_counter()
    processing_time_ms = round((end_time - start_time) * 1000, 2)

    # Collect retrieved passages used in context
    retrieved_passages = []
    for chunk in doc["chunks"]:
        if any(char in chunk for char in MetadataService.extract_character_info(chunk)):
            retrieved_passages.append(chunk)
    retrieved_context = retrieved_passages[:3] if retrieved_passages else doc["chunks"][:3]

    return QAResponse(
        document_id=request.document_id,
        question=request.question,
        answer=answer,
        retrieved_context=retrieved_context,
        processing_time_ms=processing_time_ms
    )
