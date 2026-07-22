import time
from fastapi import APIRouter
from app.models.schemas import QARequest, QAResponse
from app.storage.document_store import DocumentStore
from app.services.model_service import ModelService
from app.services.qa_service import QAService
from app.services.metadata_service import MetadataService
from app.core.exceptions import DocumentNotFoundError, ModelInitializationError
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/qa", tags=["QA"])

@router.post("", response_model=QAResponse)
def answer_question(request: QARequest):
    """Execute question answering over processed document chunks."""
    store = DocumentStore.get_instance()
    doc = store.get_document(request.document_id)
    if not doc:
        logger.warning(f"QA requested for non-existent document ID '{request.document_id}'")
        raise DocumentNotFoundError(request.document_id)

    models = ModelService.get_model_container()
    if models is None:
        logger.error("QA execution failed due to uninitialized backend models")
        raise ModelInitializationError("Backend ML models could not be loaded.")

    start_time = time.perf_counter()
    answer = QAService.answer_question(
        question=request.question,
        docs=doc["chunks"],
        tokenizer=models.tokenizer,
        model=models.qa_model
    )
    end_time = time.perf_counter()
    processing_time_ms = round((end_time - start_time) * 1000, 2)

    retrieved_passages = []
    for chunk in doc["chunks"]:
        if any(entity in chunk for entity in MetadataService.extract_entities(chunk)):
            retrieved_passages.append(chunk)
    retrieved_context = retrieved_passages[:3] if retrieved_passages else doc["chunks"][:3]

    logger.info(f"Executed QA query for document '{request.document_id}' in {processing_time_ms} ms")

    return QAResponse(
        document_id=request.document_id,
        question=request.question,
        answer=answer,
        retrieved_context=retrieved_context,
        processing_time_ms=processing_time_ms
    )
