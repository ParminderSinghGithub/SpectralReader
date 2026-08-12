from fastapi import APIRouter
from app.models.schemas import QARequest, QAResponse
from app.storage.document_store import DocumentStore
from app.services.qa_service import QAService
from app.core.exceptions import DocumentNotFoundError
from app.core.observability import StageTracker
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/qa", tags=["QA"])

@router.post("", response_model=QAResponse)
def answer_question(request: QARequest):
    """Execute question answering over processed document chunks using provider-agnostic generation."""
    store = DocumentStore.get_instance()
    doc = store.get_document(request.document_id)
    if not doc:
        logger.warning(f"QA requested for non-existent document ID '{request.document_id}'")
        raise DocumentNotFoundError(request.document_id)

    tracker = StageTracker()

    with tracker.measure_stage("qa_generation"):
        answer_text, retrieved_context, llm_resp = QAService.answer_question(
            question=request.question,
            docs=doc["chunks"]
        )

    processing_time_ms = tracker.total_elapsed_ms()

    token_usage_dict = {
        "prompt_tokens": llm_resp.token_usage.prompt_tokens,
        "completion_tokens": llm_resp.token_usage.completion_tokens,
        "total_tokens": llm_resp.token_usage.total_tokens
    }

    logger.info(f"Executed QA query for document '{request.document_id}' in {processing_time_ms} ms via provider '{llm_resp.provider_name}'")

    return QAResponse(
        document_id=request.document_id,
        question=request.question,
        answer=answer_text,
        retrieved_context=retrieved_context,
        processing_time_ms=processing_time_ms,
        llm_provider=llm_resp.provider_name,
        model_name=llm_resp.model_name,
        token_usage=token_usage_dict,
        stage_latencies=tracker.stages
    )
