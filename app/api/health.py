from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.services.model_service import ModelService
from app.llm.factory import LLMProviderFactory
from app.ocr.ocr_service import OCRService
from app.core.config import settings

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
def get_health():
    """Return backend service health status, vector model status, and provider availability."""
    models = ModelService.get_model_container()
    models_loaded = models is not None

    llm_provider_name = settings.LLM_PROVIDER
    llm_available = False
    llm_model_name = settings.GEMINI_DEFAULT_MODEL

    try:
        provider = LLMProviderFactory.get_provider(llm_provider_name)
        llm_available = provider.health_check()
        llm_model_name = provider.model_name
    except Exception:
        llm_available = False

    ocr_provider_name = settings.OCR_PROVIDER
    ocr_available = OCRService.is_available(ocr_provider_name)

    components = {
        "embedding_model": {
            "status": "loaded" if models_loaded and models.embeddings else "uninitialized",
            "model": settings.EMBEDDING_MODEL_NAME
        },
        "reranker_model": {
            "status": "loaded" if models_loaded and models.reranker else "uninitialized",
            "model": settings.RERANKER_MODEL_NAME
        },
        "active_llm_provider": {
            "name": llm_provider_name,
            "model": llm_model_name,
            "available": llm_available
        },
        "ocr_provider": {
            "name": ocr_provider_name,
            "enabled": settings.ENABLE_OCR,
            "available": ocr_available
        }
    }

    return HealthResponse(
        status="ok",
        service="SpectralReader Document Intelligence API",
        version="1.1.0",
        models_loaded=models_loaded,
        components=components
    )
