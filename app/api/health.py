from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.services.model_service import ModelService

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
def get_health():
    """Return backend service health status and model load status."""
    models = ModelService.get_model_container()
    return HealthResponse(
        status="ok",
        service="SpectralReader Document Intelligence API",
        version="1.0.0",
        models_loaded=models is not None
    )
