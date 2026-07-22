import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure repository root is on sys.path for app module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.api import health, documents, search, qa
from app.services.model_service import ModelService
from app.core.logger import get_logger

logger = get_logger("SpectralReader-API")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to pre-warm backend models on server startup."""
    logger.info("Starting SpectralReader Document Intelligence API backend...")
    models = ModelService.get_model_container()
    if models is not None:
        logger.info("Backend ML models successfully initialized.")
    else:
        logger.warning("Failed to initialize backend models on startup.")
    yield
    logger.info("Shutting down SpectralReader API backend.")

app = FastAPI(
    title="SpectralReader Document Intelligence API",
    description="REST API service for document extraction, metadata analysis, chunking, and question answering.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(qa.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main_api:app", host="0.0.0.0", port=8000, reload=True)
