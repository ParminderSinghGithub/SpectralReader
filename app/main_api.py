import os
import sys
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure repository root is on sys.path for app module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.api import health, documents, search, qa
from app.services.model_service import ModelService
from app.core.config import settings
from app.core.exceptions import SpectralReaderException
from app.models.schemas import ErrorResponse
from app.core.logger import get_logger

logger = get_logger("SpectralReader-API")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle startup logging and model pre-warming."""
    logger.info("Starting SpectralReader Document Intelligence API backend...")
    logger.info(f"Configuration -> HOST: {settings.HOST}, PORT: {settings.PORT}, LOG_LEVEL: {settings.LOG_LEVEL}, API_BASE_URL: {settings.API_BASE_URL}")
    
    models = ModelService.get_model_container()
    if models is not None:
        logger.info("Backend ML models successfully pre-warmed and loaded.")
    else:
        logger.warning("Backend ML models failed to initialize on startup.")
    
    yield
    logger.info("Shutting down SpectralReader API backend.")

tags_metadata = [
    {
        "name": "Health",
        "description": "Service health monitoring and model initialization status verification.",
    },
    {
        "name": "Documents",
        "description": "Document ingestion, text parsing, entity metadata extraction, and storage management.",
    },
    {
        "name": "Search",
        "description": "Candidate passage search and chunk filtering over document text.",
    },
    {
        "name": "QA",
        "description": "Generative question answering over document passages using FLAN-T5.",
    },
]

app = FastAPI(
    title="SpectralReader Document Intelligence API",
    description="""
    ### 📖 SpectralReader Document Intelligence API
    
    Production-grade REST microservice providing:
    - **PDF Ingestion & Text Parsing** (`/documents`)
    - **Entity Metadata Extraction** (`/documents`)
    - **Passage Retrieval & Search** (`/search`)
    - **Generative Question Answering** (`/qa`)
    - **Service Health Probes** (`/health`)
    """,
    version="1.0.0",
    openapi_tags=tags_metadata,
    lifespan=lifespan
)

# Enable CORS for configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Middleware for Request Logging and Timing
@app.middleware("http")
async def log_requests_and_timing(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Process-Time"] = f"{process_time_ms}ms"
    logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} - Duration: {process_time_ms}ms")
    return response

# Custom Exception Handlers
@app.exception_handler(SpectralReaderException)
async def spectral_reader_exception_handler(request: Request, exc: SpectralReaderException):
    logger.warning(f"Domain Exception [{exc.status_code}] on {request.method} {request.url.path}: {exc.message}")
    error_payload = ErrorResponse(
        status="error",
        message=exc.message,
        detail=exc.detail,
        status_code=exc.status_code
    )
    return JSONResponse(status_code=exc.status_code, content=error_payload.model_dump())

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP Exception [{exc.status_code}] on {request.method} {request.url.path}: {exc.detail}")
    error_payload = ErrorResponse(
        status="error",
        message=str(exc.detail),
        detail=str(exc.detail),
        status_code=exc.status_code
    )
    return JSONResponse(status_code=exc.status_code, content=error_payload.model_dump())

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception on {request.method} {request.url.path}: {str(exc)}", exc_info=True)
    error_payload = ErrorResponse(
        status="error",
        message="An unexpected internal server error occurred.",
        detail=str(exc),
        status_code=500
    )
    return JSONResponse(status_code=500, content=error_payload.model_dump())

# Include API routers
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(qa.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main_api:app", host=settings.HOST, port=settings.PORT, reload=True)
