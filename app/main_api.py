import os
import sys
import time
import uuid
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
    logger.info(
        f"Configuration -> HOST: {settings.HOST}, PORT: {settings.PORT}, LOG_LEVEL: {settings.LOG_LEVEL}, "
        f"LLM_PROVIDER: {settings.LLM_PROVIDER}, OCR_PROVIDER: {settings.OCR_PROVIDER}"
    )
    
    models = ModelService.get_model_container()
    if models is not None:
        logger.info("Backend ML vector models successfully pre-warmed and loaded.")
    else:
        logger.warning("Backend ML vector models failed to initialize on startup.")
    
    yield
    logger.info("Shutting down SpectralReader API backend.")

tags_metadata = [
    {
        "name": "Health",
        "description": "Service health monitoring, vector models, active LLM provider, and OCR engine status.",
    },
    {
        "name": "Documents",
        "description": "Document ingestion, PDF detection, parser/OCR extraction, metadata extraction, and storage.",
    },
    {
        "name": "Search",
        "description": "Candidate passage search and cross-encoder reranking over document text.",
    },
    {
        "name": "QA",
        "description": "Provider-agnostic generative question answering powered by Google Gemini.",
    },
]

app = FastAPI(
    title="SpectralReader Document Intelligence API",
    description="""
    ### 📖 SpectralReader Document Intelligence API
    
    Production-grade REST microservice providing:
    - **PDF Ingestion, Detection & OCR** (`/documents`)
    - **Entity Metadata Extraction** (`/documents`)
    - **Passage Retrieval & Reranking** (`/search`)
    - **Provider-Agnostic Question Answering** (`/qa`)
    - **Service Health Probes & Component Status** (`/health`)
    """,
    version="1.1.0",
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

# Custom Middleware for Request Tracing and Timing
@app.middleware("http")
async def log_requests_and_timing(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Process-Time"] = f"{process_time_ms}ms"
    response.headers["X-Request-ID"] = req_id
    logger.info(f"[{req_id}] {request.method} {request.url.path} - Status: {response.status_code} - Duration: {process_time_ms}ms")
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
