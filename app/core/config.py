import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file centrally
load_dotenv()

@dataclass(frozen=True)
class Settings:
    # Deployment-specific configuration (driven by environment variables with sensible defaults)
    HOST: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    PORT: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    API_BASE_URL: str = field(default_factory=lambda: os.getenv("API_BASE_URL", "http://localhost:8000"))
    STREAMLIT_BACKEND_URL: str = field(
        default_factory=lambda: os.getenv("STREAMLIT_BACKEND_URL", os.getenv("API_BASE_URL", "http://localhost:8000"))
    )
    CORS_ORIGINS: List[str] = field(
        default_factory=lambda: [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")]
    )
    HF_TOKEN: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN", None))
    MODEL_CACHE_DIR: Optional[str] = field(default_factory=lambda: os.getenv("MODEL_CACHE_DIR", None))

    # LLM Provider Configuration
    LLM_PROVIDER: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini").lower())
    GEMINI_API_KEY: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", None)))
    GEMINI_DEFAULT_MODEL: str = field(default_factory=lambda: os.getenv("GEMINI_DEFAULT_MODEL", "gemini-3.1-flash-lite"))
    GEMINI_FALLBACK_MODELS: List[str] = field(
        default_factory=lambda: [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-3.6-flash").split(",") if m.strip()]
    )

    # OCR Configuration
    ENABLE_OCR: bool = field(default_factory=lambda: os.getenv("ENABLE_OCR", "true").lower() == "true")
    OCR_PROVIDER: str = field(default_factory=lambda: os.getenv("OCR_PROVIDER", "tesseract").lower())
    OCR_MIN_TEXT_CHARS_PER_PAGE: int = field(default_factory=lambda: int(os.getenv("OCR_MIN_TEXT_CHARS_PER_PAGE", "50")))

    # Retrieval & Generation Parameters
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 300
    MAX_PROMPT_CONTEXT_CHARS: int = 4000
    MAX_CONTEXT_WINDOW_TOKENS: int = 8000
    MAX_GEN_LENGTH: int = 512
    GEN_TEMPERATURE: float = 0.4
    GEN_TOP_P: float = 0.9

settings = Settings()
