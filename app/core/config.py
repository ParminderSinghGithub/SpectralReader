import os
from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class Settings:
    # Deployment-specific configuration (driven by environment variables with defaults)
    HOST: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    PORT: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    API_BASE_URL: str = field(default_factory=lambda: os.getenv("API_BASE_URL", "http://localhost:8000"))
    CORS_ORIGINS: List[str] = field(
        default_factory=lambda: [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")]
    )

    # Internal system behavior configuration (constants)
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    GENERATIVE_MODEL_NAME: str = "google/flan-t5-large"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 300
    MAX_PROMPT_CONTEXT_CHARS: int = 4000
    MAX_GEN_LENGTH: int = 512
    GEN_TEMPERATURE: float = 0.4
    GEN_TOP_P: float = 0.9

settings = Settings()
