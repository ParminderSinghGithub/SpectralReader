from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    API_BASE_URL: str = "http://localhost:8000"
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
