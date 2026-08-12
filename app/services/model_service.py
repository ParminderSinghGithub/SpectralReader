from typing import NamedTuple, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

class ModelContainer(NamedTuple):
    embeddings: HuggingFaceEmbeddings
    reranker: CrossEncoder

class ModelService:
    _instance: Optional[ModelContainer] = None

    @classmethod
    def get_model_container(cls) -> Optional[ModelContainer]:
        """Singleton getter for loaded backend vector and reranker models."""
        if cls._instance is None:
            cls._instance = cls._load_models()
        return cls._instance

    @classmethod
    def _load_models(cls) -> Optional[ModelContainer]:
        """Internal model initialization logic."""
        try:
            cache_dir = settings.MODEL_CACHE_DIR if settings.MODEL_CACHE_DIR else None

            logger.info("Loading embedding model...")
            embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'}
            )

            logger.info("Loading cross-encoder reranker...")
            reranker_kwargs = {}
            if cache_dir:
                reranker_kwargs['cache_folder'] = cache_dir
            reranker = CrossEncoder(settings.RERANKER_MODEL_NAME, **reranker_kwargs)

            logger.info("All backend vector models loaded successfully.")
            return ModelContainer(
                embeddings=embeddings,
                reranker=reranker
            )
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return None
