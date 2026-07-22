import torch
from typing import NamedTuple, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import CrossEncoder
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

class ModelContainer(NamedTuple):
    embeddings: HuggingFaceEmbeddings
    tokenizer: AutoTokenizer
    qa_model: AutoModelForSeq2SeqLM
    reranker: CrossEncoder

class ModelService:
    _instance: Optional[ModelContainer] = None

    @classmethod
    def get_model_container(cls) -> Optional[ModelContainer]:
        """Singleton getter for loaded backend models."""
        if cls._instance is None:
            cls._instance = cls._load_models()
        return cls._instance

    @classmethod
    def _load_models(cls) -> Optional[ModelContainer]:
        """Internal model initialization logic."""
        try:
            logger.info("Loading embedding model...")
            embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'}
            )
            logger.info("Loading tokenizer & generative model...")
            tokenizer = AutoTokenizer.from_pretrained(settings.GENERATIVE_MODEL_NAME)
            qa_model = AutoModelForSeq2SeqLM.from_pretrained(
                settings.GENERATIVE_MODEL_NAME,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            logger.info("Loading cross-encoder reranker...")
            reranker = CrossEncoder(settings.RERANKER_MODEL_NAME)
            logger.info("All backend models loaded successfully.")
            return ModelContainer(
                embeddings=embeddings,
                tokenizer=tokenizer,
                qa_model=qa_model,
                reranker=reranker
            )
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return None
