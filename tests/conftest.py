import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main_api import app
from app.storage.document_store import DocumentStore
from app.services.model_service import ModelContainer
from app.llm.base import BaseLLMProvider, LLMResponse, GenerationConfig, TokenUsage

class MockLLMProvider(BaseLLMProvider):
    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return "gemini-2.5-flash"

    def generate(self, prompt: str, config=None) -> LLMResponse:
        return LLMResponse(
            text="This is a mocked answer for testing context and generation.",
            provider_name="gemini",
            model_name="gemini-2.5-flash",
            token_usage=TokenUsage(10, 20, 30)
        )

    def health_check(self) -> bool:
        return True

@pytest.fixture
def reset_document_store():
    """Reset the in-memory document store and retrieval index state before and after each test."""
    from app.services.retrieval_service import RetrievalService
    store = DocumentStore.get_instance()
    retrieval = RetrievalService.get_instance()
    store._documents.clear()
    retrieval.clear()
    yield store
    store._documents.clear()
    retrieval.clear()

@pytest.fixture
def mock_model_container():
    """Mocked backend model container for fast, deterministic testing without loading ML weights."""
    mock_embeddings = MagicMock()
    mock_reranker = MagicMock()

    # Deterministic mock vectors and reranker predictions
    def fake_embed_documents(texts):
        return [[float(i + 1) / (j + 1) for j in range(768)] for i, _ in enumerate(texts)]

    def fake_embed_query(text):
        return [1.0 / (j + 1) for j in range(768)]

    def fake_predict(pairs):
        import numpy as np
        return np.array([float(10.0 - i) for i, _ in enumerate(pairs)], dtype=np.float32)

    mock_embeddings.embed_documents.side_effect = fake_embed_documents
    mock_embeddings.embed_query.side_effect = fake_embed_query
    mock_reranker.predict.side_effect = fake_predict

    return ModelContainer(
        embeddings=mock_embeddings,
        reranker=mock_reranker
    )

@pytest.fixture
def test_client(mock_model_container, reset_document_store):
    """FastAPI TestClient with pre-mocked ModelService backend and LLM provider."""
    with patch("app.services.model_service.ModelService.get_model_container", return_value=mock_model_container):
        with patch("app.llm.factory.LLMProviderFactory.get_provider", return_value=MockLLMProvider()):
            with TestClient(app) as client:
                yield client
