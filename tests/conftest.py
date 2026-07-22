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

@pytest.fixture
def reset_document_store():
    """Reset the in-memory document store state before and after each test."""
    store = DocumentStore.get_instance()
    store._documents.clear()
    yield store
    store._documents.clear()

@pytest.fixture
def mock_model_container():
    """Mocked backend model container for fast, deterministic testing without loading ML weights."""
    mock_embeddings = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = MagicMock(input_ids=MagicMock(to=MagicMock(return_value="mock_tensor")))
    mock_tokenizer.decode.return_value = "The main protagonist is Arthur Vance, a brilliant researcher."

    mock_qa_model = MagicMock()
    mock_qa_model.device = "cpu"
    mock_qa_model.generate.return_value = ["generated_output_tensor"]

    mock_reranker = MagicMock()

    return ModelContainer(
        embeddings=mock_embeddings,
        tokenizer=mock_tokenizer,
        qa_model=mock_qa_model,
        reranker=mock_reranker
    )

@pytest.fixture
def test_client(mock_model_container, reset_document_store):
    """FastAPI TestClient with pre-mocked ModelService backend."""
    with patch("app.services.model_service.ModelService.get_model_container", return_value=mock_model_container):
        with TestClient(app) as client:
            yield client
