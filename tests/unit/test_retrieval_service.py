import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from app.services.retrieval_service import RetrievalService
from app.services.model_service import ModelContainer
from app.storage.document_store import DocumentStore
from app.core.exceptions import DocumentNotFoundError, RetrievalError
from app.core.config import settings

@pytest.fixture
def clean_retrieval_service():
    """Ensure clean retrieval service state before and after each test."""
    retrieval = RetrievalService.get_instance()
    retrieval.clear()
    yield retrieval
    retrieval.clear()

def test_index_document_success(clean_retrieval_service, mock_model_container):
    """Verify document chunks are embedded, L2 normalized, and added to FAISS index."""
    chunks = [
        "First chunk discussing transformer self-attention.",
        "Second chunk covering multi-head attention mechanisms.",
        "Third chunk describing positional encodings."
    ]
    with patch("app.services.model_service.ModelService.get_model_container", return_value=mock_model_container):
        count = clean_retrieval_service.index_document("doc-1", chunks)
        assert count == 3
        assert "doc-1" in clean_retrieval_service._indices
        index = clean_retrieval_service._indices["doc-1"]
        assert index is not None
        assert index.ntotal == 3

def test_index_document_empty_chunks(clean_retrieval_service, mock_model_container):
    """Verify empty chunk list is handled cleanly without errors."""
    with patch("app.services.model_service.ModelService.get_model_container", return_value=mock_model_container):
        count = clean_retrieval_service.index_document("empty-doc", [])
        assert count == 0
        assert clean_retrieval_service._indices["empty-doc"] is None
        assert clean_retrieval_service._chunks["empty-doc"] == []

def test_index_document_missing_models(clean_retrieval_service):
    """Verify RetrievalError is raised when ModelService returns None."""
    with patch("app.services.model_service.ModelService.get_model_container", return_value=None):
        with pytest.raises(RetrievalError) as exc:
            clean_retrieval_service.index_document("doc-err", ["Some chunk"])
        assert "Embedding model is unavailable" in str(exc.value)

def test_retrieve_candidates_success(clean_retrieval_service, mock_model_container):
    """Verify candidate retrieval returns ranked candidate dicts with dense_score and metadata."""
    chunks = [f"Passage chunk content number {i}" for i in range(15)]
    with patch("app.services.model_service.ModelService.get_model_container", return_value=mock_model_container):
        clean_retrieval_service.index_document("doc-15", chunks)
        candidates = clean_retrieval_service.retrieve_candidates("doc-15", "Passage query", top_k=10)
        
        assert len(candidates) == 10
        for c in candidates:
            assert "chunk_id" in c
            assert "text" in c
            assert "dense_score" in c
            assert isinstance(c["dense_score"], float)
            assert c["chunk_id"].startswith("doc-15_")

def test_document_isolation(clean_retrieval_service, mock_model_container):
    """Verify strict document isolation: querying doc A never retrieves doc B chunks."""
    doc_a_chunks = ["Doc A exclusive secret Alpha", "Doc A exclusive secret Beta"]
    doc_b_chunks = ["Doc B exclusive info Gamma", "Doc B exclusive info Delta"]

    with patch("app.services.model_service.ModelService.get_model_container", return_value=mock_model_container):
        clean_retrieval_service.index_document("doc-A", doc_a_chunks)
        clean_retrieval_service.index_document("doc-B", doc_b_chunks)

        candidates_a = clean_retrieval_service.retrieve_candidates("doc-A", "secret", top_k=10)
        candidates_b = clean_retrieval_service.retrieve_candidates("doc-B", "secret", top_k=10)

        for c in candidates_a:
            assert c["text"] in doc_a_chunks
            assert "Doc B" not in c["text"]
            assert c["chunk_id"].startswith("doc-A")

        for c in candidates_b:
            assert c["text"] in doc_b_chunks
            assert "Doc A" not in c["text"]
            assert c["chunk_id"].startswith("doc-B")

def test_retrieve_candidates_unindexed_doc_not_found(clean_retrieval_service):
    """Verify DocumentNotFoundError is raised for non-existent document ID."""
    with pytest.raises(DocumentNotFoundError):
        clean_retrieval_service.retrieve_candidates("ghost-doc-id-404", "query")

def test_retrieve_candidates_empty_query(clean_retrieval_service, mock_model_container):
    """Verify empty or whitespace query returns empty list without error."""
    chunks = ["Sample chunk content"]
    with patch("app.services.model_service.ModelService.get_model_container", return_value=mock_model_container):
        clean_retrieval_service.index_document("doc-query", chunks)
        assert clean_retrieval_service.retrieve_candidates("doc-query", "") == []
        assert clean_retrieval_service.retrieve_candidates("doc-query", "   ") == []

def test_retrieve_fewer_chunks_than_top_k(clean_retrieval_service, mock_model_container):
    """Verify retrieval when total chunks < top_k returns all available chunks."""
    chunks = ["Only chunk 1", "Only chunk 2"]
    with patch("app.services.model_service.ModelService.get_model_container", return_value=mock_model_container):
        clean_retrieval_service.index_document("doc-small", chunks)
        candidates = clean_retrieval_service.retrieve_candidates("doc-small", "query", top_k=10)
        assert len(candidates) == 2

def test_cross_encoder_reranking(clean_retrieval_service, mock_model_container):
    """Verify CrossEncoder scores candidate pairs and sorts descending."""
    candidates = [
        {"chunk_id": "c1", "text": "Irrelevant text about gardening", "dense_score": 0.9},
        {"chunk_id": "c2", "text": "Highly relevant answer about self-attention", "dense_score": 0.8},
        {"chunk_id": "c3", "text": "Moderately relevant text", "dense_score": 0.7},
    ]

    # Configure mock reranker to score c2 highest, then c3, then c1
    def custom_predict(pairs):
        scores = []
        for q, text in pairs:
            if "Highly relevant" in text:
                scores.append(5.0)
            elif "Moderately" in text:
                scores.append(2.0)
            else:
                scores.append(-1.0)
        return np.array(scores, dtype=np.float32)

    mock_model_container.reranker.predict.side_effect = custom_predict

    with patch("app.services.model_service.ModelService.get_model_container", return_value=mock_model_container):
        reranked = clean_retrieval_service.rerank("What is self-attention?", candidates, top_n=2)
        assert len(reranked) == 2
        assert "Highly relevant" in reranked[0]["text"]
        assert reranked[0]["score"] == 5.0
        assert "Moderately" in reranked[1]["text"]
        assert reranked[1]["score"] == 2.0

def test_delete_and_clear_index(clean_retrieval_service, mock_model_container):
    """Verify document deletion and clear methods remove indices."""
    with patch("app.services.model_service.ModelService.get_model_container", return_value=mock_model_container):
        clean_retrieval_service.index_document("doc-del-1", ["Chunk 1"])
        clean_retrieval_service.index_document("doc-del-2", ["Chunk 2"])

        assert clean_retrieval_service.delete_index("doc-del-1") is True
        assert "doc-del-1" not in clean_retrieval_service._indices
        assert "doc-del-2" in clean_retrieval_service._indices

        clean_retrieval_service.clear()
        assert len(clean_retrieval_service._indices) == 0
        assert len(clean_retrieval_service._chunks) == 0

def test_qa_live_path_retrieval_execution(clean_retrieval_service, mock_model_container):
    """Verify live-path QA: when document_id is supplied, QA executes FAISS -> CrossEncoder -> top-N passages -> ContextBuilder, rather than passing all document chunks."""
    from app.services.qa_service import QAService
    from app.generation.context_builder import ContextBuilder

    # Create 12 distinct chunks to verify that only top-3 reranked reach ContextBuilder / Gemini
    chunks = [f"Unique Document Chunk #{i} containing specific content" for i in range(12)]
    doc_id = "qa-integration-test-doc"

    with patch("app.services.model_service.ModelService.get_model_container", return_value=mock_model_container):
        clean_retrieval_service.index_document(doc_id, chunks)

        with patch.object(clean_retrieval_service, "retrieve_candidates", wraps=clean_retrieval_service.retrieve_candidates) as spy_retrieve:
            with patch.object(clean_retrieval_service, "rerank_texts", wraps=clean_retrieval_service.rerank_texts) as spy_rerank:
                with patch.object(ContextBuilder, "prepare_context", wraps=ContextBuilder(max_context_chars=4000).prepare_context) as spy_builder:
                    with patch("app.llm.factory.LLMProviderFactory.get_provider") as mock_get_provider:
                        mock_provider = MagicMock()
                        mock_provider.provider_name = "gemini"
                        mock_resp = MagicMock()
                        mock_resp.text = "Mocked answer based on top-3 retrieved passages."
                        mock_resp.provider_name = "gemini"
                        mock_resp.model_name = "gemini-3.1-flash-lite"
                        mock_provider.generate.return_value = mock_resp
                        mock_get_provider.return_value = mock_provider

                        answer, retrieved_context, llm_resp = QAService.answer_question(
                            question="What is the content of chunk 2?",
                            document_id=doc_id,
                            docs=chunks
                        )

                        # 1. Verify candidate retrieval was invoked with configured RETRIEVAL_TOP_K (10)
                        spy_retrieve.assert_called_once_with(
                            doc_id=doc_id,
                            query="What is the content of chunk 2?",
                            top_k=settings.RETRIEVAL_TOP_K
                        )

                        # 2. Verify CrossEncoder reranking was invoked with candidates and top_n=RERANK_TOP_N (3)
                        spy_rerank.assert_called_once()
                        assert spy_rerank.call_args[1]["top_n"] == settings.RERANK_TOP_N

                        # 3. Verify ContextBuilder received ONLY the top-3 reranked passages, NOT all 12 chunks!
                        spy_builder.assert_called_once()
                        passages_passed_to_builder = spy_builder.call_args[0][0]
                        assert len(passages_passed_to_builder) <= settings.RERANK_TOP_N
                        assert len(passages_passed_to_builder) < len(chunks)

                        # 4. Verify context returned in QA matches the selected passages
                        assert len(retrieved_context) <= settings.RERANK_TOP_N
                        assert answer == "Mocked answer based on top-3 retrieved passages."
