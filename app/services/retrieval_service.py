from typing import Dict, List, Optional, Any, Union
import numpy as np
import faiss

from app.core.config import settings
from app.core.exceptions import RetrievalError, DocumentNotFoundError
from app.core.logger import get_logger
from app.services.model_service import ModelService

logger = get_logger(__name__)

class RetrievalService:
    """Singleton service managing in-memory FAISS vector indexing, dense retrieval, and CrossEncoder reranking."""
    _instance: Optional['RetrievalService'] = None

    def __init__(self):
        # Per-document FAISS index mapping for strict document isolation
        self._indices: Dict[str, Optional[faiss.IndexFlatIP]] = {}
        # Per-document chunk metadata mapping: doc_id -> list of chunk dicts
        self._chunks: Dict[str, List[Dict[str, Any]]] = {}

    @classmethod
    def get_instance(cls) -> 'RetrievalService':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def index_document(self, doc_id: str, chunks: List[str]) -> int:
        """Embed document chunks using HuggingFaceEmbeddings, normalize vectors, and index in in-memory FAISS IndexFlatIP."""
        if not chunks:
            logger.warning(f"Indexing requested for document '{doc_id}' with zero chunks.")
            self._indices[doc_id] = None
            self._chunks[doc_id] = []
            return 0

        model_container = ModelService.get_model_container()
        if model_container is None or model_container.embeddings is None:
            logger.error("Failed to index document: Embedding model is not loaded in ModelService.")
            raise RetrievalError("Embedding model is unavailable in ModelService.")

        try:
            raw_embeddings = model_container.embeddings.embed_documents(chunks)
            vectors = np.array(raw_embeddings, dtype=np.float32)

            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)

            dimension = vectors.shape[1]

            # L2 normalize vectors so Inner Product (IndexFlatIP) calculates Cosine Similarity
            faiss.normalize_L2(vectors)

            index = faiss.IndexFlatIP(dimension)
            index.add(vectors)

            self._indices[doc_id] = index
            self._chunks[doc_id] = [
                {
                    "chunk_id": f"{doc_id}_{idx}",
                    "chunk_index": idx,
                    "text": chunk
                }
                for idx, chunk in enumerate(chunks)
            ]

            logger.info(f"Successfully indexed document '{doc_id}' with {index.ntotal} vectors (dim={dimension}).")
            return int(index.ntotal)

        except RetrievalError:
            raise
        except Exception as e:
            logger.error(f"Vector indexing failed for document '{doc_id}': {str(e)}", exc_info=True)
            raise RetrievalError(f"Vector indexing failed: {str(e)}")

    def retrieve_candidates(
        self,
        doc_id: str,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k candidate chunks from the document's FAISS index using normalized inner-product cosine similarity."""
        k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K

        # Check document isolation mapping
        if doc_id not in self._indices:
            from app.storage.document_store import DocumentStore
            doc = DocumentStore.get_instance().get_document(doc_id)
            if not doc:
                logger.warning(f"Retrieval attempted for non-existent document ID '{doc_id}'.")
                raise DocumentNotFoundError(doc_id)

            # Lazy indexing fallback if document was stored but index not yet created
            if doc.get("chunks"):
                self.index_document(doc_id, doc["chunks"])
            else:
                return []

        if not query or not query.strip():
            logger.info(f"Empty or whitespace query provided for document '{doc_id}'. Returning empty candidate list.")
            return []

        index = self._indices.get(doc_id)
        if index is None or index.ntotal == 0:
            logger.info(f"Document '{doc_id}' has an empty vector index.")
            return []

        model_container = ModelService.get_model_container()
        if model_container is None or model_container.embeddings is None:
            logger.error("Embedding model is unavailable for query embedding.")
            raise RetrievalError("Embedding model is unavailable in ModelService.")

        try:
            query_embedding = model_container.embeddings.embed_query(query)
            query_vec = np.array([query_embedding], dtype=np.float32)

            if query_vec.ndim == 1:
                query_vec = query_vec.reshape(1, -1)

            faiss.normalize_L2(query_vec)

            search_k = min(k, index.ntotal)
            distances, indices = index.search(query_vec, search_k)

            doc_chunks = self._chunks.get(doc_id, [])
            candidates: List[Dict[str, Any]] = []

            for idx_val, score in zip(indices[0], distances[0]):
                if 0 <= idx_val < len(doc_chunks):
                    chunk_item = doc_chunks[idx_val].copy()
                    chunk_item["dense_score"] = float(score)
                    candidates.append(chunk_item)

            logger.info(f"Retrieved {len(candidates)} dense candidates for document '{doc_id}' with query '{query[:50]}'.")
            return candidates

        except RetrievalError:
            raise
        except Exception as e:
            logger.error(f"Candidate retrieval failed for document '{doc_id}': {str(e)}", exc_info=True)
            raise RetrievalError(f"Candidate retrieval failed: {str(e)}")

    def rerank(
        self,
        query: str,
        candidates: List[Union[str, Dict[str, Any]]],
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Score (query, passage) candidate pairs using CrossEncoder and return top-n reranked items."""
        n = top_n if top_n is not None else settings.RERANK_TOP_N

        if not candidates:
            return []

        if not query or not query.strip():
            logger.info("Empty query provided for reranking. Returning unranked candidates.")
            return [
                c if isinstance(c, dict) else {"chunk_id": None, "text": str(c), "score": 0.0}
                for c in candidates[:n]
            ]

        model_container = ModelService.get_model_container()
        if model_container is None or model_container.reranker is None:
            logger.warning("CrossEncoder reranker unavailable. Returning raw candidate order.")
            return [
                c if isinstance(c, dict) else {"chunk_id": None, "text": str(c), "score": 0.0}
                for c in candidates[:n]
            ]

        try:
            pairs = []
            for c in candidates:
                text = c["text"] if isinstance(c, dict) else str(c)
                pairs.append([query, text])

            scores = model_container.reranker.predict(pairs)

            scored_candidates: List[Dict[str, Any]] = []
            for c, score in zip(candidates, scores):
                score_val = float(score)
                if isinstance(c, dict):
                    item = c.copy()
                    item["score"] = score_val
                    item["rerank_score"] = score_val
                else:
                    item = {
                        "chunk_id": None,
                        "text": str(c),
                        "score": score_val,
                        "rerank_score": score_val
                    }
                scored_candidates.append(item)

            # Sort descending by CrossEncoder score
            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
            reranked = scored_candidates[:n]

            logger.info(f"Reranked {len(candidates)} candidates down to top {len(reranked)} passages.")
            return reranked

        except Exception as e:
            logger.error(f"CrossEncoder reranking failed: {str(e)}", exc_info=True)
            raise RetrievalError(f"Reranking failed: {str(e)}")

    def rerank_texts(
        self,
        query: str,
        candidates: List[Union[str, Dict[str, Any]]],
        top_n: Optional[int] = None
    ) -> List[str]:
        """Convenience helper returning just the text strings of top-n reranked passages."""
        reranked = self.rerank(query, candidates, top_n)
        return [item["text"] for item in reranked]

    def delete_index(self, doc_id: str) -> bool:
        """Remove in-memory index and chunk metadata for the specified document."""
        existed = False
        if doc_id in self._indices:
            del self._indices[doc_id]
            existed = True
        if doc_id in self._chunks:
            del self._chunks[doc_id]
            existed = True
        if existed:
            logger.info(f"Deleted vector index and metadata for document '{doc_id}'.")
        return existed

    def clear(self) -> None:
        """Clear all in-memory retrieval indices and chunk metadata across all documents."""
        self._indices.clear()
        self._chunks.clear()
        logger.info("Cleared all in-memory retrieval indices and metadata.")
