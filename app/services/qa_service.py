from typing import List, Tuple, Optional
from app.generation.context_builder import ContextBuilder
from app.generation.prompt_builder import PromptBuilder
from app.llm.factory import LLMProviderFactory
from app.llm.base import GenerationConfig, LLMResponse
from app.services.retrieval_service import RetrievalService
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

class QAService:
    """Decoupled RAG question answering service orchestrating vector retrieval, reranking, context building, prompt rendering, and LLM generation."""

    @staticmethod
    def answer_question(
        question: str,
        docs: Optional[List[str]] = None,
        document_id: Optional[str] = None,
        provider_name: Optional[str] = None
    ) -> Tuple[str, List[str], LLMResponse]:
        """Execute RAG pipeline: Retrieval (FAISS -> CrossEncoder) -> ContextBuilder -> PromptBuilder -> LLMProvider."""
        retrieval_service = RetrievalService.get_instance()
        selected_passages: List[str] = []

        if document_id:
            logger.info(f"QAService: Executing semantic retrieval for doc_id '{document_id}' (top_k={settings.RETRIEVAL_TOP_K}, top_n={settings.RERANK_TOP_N})")
            candidates = retrieval_service.retrieve_candidates(
                doc_id=document_id,
                query=question,
                top_k=settings.RETRIEVAL_TOP_K
            )
            selected_passages = retrieval_service.rerank_texts(
                query=question,
                candidates=candidates,
                top_n=settings.RERANK_TOP_N
            )
        elif docs:
            logger.info(f"QAService: Executing reranking over {len(docs)} provided passages (top_n={settings.RERANK_TOP_N})")
            if len(docs) > settings.RERANK_TOP_N:
                selected_passages = retrieval_service.rerank_texts(
                    query=question,
                    candidates=docs,
                    top_n=settings.RERANK_TOP_N
                )
            else:
                selected_passages = list(docs)

        # Context selection: enforce approximately 4,000-char context budget on selected passages
        context_builder = ContextBuilder(max_context_chars=settings.MAX_PROMPT_CONTEXT_CHARS)
        prepared_passages, char_count = context_builder.prepare_context(selected_passages)

        if not prepared_passages:
            fallback_resp = LLMResponse(
                text="The provided document passages do not contain sufficient information to answer this question.",
                provider_name=provider_name or settings.LLM_PROVIDER,
                model_name=settings.GEMINI_DEFAULT_MODEL
            )
            return fallback_resp.text, [], fallback_resp

        prompt = PromptBuilder.build_qa_prompt(prepared_passages, question)

        provider = LLMProviderFactory.get_provider(provider_name)
        gen_config = GenerationConfig(
            temperature=settings.GEN_TEMPERATURE,
            top_p=settings.GEN_TOP_P,
            max_tokens=settings.MAX_GEN_LENGTH
        )

        response = provider.generate(prompt, config=gen_config)
        logger.info(f"QAService generated answer ({len(response.text)} chars) via LLM provider '{provider.provider_name}'")

        return response.text, prepared_passages, response

