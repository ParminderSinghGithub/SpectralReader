from typing import List, Tuple, Optional
from app.generation.context_builder import ContextBuilder
from app.generation.prompt_builder import PromptBuilder
from app.llm.factory import LLMProviderFactory
from app.llm.base import GenerationConfig, LLMResponse
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

class QAService:
    """Decoupled RAG question answering service orchestrating context building, prompt rendering, and LLM generation."""

    @staticmethod
    def answer_question(
        question: str,
        docs: List[str],
        provider_name: Optional[str] = None
    ) -> Tuple[str, List[str], LLMResponse]:
        """Execute RAG pipeline: ContextBuilder -> PromptBuilder -> LLMProvider."""
        context_builder = ContextBuilder(max_context_chars=settings.MAX_PROMPT_CONTEXT_CHARS)
        prepared_passages, char_count = context_builder.prepare_context(docs)

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
