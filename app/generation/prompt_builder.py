from typing import List
from app.generation.templates import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE, SUMMARIZE_TEMPLATE, EXTRACT_TEMPLATE

class PromptBuilder:
    """Renders structural system and user prompt strings."""

    @staticmethod
    def build_qa_prompt(context_passages: List[str], question: str) -> str:
        """Format context passages and question into complete LLM QA prompt."""
        formatted_context = "\n\n".join(
            f"[Passage {i+1}]: {p}" for i, p in enumerate(context_passages)
        )
        user_body = QA_USER_TEMPLATE.format(context=formatted_context, question=question)
        return f"{QA_SYSTEM_PROMPT}\n\n{user_body}"

    @staticmethod
    def build_summary_prompt(context_passages: List[str]) -> str:
        formatted_context = "\n\n".join(context_passages)
        return SUMMARIZE_TEMPLATE.format(context=formatted_context)

    @staticmethod
    def build_extraction_prompt(context_passages: List[str]) -> str:
        formatted_context = "\n\n".join(context_passages)
        return EXTRACT_TEMPLATE.format(context=formatted_context)
