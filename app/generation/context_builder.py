from typing import List, Tuple
from app.generation.token_budget import TokenBudgetManager
from app.core.config import settings

class ContextBuilder:
    """Handles chunk deduplication, citation tagging, and context budgeting."""

    def __init__(self, max_context_chars: int = 4000):
        self.budget_manager = TokenBudgetManager(max_context_chars=max_context_chars)

    def prepare_context(self, retrieved_chunks: List[str]) -> Tuple[List[str], int]:
        """Deduplicate chunks, enforce character budget, and return (final_passages, char_count)."""
        # Deduplicate while preserving order
        seen = set()
        deduped: List[str] = []
        for chunk in retrieved_chunks:
            cleaned = chunk.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                deduped.append(cleaned)

        budgeted_passages, total_chars = self.budget_manager.enforce_char_budget(deduped)
        return budgeted_passages, total_chars
