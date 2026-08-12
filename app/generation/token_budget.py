from typing import List, Tuple
from app.core.config import settings

class TokenBudgetManager:
    """Manages token window allocation and passage context truncation."""

    def __init__(self, max_context_chars: int = 4000, max_tokens: int = 8000):
        self.max_context_chars = max_context_chars
        self.max_tokens = max_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Lightweight heuristic estimate of token count (approx. 4 chars per token)."""
        return max(1, len(text) // 4)

    def enforce_char_budget(self, passages: List[str]) -> Tuple[List[str], int]:
        """Fit passages into total max_context_chars budget without cutting sentences mid-way."""
        budgeted: List[str] = []
        current_chars = 0

        for passage in passages:
            p_len = len(passage)
            if current_chars + p_len <= self.max_context_chars:
                budgeted.append(passage)
                current_chars += p_len + 2  # include separator space
            else:
                remaining = self.max_context_chars - current_chars
                if remaining > 100:
                    budgeted.append(passage[:remaining] + "...")
                    current_chars += len(budgeted[-1])
                break

        return budgeted, current_chars
