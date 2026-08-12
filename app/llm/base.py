from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass
class GenerationConfig:
    temperature: float = 0.4
    top_p: float = 0.9
    max_tokens: int = 512

@dataclass
class LLMResponse:
    text: str
    provider_name: str
    model_name: str
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseLLMProvider(ABC):
    """Abstract interface for all LLM generation providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'gemini')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of active model (e.g. 'gemini-2.5-flash')."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> LLMResponse:
        """Generate response from text prompt."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Verify API key and connectivity to provider endpoint."""
        pass
