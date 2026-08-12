from typing import Dict, Type, Optional
from app.llm.base import BaseLLMProvider
from app.core.config import settings
from app.core.exceptions import LLMProviderError
from app.core.logger import get_logger

logger = get_logger(__name__)

class LLMProviderFactory:
    """Registry and factory for LLM providers."""
    _providers: Dict[str, Type[BaseLLMProvider]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a provider implementation class."""
        def decorator(provider_cls: Type[BaseLLMProvider]):
            cls._providers[name.lower()] = provider_cls
            return provider_cls
        return decorator

    @classmethod
    def get_provider(
        cls,
        name: Optional[str] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """Instantiate registered provider by name (defaults to settings.LLM_PROVIDER)."""
        provider_name = (name or settings.LLM_PROVIDER).lower()
        if provider_name not in cls._providers:
            raise LLMProviderError(
                provider_name,
                f"Unsupported provider '{provider_name}'. Registered providers: {list(cls._providers.keys())}"
            )
        provider_cls = cls._providers[provider_name]
        return provider_cls(**kwargs)
