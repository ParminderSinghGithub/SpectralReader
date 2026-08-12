import pytest
from app.llm.factory import LLMProviderFactory
from app.llm.providers.gemini import GeminiProvider
from app.core.exceptions import LLMProviderError

def test_llm_factory_get_gemini_provider():
    provider = LLMProviderFactory.get_provider("gemini")
    assert isinstance(provider, GeminiProvider)
    assert provider.provider_name == "gemini"
    assert provider.model_name == "gemini-3.1-flash-lite"

def test_llm_factory_unsupported_provider():
    with pytest.raises(LLMProviderError):
        LLMProviderFactory.get_provider("non_existent_provider_xyz")
