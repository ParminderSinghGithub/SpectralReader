import requests
from typing import Optional
from app.llm.base import BaseLLMProvider, LLMResponse, GenerationConfig, TokenUsage
from app.llm.factory import LLMProviderFactory
from app.core.config import settings
from app.core.exceptions import LLMProviderError
from app.core.logger import get_logger

logger = get_logger(__name__)

@LLMProviderFactory.register("gemini")
class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM Provider using official Gemini REST API endpoints with 429 model fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        fallback_models: Optional[list] = None
    ):
        self._api_key = api_key or settings.GEMINI_API_KEY
        self._model_name = model_name or settings.GEMINI_DEFAULT_MODEL
        self._fallback_models = fallback_models if fallback_models is not None else settings.GEMINI_FALLBACK_MODELS
        self._base_url = "https://generativelanguage.googleapis.com/v1beta"

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> LLMResponse:
        if not self._api_key:
            raise LLMProviderError(self.provider_name, "GEMINI_API_KEY environment variable is missing or empty.")

        cfg = config or GenerationConfig()
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": cfg.temperature,
                "topP": cfg.top_p,
                "maxOutputTokens": cfg.max_tokens
            }
        }

        # Build deduplicated ordered candidate model chain: primary -> secondary -> tertiary
        candidate_models = [self._model_name]
        for fb in self._fallback_models:
            if fb and fb not in candidate_models:
                candidate_models.append(fb)

        last_429_detail = None

        for current_model in candidate_models:
            endpoint = f"{self._base_url}/models/{current_model}:generateContent?key={self._api_key}"

            try:
                response = requests.post(endpoint, json=payload, timeout=60)

                # Fallback ONLY on HTTP 429 Rate Limit / Quota Exhaustion
                if response.status_code == 429:
                    last_429_detail = f"Model '{current_model}' 429 Rate Limit: {response.text}"
                    logger.warning(
                        f"Gemini API 429 Rate Limit encountered for model '{current_model}'. "
                        f"Attempting fallback to next model..."
                    )
                    continue

                if response.status_code != 200:
                    error_msg = f"Gemini API returned status {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    raise LLMProviderError(self.provider_name, error_msg)

                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise LLMProviderError(self.provider_name, f"Gemini API returned response with no candidates for model '{current_model}'.")

                parts = candidates[0].get("content", {}).get("parts", [])
                text_output = "".join([p.get("text", "") for p in parts if "text" in p]).strip()
                if not text_output:
                    finish_reason = candidates[0].get("finishReason", "UNKNOWN")
                    raise LLMProviderError(self.provider_name, f"Gemini API candidate contained no text parts (finishReason: {finish_reason}).")

                usage_metadata = data.get("usageMetadata", {})
                token_usage = TokenUsage(
                    prompt_tokens=usage_metadata.get("promptTokenCount", 0),
                    completion_tokens=usage_metadata.get("candidatesTokenCount", 0),
                    total_tokens=usage_metadata.get("totalTokenCount", 0)
                )

                return LLMResponse(
                    text=text_output,
                    provider_name=self.provider_name,
                    model_name=current_model,
                    token_usage=token_usage
                )

            except LLMProviderError:
                raise
            except Exception as e:
                logger.error(f"Failed to communicate with Gemini API model '{current_model}': {str(e)}")
                raise LLMProviderError(self.provider_name, str(e))

        # If all configured fallback models encountered 429 quota exhaustion
        error_msg = f"All configured Gemini models exceeded rate limits (429). Last error: {last_429_detail}"
        logger.error(error_msg)
        raise LLMProviderError(self.provider_name, error_msg)

    def health_check(self) -> bool:
        return bool(self._api_key and len(self._api_key.strip()) > 0)
