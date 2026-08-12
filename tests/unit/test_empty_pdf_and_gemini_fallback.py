import pytest
from unittest.mock import MagicMock, patch
from io import BytesIO

from app.core.exceptions import InvalidDocumentError, LLMProviderError
from app.ocr.pdf_detector import PDFDetector
from app.llm.providers.gemini import GeminiProvider

def test_empty_pdf_detection_fast_400():
    """Verify 0-byte or empty PDF streams raise InvalidDocumentError fast (mapped to HTTP 400)."""
    empty_stream = BytesIO(b"")
    with pytest.raises(InvalidDocumentError) as exc_info:
        PDFDetector.inspect_pdf(empty_stream)
    assert exc_info.value.status_code == 400

def test_zero_byte_tesseract_fast_400():
    """Verify 0-byte streams raise InvalidDocumentError in TesseractProvider fast (mapped to HTTP 400)."""
    from app.ocr.providers.tesseract import TesseractProvider
    provider = TesseractProvider()
    empty_stream = BytesIO(b"")
    with pytest.raises(InvalidDocumentError) as exc_info:
        provider.extract_text(empty_stream)
    assert exc_info.value.status_code == 400

@patch("requests.post")
def test_gemini_fallback_primary_success(mock_post):
    """Verify primary model (gemini-3.1-flash-lite) succeeds on first attempt."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Answer from primary model"}]}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
    }
    mock_post.return_value = mock_response

    provider = GeminiProvider(
        api_key="test_key",
        model_name="gemini-3.1-flash-lite",
        fallback_models=["gemini-3.5-flash-lite", "gemini-3.6-flash"]
    )
    res = provider.generate("Test prompt")

    assert res.text == "Answer from primary model"
    assert res.model_name == "gemini-3.1-flash-lite"
    assert mock_post.call_count == 1
    assert "gemini-3.1-flash-lite" in mock_post.call_args[0][0]

@patch("requests.post")
def test_gemini_fallback_primary_429_secondary_success(mock_post):
    """Verify primary 429 triggers fallback to secondary (gemini-3.5-flash-lite)."""
    res_429 = MagicMock()
    res_429.status_code = 429
    res_429.text = "Quota exceeded"

    res_200 = MagicMock()
    res_200.status_code = 200
    res_200.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Answer from secondary model"}]}}],
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 6, "totalTokenCount": 18}
    }

    mock_post.side_effect = [res_429, res_200]

    provider = GeminiProvider(
        api_key="test_key",
        model_name="gemini-3.1-flash-lite",
        fallback_models=["gemini-3.5-flash-lite", "gemini-3.6-flash"]
    )
    res = provider.generate("Test prompt")

    assert res.text == "Answer from secondary model"
    assert res.model_name == "gemini-3.5-flash-lite"
    assert mock_post.call_count == 2

@patch("requests.post")
def test_gemini_fallback_primary_and_secondary_429_tertiary_success(mock_post):
    """Verify primary 429 + secondary 429 triggers fallback to tertiary (gemini-3.6-flash)."""
    res_429 = MagicMock()
    res_429.status_code = 429
    res_429.text = "Quota exceeded"

    res_200 = MagicMock()
    res_200.status_code = 200
    res_200.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Answer from tertiary model"}]}}],
        "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 8, "totalTokenCount": 23}
    }

    mock_post.side_effect = [res_429, res_429, res_200]

    provider = GeminiProvider(
        api_key="test_key",
        model_name="gemini-3.1-flash-lite",
        fallback_models=["gemini-3.5-flash-lite", "gemini-3.6-flash"]
    )
    res = provider.generate("Test prompt")

    assert res.text == "Answer from tertiary model"
    assert res.model_name == "gemini-3.6-flash"
    assert mock_post.call_count == 3

@patch("requests.post")
def test_gemini_fallback_all_429_raises_error(mock_post):
    """Verify LLMProviderError is raised if all 3 models return 429."""
    res_429 = MagicMock()
    res_429.status_code = 429
    res_429.text = "Quota exceeded"

    mock_post.side_effect = [res_429, res_429, res_429]

    provider = GeminiProvider(
        api_key="test_key",
        model_name="gemini-3.1-flash-lite",
        fallback_models=["gemini-3.5-flash-lite", "gemini-3.6-flash"]
    )
    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate("Test prompt")

    assert "exceeded rate limits" in str(exc_info.value)
    assert mock_post.call_count == 3

@patch("requests.post")
def test_gemini_fallback_non_429_error_no_fallback(mock_post):
    """Verify non-429 error (e.g. 401 Unauthorized) raises error immediately without fallback."""
    res_401 = MagicMock()
    res_401.status_code = 401
    res_401.text = "Unauthorized API key"

    mock_post.return_value = res_401

    provider = GeminiProvider(
        api_key="invalid_key",
        model_name="gemini-3.1-flash-lite",
        fallback_models=["gemini-3.5-flash-lite", "gemini-3.6-flash"]
    )
    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate("Test prompt")

    assert "status 401" in str(exc_info.value)
    assert mock_post.call_count == 1  # No fallback attempted for 401!
