from typing import Optional

class SpectralReaderException(Exception):
    """Base exception class for SpectralReader backend application."""
    def __init__(self, message: str, status_code: int = 500, detail: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail or message

class DocumentNotFoundError(SpectralReaderException):
    """Raised when a requested document ID is not present in storage."""
    def __init__(self, document_id: str):
        super().__init__(
            message=f"Document '{document_id}' not found.",
            status_code=404
        )

class InvalidDocumentError(SpectralReaderException):
    """Raised when a document payload or format is invalid."""
    def __init__(self, detail: str):
        super().__init__(
            message=detail,
            status_code=400
        )

class DocumentProcessingError(SpectralReaderException):
    """Raised when document extraction, cleaning, or chunking fails."""
    def __init__(self, detail: str):
        super().__init__(
            message=f"Document processing failed: {detail}",
            status_code=422
        )

class ModelInitializationError(SpectralReaderException):
    """Raised when backend ML model initialization fails."""
    def __init__(self, detail: str):
        super().__init__(
            message=f"Model initialization failed: {detail}",
            status_code=500
        )

class LLMProviderError(SpectralReaderException):
    """Raised when an LLM provider generation request fails."""
    def __init__(self, provider_name: str, detail: str):
        super().__init__(
            message=f"LLM Provider '{provider_name}' error: {detail}",
            status_code=502
        )

class OCRError(SpectralReaderException):
    """Raised when OCR extraction fails."""
    def __init__(self, detail: str):
        super().__init__(
            message=f"OCR extraction failed: {detail}",
            status_code=422
        )

class PipelineError(SpectralReaderException):
    """Raised when pipeline orchestration fails."""
    def __init__(self, detail: str):
        super().__init__(
            message=f"Pipeline error: {detail}",
            status_code=500
        )
