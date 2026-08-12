from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    service: str = Field("SpectralReader Document Intelligence API")
    version: str = Field("1.1.0")
    models_loaded: bool = Field(..., description="Whether backend ML vector models are initialized")
    components: Optional[Dict[str, Any]] = Field(None, description="Detailed backend component health matrix")

class DocumentUploadResponse(BaseModel):
    document_id: str = Field(..., description="Unique document UUID")
    filename: str = Field(..., description="Original filename")
    num_pages: int = Field(..., description="Total pages extracted")
    num_chunks: int = Field(..., description="Total chunks generated")
    entities: List[str] = Field(..., description="Extracted document entities")
    is_scanned: bool = Field(False, description="Whether document was classified as a scanned image PDF")
    ocr_used: bool = Field(False, description="Whether OCR pipeline was executed for text extraction")
    created_at: str = Field(..., description="ISO creation timestamp")

class DocumentMetadataResponse(BaseModel):
    document_id: str = Field(..., description="Unique document UUID")
    filename: str = Field(..., description="Original filename")
    num_pages: int = Field(..., description="Total pages extracted")
    num_chunks: int = Field(..., description="Total chunks generated")
    entities: List[str] = Field(..., description="Extracted document entities")
    is_scanned: bool = Field(False, description="Whether document was classified as a scanned image PDF")
    ocr_used: bool = Field(False, description="Whether OCR pipeline was executed for text extraction")
    created_at: str = Field(..., description="ISO creation timestamp")

class SearchRequest(BaseModel):
    document_id: str = Field(..., description="ID of document to search within")
    query: str = Field(..., description="Search query string")
    top_k: Optional[int] = Field(3, description="Maximum number of candidate chunks to retrieve")

class SearchResponse(BaseModel):
    document_id: str = Field(...)
    query: str = Field(...)
    results: List[str] = Field(..., description="Matching document passages")

class QARequest(BaseModel):
    document_id: str = Field(..., description="ID of target document")
    question: str = Field(..., description="Question prompt")

class QAResponse(BaseModel):
    document_id: str = Field(...)
    question: str = Field(...)
    answer: str = Field(..., description="Generated answer text")
    retrieved_context: List[str] = Field(..., description="Passages passed to QA model")
    processing_time_ms: float = Field(..., description="Total QA processing duration in milliseconds")
    llm_provider: Optional[str] = Field(None, description="Active LLM provider used")
    model_name: Optional[str] = Field(None, description="Active model name")
    token_usage: Optional[Dict[str, Any]] = Field(None, description="Token usage details")
    stage_latencies: Optional[Dict[str, Any]] = Field(None, description="Detailed stage latency measurements")

class DeleteDocumentResponse(BaseModel):
    document_id: str = Field(...)
    message: str = Field(...)

class ErrorResponse(BaseModel):
    status: str = Field("error", example="error")
    message: str = Field(..., description="High-level error summary")
    detail: Optional[str] = Field(None, description="Detailed error description")
    status_code: int = Field(..., description="HTTP status code")
