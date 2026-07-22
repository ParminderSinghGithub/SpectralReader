from pydantic import BaseModel, Field
from typing import List, Optional

class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    service: str = Field("SpectralReader Document Intelligence API")
    version: str = Field("1.0.0")
    models_loaded: bool = Field(..., description="Whether backend ML models are initialized")

class DocumentUploadResponse(BaseModel):
    document_id: str = Field(..., description="Unique document UUID")
    filename: str = Field(..., description="Original filename")
    num_pages: int = Field(..., description="Total pages extracted")
    num_chunks: int = Field(..., description="Total chunks generated")
    entities: List[str] = Field(..., description="Extracted document entities")
    created_at: str = Field(..., description="ISO creation timestamp")

class DocumentMetadataResponse(BaseModel):
    document_id: str = Field(..., description="Unique document UUID")
    filename: str = Field(..., description="Original filename")
    num_pages: int = Field(..., description="Total pages extracted")
    num_chunks: int = Field(..., description="Total chunks generated")
    entities: List[str] = Field(..., description="Extracted document entities")
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

class DeleteDocumentResponse(BaseModel):
    document_id: str = Field(...)
    message: str = Field(...)
