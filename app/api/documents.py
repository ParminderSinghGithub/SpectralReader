from fastapi import APIRouter, File, UploadFile, status
from io import BytesIO
from app.models.schemas import DocumentUploadResponse, DocumentMetadataResponse, DeleteDocumentResponse
from app.services.document_service import DocumentService
from app.services.processing_service import ProcessingService
from app.services.metadata_service import MetadataService
from app.storage.document_store import DocumentStore
from app.core.exceptions import DocumentNotFoundError, InvalidDocumentError, DocumentProcessingError
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """Accept a PDF document upload, extract text/chunks/metadata, and store in document store."""
    if not file.filename.lower().endswith(".pdf"):
        logger.warning(f"Rejected non-PDF upload attempt: '{file.filename}'")
        raise InvalidDocumentError("Only PDF documents are supported.")

    content = await file.read()
    pdf_bytes = BytesIO(content)

    full_text, num_pages = DocumentService.extract_full_text(pdf_bytes)
    if not full_text.strip():
        logger.error(f"Text extraction yielded empty output for file '{file.filename}'")
        raise DocumentProcessingError("Failed to extract text from provided PDF document.")

    chunks = ProcessingService.process_text(full_text)
    entities = MetadataService.extract_entities(full_text)

    store = DocumentStore.get_instance()
    doc_data = store.add_document(
        filename=file.filename,
        full_text=full_text,
        num_pages=num_pages,
        chunks=chunks,
        entities=entities
    )

    logger.info(f"Successfully processed document '{file.filename}' (ID: {doc_data['document_id']}) with {num_pages} pages and {len(chunks)} chunks.")

    return DocumentUploadResponse(
        document_id=doc_data["document_id"],
        filename=doc_data["filename"],
        num_pages=doc_data["num_pages"],
        num_chunks=doc_data["num_chunks"],
        entities=doc_data["entities"],
        created_at=doc_data["created_at"]
    )

@router.get("/{document_id}", response_model=DocumentMetadataResponse)
def get_document_metadata(document_id: str):
    """Retrieve metadata for a processed document by ID."""
    store = DocumentStore.get_instance()
    doc = store.get_document(document_id)
    if not doc:
        logger.warning(f"Requested metadata for non-existent document ID '{document_id}'")
        raise DocumentNotFoundError(document_id)
    return DocumentMetadataResponse(
        document_id=doc["document_id"],
        filename=doc["filename"],
        num_pages=doc["num_pages"],
        num_chunks=doc["num_chunks"],
        entities=doc.get("entities", doc.get("characters", [])),
        created_at=doc["created_at"]
    )

@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(document_id: str):
    """Remove document from the in-memory storage."""
    store = DocumentStore.get_instance()
    deleted = store.delete_document(document_id)
    if not deleted:
        logger.warning(f"Attempted to delete non-existent document ID '{document_id}'")
        raise DocumentNotFoundError(document_id)
    return DeleteDocumentResponse(
        document_id=document_id,
        message=f"Document '{document_id}' successfully removed from memory storage."
    )
