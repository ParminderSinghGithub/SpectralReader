from fastapi import APIRouter, File, UploadFile, status
from io import BytesIO
from app.models.schemas import DocumentUploadResponse, DocumentMetadataResponse, DeleteDocumentResponse
from app.pipelines.document_pipeline import DocumentPipeline
from app.storage.document_store import DocumentStore
from app.core.exceptions import DocumentNotFoundError, InvalidDocumentError
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """Accept a PDF document upload, execute DocumentPipeline, and return metadata."""
    if not file.filename.lower().endswith(".pdf"):
        logger.warning(f"Rejected non-PDF upload attempt: '{file.filename}'")
        raise InvalidDocumentError("Only PDF documents are supported.")

    content = await file.read()
    if not content or len(content.strip()) == 0:
        logger.warning(f"Rejected empty PDF file: '{file.filename}'")
        raise InvalidDocumentError("Uploaded PDF document is empty.")

    pdf_bytes = BytesIO(content)

    doc_data = DocumentPipeline.execute(filename=file.filename, pdf_file=pdf_bytes)

    return DocumentUploadResponse(
        document_id=doc_data["document_id"],
        filename=doc_data["filename"],
        num_pages=doc_data["num_pages"],
        num_chunks=doc_data["num_chunks"],
        entities=doc_data["entities"],
        is_scanned=doc_data.get("is_scanned", False),
        ocr_used=doc_data.get("ocr_used", False),
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
        is_scanned=doc.get("is_scanned", False),
        ocr_used=doc.get("ocr_used", False),
        created_at=doc["created_at"]
    )

@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(document_id: str):
    """Remove document from memory storage."""
    store = DocumentStore.get_instance()
    deleted = store.delete_document(document_id)
    if not deleted:
        logger.warning(f"Attempted to delete non-existent document ID '{document_id}'")
        raise DocumentNotFoundError(document_id)
    return DeleteDocumentResponse(
        document_id=document_id,
        message=f"Document '{document_id}' successfully removed from memory storage."
    )
