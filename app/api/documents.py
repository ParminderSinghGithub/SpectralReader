from fastapi import APIRouter, File, UploadFile, HTTPException, status
from io import BytesIO
from app.models.schemas import DocumentUploadResponse, DocumentMetadataResponse, DeleteDocumentResponse
from app.services.document_service import DocumentService
from app.services.processing_service import ProcessingService
from app.services.metadata_service import MetadataService
from app.storage.document_store import DocumentStore

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """Accept a PDF upload, process text/chunks/metadata, and store in document store."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are supported."
        )

    content = await file.read()
    pdf_bytes = BytesIO(content)

    full_text, num_pages = DocumentService.extract_full_text(pdf_bytes)
    if not full_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to extract text from provided PDF document."
        )

    chunks = ProcessingService.process_text(full_text)
    characters = MetadataService.extract_character_info(full_text)

    store = DocumentStore.get_instance()
    doc_data = store.add_document(
        filename=file.filename,
        full_text=full_text,
        num_pages=num_pages,
        chunks=chunks,
        characters=characters
    )

    return DocumentUploadResponse(
        document_id=doc_data["document_id"],
        filename=doc_data["filename"],
        num_pages=doc_data["num_pages"],
        num_chunks=doc_data["num_chunks"],
        characters_identified=doc_data["characters"],
        created_at=doc_data["created_at"]
    )

@router.get("/{document_id}", response_model=DocumentMetadataResponse)
def get_document_metadata(document_id: str):
    """Retrieve metadata for a processed document by ID."""
    store = DocumentStore.get_instance()
    doc = store.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found."
        )
    return DocumentMetadataResponse(
        document_id=doc["document_id"],
        filename=doc["filename"],
        num_pages=doc["num_pages"],
        num_chunks=doc["num_chunks"],
        characters_identified=doc["characters"],
        created_at=doc["created_at"]
    )

@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(document_id: str):
    """Remove document from the in-memory storage."""
    store = DocumentStore.get_instance()
    deleted = store.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found."
        )
    return DeleteDocumentResponse(
        document_id=document_id,
        message=f"Document '{document_id}' successfully removed from memory storage."
    )
