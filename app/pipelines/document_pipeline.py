from io import BytesIO
from typing import Dict, Any
from app.ocr.pdf_detector import PDFDetector
from app.ocr.ocr_service import OCRService
from app.services.document_service import DocumentService
from app.services.processing_service import ProcessingService
from app.services.metadata_service import MetadataService
from app.storage.document_store import DocumentStore
from app.core.config import settings
from app.core.exceptions import DocumentProcessingError, InvalidDocumentError
from app.core.logger import get_logger

logger = get_logger(__name__)

class DocumentPipeline:
    """Orchestrates PDF structure detection, parser/OCR extraction, chunking, entity extraction, and storage."""

    @staticmethod
    def execute(filename: str, pdf_file: BytesIO) -> Dict[str, Any]:
        """Execute end-to-end document ingestion pipeline."""
        # 1. Inspect PDF structure (Searchable vs Scanned)
        is_scanned, num_pages, extracted_text = PDFDetector.inspect_pdf(pdf_file)

        ocr_used = False
        full_text = ""

        # Reset buffer pointer
        if hasattr(pdf_file, "seek"):
            pdf_file.seek(0)

        # 2. Extract text via Parser or OCR Engine
        if is_scanned and settings.ENABLE_OCR:
            logger.info(f"Pipeline: Executing OCR extraction for scanned document '{filename}'")
            try:
                ocr_res = OCRService.extract(pdf_file, provider_name=settings.OCR_PROVIDER)
                full_text = ocr_res.text
                num_pages = max(num_pages, ocr_res.num_pages)
                ocr_used = True
            except InvalidDocumentError:
                raise
            except Exception as e:
                logger.warning(f"OCR extraction failed for '{filename}': {str(e)}. Falling back to standard parser.")
                if hasattr(pdf_file, "seek"):
                    pdf_file.seek(0)
                full_text, parsed_pages = DocumentService.extract_full_text(pdf_file)
                num_pages = max(num_pages, parsed_pages)
        else:
            logger.info(f"Pipeline: Extracting text via native PDF parser for '{filename}'")
            full_text, parsed_pages = DocumentService.extract_full_text(pdf_file)
            num_pages = max(num_pages, parsed_pages)

        # Re-verify extracted text
        if not full_text or not full_text.strip():
            logger.error(f"Text extraction yielded empty output for document '{filename}'")
            raise DocumentProcessingError(f"Failed to extract text from provided PDF document '{filename}'.")


        # 3. Processing & Chunking
        chunks = ProcessingService.process_text(full_text)

        # 4. Entity Metadata Extraction
        entities = MetadataService.extract_entities(full_text)

        # 5. Index & Store Document
        store = DocumentStore.get_instance()
        doc_data = store.add_document(
            filename=filename,
            full_text=full_text,
            num_pages=num_pages,
            chunks=chunks,
            entities=entities,
            is_scanned=is_scanned,
            ocr_used=ocr_used
        )

        logger.info(
            f"Pipeline: Successfully stored document '{filename}' (ID: {doc_data['document_id']}) "
            f"is_scanned={is_scanned}, ocr_used={ocr_used}"
        )
        return doc_data
