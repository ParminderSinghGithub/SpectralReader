from app.storage.document_store import DocumentStore

def test_document_store_crud_lifecycle(reset_document_store):
    store = reset_document_store
    assert len(store.list_documents()) == 0

    doc_data = store.add_document(
        filename="test_report.pdf",
        full_text="This is a test document text.",
        num_pages=5,
        chunks=["This is a test", "document text."],
        entities=["Arthur Vance"]
    )

    doc_id = doc_data["document_id"]
    assert doc_id is not None
    assert doc_data["filename"] == "test_report.pdf"
    assert doc_data["num_pages"] == 5
    assert doc_data["num_chunks"] == 2
    assert doc_data["entities"] == ["Arthur Vance"]

    retrieved = store.get_document(doc_id)
    assert retrieved is not None
    assert retrieved["document_id"] == doc_id

    assert len(store.list_documents()) == 1

    deleted = store.delete_document(doc_id)
    assert deleted is True
    assert store.get_document(doc_id) is None
    assert len(store.list_documents()) == 0

def test_document_store_non_existent_id(reset_document_store):
    store = reset_document_store
    assert store.get_document("non-existent-uuid") is None
    assert store.delete_document("non-existent-uuid") is False
