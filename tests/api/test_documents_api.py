from unittest.mock import patch

def test_upload_document_success(test_client):
    mock_text = "Arthur Vance. Arthur Vance. Arthur Vance. Arthur Vance signed the contract."
    with patch("app.services.document_service.DocumentService.extract_full_text", return_value=(mock_text, 3)):
        files = {"file": ("contract.pdf", b"%PDF-1.4 sample content", "application/pdf")}
        response = test_client.post("/documents", files=files)
        assert response.status_code == 201
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == "contract.pdf"
        assert data["num_pages"] == 3
        assert data["num_chunks"] > 0
        assert "Arthur Vance" in data["entities"]

def test_upload_document_invalid_extension(test_client):
    files = {"file": ("script.py", b"print('hello')", "text/plain")}
    response = test_client.post("/documents", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "Only PDF documents are supported" in data["message"]

def test_upload_document_empty_extraction(test_client):
    with patch("app.services.document_service.DocumentService.extract_full_text", return_value=("   ", 0)):
        files = {"file": ("empty.pdf", b"%PDF-1.4 empty", "application/pdf")}
        response = test_client.post("/documents", files=files)
        assert response.status_code == 422
        data = response.json()
        assert data["status"] == "error"
        assert "Failed to extract text" in data["message"]

def test_get_document_metadata(test_client):
    mock_text = "Arthur Vance. Arthur Vance. Arthur Vance. Arthur Vance signed the agreement."
    with patch("app.services.document_service.DocumentService.extract_full_text", return_value=(mock_text, 2)):
        files = {"file": ("agreement.pdf", b"%PDF-1.4 sample", "application/pdf")}
        upload_resp = test_client.post("/documents", files=files)
        doc_id = upload_resp.json()["document_id"]

        response = test_client.get(f"/documents/{doc_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == doc_id
        assert data["filename"] == "agreement.pdf"

def test_get_document_metadata_not_found(test_client):
    response = test_client.get("/documents/non-existent-uuid-12345")
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "error"
    assert "not found" in data["message"]

def test_delete_document_success_and_not_found(test_client):
    mock_text = "Arthur Vance. Arthur Vance. Arthur Vance. Arthur Vance signed the document."
    with patch("app.services.document_service.DocumentService.extract_full_text", return_value=(mock_text, 1)):
        files = {"file": ("doc.pdf", b"%PDF-1.4 sample", "application/pdf")}
        upload_resp = test_client.post("/documents", files=files)
        doc_id = upload_resp.json()["document_id"]

        del_resp = test_client.delete(f"/documents/{doc_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["document_id"] == doc_id

        # Repeated delete returns 404
        repeat_del = test_client.delete(f"/documents/{doc_id}")
        assert repeat_del.status_code == 404
