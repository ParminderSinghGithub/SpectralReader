from unittest.mock import patch

def test_search_passages_success(test_client):
    mock_text = "Arthur Vance Arthur Vance Arthur Vance signed the report."
    with patch("app.services.document_service.DocumentService.extract_full_text", return_value=(mock_text, 1)):
        files = {"file": ("report.pdf", b"%PDF-1.4 sample", "application/pdf")}
        upload_resp = test_client.post("/documents", files=files)
        doc_id = upload_resp.json()["document_id"]

        search_payload = {
            "document_id": doc_id,
            "query": "Who signed the report?",
            "top_k": 2
        }
        response = test_client.post("/search", json=search_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == doc_id
        assert len(data["results"]) > 0

def test_search_passages_not_found(test_client):
    search_payload = {
        "document_id": "invalid-doc-id-9999",
        "query": "Test query"
    }
    response = test_client.post("/search", json=search_payload)
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "error"
