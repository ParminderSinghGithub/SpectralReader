from unittest.mock import patch

def test_qa_answer_question_success(test_client):
    mock_text = "Arthur Vance Arthur Vance Arthur Vance signed the report."
    with patch("app.services.document_service.DocumentService.extract_full_text", return_value=(mock_text, 1)):
        files = {"file": ("report.pdf", b"%PDF-1.4 sample", "application/pdf")}
        upload_resp = test_client.post("/documents", files=files)
        doc_id = upload_resp.json()["document_id"]

        qa_payload = {
            "document_id": doc_id,
            "question": "Who signed the report?"
        }
        response = test_client.post("/qa", json=qa_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == doc_id
        assert "answer" in data
        assert "retrieved_context" in data
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] >= 0
        assert data["llm_provider"] == "gemini"

def test_qa_answer_question_not_found(test_client):
    qa_payload = {
        "document_id": "invalid-doc-id-0000",
        "question": "What is the summary?"
    }
    response = test_client.post("/qa", json=qa_payload)
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "error"
