from unittest.mock import patch

def test_get_health_success(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "SpectralReader Document Intelligence API"
    assert data["version"] == "1.1.0"
    assert data["models_loaded"] is True
    assert "components" in data

def test_get_health_models_unloaded(test_client):
    with patch("app.services.model_service.ModelService.get_model_container", return_value=None):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["models_loaded"] is False
