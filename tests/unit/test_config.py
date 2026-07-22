import os
from app.core.config import Settings

def test_default_config_settings():
    settings = Settings()
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8000
    assert settings.LOG_LEVEL == "INFO"
    assert settings.CHUNK_SIZE == 1500
    assert settings.CHUNK_OVERLAP == 300
    assert settings.MAX_GEN_LENGTH == 512

def test_env_override_config_settings(monkeypatch):
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "9090")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000,http://app.domain.com")

    settings = Settings()
    assert settings.HOST == "127.0.0.1"
    assert settings.PORT == 9090
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.CORS_ORIGINS == ["http://localhost:3000", "http://app.domain.com"]
