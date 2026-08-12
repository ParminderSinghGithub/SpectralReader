"""Configuration Loader for Validation Cases and Global Settings."""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class WarningThresholds:
    upload_latency_ms: float = 30000.0
    qa_latency_ms: float = 15000.0
    warn_on_empty_entities: bool = True
    warn_on_empty_passages: bool = True

@dataclass
class Settings:
    backend_url: str = "http://localhost:8000"
    timeout_seconds: float = 60.0
    upload_timeout_seconds: float = 600.0
    raw_archive_enabled: bool = True
    output_dir: str = "reports"
    warning_thresholds: WarningThresholds = field(default_factory=WarningThresholds)

@dataclass
class PDFCase:
    filename: str
    purpose: str
    expected_upload_success: bool = True
    expected_ocr_used: Optional[bool] = None
    search_queries: List[str] = field(default_factory=list)
    qa_questions: List[str] = field(default_factory=list)

@dataclass
class ValidationConfig:
    settings: Settings
    pdf_cases: List[PDFCase]

def load_config(config_path: str) -> ValidationConfig:
    """Load configuration YAML and return typed ValidationConfig object."""
    if not os.path.isabs(config_path):
        # Resolve relative to validation directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.abspath(os.path.join(base_dir, config_path))

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Validation configuration file not found at: '{config_path}'")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_settings = data.get("settings", {})
    raw_thresh = raw_settings.get("warning_thresholds", {})
    
    thresholds = WarningThresholds(
        upload_latency_ms=float(raw_thresh.get("upload_latency_ms", 30000.0)),
        qa_latency_ms=float(raw_thresh.get("qa_latency_ms", 15000.0)),
        warn_on_empty_entities=bool(raw_thresh.get("warn_on_empty_entities", True)),
        warn_on_empty_passages=bool(raw_thresh.get("warn_on_empty_passages", True))
    )

    settings = Settings(
        backend_url=str(raw_settings.get("backend_url", "http://localhost:8000")).rstrip("/"),
        timeout_seconds=float(raw_settings.get("timeout_seconds", 60.0)),
        upload_timeout_seconds=float(raw_settings.get("upload_timeout_seconds", 600.0)),
        raw_archive_enabled=bool(raw_settings.get("raw_archive_enabled", True)),
        output_dir=str(raw_settings.get("output_dir", "reports")),
        warning_thresholds=thresholds
    )

    pdf_cases = []
    for item in data.get("pdf_cases", []):
        pdf_cases.append(
            PDFCase(
                filename=item.get("filename", ""),
                purpose=item.get("purpose", "Document Test"),
                expected_upload_success=item.get("expected_upload_success", True),
                expected_ocr_used=item.get("expected_ocr_used", None),
                search_queries=item.get("search_queries", []),
                qa_questions=item.get("qa_questions", [])
            )
        )

    return ValidationConfig(settings=settings, pdf_cases=pdf_cases)
