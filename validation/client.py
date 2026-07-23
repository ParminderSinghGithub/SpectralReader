"""REST API Client with Raw Request/Response Archiving."""

import os
import json
import time
import requests
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, Optional
from config_loader import Settings

class APIClient:
    """HTTP REST API Client for SpectralReader Backend."""
    def __init__(self, settings: Settings, raw_dir: str):
        self.settings = settings
        self.base_url = settings.backend_url.rstrip("/")
        self.timeout = settings.timeout_seconds
        self.raw_dir = raw_dir
        self.session = requests.Session()
        self.request_counter = 0

        if self.settings.raw_archive_enabled and self.raw_dir:
            os.makedirs(self.raw_dir, exist_ok=True)

    def _archive_request(
        self,
        action_name: str,
        method: str,
        endpoint: str,
        request_payload: Any,
        filename_label: Optional[str],
        status_code: int,
        response_body: Any,
        latency_ms: float
    ):
        """Save a raw request/response snapshot into raw_dir as JSON."""
        if not self.settings.raw_archive_enabled or not self.raw_dir:
            return

        self.request_counter += 1
        safe_label = (filename_label or action_name).replace(".pdf", "").replace(" ", "_").lower()
        safe_action = action_name.replace(" ", "_").lower()
        raw_filename = f"{self.request_counter:03d}_{safe_action}_{safe_label}.json"
        raw_filepath = os.path.join(self.raw_dir, raw_filename)

        record = {
            "request_number": self.request_counter,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "url": f"{self.base_url}{endpoint}",
            "action": action_name,
            "filename_associated": filename_label,
            "request_payload": request_payload,
            "response_status_code": status_code,
            "response_body": response_body,
            "latency_ms": round(latency_ms, 2)
        }

        try:
            with open(raw_filepath, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, default=str)
        except Exception as e:
            # Fallback error writing archive
            pass

    def health_check(self) -> Tuple[int, Dict[str, Any], float]:
        """GET /health"""
        endpoint = "/health"
        url = f"{self.base_url}{endpoint}"
        start = time.perf_counter()
        try:
            res = self.session.get(url, timeout=self.timeout)
            latency = (time.perf_counter() - start) * 1000.0
            try:
                body = res.json()
            except Exception:
                body = {"raw": res.text}
            self._archive_request("health", "GET", endpoint, None, "health", res.status_code, body, latency)
            return res.status_code, body, latency
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000.0
            body = {"error": str(e)}
            self._archive_request("health", "GET", endpoint, None, "health", 0, body, latency)
            return 0, body, latency

    def upload_document(self, file_path: str) -> Tuple[int, Dict[str, Any], float]:
        """POST /documents"""
        endpoint = "/documents"
        url = f"{self.base_url}{endpoint}"
        filename = os.path.basename(file_path)

        if not os.path.exists(file_path):
            return 0, {"error": f"Local file not found: '{file_path}'"}, 0.0

        start = time.perf_counter()
        try:
            with open(file_path, "rb") as f:
                files = {"file": (filename, f, "application/pdf")}
                res = self.session.post(url, files=files, timeout=self.timeout)
            latency = (time.perf_counter() - start) * 1000.0
            try:
                body = res.json()
            except Exception:
                body = {"raw": res.text}
            self._archive_request("upload", "POST", endpoint, {"filename": filename}, filename, res.status_code, body, latency)
            return res.status_code, body, latency
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000.0
            body = {"error": str(e)}
            self._archive_request("upload", "POST", endpoint, {"filename": filename}, filename, 0, body, latency)
            return 0, body, latency

    def get_metadata(self, document_id: str, label: Optional[str] = None) -> Tuple[int, Dict[str, Any], float]:
        """GET /documents/{id}"""
        endpoint = f"/documents/{document_id}"
        url = f"{self.base_url}{endpoint}"
        start = time.perf_counter()
        try:
            res = self.session.get(url, timeout=self.timeout)
            latency = (time.perf_counter() - start) * 1000.0
            try:
                body = res.json()
            except Exception:
                body = {"raw": res.text}
            self._archive_request("metadata", "GET", endpoint, None, label or document_id, res.status_code, body, latency)
            return res.status_code, body, latency
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000.0
            body = {"error": str(e)}
            self._archive_request("metadata", "GET", endpoint, None, label or document_id, 0, body, latency)
            return 0, body, latency

    def search(self, document_id: str, query: str, top_k: int = 3, label: Optional[str] = None) -> Tuple[int, Dict[str, Any], float]:
        """POST /search"""
        endpoint = "/search"
        url = f"{self.base_url}{endpoint}"
        payload = {"document_id": document_id, "query": query, "top_k": top_k}
        start = time.perf_counter()
        try:
            res = self.session.post(url, json=payload, timeout=self.timeout)
            latency = (time.perf_counter() - start) * 1000.0
            try:
                body = res.json()
            except Exception:
                body = {"raw": res.text}
            query_label = f"{label or document_id}_{query[:15]}"
            self._archive_request("search", "POST", endpoint, payload, query_label, res.status_code, body, latency)
            return res.status_code, body, latency
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000.0
            body = {"error": str(e)}
            self._archive_request("search", "POST", endpoint, payload, label or document_id, 0, body, latency)
            return 0, body, latency

    def qa(self, document_id: str, question: str, label: Optional[str] = None) -> Tuple[int, Dict[str, Any], float]:
        """POST /qa"""
        endpoint = "/qa"
        url = f"{self.base_url}{endpoint}"
        payload = {"document_id": document_id, "question": question}
        start = time.perf_counter()
        try:
            res = self.session.post(url, json=payload, timeout=self.timeout)
            latency = (time.perf_counter() - start) * 1000.0
            try:
                body = res.json()
            except Exception:
                body = {"raw": res.text}
            q_label = f"{label or document_id}_{question[:15]}"
            self._archive_request("qa", "POST", endpoint, payload, q_label, res.status_code, body, latency)
            return res.status_code, body, latency
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000.0
            body = {"error": str(e)}
            self._archive_request("qa", "POST", endpoint, payload, label or document_id, 0, body, latency)
            return 0, body, latency

    def delete_document(self, document_id: str, label: Optional[str] = None) -> Tuple[int, Dict[str, Any], float]:
        """DELETE /documents/{id}"""
        endpoint = f"/documents/{document_id}"
        url = f"{self.base_url}{endpoint}"
        start = time.perf_counter()
        try:
            res = self.session.delete(url, timeout=self.timeout)
            latency = (time.perf_counter() - start) * 1000.0
            try:
                body = res.json()
            except Exception:
                body = {"raw": res.text}
            self._archive_request("delete", "DELETE", endpoint, None, label or document_id, res.status_code, body, latency)
            return res.status_code, body, latency
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000.0
            body = {"error": str(e)}
            self._archive_request("delete", "DELETE", endpoint, None, label or document_id, 0, body, latency)
            return 0, body, latency
