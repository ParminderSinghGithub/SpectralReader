"""Validation Data Models and Dataclasses."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import statistics

def evaluate_answer_quality(answer: str) -> str:
    """Evaluate QA answer quality using lightweight heuristics (no LLM).
    
    Possible Return Values:
    - GOOD: Meaningful multi-word answer (>= 4 words with alphanumeric characters).
    - PARTIAL: Short answer (1-3 words).
    - POOR: Ultra-short, punctuation-only (e.g. '.', 'd).', '(iv)'), or empty string.
    - NOT EVALUATED: Fallback.
    """
    if not answer or not answer.strip():
        return "POOR"
    cleaned = answer.strip()
    if len(cleaned) <= 3 or cleaned in (".", "d).", "(iv)", "iv.", "(d).", "d)", "c)", "a)"):
        return "POOR"
    words = [w for w in cleaned.split() if any(c.isalnum() for c in w)]
    if len(words) >= 4:
        return "GOOD"
    elif len(words) >= 1:
        return "PARTIAL"
    return "POOR"

@dataclass
class SystemInfo:
    timestamp: str
    python_version: str
    platform: str
    os_name: str
    cpu_architecture: str
    machine_name: str
    total_ram_gb: Optional[float] = None
    available_ram_gb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "python_version": self.python_version,
            "platform": self.platform,
            "os_name": self.os_name,
            "cpu_architecture": self.cpu_architecture,
            "machine_name": self.machine_name,
            "total_ram_gb": round(self.total_ram_gb, 2) if self.total_ram_gb else None,
            "available_ram_gb": round(self.available_ram_gb, 2) if self.available_ram_gb else None
        }

@dataclass
class MetricStats:
    min_ms: float = 0.0
    max_ms: float = 0.0
    avg_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    count: int = 0

    @classmethod
    def calculate(cls, latencies: List[float]) -> 'MetricStats':
        if not latencies:
            return cls()
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        p95_idx = min(int(0.95 * n), n - 1)
        return cls(
            min_ms=round(sorted_lat[0], 2),
            max_ms=round(sorted_lat[-1], 2),
            avg_ms=round(statistics.mean(sorted_lat), 2),
            median_ms=round(statistics.median(sorted_lat), 2),
            p95_ms=round(sorted_lat[p95_idx], 2),
            count=n
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "avg_ms": self.avg_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "count": self.count
        }

@dataclass
class SearchTestResult:
    query: str
    status_code: int
    passed: bool
    passage_count: int
    latency_ms: float
    result_category: str = "PASS"  # PASS, WARNING, APPLICATION FAILURE
    warning: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "status_code": self.status_code,
            "passed": self.passed,
            "result_category": self.result_category,
            "passage_count": self.passage_count,
            "latency_ms": round(self.latency_ms, 2),
            "warning": self.warning,
            "error": self.error
        }

@dataclass
class QATestResult:
    question: str
    status_code: int
    passed: bool
    answer: str
    retrieved_context_count: int
    processing_time_ms: float
    latency_ms: float
    answer_quality: str = "GOOD"  # GOOD, PARTIAL, POOR, NOT EVALUATED
    result_category: str = "PASS"  # PASS, WARNING, APPLICATION FAILURE
    warning: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "status_code": self.status_code,
            "passed": self.passed,
            "result_category": self.result_category,
            "answer_preview": self.answer[:100] + "..." if len(self.answer) > 100 else self.answer,
            "answer_quality": self.answer_quality,
            "retrieved_context_count": self.retrieved_context_count,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "latency_ms": round(self.latency_ms, 2),
            "warning": self.warning,
            "error": self.error
        }

@dataclass
class PDFValidationResult:
    filename: str
    purpose: str
    expected_upload_success: bool
    result_category: str = "PASS"  # PASS, WARNING, EXPECTED FAILURE, APPLICATION FAILURE
    upload_passed: bool = False
    upload_status_code: int = 0
    upload_latency_ms: float = 0.0
    document_id: Optional[str] = None
    num_pages: int = 0
    num_chunks: int = 0
    entities: List[str] = field(default_factory=list)
    metadata_passed: bool = False
    metadata_status_code: int = 0
    metadata_latency_ms: float = 0.0
    search_passed: bool = False
    search_results: List[SearchTestResult] = field(default_factory=list)
    search_avg_latency_ms: float = 0.0
    qa_passed: bool = False
    qa_results: List[QATestResult] = field(default_factory=list)
    qa_avg_latency_ms: float = 0.0
    delete_passed: bool = False
    delete_status_code: int = 0
    delete_latency_ms: float = 0.0
    verify_delete_passed: bool = False
    is_scanned: bool = False
    ocr_used: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    overall_passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "purpose": self.purpose,
            "expected_upload_success": self.expected_upload_success,
            "result_category": self.result_category,
            "upload_passed": self.upload_passed,
            "upload_status_code": self.upload_status_code,
            "upload_latency_ms": round(self.upload_latency_ms, 2),
            "document_id": self.document_id,
            "num_pages": self.num_pages,
            "num_chunks": self.num_chunks,
            "entities_count": len(self.entities),
            "metadata_passed": self.metadata_passed,
            "metadata_status_code": self.metadata_status_code,
            "metadata_latency_ms": round(self.metadata_latency_ms, 2),
            "search_passed": self.search_passed,
            "search_avg_latency_ms": round(self.search_avg_latency_ms, 2),
            "search_test_count": len(self.search_results),
            "qa_passed": self.qa_passed,
            "qa_avg_latency_ms": round(self.qa_avg_latency_ms, 2),
            "qa_test_count": len(self.qa_results),
            "delete_passed": self.delete_passed,
            "verify_delete_passed": self.verify_delete_passed,
            "is_scanned": self.is_scanned,
            "ocr_used": self.ocr_used,
            "warnings": self.warnings,
            "errors": self.errors,
            "overall_passed": self.overall_passed,
            "search_details": [r.to_dict() for r in self.search_results],
            "qa_details": [r.to_dict() for r in self.qa_results]
        }

@dataclass
class ErrorTestResult:
    test_name: str
    endpoint: str
    method: str
    expected_status: int
    actual_status: int
    passed: bool
    latency_ms: float
    result_category: str = "PASS"  # PASS, EXPECTED FAILURE, APPLICATION FAILURE
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "endpoint": self.endpoint,
            "method": self.method,
            "expected_status": self.expected_status,
            "actual_status": self.actual_status,
            "passed": self.passed,
            "result_category": self.result_category,
            "latency_ms": round(self.latency_ms, 2),
            "detail": self.detail
        }

@dataclass
class ValidationReport:
    timestamp: str
    backend_url: str
    system_info: SystemInfo
    total_pdfs: int
    passed_pdfs: int
    failed_pdfs: int
    pdf_results: List[PDFValidationResult] = field(default_factory=list)
    error_test_results: List[ErrorTestResult] = field(default_factory=list)
    global_warnings: List[str] = field(default_factory=list)
    global_errors: List[str] = field(default_factory=list)
    endpoint_latencies: Dict[str, MetricStats] = field(default_factory=dict)
    readiness_checklist: Dict[str, bool] = field(default_factory=dict)
    overall_verdict: str = "NOT READY"
    total_api_calls: int = 0
    total_runtime_seconds: float = 0.0
    average_upload_latency_ms: float = 0.0
    average_search_latency_ms: float = 0.0
    average_qa_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "backend_url": self.backend_url,
            "system_info": self.system_info.to_dict(),
            "performance_summary": {
                "total_runtime_seconds": round(self.total_runtime_seconds, 2),
                "average_upload_latency_ms": round(self.average_upload_latency_ms, 2),
                "average_search_latency_ms": round(self.average_search_latency_ms, 2),
                "average_qa_latency_ms": round(self.average_qa_latency_ms, 2),
                "total_api_calls": self.total_api_calls
            },
            "summary": {
                "total_pdfs": self.total_pdfs,
                "passed_pdfs": self.passed_pdfs,
                "failed_pdfs": self.failed_pdfs,
                "total_api_calls": self.total_api_calls,
                "overall_verdict": self.overall_verdict
            },
            "readiness_checklist": {
                name: "PASS" if passed else "FAIL"
                for name, passed in self.readiness_checklist.items()
            },
            "endpoint_performance_ms": {
                endpoint: stats.to_dict()
                for endpoint, stats in self.endpoint_latencies.items()
            },
            "pdf_results": [pdf.to_dict() for pdf in self.pdf_results],
            "error_test_results": [e.to_dict() for e in self.error_test_results],
            "warnings": self.global_warnings,
            "errors": self.global_errors
        }
