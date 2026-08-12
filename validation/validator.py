"""Core Validation Engine and Execution Orchestrator."""

import os
import sys
import uuid
import time
import signal
import platform
import statistics
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from rich.console import Console

from config_loader import ValidationConfig, PDFCase
from client import APIClient
from models import (
    SystemInfo, MetricStats, SearchTestResult, QATestResult,
    PDFValidationResult, ErrorTestResult, ValidationReport,
    evaluate_answer_quality
)
from reporter import ReportGenerator

console = Console(force_terminal=True)

def get_system_info() -> SystemInfo:
    """Collect platform hardware, OS, Python version, and memory information."""
    now_str = datetime.now(timezone.utc).isoformat()
    python_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    platform_name = platform.platform()
    os_name = os.name
    cpu_arch = platform.machine() or platform.processor()
    machine_name = platform.node()

    total_ram_gb = None
    avail_ram_gb = None
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024 ** 3)
        avail_ram_gb = mem.available / (1024 ** 3)
    except Exception:
        pass

    return SystemInfo(
        timestamp=now_str,
        python_version=python_ver,
        platform=platform_name,
        os_name=os_name,
        cpu_architecture=cpu_arch,
        machine_name=machine_name,
        total_ram_gb=total_ram_gb,
        available_ram_gb=avail_ram_gb
    )

class Validator:
    def __init__(self, config: ValidationConfig, pdfs_dir: str, reports_dir: str):
        self.config = config
        self.settings = config.settings
        self.pdf_cases = config.pdf_cases
        self.pdfs_dir = pdfs_dir
        self.reports_dir = reports_dir
        self.raw_dir = os.path.join(self.reports_dir, "raw")
        self.client = APIClient(self.settings, self.raw_dir)
        self.system_info = get_system_info()
        self.start_time = time.perf_counter()

        # Results tracking
        self.pdf_results: List[PDFValidationResult] = []
        self.error_results: List[ErrorTestResult] = []
        self.global_warnings: List[str] = []
        self.global_errors: List[str] = []
        
        # Endpoint latency accumulators
        self.latencies: Dict[str, List[float]] = {
            "GET /health": [],
            "POST /documents": [],
            "GET /documents/{id}": [],
            "POST /search": [],
            "POST /qa": [],
            "DELETE /documents/{id}": []
        }

        # Interrupt safety
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """Signal handler to ensure report generation on Ctrl+C."""
        console.print("\n[bold red]⚠️ Validation interrupted by user (Ctrl+C). Generating partial report...[/bold red]\n")
        self.finalize_and_report()
        sys.exit(1)

    def run_health_check(self) -> bool:
        """Execute pre-flight GET /health probe."""
        console.print(f"[bold cyan]🔍 Pre-flight Health Check on backend:[/bold cyan] [underline]{self.settings.backend_url}[/underline]")
        status, body, latency = self.client.health_check()
        self.latencies["GET /health"].append(latency)

        if status == 200 and body.get("status") == "ok" and body.get("models_loaded") is True:
            console.print(f"[bold green]✅ Backend reachable and ML models initialized ({latency:.1f} ms)[/bold green]\n")
            return True
        else:
            console.print(f"[bold red]❌ Pre-flight Health Check Failed![/bold red]")
            console.print(f"Status Code: {status}, Response Body: {body}")
            console.print("[bold yellow]Please start the backend server using `uvicorn app.main_api:app --reload` before running validation.[/bold yellow]\n")
            return False

    def validate_pdf_case(self, case: PDFCase) -> PDFValidationResult:
        """Run 6-step validation flow for a single PDF benchmark case."""
        result = PDFValidationResult(
            filename=case.filename,
            purpose=case.purpose,
            expected_upload_success=case.expected_upload_success
        )

        pdf_path = os.path.join(self.pdfs_dir, case.filename)

        # Expected Graceful Failure Case (e.g. empty.pdf)
        if not case.expected_upload_success:
            status, body, lat = self.client.upload_document(pdf_path)
            self.latencies["POST /documents"].append(lat)
            result.upload_status_code = status
            result.upload_latency_ms = lat

            if status in (400, 422):
                result.upload_passed = True
                result.result_category = "EXPECTED FAILURE"
                result.overall_passed = True
                console.print(f"  [green]✓ {case.filename} upload failed gracefully as expected ({status})[/green]")
            else:
                result.upload_passed = False
                result.result_category = "APPLICATION FAILURE"
                result.overall_passed = False
                result.errors.append(f"Empty PDF upload expected graceful error (400/422), got status {status}")
                console.print(f"  [red]✗ {case.filename} upload expected error, got status {status}[/red]")
            return result

        # Step 1: Upload
        if not os.path.exists(pdf_path):
            result.errors.append(f"PDF file missing from pdfs directory: '{pdf_path}'")
            result.result_category = "APPLICATION FAILURE"
            result.overall_passed = False
            return result

        status, body, lat = self.client.upload_document(pdf_path)
        self.latencies["POST /documents"].append(lat)
        result.upload_status_code = status
        result.upload_latency_ms = lat

        if status == 201 and "document_id" in body:
            result.upload_passed = True
            result.document_id = body["document_id"]
            result.num_pages = body.get("num_pages", 0)
            result.num_chunks = body.get("num_chunks", 0)
            result.entities = body.get("entities", body.get("characters_identified", []))
            result.is_scanned = body.get("is_scanned", False)
            result.ocr_used = body.get("ocr_used", False)

            if case.expected_ocr_used is not None:
                if result.ocr_used != case.expected_ocr_used:
                    warn_msg = f"OCR invocation mismatch for {case.filename}: expected_ocr_used={case.expected_ocr_used}, actual ocr_used={result.ocr_used}"
                    result.warnings.append(warn_msg)
                    console.print(f"  [yellow]⚠️ {case.filename}: {warn_msg}[/yellow]")
                else:
                    ocr_status_str = "OCR Engaged (Scanned PDF)" if result.ocr_used else "Standard Parser (Searchable PDF)"
                    console.print(f"  [green]✓ {case.filename}: Detected as {ocr_status_str}[/green]")

            if lat > self.settings.warning_thresholds.upload_latency_ms:
                warn_msg = f"Upload latency ({lat:.0f} ms) exceeded threshold ({self.settings.warning_thresholds.upload_latency_ms:.0f} ms)"
                result.warnings.append(warn_msg)

            if not result.entities and self.settings.warning_thresholds.warn_on_empty_entities:
                result.warnings.append("No entities identified during upload metadata extraction")

        else:
            result.upload_passed = False
            result.result_category = "APPLICATION FAILURE"
            result.errors.append(f"Upload failed with status {status}: {body.get('detail', body)}")
            result.overall_passed = False
            return result

        # Step 2: Metadata GET
        doc_id = result.document_id
        status, body, lat = self.client.get_metadata(doc_id, case.filename)
        self.latencies["GET /documents/{id}"].append(lat)
        result.metadata_status_code = status
        result.metadata_latency_ms = lat

        if status == 200 and body.get("document_id") == doc_id and body.get("filename") == case.filename:
            result.metadata_passed = True
        else:
            result.metadata_passed = False
            result.errors.append(f"Metadata verification failed ({status}): expected document_id '{doc_id}'")

        # Step 3: Search Queries Execution
        search_successes = 0
        search_latencies = []
        for query in case.search_queries:
            s_status, s_body, s_lat = self.client.search(doc_id, query, top_k=3, label=case.filename)
            self.latencies["POST /search"].append(s_lat)
            search_latencies.append(s_lat)

            s_results = s_body.get("results", [])
            s_passed = (s_status == 200 and isinstance(s_results, list))
            s_category = "PASS" if s_passed else "APPLICATION FAILURE"
            s_warn = None
            if s_passed and len(s_results) == 0:
                s_category = "WARNING"
                s_warn = f"Zero passages returned for search query '{query}'"
                result.warnings.append(s_warn)

            if s_passed:
                search_successes += 1

            result.search_results.append(
                SearchTestResult(
                    query=query,
                    status_code=s_status,
                    passed=s_passed,
                    passage_count=len(s_results),
                    latency_ms=s_lat,
                    result_category=s_category,
                    warning=s_warn,
                    error=None if s_passed else str(s_body)
                )
            )

        result.search_passed = (search_successes == len(case.search_queries)) if case.search_queries else True
        result.search_avg_latency_ms = statistics.mean(search_latencies) if search_latencies else 0.0

        # Step 4: QA Questions Execution
        qa_successes = 0
        qa_latencies = []
        for question in case.qa_questions:
            q_status, q_body, q_lat = self.client.qa(doc_id, question, label=case.filename)
            self.latencies["POST /qa"].append(q_lat)
            qa_latencies.append(q_lat)

            answer = q_body.get("answer", "")
            retrieved_ctx = q_body.get("retrieved_context", [])
            proc_time = q_body.get("processing_time_ms", 0.0)

            q_passed = (
                q_status == 200 and
                bool(answer.strip()) and
                isinstance(retrieved_ctx, list) and
                len(retrieved_ctx) > 0
            )

            q_category = "PASS" if q_passed else "APPLICATION FAILURE"
            q_quality = evaluate_answer_quality(answer)

            q_warn = None
            if q_lat > self.settings.warning_thresholds.qa_latency_ms:
                q_warn = f"QA latency ({q_lat:.0f} ms) exceeded threshold ({self.settings.warning_thresholds.qa_latency_ms:.0f} ms)"
                result.warnings.append(q_warn)

            if q_passed:
                qa_successes += 1

            result.qa_results.append(
                QATestResult(
                    question=question,
                    status_code=q_status,
                    passed=q_passed,
                    answer=answer,
                    retrieved_context_count=len(retrieved_ctx),
                    processing_time_ms=proc_time,
                    latency_ms=q_lat,
                    answer_quality=q_quality,
                    result_category=q_category,
                    warning=q_warn,
                    error=None if q_passed else str(q_body)
                )
            )

        result.qa_passed = (qa_successes == len(case.qa_questions)) if case.qa_questions else True
        result.qa_avg_latency_ms = statistics.mean(qa_latencies) if qa_latencies else 0.0

        # Step 5: Delete Document
        d_status, d_body, d_lat = self.client.delete_document(doc_id, case.filename)
        self.latencies["DELETE /documents/{id}"].append(d_lat)
        result.delete_status_code = d_status
        result.delete_latency_ms = d_lat
        result.delete_passed = (d_status == 200)

        # Step 6: Verify Deletion (Must return 404)
        v_status, v_body, v_lat = self.client.get_metadata(doc_id, f"verify_{case.filename}")
        self.latencies["GET /documents/{id}"].append(v_lat)
        result.verify_delete_passed = (v_status == 404)

        if not result.verify_delete_passed:
            result.errors.append(f"Post-delete verification failed: GET returned status {v_status} instead of 404")

        # Overall PDF result verdict and classification
        result.overall_passed = (
            result.upload_passed and
            result.metadata_passed and
            result.search_passed and
            result.qa_passed and
            result.delete_passed and
            result.verify_delete_passed
        )

        if not result.overall_passed:
            result.result_category = "APPLICATION FAILURE"
        elif result.warnings:
            result.result_category = "WARNING"
        else:
            result.result_category = "PASS"

        return result

    def run_error_edge_tests(self) -> List[ErrorTestResult]:
        """Execute automated error and edge case tests."""
        console.print("[bold cyan]🧪 Running Automated Edge Case & Error Handling Tests...[/bold cyan]")
        results = []
        dummy_uuid = str(uuid.uuid4())

        # Test 1: GET non-existent metadata -> 404
        status, body, lat = self.client.get_metadata(dummy_uuid, "edge_get_non_existent")
        self.latencies["GET /documents/{id}"].append(lat)
        p1 = (status == 404)
        results.append(ErrorTestResult(
            test_name="Non-Existent Document Metadata (GET 404)",
            endpoint=f"/documents/{dummy_uuid}",
            method="GET",
            expected_status=404,
            actual_status=status,
            passed=p1,
            result_category="EXPECTED FAILURE" if p1 else "APPLICATION FAILURE",
            latency_ms=lat,
            detail="Verified backend returns 404 for invalid UUID" if p1 else f"Failed: status {status}"
        ))

        # Test 2: Double Delete same non-existent document -> 404
        status1, _, lat1 = self.client.delete_document(dummy_uuid, "edge_del_1")
        status2, body2, lat2 = self.client.delete_document(dummy_uuid, "edge_del_2")
        self.latencies["DELETE /documents/{id}"].extend([lat1, lat2])
        p2 = (status2 == 404)
        results.append(ErrorTestResult(
            test_name="Duplicate Delete Non-Existent Document (DELETE 404)",
            endpoint=f"/documents/{dummy_uuid}",
            method="DELETE",
            expected_status=404,
            actual_status=status2,
            passed=p2,
            result_category="EXPECTED FAILURE" if p2 else "APPLICATION FAILURE",
            latency_ms=lat2,
            detail="Verified backend returns 404 for second delete" if p2 else f"Failed: status {status2}"
        ))

        # Test 3: Search non-existent document -> 404
        status, body, lat = self.client.search(dummy_uuid, "Test query", top_k=3, label="edge_search_invalid")
        self.latencies["POST /search"].append(lat)
        p3 = (status == 404)
        results.append(ErrorTestResult(
            test_name="Search Non-Existent Document (POST 404)",
            endpoint="/search",
            method="POST",
            expected_status=404,
            actual_status=status,
            passed=p3,
            result_category="EXPECTED FAILURE" if p3 else "APPLICATION FAILURE",
            latency_ms=lat,
            detail="Verified search fails gracefully for invalid document ID" if p3 else f"Failed: status {status}"
        ))

        # Test 4: QA non-existent document -> 404
        status, body, lat = self.client.qa(dummy_uuid, "Test question?", label="edge_qa_invalid")
        self.latencies["POST /qa"].append(lat)
        p4 = (status == 404)
        results.append(ErrorTestResult(
            test_name="QA Non-Existent Document (POST 404)",
            endpoint="/qa",
            method="POST",
            expected_status=404,
            actual_status=status,
            passed=p4,
            result_category="EXPECTED FAILURE" if p4 else "APPLICATION FAILURE",
            latency_ms=lat,
            detail="Verified QA fails gracefully for invalid document ID" if p4 else f"Failed: status {status}"
        ))

        for r in results:
            mark = "[green]✓[/green]" if r.passed else "[red]✗[/red]"
            console.print(f"  {mark} {r.test_name} ({r.latency_ms:.1f} ms)")

        console.print("")
        return results

    def finalize_and_report(self) -> ValidationReport:
        """Compute latency metrics, readiness checklist, verdicts, and generate reports."""
        total_runtime = time.perf_counter() - self.start_time

        endpoint_metrics = {}
        for ep, lats in self.latencies.items():
            endpoint_metrics[ep] = MetricStats.calculate(lats)

        passed_pdfs = sum(1 for p in self.pdf_results if p.overall_passed)
        failed_pdfs = len(self.pdf_results) - passed_pdfs
        total_calls = self.client.request_counter

        # Evaluate deployment readiness checklist (ONLY FOR SUPPORTED BENCHMARK CASES)
        health_ok = endpoint_metrics.get("GET /health", MetricStats()).count > 0
        all_err_tests_ok = all(e.passed for e in self.error_results)
        
        pdf_by_name = {p.filename: p for p in self.pdf_results}

        checklist = {
            "Backend Reachable": health_ok,
            "Upload Endpoint": all(p.upload_passed for p in self.pdf_results if p.expected_upload_success),
            "Metadata Endpoint": all(p.metadata_passed for p in self.pdf_results if p.expected_upload_success),
            "Search Endpoint": all(p.search_passed for p in self.pdf_results if p.expected_upload_success),
            "QA Endpoint": all(p.qa_passed for p in self.pdf_results if p.expected_upload_success),
            "Delete Endpoint": all(p.delete_passed and p.verify_delete_passed for p in self.pdf_results if p.expected_upload_success),
            "Invalid Input Handling": all_err_tests_ok and pdf_by_name.get("empty.pdf", PDFValidationResult("empty.pdf", "", False)).overall_passed,
            "Technical Paper Benchmark": pdf_by_name.get("attention_is_all_you_need.pdf", PDFValidationResult("attention_is_all_you_need.pdf", "", True)).overall_passed,
            "Scanned OCR Benchmark": pdf_by_name.get("test_ocr.pdf", PDFValidationResult("test_ocr.pdf", "", True)).overall_passed,
            "Business Report Benchmark": pdf_by_name.get("2025_AnnualReport.pdf", PDFValidationResult("2025_AnnualReport.pdf", "", True)).overall_passed,
            "Story Benchmark": pdf_by_name.get("the_canterville_ghost.pdf", PDFValidationResult("the_canterville_ghost.pdf", "", True)).overall_passed,
            "Medium Document Benchmark": pdf_by_name.get("sample-100pages.pdf", PDFValidationResult("sample-100pages.pdf", "", True)).overall_passed,
            "Large Document Benchmark": pdf_by_name.get("sample-1000pages.pdf", PDFValidationResult("sample-1000pages.pdf", "", True)).overall_passed,
        }


        overall_verdict = "READY FOR DEPLOYMENT" if all(checklist.values()) else "NOT READY FOR DEPLOYMENT"

        report = ValidationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            backend_url=self.settings.backend_url,
            system_info=self.system_info,
            total_pdfs=len(self.pdf_results),
            passed_pdfs=passed_pdfs,
            failed_pdfs=failed_pdfs,
            pdf_results=self.pdf_results,
            error_test_results=self.error_results,
            global_warnings=self.global_warnings,
            global_errors=self.global_errors,
            endpoint_latencies=endpoint_metrics,
            readiness_checklist=checklist,
            overall_verdict=overall_verdict,
            total_api_calls=total_calls,
            total_runtime_seconds=total_runtime,
            average_upload_latency_ms=endpoint_metrics.get("POST /documents", MetricStats()).avg_ms,
            average_search_latency_ms=endpoint_metrics.get("POST /search", MetricStats()).avg_ms,
            average_qa_latency_ms=endpoint_metrics.get("POST /qa", MetricStats()).avg_ms
        )

        reporter = ReportGenerator(report, self.reports_dir)
        json_path = reporter.generate_json_report()
        md_path = reporter.generate_markdown_report()
        reporter.render_console()

        console.print(f"[bold green]📄 JSON Report saved to:[/bold green] [underline]{json_path}[/underline]")
        console.print(f"[bold green]📄 Markdown Report saved to:[/bold green] [underline]{md_path}[/underline]\n")

        return report

    def run_all(self) -> int:
        """Run complete E2E validation pipeline."""
        if not self.run_health_check():
            return 1

        console.print(f"[bold cyan]📁 Running validation cases across {len(self.pdf_cases)} supported benchmark PDF files...[/bold cyan]\n")

        for case in tqdm(self.pdf_cases, desc="Validating Benchmark PDFs", unit="pdf"):
            res = self.validate_pdf_case(case)
            self.pdf_results.append(res)

        self.error_results = self.run_error_edge_tests()
        report = self.finalize_and_report()

        return 0 if report.overall_verdict == "READY FOR DEPLOYMENT" else 1
