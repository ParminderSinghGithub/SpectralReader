"""Validation Report Generator for Markdown, JSON, and Rich Console Output."""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, List
from models import ValidationReport, PDFValidationResult, ErrorTestResult, MetricStats
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console(force_terminal=True)

class ReportGenerator:
    def __init__(self, report: ValidationReport, reports_dir: str):
        self.report = report
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_json_report(self) -> str:
        """Write reports/latest_report.json."""
        filepath = os.path.join(self.reports_dir, "latest_report.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.report.to_dict(), f, indent=2, default=str)
        return filepath

    def generate_markdown_report(self) -> str:
        """Write reports/latest_report.md."""
        filepath = os.path.join(self.reports_dir, "latest_report.md")
        lines = []

        lines.append("# 🧪 SpectralReader Automated End-to-End Validation Report\n")
        lines.append(f"**Generated At**: {self.report.timestamp}  ")
        lines.append(f"**Backend URL**: `{self.report.backend_url}`  ")
        lines.append(f"**Overall Verdict**: **`{self.report.overall_verdict}`**  \n")

        lines.append("---\n")

        # System Information
        lines.append("## 💻 System Information\n")
        lines.append("| Property | Value |")
        lines.append("| :--- | :--- |")
        sys_info = self.report.system_info
        lines.append(f"| **Date & Time** | {sys_info.timestamp} |")
        lines.append(f"| **Python Version** | {sys_info.python_version} |")
        lines.append(f"| **Platform / OS** | {sys_info.platform} ({sys_info.os_name}) |")
        lines.append(f"| **CPU Architecture** | {sys_info.cpu_architecture} |")
        lines.append(f"| **Machine Name** | {sys_info.machine_name} |")
        if sys_info.total_ram_gb:
            lines.append(f"| **RAM (Total / Available)** | {sys_info.total_ram_gb} GB / {sys_info.available_ram_gb} GB |")
        lines.append("\n")

        # Executive Summary Section
        lines.append("## 📊 Executive Performance & Validation Summary\n")
        lines.append("| Metric | Value |")
        lines.append("| :--- | :--- |")
        lines.append(f"| **Supported Benchmark PDFs** | {self.report.total_pdfs} |")
        lines.append(f"| **Passed PDF Test Cases** | {self.report.passed_pdfs} |")
        lines.append(f"| **Failed PDF Test Cases** | {self.report.failed_pdfs} |")
        lines.append(f"| **Total REST API Calls** | {self.report.total_api_calls} |")
        lines.append(f"| **Total Runtime** | {self.report.total_runtime_seconds:.2f} seconds |")
        lines.append(f"| **Average Upload Latency** | {self.report.average_upload_latency_ms:.2f} ms |")
        lines.append(f"| **Average Search Latency** | {self.report.average_search_latency_ms:.2f} ms |")
        lines.append(f"| **Average QA Latency** | {self.report.average_qa_latency_ms:.2f} ms |")
        lines.append(f"| **Final Deployment Verdict** | **`{self.report.overall_verdict}`** |")
        lines.append("\n")

        # Deployment Readiness Checklist
        lines.append("## 🚀 Deployment Readiness Checklist\n")
        lines.append("| Requirement / Feature | Status |")
        lines.append("| :--- | :--- |")
        for req_name, passed in self.report.readiness_checklist.items():
            status_str = "✅ **PASS**" if passed else "❌ **FAIL**"
            lines.append(f"| {req_name} | {status_str} |")
        lines.append("\n")

        # Endpoint Latency Performance
        lines.append("## ⚡ REST API Performance Metrics\n")
        lines.append("| Endpoint / Action | Min Latency (ms) | Max Latency (ms) | Average Latency (ms) | Median Latency (ms) | P95 Latency (ms) | Call Count |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for endpoint, stats in self.report.endpoint_latencies.items():
            lines.append(f"| `{endpoint}` | {stats.min_ms} | {stats.max_ms} | {stats.avg_ms} | {stats.median_ms} | {stats.p95_ms} | {stats.count} |")
        lines.append("\n")

        # Detailed PDF Results
        lines.append("## 📄 Detailed PDF Validation Results\n")
        for pdf in self.report.pdf_results:
            status_icon = "✅ PASS" if pdf.overall_passed else "❌ FAIL"
            lines.append(f"### {pdf.filename} ({pdf.purpose}) - {status_icon} (`{pdf.result_category}`)\n")
            lines.append(f"- **Expected Upload Success**: `{pdf.expected_upload_success}`")
            lines.append(f"- **Upload Status**: `{pdf.upload_status_code}` (Latency: `{pdf.upload_latency_ms:.2f} ms`)")
            lines.append(f"- **Document ID**: `{pdf.document_id or 'N/A'}`")
            lines.append(f"- **Pages / Chunks**: `{pdf.num_pages}` pages / `{pdf.num_chunks}` chunks")
            lines.append(f"- **Identified Entities**: `{len(pdf.entities)}` extracted")

            if pdf.search_results:
                lines.append("\n#### Search Queries Executed:")
                lines.append("| Search Query | Status | Category | Passages Found | Latency (ms) | Warning |")
                lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
                for s in pdf.search_results:
                    pass_str = "PASS" if s.passed else "FAIL"
                    lines.append(f"| `{s.query}` | {pass_str} ({s.status_code}) | `{s.result_category}` | {s.passage_count} | {s.latency_ms:.2f} | {s.warning or '-'} |")

            if pdf.qa_results:
                lines.append("\n#### QA Questions Executed:")
                lines.append("| QA Question | Status | Context Count | Model Time (ms) | Latency (ms) | Answer Quality | Answer Preview |")
                lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
                for q in pdf.qa_results:
                    pass_str = "PASS" if q.passed else "FAIL"
                    ans_preview = q.answer.replace("\n", " ")[:60] + "..." if len(q.answer) > 60 else q.answer
                    lines.append(f"| `{q.question}` | {pass_str} ({q.status_code}) | {q.retrieved_context_count} | {q.processing_time_ms:.2f} | {q.latency_ms:.2f} | **`{q.answer_quality}`** | {ans_preview} |")

            lines.append(f"\n- **Delete Status**: `{pdf.delete_status_code}` | **Verify Delete (404 Check)**: {'PASS' if pdf.verify_delete_passed else 'FAIL'}")

            if pdf.warnings:
                lines.append(f"- **Warnings**: {', '.join(pdf.warnings)}")
            if pdf.errors:
                lines.append(f"- **Errors**: {', '.join(pdf.errors)}")

            lines.append("\n---\n")

        # Error Tests
        lines.append("## 🧪 Edge Case & Error Handling Validation\n")
        lines.append("| Error Test | Method & Endpoint | Expected | Actual | Verdict | Category | Latency (ms) | Detail |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for err in self.report.error_test_results:
            pass_str = "✅ PASS" if err.passed else "❌ FAIL"
            lines.append(f"| {err.test_name} | `{err.method} {err.endpoint}` | `{err.expected_status}` | `{err.actual_status}` | {pass_str} | `{err.result_category}` | {err.latency_ms:.2f} | {err.detail} |")

        lines.append("\n")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return filepath

    def render_console(self):
        """Render beautiful Rich terminal tables and summary banners."""
        console.print("\n")
        console.rule("[bold cyan]SpectralReader End-to-End Automated Validation Summary[/bold cyan]")
        console.print("\n")

        # System Info Panel
        sys_info = self.report.system_info
        info_text = (
            f"[bold]Timestamp:[/bold] {sys_info.timestamp} | "
            f"[bold]Python:[/bold] {sys_info.python_version} | "
            f"[bold]OS/Platform:[/bold] {sys_info.platform}\n"
            f"[bold]Backend URL:[/bold] {self.report.backend_url} | "
            f"[bold]Total API Calls:[/bold] {self.report.total_api_calls} | "
            f"[bold]Total Runtime:[/bold] {self.report.total_runtime_seconds:.1f}s"
        )
        console.print(Panel(info_text, title="[bold yellow]Environment & Execution Context[/bold yellow]", border_style="blue"))

        # Document Validation Table
        doc_table = Table(title="Supported Benchmark Validation Results", box=box.ROUNDED, header_style="bold magenta")
        doc_table.add_column("Benchmark PDF File", style="cyan", no_wrap=True)
        doc_table.add_column("Category", justify="center")
        doc_table.add_column("Upload", justify="center")
        doc_table.add_column("Metadata", justify="center")
        doc_table.add_column("Search", justify="center")
        doc_table.add_column("QA", justify="center")
        doc_table.add_column("Delete", justify="center")
        doc_table.add_column("Avg Latency", justify="right")
        doc_table.add_column("Verdict", justify="center")

        for pdf in self.report.pdf_results:
            cat_color = "green" if pdf.result_category == "PASS" else ("yellow" if pdf.result_category in ("WARNING", "EXPECTED FAILURE") else "red")
            cat_str = f"[{cat_color}]{pdf.result_category}[/{cat_color}]"

            u_str = "[green]PASS[/green]" if pdf.upload_passed else ("[yellow]EXPECTED FAIL[/yellow]" if not pdf.expected_upload_success else "[red]FAIL[/red]")
            m_str = "[green]PASS[/green]" if pdf.metadata_passed else ("[yellow]N/A[/yellow]" if not pdf.expected_upload_success else "[red]FAIL[/red]")
            s_str = "[green]PASS[/green]" if pdf.search_passed else ("[yellow]N/A[/yellow]" if not pdf.expected_upload_success else "[red]FAIL[/red]")
            q_str = "[green]PASS[/green]" if pdf.qa_passed else ("[yellow]N/A[/yellow]" if not pdf.expected_upload_success else "[red]FAIL[/red]")
            d_str = "[green]PASS[/green]" if (pdf.delete_passed and pdf.verify_delete_passed) else ("[yellow]N/A[/yellow]" if not pdf.expected_upload_success else "[red]FAIL[/red]")
            
            lat_avg = (pdf.upload_latency_ms + pdf.metadata_latency_ms + pdf.search_avg_latency_ms + pdf.qa_avg_latency_ms + pdf.delete_latency_ms)
            if pdf.search_results or pdf.qa_results:
                divisor = (1 + 1 + len(pdf.search_results) + len(pdf.qa_results) + 1)
                lat_avg = lat_avg / max(divisor, 1)
            
            v_str = "[bold green]PASS[/bold green]" if pdf.overall_passed else "[bold red]FAIL[/bold red]"

            doc_table.add_row(
                pdf.filename,
                cat_str,
                u_str,
                m_str,
                s_str,
                q_str,
                d_str,
                f"{lat_avg:.1f} ms",
                v_str
            )

        console.print(doc_table)
        console.print("\n")

        # Endpoint Latency Table
        perf_table = Table(title="REST API Latency Statistics", box=box.SIMPLE_HEAD, header_style="bold blue")
        perf_table.add_column("Endpoint", style="bold white")
        perf_table.add_column("Min (ms)", justify="right")
        perf_table.add_column("Max (ms)", justify="right")
        perf_table.add_column("Average (ms)", justify="right")
        perf_table.add_column("Median (ms)", justify="right")
        perf_table.add_column("P95 (ms)", justify="right")
        perf_table.add_column("Count", justify="right")

        for ep, stats in self.report.endpoint_latencies.items():
            perf_table.add_row(
                ep,
                f"{stats.min_ms:.1f}",
                f"{stats.max_ms:.1f}",
                f"{stats.avg_ms:.1f}",
                f"{stats.median_ms:.1f}",
                f"{stats.p95_ms:.1f}",
                str(stats.count)
            )

        console.print(perf_table)
        console.print("\n")

        # Deployment Readiness Checklist Table
        ready_table = Table(title="Deployment Readiness Checklist", box=box.DOUBLE_EDGE, header_style="bold yellow")
        ready_table.add_column("Checklist Requirement", style="white")
        ready_table.add_column("Status", justify="center")

        for req_name, passed in self.report.readiness_checklist.items():
            status_text = "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]"
            ready_table.add_row(req_name, status_text)

        console.print(ready_table)
        console.print("\n")

        # Overall Verdict Banner
        if self.report.overall_verdict == "READY FOR DEPLOYMENT":
            verdict_panel = Panel(
                Text("🚀 OVERALL VERDICT: READY FOR DEPLOYMENT", justify="center", style="bold white on green"),
                border_style="green"
            )
        else:
            verdict_panel = Panel(
                Text("❌ OVERALL VERDICT: NOT READY FOR DEPLOYMENT", justify="center", style="bold white on red"),
                border_style="red"
            )
        console.print(verdict_panel)
        console.print("\n")
