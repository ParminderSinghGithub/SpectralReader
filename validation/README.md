# 🧪 SpectralReader Automated End-to-End Validation Framework

An extensible, production-quality automated end-to-end (E2E) validation framework for **SpectralReader**.

Unlike unit tests or mocks, this framework exercises the running **FastAPI REST backend** over HTTP exactly like a real user interface or external API consumer. It validates document uploads, metadata parsing, passage search queries, FLAN-T5 generative question answering, document deletion, edge cases, error handling, and performance metrics across supported machine-readable PDF document types.

> [!NOTE]
> **Scope & Document Support Notice**:
> The validation benchmark suite currently tests **machine-readable PDFs** containing native text streams.
> Image-only or scanned PDF documents requiring Optical Character Recognition (OCR) are intentionally outside the current project scope and are excluded from benchmark deployment validation. OCR support is documented on the product roadmap for future releases.

---

## 📂 Framework Folder Structure

```
validation/
├── pdfs/                           # Benchmark PDF test documents
│   ├── 2025_AnnualReport.pdf       # Business Report Benchmark (80 pages)
│   ├── the_canterville_ghost.pdf   # Story / Literary Benchmark (53 pages)
│   ├── sample-100pages.pdf         # Medium Document Stress Test (100 pages)
│   ├── sample-1000pages.pdf        # Large Document Stress Test (1000 pages)
│   └── empty.pdf                   # Graceful Input Error Test (empty/unparseable PDF)
├── configs/
│   └── validation_cases.yaml       # YAML configuration driven test suite
├── reports/
│   ├── latest_report.md            # Human-readable Markdown summary report
│   ├── latest_report.json          # Machine-readable JSON summary report
│   └── raw/                        # Individual HTTP request/response JSON archives
├── validate.py                     # Main CLI execution entry point
├── config_loader.py                # YAML configuration loader & schema parser
├── client.py                       # HTTP REST API client with timing & archiving
├── models.py                       # Dataclasses, metrics, & report data models
├── validator.py                    # Core E2E validation engine & test runner
├── reporter.py                     # Markdown, JSON, and Rich console reporter
├── requirements.txt                # Validation framework Python dependencies
└── README.md                       # Validation framework documentation
```

---

## 🛠️ Requirements & Installation

The validation framework requires Python 3.12+ and the following packages:

- `requests` (HTTP REST client)
- `pyyaml` (YAML configuration parser)
- `rich` (Terminal formatting, tables, panels, and colored output)
- `tqdm` (Progress bar rendering)
- `psutil` (System hardware memory monitoring)

Install dependencies into your Virtual Environment:

```bash
pip install -r validation/requirements.txt
```

---

## 🚀 How to Run Validation

> [!IMPORTANT]
> The FastAPI backend service must be running before executing validation.  
> Start the server in a separate terminal window:  
> `uvicorn app.main_api:app --reload --host 0.0.0.0 --port 8000`

### 1. Default Execution
Run the full benchmark validation suite against `http://localhost:8000`:

```bash
python validation/validate.py
```

### 2. Custom Backend URL or Timeout
Override backend endpoint or HTTP timeout:

```bash
python validation/validate.py --backend-url http://localhost:8000 --timeout 180
```

### 3. Custom Configuration or Output Directory

```bash
python validation/validate.py --config validation/configs/validation_cases.yaml --output-dir reports
```

---

## 🔄 6-Step Validation Flow

For every supported PDF benchmark test case in `configs/validation_cases.yaml`, the framework executes the following 6-step lifecycle:

1. **Step 1: Upload (`POST /documents`)**
   - Uploads binary PDF.
   - Verifies HTTP status `201 Created` and schema (`document_id`, `filename`, `num_pages`, `num_chunks`, `entities`).
   - Measures upload latency.
2. **Step 2: Metadata (`GET /documents/{id}`)**
   - Fetches document metadata.
   - Verifies HTTP status `200 OK` and matching `document_id` and `filename`.
3. **Step 3: Search (`POST /search`)**
   - Executes all configured search queries.
   - Verifies HTTP status `200 OK`, schema, returned passages, and latency.
   - Differentiates warnings (e.g. zero passages returned) without failing valid API responses.
4. **Step 4: Question Answering (`POST /qa`)**
   - Executes all configured QA questions.
   - Verifies HTTP status `200 OK`, schema, non-empty answer string, non-empty retrieved context list, and model processing time.
   - Evaluates answer quality using lightweight heuristics (`GOOD`, `PARTIAL`, `POOR`) for informational insight without LLM evaluation.
5. **Step 5: Delete (`DELETE /documents/{id}`)**
   - Deletes the document from in-memory storage.
   - Verifies HTTP status `200 OK`.
6. **Step 6: Deletion Verification (`GET /documents/{id}`)**
   - Re-fetches metadata for the deleted document.
   - Verifies HTTP status `404 Not Found` (fails if document persists).

---

## 🏷️ Result Classification Categories

The validator categorizes every test result into one of four distinct categories:

- **`PASS`**: Endpoint responded correctly with expected success schema.
- **`WARNING`**: Endpoint succeeded, but triggered an informational warning (e.g. zero search passages or latency threshold alert).
- **`EXPECTED FAILURE`**: Endpoint returned non-200 status as explicitly expected for invalid input (e.g. `empty.pdf` returning HTTP `422`). Counts as a PASS for deployment readiness.
- **`APPLICATION FAILURE`**: Unexpected HTTP 500 error or failure to meet API contract requirements.

---

## 🧪 Edge Case & Error Handling Tests

The validator automatically executes non-PDF boundary tests:
- **Non-Existent Document Metadata**: `GET /documents/{random_uuid}` -> Expects `404`.
- **Duplicate Delete**: `DELETE /documents/{random_uuid}` twice -> Expects `404`.
- **Search Non-Existent Document**: `POST /search` with invalid UUID -> Expects `404`.
- **QA Non-Existent Document**: `POST /qa` with invalid UUID -> Expects `404`.
- **Empty PDF Ingestion**: `POST /documents` with `empty.pdf` -> Expects graceful `422` error handling.

---

## ➕ Adding New Benchmark PDF Test Cases & Questions

To add a new machine-readable PDF document test case:

1. **Add PDF File**: Place your text-based PDF file inside `validation/pdfs/` (e.g., `new_contract.pdf`).
2. **Update `validation/configs/validation_cases.yaml`**: Add a new block under `pdf_cases`:

```yaml
  - filename: "new_contract.pdf"
    purpose: "Vendor Agreement Test"
    expected_upload_success: true
    search_queries:
      - "indemnification"
      - "governing law"
    qa_questions:
      - "What is the contract duration?"
      - "What are the payment terms?"
```

3. **Rerun Validator**: Execute `python validation/validate.py`.

---

## 📊 Reports & Raw Request Archive

Upon completion, the framework writes:

1. **Markdown Report** (`validation/reports/latest_report.md`):
   - Executive summary table with total runtime and average upload/search/QA latencies.
   - System environment details (CPU, RAM, Python version, OS).
   - Deployment Readiness Checklist.
   - Endpoint latency statistics (Min, Max, Average, Median, P95).
   - Detailed breakdown per PDF case (including QA Answer Quality) and error handling tests.
2. **JSON Report** (`validation/reports/latest_report.json`):
   - Machine-readable snapshot containing full metrics, per-query details, and checklist statuses.
3. **Raw Request/Response Archive** (`validation/reports/raw/*.json`):
   - Every single HTTP call made during validation is archived as an individual JSON file (e.g. `001_upload_2025_annualreport.json`, `002_metadata_2025_annualreport.json`, etc.) for auditability and offline debugging.

---

## 🛑 Exit Codes

The validator returns standard shell exit codes for CI/CD integration:

- **Exit Code 0**: Overall verdict is **`READY FOR DEPLOYMENT`** (all supported benchmark endpoints, test cases, deletion checks, and error tests passed).
- **Exit Code 1**: Overall verdict is **`NOT READY FOR DEPLOYMENT`** (pre-flight backend unreachable, unhandled exception, HTTP status mismatch, or post-delete persistence failure).
