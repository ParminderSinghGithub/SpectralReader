# 📖 SpectralReader - Document Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139-emerald.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Client-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**SpectralReader** is an AI-powered Document Intelligence application designed for document understanding, PDF text extraction, entity metadata recognition, passage retrieval, and context-aware question answering. Built with a modular Python backend powered by **FastAPI** and **FLAN-T5**, it decouples machine learning inference and document processing from its interactive **Streamlit** user interface.

---

## 🎯 Why SpectralReader?

Unstructured text trapped in PDF documents—such as research papers, legal contracts, technical manuals, and corporate reports—is difficult to search and analyze efficiently. **SpectralReader** addresses this challenge by providing a structured Document Intelligence API and interactive client that extracts structural content, identifies key entities, and answers natural language questions over document passages.

### Key Engineering Concepts Demonstrated:
- **Clean Microservice Architecture**: Complete separation of UI presentation from backend logic, data parsing, and model execution.
- **RESTful API Design**: Single source of truth API built with FastAPI, Pydantic validation, and OpenAPI specification.
- **ML Dependency Isolation**: Decoupled model lifecycle management with fast, deterministic unit/integration testing via dependency mocking.
- **Deployment-Ready Engineering**: Environment-driven configurations, structured logging, multi-stage Docker builds, and cloud deployment guides for Render and Streamlit Cloud.

---

## 🏛️ System Architecture

SpectralReader uses a decoupled client-service architecture. The **FastAPI REST API** serves as the primary engine for document parsing, entity extraction, and FLAN-T5 generation, while the **Streamlit Web Application** functions strictly as an API client.

```mermaid
graph TD
    User([User / External Client]) -->|HTTP / REST| Frontend[Streamlit Frontend Client]
    Frontend -->|POST /documents| API[FastAPI Backend Server]
    Frontend -->|POST /qa| API
    Frontend -->|GET /health| API

    subgraph Backend Microservice Tier
        API -->|Parse PDF| DocService[DocumentService]
        API -->|Chunk Text| ProcService[ProcessingService]
        API -->|Extract Entities| MetaService[MetadataService]
        API -->|In-Memory Store| Storage[DocumentStore]
        API -->|QA Inference| QAService[QAService]
        
        QAService -->|FLAN-T5 Gen| Models[ModelService Singleton]
    end
```

---

## 💡 Architecture Decisions

- **FastAPI as Core Backend**: Chosen for high performance, automatic Pydantic request/response validation, native OpenAPI/Swagger generation, and clean asynchronous request routing.
- **Streamlit as Official Frontend**: Streamlit provides a responsive interface for document uploads and interactive analysis without adding complex frontend JavaScript build pipelines.
- **Backend as Single Source of Truth**: All PDF parsing, text cleaning, chunking, entity extraction, model loading, and QA generation reside strictly within backend services.
- **Exclusive REST API Communication**: The Streamlit client communicates with the backend exclusively over HTTP REST endpoints. If the backend is offline, Streamlit prompts the operator to start the server rather than silently running local in-process fallbacks.
- **External System Integration**: Decoupling the business logic into REST endpoints enables external systems (mobile apps, CLI tools, automated batch pipelines) to consume the service independently of the Streamlit client.

---

## ✨ Features

- 📄 **PDF Text Extraction**: Page-level text parsing using `pdfplumber`.
- 🧩 **Semantic Text Chunking**: Boundary-aware document splitting with configurable chunk sizes and overlap limits.
- 🏷️ **Entity Metadata Recognition**: Pattern-based entity extraction and frequency analysis.
- 🔍 **Passage Retrieval & Search**: Candidate passage filtering across document chunks.
- 🧠 **Context-Aware Question Answering**: Generative answer synthesis using FLAN-T5-Large.
- ⚡ **REST Microservice**: Standardized JSON responses, Pydantic data validation, and global exception handling.
- 📊 **Health Probes & Metrics**: `/health` endpoint and `X-Process-Time` request timing headers.

---

## 🛠️ Technology Stack

| Category | Technology Used | Description |
| :--- | :--- | :--- |
| **Backend Framework** | FastAPI | REST API routing and OpenAPI generation |
| **Frontend Interface** | Streamlit | Interactive web user interface |
| **NLP & Language Models** | FLAN-T5-Large | Seq2Seq generative question answering |
| **Document Processing** | `pdfplumber`, LangChain | PDF text extraction and recursive text splitting |
| **ML Framework** | PyTorch, Hugging Face | Model inference and execution |
| **API & Data Validation**| Pydantic, Python-Multipart | Schema validation and file upload handling |
| **Containerization** | Docker, Docker Compose | Multi-stage build containerization |
| **Testing Suite** | Pytest, Pytest-Cov, HTTPX | Automated unit and integration testing |

---

## 📂 Repository Structure

```
SpectralReader/
├── Dockerfile                  # Multi-stage Docker build container
├── .dockerignore               # Docker context exclusion rules
├── docker-compose.yml          # Local container orchestration file
├── .env.example                # Deployment environment variable template
├── LICENSE                     # MIT License
├── README.md                   # Platform documentation
├── app/                        # Application source code
│   ├── main_api.py             # FastAPI backend entry point
│   ├── main.py                 # Streamlit frontend client entry point
│   ├── requirements.txt        # Python dependency manifest
│   ├── api/                    # REST API Endpoint Routers
│   ├── core/                   # Infrastructure Core (config, logger, exceptions)
│   ├── models/                 # Pydantic Schemas (request & response models)
│   ├── services/               # Backend Business Logic (document, processing, metadata, model, qa)
│   └── storage/                # In-memory document storage
└── tests/                      # Testing Suite
    ├── conftest.py             # Shared pytest fixtures & ML mocks
    ├── unit/                   # Service unit tests
    └── api/                    # API integration tests
```

---

## 🚀 Local Setup & Execution

### Prerequisites
- Python 3.12 or higher
- `pip` package manager

### 1. Installation
```bash
git clone https://github.com/ParminderSinghGithub/SpectralReader.git
cd SpectralReader
python -m venv venv

# On Windows
.\venv\Scripts\activate

# On Linux/macOS
source venv/bin/activate

pip install -r app/requirements.txt
```

### 2. Launch FastAPI Backend Service
```bash
uvicorn app.main_api:app --reload --host 0.0.0.0 --port 8000
```
The REST API will be accessible at `http://localhost:8000`.  
View interactive Swagger documentation at `http://localhost:8000/docs` and ReDoc at `http://localhost:8000/redoc`.

### 3. Launch Streamlit Frontend Client
In a separate terminal window:
```bash
# On Windows
.\venv\Scripts\activate

streamlit run app/main.py
```
The Streamlit interface will open at `http://localhost:8501`.

---

## 🐳 Docker Setup

### 1. Build and Run Container Locally
```bash
docker build -t spectralreader-api .
docker run -p 8000:8000 -e PORT=8000 spectralreader-api
```

### 2. Using Docker Compose
```bash
docker-compose up --build
```
Probes health status automatically via container health checks on `http://localhost:8000/health`.

---

## 🌐 Deployment Guide

### Deploying FastAPI Backend to Render
1. Sign in to [Render](https://render.com/) and click **New +** -> **Web Service**.
2. Connect your Git repository (`SpectralReader`).
3. Select **Docker** as the Runtime.
4. Set the following details:
   - **Name**: `spectralreader-api`
   - **Region**: Select your preferred region
   - **Branch**: `main`
5. Under **Environment Variables**, set:
   - `PORT` = `8000`
   - `LOG_LEVEL` = `INFO`
   - `CORS_ORIGINS` = `*`
6. Click **Create Web Service**. Render will automatically build the `Dockerfile`, expose the `$PORT`, and run health checks on `/health`.

### Deploying Streamlit Frontend to Streamlit Community Cloud
1. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud).
2. Click **New app**.
3. Select your repository (`SpectralReader`), branch (`main`), and set Main file path to:
   ```
   app/main.py
   ```
4. Under **Advanced settings...** -> **Secrets**, set your Render API backend URL:
   ```toml
   API_BASE_URL = "https://spectralreader-api.onrender.com"
   ```
5. Click **Deploy!**. The frontend client will connect to your Render backend REST API.

---

## ⚙️ Environment Variables Reference

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `HOST` | String | `0.0.0.0` | IP binding interface address |
| `PORT` | Integer | `8000` | Port exposed by backend service |
| `LOG_LEVEL` | String | `INFO` | Logger verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `API_BASE_URL` | String | `http://localhost:8000` | Base URL used by Streamlit client |
| `CORS_ORIGINS` | String | `*` | Allowed CORS origins (comma-separated) |

---

## 📡 REST API Reference & Example Requests

### 1. Health Check
```bash
curl -X GET "http://localhost:8000/health"
```
**Example Response**:
```json
{
  "status": "ok",
  "service": "SpectralReader Document Intelligence API",
  "version": "1.0.0",
  "models_loaded": true
}
```

### 2. Upload Document
```bash
curl -X POST "http://localhost:8000/documents" \
  -F "file=@/path/to/document.pdf"
```
**Example Response**:
```json
{
  "document_id": "<document_id>",
  "filename": "sample_document.pdf",
  "num_pages": "<num_pages>",
  "num_chunks": "<num_chunks>",
  "entities": ["<entity_1>", "<entity_2>"],
  "created_at": "<iso_timestamp>"
}
```

### 3. Get Document Metadata
```bash
curl -X GET "http://localhost:8000/documents/<document_id>"
```

### 4. Search Candidate Passages
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "<document_id>",
    "query": "What are the primary findings?",
    "top_k": 3
  }'
```

### 5. Generative Question Answering
```bash
curl -X POST "http://localhost:8000/qa" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "<document_id>",
    "question": "What is the primary conclusion of the report?"
  }'
```
**Example Response**:
```json
{
  "document_id": "<document_id>",
  "question": "What is the primary conclusion of the report?",
  "answer": "<generated_answer_text>",
  "retrieved_context": ["<context_passage_1>", "<context_passage_2>"],
  "processing_time_ms": "<processing_time_ms>"
}
```

### 6. Delete Document
```bash
curl -X DELETE "http://localhost:8000/documents/<document_id>"
```

---

## 🧪 Automated Testing

Execute the test suite with coverage reporting:
```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

### Testing Strategy
- **Service & Router Verification**: Core document services, storage handlers, schema validation, and REST API endpoints are covered by unit and integration tests.
- **ML Dependency Isolation**: Heavy model downloads and inference runtimes are mocked using `pytest` fixtures, allowing the test suite to execute deterministically in under one second.

---

## 🗺️ Roadmap & Future Enhancements

- 🗄️ **Persistent Document Storage**: Transition from in-memory storage to PostgreSQL or SQLite.
- 🔍 **Vector Database Integration**: Store embeddings in FAISS or Qdrant for semantic similarity retrieval.
- 📑 **OCR Integration**: Support scanned PDF documents using Tesseract OCR or pdf2image.
- 🔒 **Authentication & Authorization**: Add API key management and JWT user authentication.
- 📦 **Cloud Object Storage**: Store uploaded PDF binaries in AWS S3 or Google Cloud Storage.
- 📚 **Multi-Document Retrieval**: Enable query execution across multiple documents simultaneously.
- ⚡ **Background Processing**: Offload heavy document ingestion tasks to Celery / Redis workers.

---

## 📜 License

Released under the **MIT License** — free for academic, personal, and commercial use. See [LICENSE](LICENSE) for details.
