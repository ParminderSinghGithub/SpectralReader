# Multi-stage Dockerfile for SpectralReader Document Intelligence Microservice
# Stage 1: Build dependency wheels
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final lightweight runtime container
FROM python:3.12-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY app/ ./app/

# Deployment environment defaults (overridden dynamically by Render / cloud platforms)
ENV HOST=0.0.0.0
ENV PORT=8000
ENV LOG_LEVEL=INFO
ENV API_BASE_URL=http://localhost:8000
ENV CORS_ORIGINS=*

EXPOSE 8000

# Docker Healthcheck targeting /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Launch FastAPI application using dynamic PORT environment variable binding
CMD ["sh", "-c", "uvicorn app.main_api:app --host 0.0.0.0 --port ${PORT}"]
