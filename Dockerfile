# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Install dependencies in an isolated layer so the final image stays lean.
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools needed by chromadb / onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first — Docker cache will skip reinstall if unchanged
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Create a writable directory for ChromaDB persistent storage and logs.
# In production mount these as Docker volumes so data survives restarts.
RUN mkdir -p data/chroma_db logs && \
    chmod -R 777 data logs

# Never run as root in production
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Uvicorn port (override with -e PORT=XXXX if needed)
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
