# Production Dockerfile for ML + Quants Trading Agent (Paper Trading)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (build-essential for numpy/scipy/lightgbm if compiling wheels, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy source tree and configuration
COPY configs/ ./configs/
COPY src/ ./src/
COPY models/ ./models/
COPY scripts/ ./scripts/

# Ensure start script has execute permissions
RUN chmod +x /app/scripts/*.sh || true

# Create persistent runtime data, models, and logging directories
RUN mkdir -p /app/logs/audit /app/data /app/models

# Create non-root unprivileged service account
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app
USER trader

# Volume mounts for persistence
VOLUME ["/app/logs", "/app/data"]

EXPOSE 8000

# Default entrypoint launches start_production.sh (starts FastAPI + optional background loop)
CMD ["bash", "/app/scripts/start_production.sh"]

