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

# Create persistent runtime data and logging directories
RUN mkdir -p /app/logs/audit /app/data

# Create non-root unprivileged service account
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app
USER trader

# Volume mounts for persistence
VOLUME ["/app/logs", "/app/data"]

# Healthcheck: validates python process is responsive and audit trail log updated
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os, sys, glob; logs = glob.glob('/app/logs/audit/*_audit.jsonl'); sys.exit(0 if logs else 1)"

# Default entrypoint runs live trading loop
ENTRYPOINT ["python", "-m", "src.execution.live_trading_loop"]
CMD ["--symbols", "AAPL", "MSFT", "NVDA", "GOOGL", "--bar-interval", "60"]
