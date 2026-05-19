# Single-stage build: avoids large cross-stage venv COPY that exhausts
# Docker Desktop's WSL2 disk during BuildKit layer export.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime services + build tools (purged after pip install to reduce size)
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    supervisor \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .

# CPU-only torch first — prevents pip from pulling the CUDA variant (~2 GB).
RUN pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements-prod.txt

# Remove build toolchain; keeps only runtime shared libs
RUN apt-get purge -y --auto-remove build-essential

# Bake the cross-encoder model to eliminate cold-start latency.
RUN python3 -c "\
from sentence_transformers.cross_encoder import CrossEncoder; \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Process manager and reverse-proxy configuration
# sed converts Windows CRLF → Unix LF
COPY docker/nginx.conf /etc/nginx/nginx.conf
COPY docker/supervisord.conf /etc/supervisor/docuverse.conf
RUN sed -i 's/\r$//' /etc/nginx/nginx.conf /etc/supervisor/docuverse.conf

# Application source (filtered by .dockerignore)
COPY . .

EXPOSE 7860

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/docuverse.conf"]
