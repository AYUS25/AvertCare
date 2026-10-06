# ─────────────────────────────────────────────────────────────
# AvertCare Backend · Dockerfile
# Base: python:3.11-slim (smallest secure image with C-extension support)
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

WORKDIR /app

# System deps (curl for healthcheck probe)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependency layer (cached unless requirements.txt changes) ─
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
