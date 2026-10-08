# ─────────────────────────────────────────────────────────────────────────────
# AvertCare Backend · Dockerfile
#
# Multi-stage build:
#   builder  — installs C-extension dependencies (shap, scikit-learn, etc.)
#   runner   — minimal runtime image (no build tools)
#
# The trained model artifacts are NOT baked into this image.
# They are injected at runtime via the Docker Compose volume mount:
#   ./backend/models → /app/models  (read-only)
# This keeps the image small and lets M1 update models without rebuilding.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS builder

WORKDIR /app

# Build deps needed for C-extensions (shap, numpy, scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer — cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runner

WORKDIR /app

# Runtime system deps only (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application source
COPY app/ ./app/

# Create the models directory — it will be populated by the volume mount
RUN mkdir -p /app/models

# PATH for pip --user installs
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

# 2 workers for production; override per-env as needed
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
