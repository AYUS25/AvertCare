"""
AvertCare Backend · FastAPI Application Entry Point

Startup order:
  1. CORS middleware (allow Vercel + localhost)
  2. Prometheus instrumentation (/metrics endpoint)
  3. API router mount
  4. Health & root probes
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes import router
from app.core.config import settings

# Names match the Clinical Ops PromQL (fastapi_requests_*).
_REQUESTS = Counter(
    "fastapi_requests_total",
    "HTTP requests served by the AvertCare API.",
    labelnames=("method", "handler", "status"),
)
_LATENCY = Histogram(
    "fastapi_requests_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=("method", "handler"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)


def _record_request(info) -> None:
    _REQUESTS.labels(info.method, info.modified_handler, info.modified_status).inc()
    _LATENCY.labels(info.method, info.modified_handler).observe(info.modified_duration)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ─────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Prometheus ───────────────────────────────────────────
    # Paths are regular expressions. A bare "/" would match every route.
    Instrumentator(
        should_group_status_codes=False,
        excluded_handlers=[
            "^/metrics$",
            "^/health$",
            "^/$",
            "^/docs$",
            "^/redoc$",
            "^/openapi.json$",
        ],
    ).add(_record_request).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    # ── Routes ───────────────────────────────────────────────
    app.include_router(router)

    # ── Probes ───────────────────────────────────────────────
    @app.get("/health", tags=["Ops"], include_in_schema=False)
    async def health() -> dict:
        return {"status": "ok", "version": settings.APP_VERSION}

    @app.get("/", tags=["Ops"], include_in_schema=False)
    async def root() -> dict:
        return {"service": settings.APP_NAME, "docs": "/docs"}

    return app


app = create_app()
