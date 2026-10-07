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
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes import router
from app.core.config import settings


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
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

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
