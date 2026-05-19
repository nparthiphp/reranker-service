"""
app/main.py — Reranker Service entry point.

Run:  uvicorn app.main:app --reload --port 8001
Docs: http://localhost:8001/docs
"""
from __future__ import annotations
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.rerank_router import router as rerank_router
from app.core.settings import get_settings
from app.models.schemas import HealthResponse, ModelStatus

settings   = get_settings()
_START_TIME = time.perf_counter()


# ── Lifespan: startup + shutdown ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    if settings.prewarm_on_startup:
        print(f"[{settings.app_name}] Pre-warming models...")
        try:
            import app.services.cross_encoder as ce
            ce.warmup()
            print(f"[{settings.app_name}] Cross-encoder ready: {settings.cross_encoder_model}")
        except Exception as e:
            print(f"[{settings.app_name}] Cross-encoder warmup failed: {e}")
        try:
            import app.services.mmr as mmr
            mmr.warmup()
            print(f"[{settings.app_name}] Embedding model ready: {settings.embedding_model}")
        except Exception as e:
            print(f"[{settings.app_name}] MMR warmup failed: {e}")

    print(f"[{settings.app_name}] v{settings.app_version} ready — http://localhost:{settings.port}/docs")
    yield
    # ── Shutdown ──
    print(f"[{settings.app_name}] Shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Reranker Service",
    description=(
        "**Enterprise-grade standalone reranking microservice.**\n\n"
        "Takes retrieved candidates from any RAG system and returns "
        "precision-ranked results using:\n\n"
        "- **Cross-Encoder** — BERT reads query + chunk together → precise relevance score\n"
        "- **MMR** — Maximum Marginal Relevance → removes duplicate content\n"
        "- **Ensemble** — Cross-Encoder then MMR (recommended for production)\n\n"
        "**Authentication:** Pass `X-API-Key` header (set `API_KEY` in .env)\n\n"
        "**Integration:** Works with EduRAG, DocChunker, or any RAG pipeline."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_middleware(request: Request, call_next) -> Response:
    """Inject correlation ID and timing into every response."""
    correlation_id = (
        request.headers.get(settings.correlation_id_header)
        or str(uuid.uuid4())[:8]
    )
    t0       = time.perf_counter()
    response = await call_next(request)
    elapsed  = round((time.perf_counter() - t0) * 1000, 2)
    response.headers[settings.correlation_id_header] = correlation_id
    response.headers["X-Response-Time-Ms"]            = str(elapsed)
    response.headers["X-Service"]                     = settings.app_name
    response.headers["X-Version"]                     = settings.app_version
    return response


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(rerank_router, prefix="/api/v1")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Service health — checks model status",
)
async def health() -> HealthResponse:
    import app.services.cross_encoder as ce
    import app.services.mmr as mmr_svc

    models = [
        ModelStatus(
            name=settings.cross_encoder_model,
            loaded=ce.is_loaded(),
        ),
        ModelStatus(
            name=settings.embedding_model,
            loaded=mmr_svc.is_loaded(),
        ),
    ]

    all_loaded = all(m.loaded for m in models)
    status_str = "healthy" if all_loaded else "degraded"

    return HealthResponse(
        status=status_str,
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        models=models,
        supported_methods=["cross_encoder", "mmr", "ensemble"],
        uptime_seconds=round(time.perf_counter() - _START_TIME, 1),
    )


# ── Metrics ───────────────────────────────────────────────────────────────────
@app.get(
    "/api/v1/metrics",
    tags=["observability"],
    summary="Prometheus metrics",
    include_in_schema=settings.enable_metrics,
)
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    try:
        from prometheus_client import (
            CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
        )
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        return JSONResponse({"error": "prometheus_client not installed"})


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({
        "service": settings.app_name,
        "version": settings.app_version,
        "docs":    "/docs",
        "health":  "/api/v1/health",
        "endpoints": {
            "rerank":         "POST /api/v1/rerank",
            "cross_encoder":  "POST /api/v1/rerank/cross-encoder",
            "mmr":            "POST /api/v1/rerank/mmr",
            "health":         "GET  /api/v1/health",
            "metrics":        "GET  /api/v1/metrics",
        },
    })
