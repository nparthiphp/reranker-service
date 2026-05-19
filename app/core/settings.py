"""app/core/settings.py — Centralised config. Python 3.9+."""
from __future__ import annotations
from enum import Enum
from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    DEBUG   = "DEBUG"
    INFO    = "INFO"
    WARNING = "WARNING"
    ERROR   = "ERROR"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        case_sensitive=False, extra="ignore"
    )

    # ── Service identity ───────────────────────────────────────
    app_name:    str = "Reranker Service"
    app_version: str = "1.0.0"
    environment: str = "development"   # development | staging | production
    port:        int = 8001
    debug:       bool = False

    # ── Cross-encoder model ─────────────────────────────────────
    # Swap to a larger model for better accuracy (needs more RAM)
    # ms-marco-MiniLM-L-6-v2   67MB  MRR=0.323  (default — fast)
    # ms-marco-MiniLM-L-12-v2  130MB MRR=0.332  (balanced)
    # ms-marco-electra-base     420MB MRR=0.341  (best quality)
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_max_length: int = 512   # max tokens per (query+chunk) pair

    # ── Embedding model for MMR ─────────────────────────────────
    # MUST match the bi-encoder used by your RAG pipeline
    embedding_model: str = "multi-qa-MiniLM-L6-cos-v1"

    # ── Reranking defaults ──────────────────────────────────────
    default_top_k:       int   = 3
    default_mmr_lambda:  float = 0.7    # 1.0=pure relevance, 0.0=pure diversity
    max_input_chunks:    int   = 100    # reject requests larger than this
    min_chunk_chars:     int   = 10     # skip very short chunks

    # ── Security ────────────────────────────────────────────────
    # Set API_KEY in .env to enable authentication
    # Leave empty to disable auth (development only)
    api_key: Optional[str] = Field(None, env="API_KEY")

    # ── Rate limiting ───────────────────────────────────────────
    rate_limit_per_minute: int = 100    # per client IP

    # ── Observability ───────────────────────────────────────────
    log_level:         LogLevel = LogLevel.INFO
    log_json:          bool     = True    # structured JSON logs
    enable_metrics:    bool     = True    # /api/v1/metrics endpoint
    correlation_id_header: str  = "X-Correlation-ID"

    # ── Performance ─────────────────────────────────────────────
    # Pre-warm models at startup so first request is not slow
    prewarm_on_startup: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
