"""app/models/schemas.py — All Pydantic models. Python 3.9+."""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class RerankMethod(str, Enum):
    CROSS_ENCODER = "cross_encoder"   # BERT reads Q+chunk together
    MMR           = "mmr"             # max marginal relevance (diversity)
    ENSEMBLE      = "ensemble"        # cross_encoder → MMR (default)


class RankReason(str, Enum):
    HIGH_RELEVANCE    = "high_relevance"      # CE score >= 0.8
    MODERATE_RELEVANCE = "moderate_relevance" # CE score 0.5-0.8
    LOW_RELEVANCE     = "low_relevance"       # CE score 0.3-0.5
    DIVERSITY_SELECTED = "diversity_selected" # chosen by MMR for coverage
    DIVERSITY_FILTERED = "diversity_filtered" # removed by MMR (duplicate)
    BEST_AVAILABLE    = "best_available"      # all scores low, best we have


# ── Request models ─────────────────────────────────────────────────────────────

class CandidateChunk(BaseModel):
    """
    One candidate chunk from vector search.
    Passed to reranker for precise scoring.
    """
    chunk_id:       str               # unique ID — returned unchanged
    text:           str               # text to score against query
    original_score: float = 0.0      # bi-encoder cosine score from Qdrant
    metadata:       Dict[str, Any] = Field(default_factory=dict)  # passthrough

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("chunk text cannot be empty")
        return v.strip()


class RerankRequest(BaseModel):
    """
    Main reranking request.

    Send top-20 candidates from your vector search.
    Get back top_k precisely ranked with scores and reasons.
    """
    request_id:  str   = Field(default_factory=lambda: str(uuid4())[:8])
    query:       str   = Field(..., min_length=1, max_length=1000)
    chunks:      List[CandidateChunk] = Field(..., min_length=1, max_length=100)
    method:      RerankMethod = RerankMethod.ENSEMBLE
    top_k:       int   = Field(3, ge=1, le=20)

    # MMR tuning (only used when method includes mmr)
    mmr_lambda:  float = Field(0.7, ge=0.0, le=1.0,
        description="0.0=pure diversity, 1.0=pure relevance, 0.7=recommended")

    # Filter out low-score chunks before returning
    min_score:   float = Field(0.0, ge=0.0, le=1.0)

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query cannot be empty")
        return v.strip()


# ── Response models ────────────────────────────────────────────────────────────

class SimilarityInfo(BaseModel):
    """Debug info showing why this chunk was ranked here."""
    cross_encoder_score:    Optional[float] = None   # raw CE logit score
    mmr_score:              Optional[float] = None   # MMR combined score
    similarity_to_query:    Optional[float] = None   # cosine(query, chunk)
    similarity_to_selected: Optional[float] = None   # max cosine to already-selected


class RankedChunk(BaseModel):
    """One chunk with full ranking metadata."""
    chunk_id:       str
    text:           str
    metadata:       Dict[str, Any] = Field(default_factory=dict)
    rank:           int             # 1 = best
    original_score: float           # from Qdrant bi-encoder
    rerank_score:   float           # from cross-encoder or MMR
    score_delta:    float           # rerank_score - original_score (+ = moved up)
    rank_reason:    RankReason
    rank_explanation: str           # human readable explanation
    similarity_info: Optional[SimilarityInfo] = None


class RerankStats(BaseModel):
    """Performance and quality stats for this reranking request."""
    total_input:          int
    total_output:         int
    chunks_filtered:      int         # dropped by min_score
    cross_encoder_ms:     float       # time for CE scoring
    mmr_ms:               float       # time for MMR filtering
    total_ms:             float
    score_improvement:    float       # avg rerank_score - avg original_score
    diversity_score:      float       # avg pairwise distance in result set (MMR)


class RerankResponse(BaseModel):
    """Full reranking response."""
    request_id:   str
    query:        str
    method_used:  RerankMethod
    stats:        RerankStats
    chunks:       List[RankedChunk]


# ── Health ─────────────────────────────────────────────────────────────────────

class ModelStatus(BaseModel):
    name:    str
    loaded:  bool
    size_mb: Optional[float] = None


class HealthResponse(BaseModel):
    status:            str    # healthy | degraded | unhealthy
    service:           str
    version:           str
    environment:       str
    models:            List[ModelStatus]
    supported_methods: List[str]
    uptime_seconds:    float


# ── Error ──────────────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    code:    str
    message: str
    field:   Optional[str] = None


class ErrorResponse(BaseModel):
    request_id: str
    errors:     List[ErrorDetail]
