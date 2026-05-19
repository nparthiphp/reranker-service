"""
app/services/ensemble.py
-------------------------
Ensemble pipeline — Cross-Encoder then MMR.

Stage 1 — Cross-Encoder:
  Input:  top-N candidates from vector search
  Action: scores each (query, chunk) pair → precise relevance
  Output: top-10 by CE score (eliminates clearly irrelevant)

Stage 2 — MMR:
  Input:  top-10 from CE
  Action: removes near-duplicate chunks while preserving relevance
  Output: top-k diverse, relevant chunks for LLM

Why this order matters:
  CE first: eliminates off-topic chunks → MMR only sees relevant ones
  MMR second: adds diversity to already-relevant set

  If MMR runs first on raw candidates:
    may select diverse but irrelevant chunks
  If CE runs first:
    MMR selects from already-validated relevant set → much better
"""
from __future__ import annotations

import time
from typing import List, Tuple

from app.core.settings import get_settings
from app.models.schemas import (
    CandidateChunk, RankedChunk, RerankMethod,
    RerankStats, RerankResponse
)
import app.services.cross_encoder as ce_service
import app.services.mmr as mmr_service

settings = get_settings()


def _compute_score_improvement(
    chunks:  List[CandidateChunk],
    results: List[RankedChunk],
) -> float:
    """Calculate average improvement from reranking."""
    if not results:
        return 0.0
    orig_avg   = sum(c.original_score for c in chunks) / len(chunks)
    rerank_avg = sum(r.rerank_score for r in results) / len(results)
    return round(rerank_avg - orig_avg, 4)


def run_ensemble(
    query:      str,
    chunks:     List[CandidateChunk],
    top_k:      int   = 3,
    mmr_lambda: float = 0.7,
    min_score:  float = 0.0,
    request_id: str   = "",
) -> RerankResponse:
    """
    Full ensemble pipeline: cross-encoder → MMR.

    Steps:
      1. Cross-encoder scores all N chunks → sort → keep top-10
      2. MMR selects top_k diverse chunks from those 10
      3. Return with full metadata

    Parameters
    ----------
    query      : user question
    chunks     : all candidates (typically 20 from hybrid search)
    top_k      : final count to return (default 3)
    mmr_lambda : diversity tuning (0.7 = recommended)
    min_score  : filter threshold
    """
    total_start = time.perf_counter()

    # ── Stage 1: Cross-encoder ────────────────────────────────────────────────
    # Score all chunks, keep top min(10, len(chunks)) for MMR
    ce_top_k   = min(10, len(chunks))
    ce_results, ce_ms = ce_service.rerank(
        query=query,
        chunks=chunks,
        top_k=ce_top_k,
        min_score=min_score,
    )

    # Convert CE results back to CandidateChunk for MMR input
    # Use CE rerank_score as the new original_score
    ce_as_candidates = [
        CandidateChunk(
            chunk_id=r.chunk_id,
            text=r.text,
            metadata=r.metadata,
            original_score=r.rerank_score,  # CE score becomes input to MMR
        )
        for r in ce_results
    ]

    # ── Stage 2: MMR diversity filter ─────────────────────────────────────────
    mmr_results, mmr_ms, diversity_score = mmr_service.rerank(
        query=query,
        chunks=ce_as_candidates,
        top_k=top_k,
        lambda_=mmr_lambda,
        min_score=0.0,   # CE already filtered — don't double-filter
    )

    # Fix rank numbers after MMR reorder
    for i, chunk in enumerate(mmr_results, start=1):
        chunk.rank = i

    total_ms   = round((time.perf_counter() - total_start) * 1000, 2)
    filtered   = len(chunks) - len(ce_results)
    improvement = _compute_score_improvement(chunks, mmr_results)

    stats = RerankStats(
        total_input=len(chunks),
        total_output=len(mmr_results),
        chunks_filtered=filtered,
        cross_encoder_ms=ce_ms,
        mmr_ms=mmr_ms,
        total_ms=total_ms,
        score_improvement=improvement,
        diversity_score=diversity_score,
    )

    return RerankResponse(
        request_id=request_id,
        query=query,
        method_used=RerankMethod.ENSEMBLE,
        stats=stats,
        chunks=mmr_results,
    )


def run_cross_encoder_only(
    query:      str,
    chunks:     List[CandidateChunk],
    top_k:      int   = 3,
    min_score:  float = 0.0,
    request_id: str   = "",
) -> RerankResponse:
    """Cross-encoder only — no MMR diversity step."""
    results, ce_ms = ce_service.rerank(
        query=query, chunks=chunks,
        top_k=top_k, min_score=min_score,
    )
    stats = RerankStats(
        total_input=len(chunks),
        total_output=len(results),
        chunks_filtered=len(chunks) - len(results),
        cross_encoder_ms=ce_ms,
        mmr_ms=0.0,
        total_ms=ce_ms,
        score_improvement=_compute_score_improvement(chunks, results),
        diversity_score=0.0,
    )
    return RerankResponse(
        request_id=request_id, query=query,
        method_used=RerankMethod.CROSS_ENCODER,
        stats=stats, chunks=results,
    )


def run_mmr_only(
    query:      str,
    chunks:     List[CandidateChunk],
    top_k:      int   = 3,
    mmr_lambda: float = 0.7,
    min_score:  float = 0.0,
    request_id: str   = "",
) -> RerankResponse:
    """MMR only — uses bi-encoder similarity (no cross-encoder)."""
    results, mmr_ms, diversity_score = mmr_service.rerank(
        query=query, chunks=chunks, top_k=top_k,
        lambda_=mmr_lambda, min_score=min_score,
    )
    stats = RerankStats(
        total_input=len(chunks),
        total_output=len(results),
        chunks_filtered=len(chunks) - len(results),
        cross_encoder_ms=0.0,
        mmr_ms=mmr_ms,
        total_ms=mmr_ms,
        score_improvement=_compute_score_improvement(chunks, results),
        diversity_score=diversity_score,
    )
    return RerankResponse(
        request_id=request_id, query=query,
        method_used=RerankMethod.MMR,
        stats=stats, chunks=results,
    )
