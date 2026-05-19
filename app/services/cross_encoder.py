"""
app/services/cross_encoder.py
-------------------------------
Cross-encoder reranking service.

Architecture:
  Input:  [(query, chunk1), (query, chunk2), ...]   ← 20 pairs typically
  Model:  BERT reads BOTH query and chunk together
          every Q token attends to every chunk token
  Output: relevance score 0-1 per pair

Why more accurate than bi-encoder:
  Bi-encoder: encodes separately → misses token interactions
  Cross-encoder: reads together → "theorem" in Q weights
                 "5.2" in chunk → exact match detected

Training: ms-marco dataset — 1M Bing queries with human relevance labels
Model size: 67MB (MiniLM-L-6) | 130MB (MiniLM-L-12) | 420MB (electra)
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

from app.core.settings import get_settings
from app.models.schemas import (
    CandidateChunk, RankedChunk, RankReason,
    SimilarityInfo
)

settings = get_settings()

# ── Singleton model cache ──────────────────────────────────────────────────────
_MODEL_CACHE: Dict = {}
_LOAD_TIME_MS: float = 0.0


def get_model():
    """Load cross-encoder once at startup, reuse for all requests."""
    global _LOAD_TIME_MS
    model_name = settings.cross_encoder_model
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import CrossEncoder
        t0 = time.perf_counter()
        _MODEL_CACHE[model_name] = CrossEncoder(
            model_name,
            max_length=settings.cross_encoder_max_length,
        )
        _LOAD_TIME_MS = round((time.perf_counter() - t0) * 1000, 2)
    return _MODEL_CACHE[model_name]


def is_loaded() -> bool:
    return settings.cross_encoder_model in _MODEL_CACHE


def warmup():
    """Pre-warm model at startup — prevents cold start on first request."""
    model = get_model()
    # Run one dummy pair to force weight loading into memory
    model.predict([("warmup query", "warmup chunk")], show_progress_bar=False)


# ── Core scoring ───────────────────────────────────────────────────────────────

def _score_pairs(
    query: str,
    chunks: List[CandidateChunk],
) -> List[Tuple[CandidateChunk, float]]:
    """
    Score all (query, chunk) pairs with cross-encoder.
    Returns [(chunk, score), ...] unsorted.
    """
    model = get_model()
    pairs = [
        (query, chunk.text[:settings.cross_encoder_max_length * 4])
        for chunk in chunks
    ]
    scores = model.predict(pairs, show_progress_bar=False)
    return [(chunk, float(score)) for chunk, score in zip(chunks, scores)]


def _classify_reason(score: float) -> Tuple[RankReason, str]:
    """Return reason enum and human explanation for a score."""
    if score >= 0.8:
        return (
            RankReason.HIGH_RELEVANCE,
            f"Cross-encoder score {score:.3f} — query directly answered by this chunk. "
            "BERT detected strong token-level match between question and content."
        )
    elif score >= 0.5:
        return (
            RankReason.MODERATE_RELEVANCE,
            f"Cross-encoder score {score:.3f} — chunk is relevant to the topic "
            "but may not fully answer the question. Good supporting context."
        )
    elif score >= 0.3:
        return (
            RankReason.LOW_RELEVANCE,
            f"Cross-encoder score {score:.3f} — related topic but indirect answer. "
            "Included as supplementary context."
        )
    else:
        return (
            RankReason.BEST_AVAILABLE,
            f"Cross-encoder score {score:.3f} — all candidates have low relevance. "
            "This is the best available match in the retrieved set."
        )


# ── Public API ─────────────────────────────────────────────────────────────────

def rerank(
    query: str,
    chunks: List[CandidateChunk],
    top_k: int = 3,
    min_score: float = 0.0,
) -> Tuple[List[RankedChunk], float]:
    """
    Rerank chunks using cross-encoder.

    Parameters
    ----------
    query      : user question
    chunks     : candidates from vector search (typically 20)
    top_k      : how many to return
    min_score  : drop chunks below this threshold

    Returns
    -------
    (ranked_chunks, elapsed_ms)
    """
    if not chunks:
        return [], 0.0

    t0 = time.perf_counter()

    # Score all pairs
    scored = _score_pairs(query, chunks)

    # Filter by minimum score
    scored = [(c, s) for c, s in scored if s >= min_score]

    # Sort descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Build RankedChunk objects
    results: List[RankedChunk] = []
    for rank, (chunk, score) in enumerate(scored[:top_k], start=1):
        reason, explanation = _classify_reason(score)
        delta = round(score - chunk.original_score, 4)
        results.append(RankedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            metadata=chunk.metadata,
            rank=rank,
            original_score=chunk.original_score,
            rerank_score=round(score, 4),
            score_delta=delta,
            rank_reason=reason,
            rank_explanation=explanation,
            similarity_info=SimilarityInfo(
                cross_encoder_score=round(score, 4),
            ),
        ))

    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    return results, elapsed
