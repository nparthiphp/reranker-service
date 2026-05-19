"""
app/services/mmr.py
--------------------
Maximum Marginal Relevance (MMR) diversity filter.

Problem it solves:
  Cross-encoder returns top 3 chunks — but if all 3 come from the
  same textbook page, LLM gets duplicate info → repetitive answer.

How MMR works:
  Iteratively selects chunks that are:
    - Relevant to the query          (high cosine sim to query)
    - Different from already selected (low cosine sim to selected)

  score(chunk) = λ × sim(query, chunk)
               - (1-λ) × max_sim(chunk, already_selected)

  λ = 0.7 → 70% relevance + 30% diversity (recommended for RAG)
  λ = 1.0 → pure relevance (same as top-k)
  λ = 0.0 → pure diversity (maximize coverage)

Example:
  CE top 5: [chunk_A, chunk_B, chunk_C, chunk_D, chunk_E]
  chunk_A: photosynthesis intro    sim_query=0.95
  chunk_B: photosynthesis detail   sim_query=0.91  sim_A=0.94 (near duplicate)
  chunk_C: chlorophyll             sim_query=0.82  sim_A=0.60 (different aspect)

  MMR selects: A (best relevance) → C (relevant AND different) → D
  Not:         A → B (too similar to A → repetitive)
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.settings import get_settings
from app.models.schemas import (
    CandidateChunk, RankedChunk, RankReason, SimilarityInfo
)

settings = get_settings()

# ── Embedding model cache ──────────────────────────────────────────────────────
_EMBED_CACHE: Dict = {}


def get_embedding_model():
    model_name = settings.embedding_model
    if model_name not in _EMBED_CACHE:
        from sentence_transformers import SentenceTransformer
        _EMBED_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBED_CACHE[model_name]


def is_loaded() -> bool:
    return settings.embedding_model in _EMBED_CACHE


def warmup():
    model = get_embedding_model()
    model.encode(["warmup"], convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=False)


# ── Cosine similarity ──────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))   # already L2-normalised → dot = cosine


def _diversity_score(result_vecs: List[np.ndarray]) -> float:
    """Avg pairwise distance in selected set — measures diversity quality."""
    if len(result_vecs) < 2:
        return 1.0
    sims = []
    for i in range(len(result_vecs)):
        for j in range(i + 1, len(result_vecs)):
            sims.append(_cosine(result_vecs[i], result_vecs[j]))
    return round(1.0 - (sum(sims) / len(sims)), 4)


# ── Core MMR algorithm ─────────────────────────────────────────────────────────

def _mmr_select(
    query_vec:   np.ndarray,
    chunk_vecs:  List[np.ndarray],
    chunks:      List[CandidateChunk],
    top_k:       int,
    lambda_:     float,
) -> List[Tuple[int, float, float]]:
    """
    MMR selection algorithm.

    Returns list of (original_index, mmr_score, sim_to_selected)
    for the top_k selected chunks.
    """
    remaining_idxs = list(range(len(chunks)))
    selected_idxs: List[int] = []
    selected_vecs: List[np.ndarray] = []
    scores: List[Tuple[int, float, float]] = []

    for _ in range(min(top_k, len(remaining_idxs))):
        best_idx  = -1
        best_mmr  = float("-inf")
        best_sim_selected = 0.0

        for idx in remaining_idxs:
            # Relevance: how similar to query
            sim_query = _cosine(query_vec, chunk_vecs[idx])

            # Redundancy: how similar to already-selected chunks
            sim_selected = 0.0
            if selected_vecs:
                sim_selected = max(
                    _cosine(chunk_vecs[idx], sel_vec)
                    for sel_vec in selected_vecs
                )

            # MMR score
            mmr = lambda_ * sim_query - (1 - lambda_) * sim_selected

            if mmr > best_mmr:
                best_mmr         = mmr
                best_idx         = idx
                best_sim_selected = sim_selected

        if best_idx == -1:
            break

        selected_idxs.append(best_idx)
        selected_vecs.append(chunk_vecs[best_idx])
        remaining_idxs.remove(best_idx)
        scores.append((best_idx, round(best_mmr, 4), round(best_sim_selected, 4)))

    return scores


# ── Public API ─────────────────────────────────────────────────────────────────

def rerank(
    query:      str,
    chunks:     List[CandidateChunk],
    top_k:      int   = 3,
    lambda_:    float = 0.7,
    min_score:  float = 0.0,
) -> Tuple[List[RankedChunk], float, float]:
    """
    Rerank chunks using Maximum Marginal Relevance.

    Parameters
    ----------
    query    : user question
    chunks   : candidates (already cross-encoder scored ideally)
    top_k    : how many to return
    lambda_  : 0.7 = 70% relevance + 30% diversity
    min_score: drop chunks below this original_score

    Returns
    -------
    (ranked_chunks, elapsed_ms, diversity_score)
    """
    if not chunks:
        return [], 0.0, 0.0

    t0    = time.perf_counter()
    model = get_embedding_model()

    # Filter by min score
    filtered = [c for c in chunks if c.original_score >= min_score]
    if not filtered:
        filtered = chunks   # fallback — use all if all below threshold

    # Embed query and all chunks
    texts_to_embed = [query] + [c.text[:512] for c in filtered]
    embeddings = model.encode(
        texts_to_embed,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )

    query_vec  = embeddings[0]
    chunk_vecs = list(embeddings[1:])

    # Run MMR selection
    selected = _mmr_select(query_vec, chunk_vecs, filtered, top_k, lambda_)

    # Build RankedChunk objects
    selected_vecs_for_diversity = [chunk_vecs[idx] for idx, _, _ in selected]
    div_score = _diversity_score(selected_vecs_for_diversity)

    results: List[RankedChunk] = []
    for rank, (orig_idx, mmr_score, sim_selected) in enumerate(selected, start=1):
        chunk  = filtered[orig_idx]
        sim_q  = round(_cosine(query_vec, chunk_vecs[orig_idx]), 4)

        if sim_selected > 0.85:
            reason = RankReason.DIVERSITY_SELECTED
            explanation = (
                f"MMR score {mmr_score:.3f} — selected for diversity. "
                f"Similarity to query {sim_q:.3f} but provides different "
                f"information from other selected chunks (max overlap {sim_selected:.3f})."
            )
        elif sim_q >= 0.7:
            reason = RankReason.HIGH_RELEVANCE
            explanation = (
                f"MMR score {mmr_score:.3f} — high query relevance ({sim_q:.3f}) "
                f"with acceptable diversity (max overlap {sim_selected:.3f})."
            )
        else:
            reason = RankReason.MODERATE_RELEVANCE
            explanation = (
                f"MMR score {mmr_score:.3f} — moderate relevance ({sim_q:.3f}), "
                f"selected to provide coverage of topic."
            )

        results.append(RankedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            metadata=chunk.metadata,
            rank=rank,
            original_score=chunk.original_score,
            rerank_score=mmr_score,
            score_delta=round(mmr_score - chunk.original_score, 4),
            rank_reason=reason,
            rank_explanation=explanation,
            similarity_info=SimilarityInfo(
                mmr_score=mmr_score,
                similarity_to_query=sim_q,
                similarity_to_selected=sim_selected,
            ),
        ))

    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    return results, elapsed, div_score
