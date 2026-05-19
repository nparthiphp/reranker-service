"""
app/api/rerank_router.py
------------------------
All reranking endpoints.

POST /api/v1/rerank               ← main endpoint (ensemble by default)
POST /api/v1/rerank/cross-encoder ← cross-encoder only
POST /api/v1/rerank/mmr           ← MMR diversity only
"""
from __future__ import annotations
import time
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from app.middleware.auth import verify_api_key
from app.models.schemas import RerankMethod, RerankRequest, RerankResponse
import app.services.ensemble as ensemble_service

router = APIRouter(prefix="/rerank", tags=["rerank"])


def _dispatch(req: RerankRequest) -> RerankResponse:
    """Route to correct service based on method."""
    if req.method == RerankMethod.CROSS_ENCODER:
        return ensemble_service.run_cross_encoder_only(
            query=req.query, chunks=req.chunks,
            top_k=req.top_k, min_score=req.min_score,
            request_id=req.request_id,
        )
    elif req.method == RerankMethod.MMR:
        return ensemble_service.run_mmr_only(
            query=req.query, chunks=req.chunks,
            top_k=req.top_k, mmr_lambda=req.mmr_lambda,
            min_score=req.min_score, request_id=req.request_id,
        )
    else:  # ENSEMBLE (default)
        return ensemble_service.run_ensemble(
            query=req.query, chunks=req.chunks,
            top_k=req.top_k, mmr_lambda=req.mmr_lambda,
            min_score=req.min_score, request_id=req.request_id,
        )


@router.post(
    "",
    response_model=RerankResponse,
    summary="Rerank candidates — main endpoint",
    description=(
        "Takes top-N candidates from your vector search and returns "
        "top_k precisely ranked chunks.\n\n"
        "**Methods:**\n"
        "- `ensemble` (default) — cross-encoder accuracy + MMR diversity\n"
        "- `cross_encoder` — BERT reads query+chunk together → score\n"
        "- `mmr` — maximise relevance while minimising duplicate content\n\n"
        "**Typical usage:** Send top-20 from Qdrant hybrid search, get back top-3."
    ),
)
async def rerank(
    request: RerankRequest,
    _auth:   Annotated[str, Depends(verify_api_key)],
) -> RerankResponse:
    try:
        return _dispatch(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reranking failed: {exc}",
        )


@router.post(
    "/cross-encoder",
    response_model=RerankResponse,
    summary="Cross-encoder reranking only",
    description="Forces cross_encoder method regardless of request body method field.",
)
async def rerank_cross_encoder(
    request: RerankRequest,
    _auth:   Annotated[str, Depends(verify_api_key)],
) -> RerankResponse:
    request.method = RerankMethod.CROSS_ENCODER
    try:
        return _dispatch(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cross-encoder reranking failed: {exc}",
        )


@router.post(
    "/mmr",
    response_model=RerankResponse,
    summary="MMR diversity reranking only",
    description=(
        "Forces mmr method. Use when you want to maximise coverage "
        "and avoid duplicate content in LLM context. "
        "Tune mmr_lambda: 0.7 = 70% relevance + 30% diversity (recommended)."
    ),
)
async def rerank_mmr(
    request: RerankRequest,
    _auth:   Annotated[str, Depends(verify_api_key)],
) -> RerankResponse:
    request.method = RerankMethod.MMR
    try:
        return _dispatch(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MMR reranking failed: {exc}",
        )
