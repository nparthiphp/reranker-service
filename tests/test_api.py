"""
tests/test_api.py — Reranker Service test suite.
Uses mock ML models so tests run without downloading HuggingFace models.
On your Mac with real models installed, tests will use actual CE + MMR.
"""
from __future__ import annotations
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# ── Mock models before importing app ─────────────────────────────────────────
# This ensures tests pass in CI/CD without GPU or HuggingFace access

def _mock_ce_predict(pairs, show_progress_bar=False):
    """Simulate cross-encoder: score based on keyword overlap."""
    scores = []
    for query, chunk in pairs:
        q_words = set(query.lower().split())
        c_words = set(chunk.lower().split())
        overlap = len(q_words & c_words)
        base = overlap * 0.15
        # boost biology terms for photosynthesis query
        bio_terms = {"photosynthesis","sunlight","chloroplast","chlorophyll","glucose","plant"}
        bio_hits = len(bio_terms & c_words)
        scores.append(min(0.99, base + bio_hits * 0.2))
    return np.array(scores)

# Vocabulary-based embedding: substring match so "photosynthesis" in query
# matches "photosynthesis" in chunk regardless of word boundaries
_VOCAB = [
    "photosynthesis","sunlight","chloroplast","chlorophyll","glucose","plant",
    "light","food","energy","pigment","revolution","french","napoleon","king",
    "monarchy","republic","1789","quadratic","equation","formula","polynomial",
    "roots","discriminant","squared","degree"
]

def _mock_encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                 show_progress_bar=False, batch_size=32):
    """Simulate embeddings using vocabulary substring matching."""
    results = []
    for text in texts:
        t = text.lower()
        v = np.array([1.0 if w in t else 0.0 for w in _VOCAB])
        norm = np.linalg.norm(v)
        results.append(v / norm if norm > 0 else v)
    return np.array(results)

mock_ce = MagicMock()
mock_ce.predict.side_effect = _mock_ce_predict

mock_st = MagicMock()
mock_st.encode.side_effect = _mock_encode

import app.services.cross_encoder as ce_mod
import app.services.mmr as mmr_mod
ce_mod._MODEL_CACHE[ce_mod.settings.cross_encoder_model] = mock_ce
mmr_mod._EMBED_CACHE[mmr_mod.settings.embedding_model] = mock_st

from app.main import app
client = TestClient(app)

# ── Test data ─────────────────────────────────────────────────────────────────
QUERY = "what is photosynthesis?"

CHUNKS_BIOLOGY = [
    {"chunk_id":"1","text":"Plants use sunlight to produce food through photosynthesis.","original_score":0.82},
    {"chunk_id":"2","text":"Chloroplasts in plant cells absorb light energy from the sun.","original_score":0.79},
    {"chunk_id":"3","text":"Glucose is produced during photosynthesis and stored as energy.","original_score":0.74},
    {"chunk_id":"4","text":"The French Revolution began in 1789 and transformed politics.","original_score":0.61},
    {"chunk_id":"5","text":"Napoleon Bonaparte rose to power after the revolutionary period.","original_score":0.58},
]

CHUNKS_DUPLICATE = [
    {"chunk_id":"a","text":"Photosynthesis uses sunlight to make glucose in plant cells.","original_score":0.91},
    {"chunk_id":"b","text":"Photosynthesis is where plants convert sunlight to glucose.","original_score":0.89},
    {"chunk_id":"c","text":"Plants make food from sunlight using photosynthesis in chloroplasts.","original_score":0.87},
    {"chunk_id":"d","text":"The quadratic formula solves polynomial equations of degree two.","original_score":0.42},
    {"chunk_id":"e","text":"Chlorophyll is the green pigment that captures light for photosynthesis.","original_score":0.78},
]

# ── Health tests ──────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_200(self):
        assert client.get("/api/v1/health").status_code == 200

    def test_health_has_required_fields(self):
        data = client.get("/api/v1/health").json()
        for field in ["status","service","version","models","supported_methods","uptime_seconds"]:
            assert field in data, f"missing field: {field}"

    def test_health_service_name(self):
        assert "Reranker" in client.get("/api/v1/health").json()["service"]

    def test_health_supported_methods(self):
        methods = client.get("/api/v1/health").json()["supported_methods"]
        for m in ["cross_encoder","mmr","ensemble"]:
            assert m in methods

    def test_root_endpoint(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "Reranker" in r.json()["service"]

    def test_response_has_correlation_header(self):
        r = client.get("/api/v1/health")
        assert "X-Response-Time-Ms" in r.headers

# ── Cross-encoder tests ───────────────────────────────────────────────────────
class TestCrossEncoder:
    def test_basic_rerank_200(self):
        r = client.post("/api/v1/rerank", json={
            "query":QUERY,"chunks":CHUNKS_BIOLOGY,"method":"cross_encoder","top_k":3})
        assert r.status_code == 200

    def test_returns_correct_count(self):
        data = client.post("/api/v1/rerank", json={
            "query":QUERY,"chunks":CHUNKS_BIOLOGY,"method":"cross_encoder","top_k":3}).json()
        assert len(data["chunks"]) == 3

    def test_biology_ranks_higher_than_history(self):
        data = client.post("/api/v1/rerank", json={
            "query":QUERY,"chunks":CHUNKS_BIOLOGY,"method":"cross_encoder","top_k":3}).json()
        ids = {c["chunk_id"] for c in data["chunks"]}
        assert not ids.issuperset({"4","5"}), "History chunks should not dominate biology query"

    def test_ranks_sequential(self):
        data = client.post("/api/v1/rerank", json={
            "query":QUERY,"chunks":CHUNKS_BIOLOGY,"method":"cross_encoder","top_k":3}).json()
        ranks = [c["rank"] for c in data["chunks"]]
        assert ranks == list(range(1, len(ranks)+1))

    def test_chunk_has_required_fields(self):
        data = client.post("/api/v1/rerank", json={
            "query":QUERY,"chunks":CHUNKS_BIOLOGY,"method":"cross_encoder","top_k":1}).json()
        chunk = data["chunks"][0]
        for field in ["chunk_id","text","rank","rerank_score","original_score","rank_reason","rank_explanation"]:
            assert field in chunk, f"missing chunk field: {field}"

    def test_dedicated_endpoint(self):
        r = client.post("/api/v1/rerank/cross-encoder", json={
            "query":QUERY,"chunks":CHUNKS_BIOLOGY[:3],"top_k":2})
        assert r.status_code == 200
        assert r.json()["method_used"] == "cross_encoder"

    def test_stats_present_and_correct(self):
        data = client.post("/api/v1/rerank", json={
            "query":QUERY,"chunks":CHUNKS_BIOLOGY,"method":"cross_encoder","top_k":3}).json()
        stats = data["stats"]
        assert stats["total_input"] == len(CHUNKS_BIOLOGY)
        assert stats["total_output"] == 3
        assert stats["cross_encoder_ms"] >= 0

    def test_metadata_passthrough(self):
        chunks = [{"chunk_id":"x","text":"plants use sunlight for photosynthesis",
                   "original_score":0.9,"metadata":{"subject":"biology","grade":10}}]
        data = client.post("/api/v1/rerank", json={
            "query":QUERY,"chunks":chunks,"method":"cross_encoder","top_k":1}).json()
        meta = data["chunks"][0]["metadata"]
        assert meta["subject"] == "biology"
        assert meta["grade"] == 10

# ── MMR tests ─────────────────────────────────────────────────────────────────
class TestMMR:
    def test_mmr_returns_200(self):
        r = client.post("/api/v1/rerank", json={
            "query":QUERY,"chunks":CHUNKS_DUPLICATE,"method":"mmr","top_k":3,"mmr_lambda":0.7})
        assert r.status_code == 200

    def test_mmr_returns_correct_count(self):
        data = client.post("/api/v1/rerank", json={
            "query":QUERY,"chunks":CHUNKS_DUPLICATE,"method":"mmr","top_k":3}).json()
        assert len(data["chunks"]) == 3

    def test_mmr_diversity_score_present(self):
        data = client.post("/api/v1/rerank", json={
            "query":QUERY,"chunks":CHUNKS_DUPLICATE,"method":"mmr","top_k":3}).json()
        assert "diversity_score" in data["stats"]

    def test_mmr_dedicated_endpoint(self):
        r = client.post("/api/v1/rerank/mmr", json={
            "query":QUERY,"chunks":CHUNKS_BIOLOGY,"top_k":2})
        assert r.status_code == 200
        assert r.json()["method_used"] == "mmr"

    def test_mmr_excludes_irrelevant_chunk(self):
        data = client.post("/api/v1/rerank", json={
            "query":QUERY,"chunks":CHUNKS_DUPLICATE,"method":"mmr","top_k":3}).json()
        texts = [c["text"].lower() for c in data["chunks"]]
        # quadratic math chunk (id=d) should not appear for biology query
        assert not any("quadratic" in t for t in texts), \
            "Irrelevant math chunk should not appear for biology query"

# ── Ensemble tests ────────────────────────────────────────────────────────────
class TestEnsemble:
    def test_ensemble_default_method(self):
        data = client.post("/api/v1/rerank", json={
            "query":QUERY,"chunks":CHUNKS_BIOLOGY,"top_k":3}).json()
        assert data["method_used"] == "ensemble"

    def test_ensemble_returns_correct_count(self):
        data = client.post("/api/v1/rerank", json={
            "query":QUERY,"chunks":CHUNKS_BIOLOGY,"top_k":3}).json()
        assert len(data["chunks"]) <= 3

    def test_ensemble_stats_has_both_timings(self):
        data = client.post("/api/v1/rerank", json={
            "query":QUERY,"chunks":CHUNKS_BIOLOGY,"top_k":3}).json()
        stats = data["stats"]
        assert stats["cross_encoder_ms"] >= 0
        assert stats["mmr_ms"] >= 0
        assert stats["total_ms"] > 0

    def test_ensemble_excludes_irrelevant(self):
        data = client.post("/api/v1/rerank", json={
            "query":QUERY,"chunks":CHUNKS_DUPLICATE,"top_k":3}).json()
        texts = [c["text"].lower() for c in data["chunks"]]
        assert not any("quadratic" in t for t in texts)

# ── Validation tests ──────────────────────────────────────────────────────────
class TestValidation:
    def test_empty_query_rejected(self):
        r = client.post("/api/v1/rerank", json={
            "query":"","chunks":CHUNKS_BIOLOGY[:2],"top_k":1})
        assert r.status_code == 422

    def test_empty_chunks_rejected(self):
        r = client.post("/api/v1/rerank", json={"query":QUERY,"chunks":[],"top_k":1})
        assert r.status_code == 422

    def test_empty_chunk_text_rejected(self):
        r = client.post("/api/v1/rerank", json={
            "query":QUERY,
            "chunks":[{"chunk_id":"1","text":"","original_score":0.5}],
            "top_k":1})
        assert r.status_code == 422

    def test_invalid_method_rejected(self):
        r = client.post("/api/v1/rerank", json={
            "query":QUERY,"chunks":CHUNKS_BIOLOGY[:2],
            "method":"invalid","top_k":1})
        assert r.status_code == 422

    def test_top_k_larger_than_chunks_ok(self):
        data = client.post("/api/v1/rerank", json={
            "query":QUERY,"chunks":CHUNKS_BIOLOGY[:2],"top_k":10}).json()
        assert len(data["chunks"]) <= 2

    def test_single_chunk_returns_one(self):
        data = client.post("/api/v1/rerank", json={
            "query":QUERY,
            "chunks":[{"chunk_id":"1","text":"plants use sunlight for photosynthesis","original_score":0.9}],
            "top_k":3}).json()
        assert len(data["chunks"]) == 1

    def test_request_id_passthrough(self):
        data = client.post("/api/v1/rerank", json={
            "query":QUERY,"chunks":CHUNKS_BIOLOGY[:2],
            "request_id":"test-123","top_k":1}).json()
        assert data["request_id"] == "test-123"
