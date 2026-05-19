# 🎯 Reranker Service

> **Enterprise-grade standalone reranking microservice** — takes retrieved candidates from any vector store and returns precision-ranked results using Cross-Encoder, MMR diversity filtering, and score fusion. Plugs into any RAG pipeline via a single HTTP call.

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-passing-brightgreen)](#testing)

---

## 🧠 What it does

Takes top-20 candidates from your vector search → returns the best 3, precisely ranked.

```
Your RAG system                    Reranker Service
─────────────────                  ────────────────────────────────────
Qdrant top 20 candidates  ──────►  Cross-Encoder  scores each pair
BM25 top 20 candidates    ──────►  MMR            removes duplicates
                                   Score Fusion   combines signals
                          ◄──────  Top 3 ranked chunks + scores + reasons
```

---

## 🚀 Quick start

```bash
# 1. Clone
git clone https://github.com/nparthiphp/reranker-service.git
cd reranker-service

# 2. Install
pip install -r requirements.txt

# 3. Run
uvicorn app.main:app --reload --port 8001

# 4. Test
curl -X POST http://localhost:8001/api/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is photosynthesis?",
    "chunks": [
      {"chunk_id": "1", "text": "Plants use sunlight to make food via photosynthesis.", "original_score": 0.82},
      {"chunk_id": "2", "text": "The French Revolution began in 1789.", "original_score": 0.71},
      {"chunk_id": "3", "text": "Chloroplasts absorb light energy in plant cells.", "original_score": 0.79}
    ],
    "method": "ensemble",
    "top_k": 2
  }'
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/rerank` | Rerank candidates — main endpoint |
| `POST` | `/api/v1/rerank/cross-encoder` | Cross-encoder only |
| `POST` | `/api/v1/rerank/mmr` | MMR diversity only |
| `GET`  | `/api/v1/health` | Service health + model status |
| `GET`  | `/api/v1/metrics` | Prometheus metrics |
| `GET`  | `/docs` | Swagger UI |

---

## 🔧 Reranking methods

| Method | How it works | Best for | Latency |
|--------|-------------|----------|---------|
| `cross_encoder` | BERT reads query+chunk together → relevance score | General purpose | ~100ms/20 |
| `mmr` | Penalises duplicate chunks → diversity | Long documents | ~20ms/20 |
| `ensemble` | Cross-encoder then MMR (default) | Production RAG | ~120ms/20 |

---

## ⚙️ Configuration

```env
# .env
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
EMBEDDING_MODEL=multi-qa-MiniLM-L6-cos-v1
DEFAULT_TOP_K=3
DEFAULT_MMR_LAMBDA=0.7
API_KEY=your-secret-key          # optional auth
RATE_LIMIT_PER_MINUTE=100
LOG_LEVEL=INFO
```

### Model options

| Model | Size | MRR@10 | Speed |
|-------|------|--------|-------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 67MB | 0.323 | ⚡ fastest |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 130MB | 0.332 | fast |
| `cross-encoder/ms-marco-electra-base` | 420MB | 0.341 | medium |

---

## 🔌 Integrate with EduRAG / DocChunker

```python
import httpx

async def rerank(query: str, chunks: list) -> list:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://reranker-service:8001/api/v1/rerank",
            json={
                "query": query,
                "chunks": chunks,
                "method": "ensemble",
                "top_k": 3
            },
            headers={"X-API-Key": "your-key"},
            timeout=5.0,
        )
        return response.json()["chunks"]
```

---

## 🏗️ Architecture

```
reranker-service/
├── app/
│   ├── main.py                        ← FastAPI entry point
│   ├── api/
│   │   └── rerank_router.py           ← All reranking endpoints
│   ├── core/
│   │   └── settings.py                ← Config via .env
│   ├── middleware/
│   │   ├── auth.py                    ← API key authentication
│   │   ├── rate_limiter.py            ← Per-client rate limiting
│   │   └── logging_middleware.py      ← Request/response logging
│   ├── models/
│   │   └── schemas.py                 ← Pydantic request/response models
│   └── services/
│       ├── cross_encoder.py           ← Cross-encoder reranking
│       ├── mmr.py                     ← MMR diversity filter
│       └── ensemble.py                ← Combines CE + MMR
├── tests/
│   ├── test_cross_encoder.py
│   ├── test_mmr.py
│   └── test_api.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## 🐳 Docker

```bash
# Single container
docker build -t reranker-service .
docker run -p 8001:8001 reranker-service

# With docker-compose (includes Redis for rate limiting)
docker-compose up
```

---

## 🧪 Testing

```bash
pytest tests/ -v --cov=app
```

---

## 📊 Performance

| Chunks | Cross-encoder | MMR | Ensemble |
|--------|--------------|-----|----------|
| 10     | ~50ms        | ~5ms | ~55ms  |
| 20     | ~100ms       | ~10ms | ~110ms |
| 50     | ~250ms       | ~20ms | ~270ms |

---

## 👤 Author

**Parthiban Natarajan** — General Manager / Solution Architect  
Jio Platforms · 20+ years · Agentic AI · RAG · LangGraph  
[LinkedIn](https://linkedin.com/in/parthibannatarajan) · [GitHub](https://github.com/nparthiphp)
