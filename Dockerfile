FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models at build time (faster container startup)
RUN python3 -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512); \
SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')"

COPY app/ ./app/

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8001/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
