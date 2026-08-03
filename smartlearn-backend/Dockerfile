FROM python:3.11-slim

WORKDIR /app

# OpenBLAS threading lib needed by torch/sentence-transformers at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the embedding model at BUILD time. The build environment can
# reach HuggingFace, and baking the snapshot into the image means the first
# upload at runtime does not stall on a ~90MB download (which used to exceed
# the gateway request timeout and return 502).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'use_safetensors': False})"

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
