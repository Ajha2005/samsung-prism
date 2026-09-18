# PRISM Theme 1 — reproducible CPU image.
#
# Build:
#   docker build -t prism-code-search .
#
# Smoke test (offline, no downloads — proves the pipeline runs end-to-end):
#   docker run --rm prism-code-search
#
# Real leaderboard eval (needs network for the model + CoIR dataset the first time):
#   docker run --rm -v "$PWD/results:/app/results" prism-code-search \
#       python -m prism.cli eval --output results/appsretrieval_results.json
#
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# System deps kept minimal; git is handy for reproducibility metadata.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching.
COPY requirements.txt requirements-eval.txt ./
# CPU-only torch keeps the image small; the rest come from PyPI.
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-eval.txt \
    && pip install pytest==8.3.3

# Copy the project and install it (exposes prism-eval / prism-demo / prism-ablation).
COPY . .
RUN pip install --no-deps -e .

# Default: run the offline test suite + demo so `docker run` verifies the build
# with no network access.
CMD ["bash", "-lc", "python -m pytest -q && python -m prism.cli demo --backend hashing"]
