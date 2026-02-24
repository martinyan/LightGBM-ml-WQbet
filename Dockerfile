# HKJC production reproduction image (template)
# Goal: run scraper + feature builder + GoldenWinBet/GoldenQbet predictors reproducibly.

FROM node:22-bullseye

# Install Python + build tools (lightgbm wheels usually exist; keep build deps minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests first for better layer caching
COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

# Copy the repo
COPY . /app

# Convenience: verify artifacts at build time is optional (can be slow). Uncomment if desired.
# RUN python3 scripts/verify_artifacts.py

CMD ["bash"]
