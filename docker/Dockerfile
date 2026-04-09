# ─────────────────────────────────────────────────────────────────────────────
# PhysicalAI — Inference / Training Image
#
# Base: CUDA 12.1 + cuDNN 8 (matches PyTorch 2.2 requirements)
# Python: 3.10 (PyTorch wheels available, stable for ML)
#
# Build:
#   docker build -f docker/Dockerfile -t physicalai:latest .
#
# Run inference:
#   docker run --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface \
#     physicalai:latest python scripts/evaluate.py --help
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3.10-dev \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python/pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python -m pip install --upgrade pip --quiet

# ── PyTorch (pinned to CUDA 12.1) ─────────────────────────────────────────────
# Install before the rest of physicalai deps so layer is cached separately
RUN pip install --quiet \
    torch==2.2.2 \
    torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Copy and install physicalai ───────────────────────────────────────────────
WORKDIR /workspace
COPY pyproject.toml ./
COPY src/ ./src/

# Install without torch/torchvision (already installed above)
RUN pip install --quiet -e . --no-deps && \
    pip install --quiet \
        numpy>=1.26 \
        einops>=0.7 \
        transformers>=4.40 \
        hydra-core>=1.3 \
        omegaconf>=2.3 \
        wandb>=0.16 \
        tqdm>=4.66 \
        Pillow>=10.0 \
        pyyaml>=6.0 \
        accelerate>=0.27

# Copy remaining project files (after deps — preserves layer cache)
COPY . .

# ── HuggingFace cache ─────────────────────────────────────────────────────────
# Mount ~/.cache/huggingface as a volume to persist model weights across runs
ENV HF_HOME=/root/.cache/huggingface
VOLUME ["/root/.cache/huggingface"]

# ── Default command ───────────────────────────────────────────────────────────
CMD ["python", "scripts/evaluate.py", "--help"]
