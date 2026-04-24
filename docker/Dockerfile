# ─────────────────────────────────────────────────────────────────────────────
# PhysicalAI — Inference / Training Image
#
# Base: CUDA 12.1 + cuDNN 8 (matches PyTorch 2.2 requirements)
# Python: 3.10 (PyTorch wheels available, stable for ML)
#
# Build:
#   docker build -f docker/Dockerfile -t physicalai:latest .
#
# Run inference (GPU):
#   docker run --gpus all \
#     -v ~/.cache/huggingface:/home/appuser/.cache/huggingface \
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
    python -m pip install --upgrade pip

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd -m -u 1000 appuser
WORKDIR /workspace
RUN chown appuser:appuser /workspace

# ── PyTorch (pinned to CUDA 12.1) — heavy layer, cached separately ────────────
# Install as root before switching user so pip cache is shared
RUN pip install \
    torch==2.2.2 \
    torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Copy and install physicalai ───────────────────────────────────────────────
# Copy only dependency files first — preserves cache when only source changes
COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser src/ ./src/

# pip sees torch already satisfies >=2.2 and skips it — no manual dep listing
RUN pip install -e ".[inference]"

# Copy remaining project files (scripts, configs, etc.)
COPY --chown=appuser:appuser . .

# ── Switch to non-root user ───────────────────────────────────────────────────
USER appuser

# ── HuggingFace cache ─────────────────────────────────────────────────────────
ENV HF_HOME=/home/appuser/.cache/huggingface
VOLUME ["/home/appuser/.cache/huggingface"]

# ── Entrypoint: override CMD at runtime to switch between inference/training ──
# Inference: docker run physicalai:latest scripts/evaluate.py --config configs/model/openvla.yaml
# Training:  docker run physicalai:latest scripts/train.py --config configs/training/openvla_lora.yaml
ENTRYPOINT ["python"]
CMD ["scripts/evaluate.py", "--help"]
