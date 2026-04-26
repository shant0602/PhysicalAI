#!/bin/bash
# One-shot Lambda Labs setup and training launcher (Docker-based).
# Run this on a fresh Lambda A100 instance:
#   bash scripts/setup_lambda.sh
#
# Or pipe directly from GitHub:
#   REPO_URL=https://github.com/... WANDB_KEY=... WANDB_ENTITY=... \
#   bash <(curl -s https://raw.githubusercontent.com/.../scripts/setup_lambda.sh)
set -euo pipefail

REPO_URL="${REPO_URL:-}"         # required if not already inside the repo
WANDB_KEY="${WANDB_KEY:-}"       # W&B API key (or leave blank to login interactively)
WANDB_ENTITY="${WANDB_ENTITY:-}" # W&B username/team — REQUIRED for training
DATASET="${DATASET:-libero}"     # 'libero' (~10GB) or 'bridge_orig' (~200GB)

echo "=== PhysicalAI — Lambda Setup (Docker) ==="

# 1. Clone repo (skip if already inside it)
if [ ! -f "pyproject.toml" ]; then
  if [ -z "$REPO_URL" ]; then
    echo "ERROR: Set REPO_URL to your GitHub repo URL"
    exit 1
  fi
  git clone "$REPO_URL" physicalai
  cd physicalai
fi

if ! git checkout feature/openvla-training-pipeline 2>/dev/null; then
  echo "WARNING: could not switch to feature/openvla-training-pipeline — running on $(git branch --show-current)"
fi

# 2. Initialise submodules — Dockerfile.train COPYs third_party/ so this must run before docker build
echo "Initialising git submodules..."
git submodule update --init --recursive

# 3. Verify Docker + nvidia-container-toolkit (Lambda ships with both)
if ! command -v docker > /dev/null 2>&1; then
  echo "ERROR: Docker not found. Lambda Labs instances should have Docker pre-installed."
  exit 1
fi

# Add current user to docker group if not already a member (avoids permission denied on socket)
if ! groups | grep -q docker; then
  echo "Adding $USER to docker group..."
  sudo usermod -aG docker "$USER"
  # Re-exec the script under the new group so remaining commands can access Docker
  exec sg docker "$0" "$@"
fi

if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia\|nvidia.*Runtimes"; then
  echo "nvidia runtime not found — installing nvidia-container-toolkit..."
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

# 4. Log in to W&B
if [ -n "$WANDB_KEY" ]; then
  pip install --quiet wandb
  wandb login "$WANDB_KEY"
else
  echo "WANDB_KEY not set — W&B logging will prompt interactively"
fi

# 5. Verify GPU on host
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')" \
  || echo "WARNING: torch GPU check failed — verify CUDA drivers"

# 6. Download dataset onto host (will be mounted into the container at /workspace/datasets)
echo "Installing git-lfs and downloading dataset: $DATASET"
sudo apt-get install -y git-lfs
git lfs install
python3 scripts/download_dataset.py "$DATASET"

# 7. Build the training Docker image (bakes OpenVLA, flash-attn, TF, physicalai)
echo "Building physicalai:train Docker image..."
make docker-build-train

# 8. Pre-create output directories so Docker doesn't create them as root
mkdir -p runs adapter-tmp

# 9. Launch training via Docker
# Preserve WANDB_ENTITY from the caller before sourcing the env file (which resets it to "")
_WANDB_ENTITY="$WANDB_ENTITY"
set -a
source configs/training/libero_lora.env
set +a
WANDB_ENTITY="${_WANDB_ENTITY:-$WANDB_ENTITY}"

echo "Starting LoRA fine-tuning inside Docker..."
WANDB_API_KEY="$WANDB_KEY" \
WANDB_ENTITY="$WANDB_ENTITY" \
docker compose --profile gpu run --rm physicalai-train
