#!/usr/bin/env bash
# One-shot setup and GR00T N1.5 LoRA training launcher for rented x86 GPU instances.
# Tested on A100 40GB / Ubuntu 22.04 with Docker pre-installed.
#
# Usage — run directly on the instance:
#   REPO_URL=https://github.com/shant0602/PhysicalAI \
#   WANDB_KEY=<key> WANDB_ENTITY=shant0602 HF_TOKEN=<token> \
#   bash scripts/setup_groot_gpu.sh
#
# Or pipe from GitHub (fresh instance, no repo yet):
#   REPO_URL=https://github.com/shant0602/PhysicalAI \
#   WANDB_KEY=<key> WANDB_ENTITY=shant0602 HF_TOKEN=<token> \
#   bash <(curl -fsSL https://raw.githubusercontent.com/shant0602/PhysicalAI/feature/openvla-training-pipeline/scripts/setup_groot_gpu.sh)
#
# Environment variables:
#   REPO_URL          GitHub repo URL (required if not already inside the repo)
#   BRANCH            Branch to checkout (default: feature/openvla-training-pipeline)
#   WANDB_KEY         W&B API key (required for experiment tracking)
#   WANDB_ENTITY      W&B username or team (required)
#   HF_TOKEN          HuggingFace token (required to download nvidia/ gated dataset)
#   FLASH_ATTN_WHEEL  Optional prebuilt wheel URL for x86 fast path (skips ~20 min compile)
#                     Leave unset to build flash-attn from source (always correct)

set -euo pipefail

REPO_URL="${REPO_URL:-}"
BRANCH="${BRANCH:-feature/openvla-training-pipeline}"
WANDB_KEY="${WANDB_KEY:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
HF_TOKEN="${HF_TOKEN:-}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-}"

echo "=== PhysicalAI — GR00T N1.5 GPU Setup ==="

# ── 1. Clone repo ─────────────────────────────────────────────────────────────
if [ ! -f "pyproject.toml" ]; then
    if [ -z "$REPO_URL" ]; then
        echo "ERROR: Set REPO_URL to your GitHub repo URL, or run from inside the repo."
        exit 1
    fi
    git clone "$REPO_URL" physicalai
    cd physicalai
fi

git checkout "$BRANCH" 2>/dev/null || \
    echo "WARNING: could not switch to $BRANCH — running on $(git branch --show-current)"

echo "Initialising git submodules (isaac_groot + openvla)..."
git submodule update --init --recursive

# ── 2. Docker + nvidia runtime ────────────────────────────────────────────────
if ! command -v docker > /dev/null 2>&1; then
    echo "ERROR: Docker not found. Install Docker before running this script."
    exit 1
fi

if ! groups | grep -q docker; then
    echo "Adding $USER to docker group..."
    sudo usermod -aG docker "$USER"
    exec sg docker "$0" "$@"
fi

if ! docker info 2>/dev/null | grep -qi nvidia; then
    echo "nvidia runtime not found — installing nvidia-container-toolkit..."
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

# ── 3. W&B credentials ───────────────────────────────────────────────────────
if [ -n "$WANDB_KEY" ]; then
    pip install --quiet wandb
    wandb login "$WANDB_KEY"
else
    echo "WARNING: WANDB_KEY not set — W&B will prompt interactively or training will fail."
fi

if [ -z "$WANDB_ENTITY" ]; then
    echo "WARNING: WANDB_ENTITY not set — set it via env var or training will warn."
fi

# ── 4. GPU check ──────────────────────────────────────────────────────────────
echo "GPU check:"
python3 -c "
import subprocess, sys
r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                   capture_output=True, text=True)
print(' ', r.stdout.strip())
"

# ── 5. Download CanSort dataset (~10 GB) ──────────────────────────────────────
echo "Downloading gr1_arms_only.CanSort from HuggingFace..."
pip install --quiet "huggingface_hub[cli]"
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi
huggingface-cli download nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
    --repo-type dataset \
    --include "gr1_arms_only.CanSort/**" \
    --local-dir ~/gr00t_dataset
echo "Dataset ready at ~/gr00t_dataset/gr1_arms_only.CanSort"

# ── 6. Build training image ───────────────────────────────────────────────────
echo "Building physicalai:train-groot Docker image..."
echo "  flash-attn: ${FLASH_ATTN_WHEEL:-building from source (~20 min)}"
docker build \
    --build-arg FLASH_ATTN_WHEEL="$FLASH_ATTN_WHEEL" \
    -f docker/Dockerfile.train.groot \
    -t physicalai:train-groot \
    .

# ── 7. Prepare output directories ─────────────────────────────────────────────
mkdir -p checkpoints

# ── 8. Launch training ────────────────────────────────────────────────────────
_ENTITY="$WANDB_ENTITY"
set -a
source configs/training/groot_lora.env
set +a
WANDB_ENTITY="${_ENTITY:-$WANDB_ENTITY}"

echo ""
echo "=== Starting GR00T LoRA fine-tuning ==="
echo "  Model:   $GROOT_MODEL_PATH"
echo "  Dataset: ~/gr00t_dataset/gr1_arms_only.CanSort"
echo "  Output:  $OUTPUT_DIR"
echo "  Steps:   $MAX_STEPS  |  Batch: $BATCH_SIZE  |  LoRA rank: $LORA_RANK"
echo "  W&B:     $WANDB_PROJECT / $WANDB_ENTITY"
echo ""

WANDB_API_KEY="$WANDB_KEY" \
WANDB_ENTITY="$WANDB_ENTITY" \
HF_TOKEN="$HF_TOKEN" \
GROOT_DATASET_DIR=~/gr00t_dataset \
docker compose --profile gpu run --rm physicalai-train-groot

echo "=== Training complete. Checkpoint at: $OUTPUT_DIR ==="
