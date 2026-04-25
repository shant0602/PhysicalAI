#!/bin/bash
# One-shot Lambda Labs setup and training launcher.
# Run this on a fresh Lambda A100 instance:
#   bash scripts/setup_lambda.sh
set -euo pipefail

REPO_URL="${REPO_URL:-}"       # set via: REPO_URL=https://github.com/... bash scripts/setup_lambda.sh
WANDB_KEY="${WANDB_KEY:-}"     # set via: WANDB_KEY=your_key bash scripts/setup_lambda.sh
WANDB_ENTITY="${WANDB_ENTITY:-}"
DATASET="${DATASET:-bridge_orig}"
DATASET_DIR="${DATASET_DIR:-datasets/open-x-embodiment}"

echo "=== PhysicalAI — Lambda Setup ==="

# 1. Clone repo (skip if already inside it)
if [ ! -f "pyproject.toml" ]; then
  if [ -z "$REPO_URL" ]; then
    echo "ERROR: Set REPO_URL to your GitHub repo URL"
    exit 1
  fi
  git clone "$REPO_URL" physicalai
  cd physicalai
fi

# 2. Initialise submodules (includes third_party/openvla)
echo "Initialising git submodules..."
git submodule update --init --recursive

# 3. Install — follow OpenVLA README exactly
echo "Installing OpenVLA and dependencies..."
pip install -e third_party/openvla --no-deps
pip install packaging ninja
pip install git+https://github.com/moojink/dlimp
pip install "flash-attn==2.5.5" --no-build-isolation

# Install physicalai inference layer
pip install -e ".[inference]"

# 4. Log in to W&B
if [ -n "$WANDB_KEY" ]; then
  wandb login "$WANDB_KEY"
else
  echo "WANDB_KEY not set — W&B logging will prompt interactively"
fi

# 5. Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

# 6. Optional: download dataset
if [ -n "$DATASET" ]; then
  echo "Downloading dataset: $DATASET"
  python scripts/download_dataset.py "$DATASET" --out_dir "$DATASET_DIR"
fi

# 7. Launch training — delegates entirely to OpenVLA's finetune.py
NUM_GPUS=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")
echo "Launching training on $NUM_GPUS GPU(s)..."
bash scripts/train.sh \
  --vla_path "openvla/openvla-7b" \
  --dataset_name "$DATASET" \
  --wandb_entity "$WANDB_ENTITY" \
  --lora_rank 32 \
  --batch_size 16 \
  --learning_rate 5e-4 \
  "$@"
