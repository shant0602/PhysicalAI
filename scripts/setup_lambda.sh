#!/bin/bash
# One-shot Lambda Labs setup and training launcher.
# Run this on a fresh Lambda A100 instance:
#   bash scripts/setup_lambda.sh
set -euo pipefail

REPO_URL="${REPO_URL:-}"   # set via: REPO_URL=https://github.com/... bash scripts/setup_lambda.sh
WANDB_KEY="${WANDB_KEY:-}" # set via: WANDB_KEY=your_key bash scripts/setup_lambda.sh

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

# 2. Install dependencies
echo "Installing dependencies..."
pip install --quiet -e ".[inference]"

# 3. Log in to W&B
if [ -n "$WANDB_KEY" ]; then
  wandb login "$WANDB_KEY"
else
  echo "WANDB_KEY not set — W&B logging will prompt interactively"
fi

# 4. Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

# 5. Launch training
echo "Starting training..."
python scripts/train.py --config configs/training/openvla_lora.yaml "$@"
