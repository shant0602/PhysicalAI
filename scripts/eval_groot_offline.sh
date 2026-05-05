#!/usr/bin/env bash
# Offline action-prediction eval for GR00T — Stage 2 sanity gate before sim.
# Runs predicted vs ground-truth joint plots + unnormalized MSE.
# No Isaac Sim required. Takes ~5 minutes on CPU or GPU.
#
# Usage:
#   bash scripts/eval_groot_offline.sh --model_path ./checkpoints/gr00t_lora_cansort
#   bash scripts/eval_groot_offline.sh --model_path nvidia/GR00T-N1.5-3B   # baseline
#
# Flags (all passed through to eval_policy.py):
#   --model_path PATH      checkpoint or HF model ID (required)
#   --dataset_path PATH    defaults to $DATASET_PATH or ~/gr00t_dataset/gr1_arms_only.CanSort
#   --data_config STR      defaults to fourier_gr1_arms_only
#   --embodiment_tag STR   defaults to gr1
#   --plot                 show joint-prediction plots (default: on)
#   --save_plot_path PATH  save plots to file instead of displaying

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_SCRIPT="$REPO_ROOT/third_party/isaac_groot/scripts/eval_policy.py"

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: eval_policy.py not found at $EVAL_SCRIPT"
    echo "Run: make submodule-init"
    exit 1
fi

DATASET_PATH="${DATASET_PATH:-${HOME}/gr00t_dataset/gr1_arms_only.CanSort}"

exec python "$EVAL_SCRIPT" \
    --dataset_path "$DATASET_PATH" \
    --data_config fourier_gr1_arms_only \
    --embodiment_tag gr1 \
    --video_backend torchvision_av \
    --plot \
    "$@"
