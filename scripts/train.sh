#!/usr/bin/env bash
# Wrapper around OpenVLA's official LoRA fine-tuning script.
# All arguments are forwarded directly to vla-scripts/finetune.py.
#
# Usage (from repo root):
#   bash scripts/train.sh \
#     --dataset_name bridge_orig \
#     --data_root_dir datasets/open-x-embodiment \
#     --wandb_entity myteam
#
# Or via make:
#   make train DATASET=bridge_orig WANDB_ENTITY=myteam
#
# Environment variables (can be set in .env or shell):
#   NUM_GPUS        — number of GPUs (default: 1)
#   DATA_ROOT_DIR   — path to downloaded OXE datasets (default: datasets/open-x-embodiment)
#   RUN_ROOT_DIR    — checkpoint/log output directory (default: runs)
#   ADAPTER_TMP_DIR — temp dir for LoRA weights (default: adapter-tmp)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FINETUNE_SCRIPT="$REPO_ROOT/third_party/openvla/vla-scripts/finetune.py"

if [ ! -f "$FINETUNE_SCRIPT" ]; then
    echo "ERROR: OpenVLA fine-tuning script not found at $FINETUNE_SCRIPT"
    echo "Run: make submodule-init"
    exit 1
fi

NUM_GPUS="${NUM_GPUS:-1}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-datasets/open-x-embodiment}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-runs}"
ADAPTER_TMP_DIR="${ADAPTER_TMP_DIR:-adapter-tmp}"

exec torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node "$NUM_GPUS" \
    "$FINETUNE_SCRIPT" \
    --data_root_dir "$DATA_ROOT_DIR" \
    --run_root_dir "$RUN_ROOT_DIR" \
    --adapter_tmp_dir "$ADAPTER_TMP_DIR" \
    "$@"
