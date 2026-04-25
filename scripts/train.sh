#!/usr/bin/env bash
# Thin wrapper around OpenVLA's official LoRA fine-tuning script.
# Reads defaults from environment (can be set via configs/training/openvla_lora.env).
# Any argument passed in "$@" overrides the corresponding env-var default (last-wins).
#
# Usage (from repo root):
#   bash scripts/train.sh                                       # use all env defaults
#   bash scripts/train.sh --dataset_name fractal20220817_data   # override one arg
#   source configs/training/openvla_lora.env && bash scripts/train.sh
#
# Or via make:
#   make train DATASET=bridge_orig WANDB_ENTITY=myteam
#
# Environment variables (loadable from configs/training/openvla_lora.env):
#   NUM_GPUS, VLA_PATH, DATASET_NAME, DATA_ROOT_DIR, RUN_ROOT_DIR, ADAPTER_TMP_DIR,
#   BATCH_SIZE, MAX_STEPS, SAVE_STEPS, LEARNING_RATE, GRAD_ACCUMULATION_STEPS,
#   USE_LORA, LORA_RANK, LORA_DROPOUT, USE_QUANTIZATION, IMAGE_AUG,
#   SHUFFLE_BUFFER_SIZE, SAVE_LATEST_CHECKPOINT_ONLY, WANDB_PROJECT, WANDB_ENTITY

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FINETUNE_SCRIPT="$REPO_ROOT/third_party/openvla/vla-scripts/finetune.py"

if [ ! -f "$FINETUNE_SCRIPT" ]; then
    echo "ERROR: OpenVLA fine-tuning script not found at $FINETUNE_SCRIPT"
    echo "Run: make submodule-init"
    exit 1
fi

if ! command -v torchrun > /dev/null 2>&1; then
    echo "ERROR: torchrun not found — install PyTorch or activate the correct venv"
    exit 1
fi

# Defaults from environment; CLI args in "$@" override these (draccus uses last-wins).
NUM_GPUS="${NUM_GPUS:-1}"
VLA_PATH="${VLA_PATH:-openvla/openvla-7b}"
DATASET_NAME="${DATASET_NAME:-bridge_orig}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-$REPO_ROOT/datasets/open-x-embodiment}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-$REPO_ROOT/runs}"
ADAPTER_TMP_DIR="${ADAPTER_TMP_DIR:-$REPO_ROOT/adapter-tmp}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_STEPS="${MAX_STEPS:-200000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
USE_LORA="${USE_LORA:-true}"
LORA_RANK="${LORA_RANK:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
USE_QUANTIZATION="${USE_QUANTIZATION:-false}"
IMAGE_AUG="${IMAGE_AUG:-true}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-100000}"
SAVE_LATEST_CHECKPOINT_ONLY="${SAVE_LATEST_CHECKPOINT_ONLY:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-physicalai-openvla}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

if [ -z "$WANDB_ENTITY" ]; then
    echo "WARNING: WANDB_ENTITY is unset — set it via the env var or --wandb_entity flag"
fi

exec torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node "$NUM_GPUS" \
    "$FINETUNE_SCRIPT" \
    --vla_path "$VLA_PATH" \
    --dataset_name "$DATASET_NAME" \
    --data_root_dir "$DATA_ROOT_DIR" \
    --run_root_dir "$RUN_ROOT_DIR" \
    --adapter_tmp_dir "$ADAPTER_TMP_DIR" \
    --batch_size "$BATCH_SIZE" \
    --max_steps "$MAX_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS" \
    --use_lora "$USE_LORA" \
    --lora_rank "$LORA_RANK" \
    --lora_dropout "$LORA_DROPOUT" \
    --use_quantization "$USE_QUANTIZATION" \
    --image_aug "$IMAGE_AUG" \
    --shuffle_buffer_size "$SHUFFLE_BUFFER_SIZE" \
    --save_latest_checkpoint_only "$SAVE_LATEST_CHECKPOINT_ONLY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" \
    "$@"
