#!/usr/bin/env bash
# Thin wrapper around GR00T N1.5's official LoRA fine-tuning script.
# Reads defaults from configs/training/groot_lora.env (last-wins with "$@").
#
# Usage (from repo root):
#   source configs/training/groot_lora.env && bash scripts/train_groot.sh
#   bash scripts/train_groot.sh --max_steps 5000   # override one arg
#
# Or via make:
#   make train-groot WANDB_ENTITY=myteam
#
# Two-step env setup required before first run:
#   conda create -n gr00t python=3.10 -y && conda activate gr00t
#   pip install -e third_party/isaac_groot[base]
#   pip install flash-attn==2.7.1.post4 --no-build-isolation
#
# Environment variables (loadable from configs/training/groot_lora.env):
#   GROOT_MODEL_PATH, EMBODIMENT_TAG, DATA_CONFIG, DATASET_PATH,
#   OUTPUT_DIR, NUM_GPUS, MAX_STEPS, BATCH_SIZE, SAVE_STEPS,
#   LEARNING_RATE, GRAD_ACCUMULATION_STEPS, DATALOADER_NUM_WORKERS,
#   LORA_RANK, LORA_ALPHA, LORA_DROPOUT, TUNE_DIFFUSION_MODEL,
#   TUNE_LLM, TUNE_VISUAL, TUNE_PROJECTOR, VIDEO_BACKEND,
#   WANDB_PROJECT, WANDB_ENTITY

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FINETUNE_SCRIPT="$REPO_ROOT/third_party/isaac_groot/scripts/gr00t_finetune.py"

if [ ! -f "$FINETUNE_SCRIPT" ]; then
    echo "ERROR: GR00T fine-tuning script not found at $FINETUNE_SCRIPT"
    echo "Run: make submodule-init"
    exit 1
fi

if ! command -v python > /dev/null 2>&1; then
    echo "ERROR: python not found — activate the gr00t conda env first"
    exit 1
fi

# Defaults from environment; CLI args in "$@" override (tyro last-wins).
GROOT_MODEL_PATH="${GROOT_MODEL_PATH:-nvidia/GR00T-N1.5-3B}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-gr1}"
DATA_CONFIG="${DATA_CONFIG:-fourier_gr1_arms_only}"
DATASET_PATH="${DATASET_PATH:-${HOME}/gr00t_dataset/gr1_arms_only.CanSort}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/gr00t_lora_cansort}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_STEPS="${MAX_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
TUNE_DIFFUSION_MODEL="${TUNE_DIFFUSION_MODEL:-false}"
TUNE_LLM="${TUNE_LLM:-false}"
TUNE_VISUAL="${TUNE_VISUAL:-false}"
TUNE_PROJECTOR="${TUNE_PROJECTOR:-true}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchvision_av}"
WANDB_PROJECT="${WANDB_PROJECT:-physicalai-groot}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

if [ -z "$WANDB_ENTITY" ]; then
    echo "WARNING: WANDB_ENTITY is unset — set it via env var or --wandb_entity flag"
fi

# Convert bash booleans to tyro boolean flags.
# tyro uses --flag / --no-flag syntax for bool fields.
_bool_flag() {
    local name="$1" val="$2"
    if [ "$val" = "true" ]; then echo "--${name}"; else echo "--no-${name}"; fi
}

exec python "$FINETUNE_SCRIPT" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --base_model_path "$GROOT_MODEL_PATH" \
    --embodiment_tag "$EMBODIMENT_TAG" \
    --data_config "$DATA_CONFIG" \
    --num_gpus "$NUM_GPUS" \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --save_steps "$SAVE_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --gradient_accumulation_steps "$GRAD_ACCUMULATION_STEPS" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    $(_bool_flag "tune_diffusion_model" "$TUNE_DIFFUSION_MODEL") \
    $(_bool_flag "tune_llm" "$TUNE_LLM") \
    $(_bool_flag "tune_visual" "$TUNE_VISUAL") \
    $(_bool_flag "tune_projector" "$TUNE_PROJECTOR") \
    --video_backend "$VIDEO_BACKEND" \
    --report_to wandb \
    "$@"
