# PhysicalAI

Research and engineering repo for embodied AI — from large language models to Vision Language Action (VLA) models for robot control.

## Architecture Overview

The diagram below traces the evolution from pure language models to modern VLA systems capable of continuous robot control via flow matching.

![LLM vs VLM vs VLA Architecture](assets/diagrams/vla_vlm_llm_arch_diagram.svg)

### VLA Architecture Diagram
**© 2026 Shantanu Singh**  
github.com/shant0602

Comparative architecture diagram: LLM vs VLM vs VLA (1st gen) vs VLA (2nd gen).  
All claims cited to 8 primary academic sources.  
Licensed CC BY 4.0 — free to share with attribution.

## Repo Structure

```
PhysicalAI/
├── assets/          # Diagrams, figures
├── docs/            # Research notes, guides
├── src/
│   ├── models/      # LLM / VLM / VLA architectures
│   ├── data/        # Datasets, transforms, loaders
│   ├── training/    # Trainer, losses, schedulers
│   ├── inference/   # Policy wrapper, inference server
│   ├── robot/       # Sim envs, hardware drivers, control
│   ├── evaluation/  # Metrics, rollout harness
│   └── utils/       # Config, logging, checkpointing, viz
├── scripts/         # train.py, evaluate.py, collect_data.py
├── configs/         # Hydra YAML experiment configs
├── notebooks/       # Exploration and demos
└── tests/           # Unit and integration tests
```

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv && source .venv/bin/activate

# Install package in editable mode
make install
```

## Usage

```bash
make train       # Launch training
make evaluate    # Run evaluation
make test        # Run test suite
make lint        # Lint + type-check
```

## LoRA Fine-Tuning on a GPU Cloud Instance

Fine-tunes OpenVLA-7B on the [LIBERO](https://lifelong-robot-learning.github.io/LIBERO/) `libero_spatial` task suite using LoRA (rank=32). Runs entirely inside Docker — no host Python environment needed.

**Tested on:** Lambda Labs GH200 (ARM64, H100 GPU, 480GB VRAM). Also works on any NVIDIA GPU with ≥27GB VRAM (A100, H100, RTX 4090, etc.).  
**Training time:** ~2h for 10,000 steps at batch size 16 on a single GPU.

---

### Step 1 — Provision a GPU instance

Spin up a cloud GPU instance with Docker pre-installed (Lambda Labs, RunPod, Vast.ai, etc.).
SSH in:

```bash
ssh ubuntu@<your-instance-ip>
```

---

### Step 2 — Clone the repo and initialise submodules

```bash
git clone https://github.com/shant0602/PhysicalAI.git
cd PhysicalAI
git submodule update --init --recursive   # pulls third_party/openvla and rlds_dataset_mod
```

---

### Step 3 — Download the base model

The training container downloads the model automatically on first run via HuggingFace.
To pre-download it manually (saves time, avoids re-downloading if container is recreated):

```bash
pip install huggingface_hub
HF_HOME=/home/ubuntu/hf-cache python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('openvla/openvla-7b', local_dir='/home/ubuntu/hf-cache/hub/models--openvla--openvla-7b')
"
```

The base model is ~14GB and takes ~15 seconds on a fast cloud connection.

---

### Step 4 — Download the LIBERO dataset

```bash
sudo apt-get install -y git-lfs
git lfs install
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds \
    /home/ubuntu/PhysicalAI/datasets/modified_libero_rlds
cd /home/ubuntu/PhysicalAI/datasets/modified_libero_rlds && git lfs pull
cd /home/ubuntu/PhysicalAI
```

Dataset is ~10GB total (all 4 LIBERO suites: Spatial, Object, Goal, Long).

---

### Step 5 — Build the Docker training image

```bash
cd /home/ubuntu/PhysicalAI
docker build -f docker/Dockerfile.train -t physicalai:train .
```

This takes ~10–15 minutes on first build. The image (~32GB) includes PyTorch 2.2, OpenVLA, TensorFlow (for RLDS data loading), and all dependencies. Once built it can be reused across runs without rebuilding.

**Note for ARM64 (GH200):** the image uses `nvcr.io/nvidia/pytorch:24.01-py3` as base, which supports both `amd64` and `aarch64`. flash-attn is intentionally uninstalled (the pre-built binary in the NVIDIA image is x86-only); training falls back to standard attention which works fine for LoRA on 480GB VRAM.

---

### Step 6 — Find your W&B entity

Your W&B entity is shown when you run:

```bash
pip install wandb
wandb login --relogin   # paste your API key from wandb.ai/settings
```

The output shows: `Currently logged in as: <username> (<entity>)`.  
**Use the value in parentheses** as your `WANDB_ENTITY` — it may differ from your username (e.g. `shantan-singh-georgia-tech-alumni-association`).

---

### Step 7 — Launch training

Create output directories and start the training container:

```bash
mkdir -p /home/ubuntu/PhysicalAI/runs /home/ubuntu/PhysicalAI/adapter-tmp /home/ubuntu/hf-cache

docker run -d \
  --name openvla-train \
  --gpus all \
  --shm-size=16g \
  -e HF_HOME=/hf-cache \
  -e WANDB_API_KEY=<your-wandb-api-key> \
  -e WANDB_MODE=offline \
  -e WANDB_DIR=/tmp \
  -e VLA_PATH=openvla/openvla-7b \
  -e DATASET_NAME=libero_spatial_no_noops \
  -e DATA_ROOT_DIR=/workspace/datasets/modified_libero_rlds \
  -e RUN_ROOT_DIR=/workspace/runs \
  -e ADAPTER_TMP_DIR=/workspace/adapter-tmp \
  -e BATCH_SIZE=16 \
  -e MAX_STEPS=10000 \
  -e SAVE_STEPS=500 \
  -e LEARNING_RATE=5e-4 \
  -e GRAD_ACCUMULATION_STEPS=1 \
  -e USE_LORA=true \
  -e LORA_RANK=32 \
  -e LORA_DROPOUT=0.0 \
  -e USE_QUANTIZATION=false \
  -e IMAGE_AUG=true \
  -e SHUFFLE_BUFFER_SIZE=10000 \
  -e SAVE_LATEST_CHECKPOINT_ONLY=true \
  -e WANDB_PROJECT=physicalai-openvla \
  -e WANDB_ENTITY=<your-wandb-entity> \
  -v /home/ubuntu/hf-cache:/hf-cache \
  -v /home/ubuntu/PhysicalAI/datasets:/workspace/datasets \
  -v /home/ubuntu/PhysicalAI/runs:/workspace/runs \
  -v /home/ubuntu/PhysicalAI/adapter-tmp:/workspace/adapter-tmp \
  physicalai:train \
  scripts/train.sh
```

> **Important:** the last argument is `scripts/train.sh`, not `bash scripts/train.sh`.
> The container ENTRYPOINT is already `/bin/bash`, so prepending `bash` causes
> `/bin/bash bash scripts/train.sh` which tries to execute the `bash` binary as a
> script and fails with `cannot execute binary file`.

Training config reference: [`configs/training/libero_lora.env`](configs/training/libero_lora.env).

---

### Step 8 — Monitor training

```bash
# Container running?
docker inspect openvla-train --format '{{.State.Status}}'

# Current step (tqdm bar is written to docker logs):
docker logs openvla-train 2>&1 | grep -oP '\d+/10000' | tail -1

# GPU utilization:
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader

# Tail raw logs:
docker logs openvla-train --tail 20 2>&1
```

Expected behavior on first run: GPU hits 100% utilization within ~30 seconds. The tqdm progress bar appears after the shuffle buffer fills (~1–2 min). Steps print at ~1.4 steps/sec on a GH200.

---

### Step 9 — Sync W&B (training runs offline by default)

W&B runs in offline mode to avoid API authentication issues. Sync to the web UI at any time while training is running (or after it finishes):

```bash
# Copy run data from container to host
docker cp openvla-train:/tmp/wandb/$(docker exec openvla-train ls /tmp/wandb/ | grep offline-run) \
  /home/ubuntu/wandb-offline-run

# Sync to W&B (use the entity value from Step 6)
WANDB_API_KEY=<your-key> wandb sync /home/ubuntu/wandb-offline-run \
  --entity <your-wandb-entity>
```

Loss curve and metrics appear at `wandb.ai/<your-entity>/physicalai-openvla`.

---

### Step 10 — Save your checkpoint before stopping the instance

Cloud instances wipe all local data on termination. The merged checkpoint (base model + LoRA weights fused) is in `runs/` and is ~15GB. Upload it to HuggingFace Hub before stopping:

```bash
pip install huggingface_hub
python3 - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("<your-hf-username>/openvla-7b-libero-spatial-lora",
                token="<your-hf-token>", private=True, exist_ok=True)
api.upload_folder(
    folder_path="/home/ubuntu/PhysicalAI/runs/openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug",
    repo_id="<your-hf-username>/openvla-7b-libero-spatial-lora",
    token="<your-hf-token>",
    commit_message="Merged checkpoint after training",
)
print("Upload complete.")
EOF
```

The dataset (`openvla/modified_libero_rlds`) and base model (`openvla/openvla-7b`) are both on HuggingFace and can be re-downloaded for free — no need to back those up.

---

### Resuming training from a checkpoint

`vla-scripts/finetune.py` has no built-in resume flag. To continue training from a saved checkpoint, use the merged checkpoint as the starting model (`--vla_path`). The model weights already include all prior learning; only the step counter and LR schedule restart from 0.

```bash
# Example: checkpoint saved at step 1500, 8500 steps remaining
docker run -d \
  --name openvla-train \
  ... \                                          # same flags as Step 7
  -e VLA_PATH=<your-hf-username>/openvla-7b-libero-spatial-lora \
  -e MAX_STEPS=8500 \
  ... \
  physicalai:train \
  scripts/train.sh
```

---

### After training — evaluate

```bash
make eval-libero CHECKPOINT=runs/<your-run-dir>
```

Runs the official LIBERO simulation benchmark (50 rollouts × 10 tasks) and reports task success rate.  
Expected result for `libero_spatial` after 10k steps: **~84.7%** (from OpenVLA paper, Table in Appendix E).

There is no loss threshold to target — 10,000 steps is the paper's validated number for LIBERO-Spatial. Check the W&B loss curve to confirm it is decreasing and not diverging.

## Roadmap

- [ ] LLM backbone implementations
- [ ] VLM vision encoders (ViT, SigLIP, CLIP, DINOv2)
- [ ] VLA action expert + flow matching decoder
- [ ] Open-X Embodiment dataset integration
- [ ] MuJoCo / Isaac Sim environment wrappers
- [ ] Real-robot hardware interface (ROS2)
- [ ] Evaluation harness + task success metrics

## License

GNU General Public License v3 — see [LICENSE](LICENSE).

---

*Author: Shantanu Singh*
