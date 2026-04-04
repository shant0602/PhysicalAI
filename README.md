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
