# VLM / VLA Architecture Overview

See the full architecture comparison diagram: [`assets/diagrams/vla_architecture_shantanu_singh_.svg`](../../assets/diagrams/vla_architecture_shantanu_singh_.svg)

## Model Families

### LLM (Large Language Model)
- **Examples:** GPT-4, LLaMA
- **Input:** Text only
- **Key components:** BPE tokenizer → Transformer backbone → LM head

### VLM (Vision Language Model)
- **Examples:** LLaVA, GPT-4o
- **Input:** Text + Images
- **Key components:** ViT/CLIP encoder + BPE tokenizer → MLP projector → Transformer backbone → LM head

### VLA 1st Gen (Vision Language Action)
- **Examples:** RT-2, OpenVLA
- **Input:** Text + Images + Robot state
- **Output:** Discrete action tokens (256 bins/dimension, autoregressive)

### VLA 2nd Gen
- **Examples:** π0, π0.5+KI
- **Input:** Text + Images + Robot state
- **Key innovations:**
  - SigLIP vision encoder
  - Block-wise self-attention fusion
  - Separate action expert (300M) with stop-gradient from VLM block (3B)
  - Flow matching decoder → continuous actions, all dims at once, action chunking
- **Output:** Continuous action chunks (raw floats)

## Key References
- RT-2 (Brohan et al., 2023)
- OpenVLA (Kim et al., 2024)
- π0 (Black et al., 2024) — arXiv:2410.24164
- Flow matching for generative modeling (Lipman et al., 2022)
