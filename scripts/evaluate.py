#!/usr/bin/env python3
"""
OpenVLA inference script.

Usage:
    # GPU (default):
    python scripts/evaluate.py \\
        --image path/to/image.jpg \\
        --instruction "pick up the red cup" \\
        --device cuda:0

    # CPU (slow, ~5-10 min):
    python scripts/evaluate.py \\
        --image path/to/image.jpg \\
        --instruction "pick up the red cup" \\
        --device cpu --dtype float32

    # 4-bit quantized GPU (needs pip install 'physicalai[inference]'):
    python scripts/evaluate.py \\
        --image path/to/image.jpg \\
        --instruction "pick up the red cup" \\
        --quantize

    # Load from YAML config (CLI args override YAML values):
    python scripts/evaluate.py \\
        --config configs/model/openvla.yaml \\
        --image path/to/image.jpg \\
        --instruction "pick up the red cup"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run OpenVLA inference and print the 7-DoF action vector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", required=True, help="Path to robot workspace image (JPEG/PNG)")
    p.add_argument("--instruction", required=True, help="Natural language task instruction")
    p.add_argument("--config", default=None, help="Path to YAML config file (optional)")
    p.add_argument("--device", default="cuda:0", help="Torch device (cuda:0 / cpu)")
    p.add_argument(
        "--quantize",
        action="store_true",
        default=False,
        help="4-bit quantization via bitsandbytes (GPU only, requires physicalai[inference])",
    )
    p.add_argument(
        "--unnorm-key",
        dest="unnorm_key",
        default="bridge_orig",
        help="Dataset key for action unnormalization",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model floating-point dtype (use float32 on CPU)",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Defer heavy imports so --help is instant
    from PIL import Image
    from physicalai.inference import ACTION_AXES, OpenVLAPolicy
    from physicalai.models.vla import OpenVLAModel
    from physicalai.utils.config import OpenVLAConfig
    from physicalai.utils.logging import get_logger

    log = get_logger("evaluate")

    # Build config: YAML base + CLI overrides
    if args.config:
        log.info("Loading config from %s", args.config)
        config = OpenVLAConfig.from_yaml(args.config)
        config.device = args.device
        config.quantize = args.quantize or config.quantize
        config.unnorm_key = args.unnorm_key
        config.dtype = args.dtype
    else:
        config = OpenVLAConfig(
            device=args.device,
            quantize=args.quantize,
            unnorm_key=args.unnorm_key,
            dtype=args.dtype,
        )

    # Validate before the expensive model download
    try:
        config.validate()
    except ValueError as e:
        log.error("%s", e)
        sys.exit(1)

    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        log.error("Image not found: %s", image_path)
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")
    log.info("Loaded image: %s (%dx%d)", image_path.name, *image.size)

    # Load model + build policy
    log.info("Loading OpenVLA model (this may take a few minutes on first run)...")
    model = OpenVLAModel(config).load()
    policy = OpenVLAPolicy(model)

    # Inference
    log.info('Running inference for instruction: "%s"', args.instruction)
    obs = {"image": image, "instruction": args.instruction}
    result = policy.step(obs)

    # Print results
    action = result["action"]
    print("\n--- OpenVLA Action Vector ---")
    for axis, value in zip(ACTION_AXES, action):
        print(f"  {axis:<10s}: {value:+.6f}")
    print(f"\n  raw : {action}")
    print("-----------------------------\n")


if __name__ == "__main__":
    main()
