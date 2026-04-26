#!/usr/bin/env python3
"""
Download datasets for OpenVLA fine-tuning.

Follows the exact download methods documented in the OpenVLA README:
  https://github.com/openvla/openvla#finetuning-openvla

LIBERO (libero):
  Clones openvla/modified_libero_rlds from HuggingFace via git-lfs (~10GB).
  Contains all four task suites: libero_spatial_no_noops, libero_object_no_noops,
  libero_goal_no_noops, libero_10_no_noops.
  Prerequisites: git-lfs (sudo apt-get install git-lfs)

Bridge V2 (bridge_orig):
  Uses wget from Berkeley RAIL — as shown in OpenVLA's README.

Other OXE datasets (fractal20220817_data, kuka, taco_play, etc.):
  Uses gsutil from gs://gresearch/robotics/ + preprocessing via rlds_dataset_mod.

Prerequisites for OXE datasets:
  gcloud components install gsutil
  make submodule-init   (clones third_party/rlds_dataset_mod)

Usage:
  python scripts/download_dataset.py libero                         # LIBERO (~10GB)
  python scripts/download_dataset.py libero --dry_run               # preview command
  python scripts/download_dataset.py bridge_orig --out_dir datasets/open-x-embodiment
  python scripts/download_dataset.py fractal20220817_data
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Bridge V2: exact wget source from OpenVLA README
BRIDGE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/"

# Other OXE datasets: from Google Cloud Storage
GCS_BUCKET = "gs://gresearch/robotics"

RLDS_MOD_SCRIPT = REPO_ROOT / "third_party" / "rlds_dataset_mod" / "modify_rlds_dataset.py"

# LIBERO: HuggingFace dataset with all four task suites in RLDS format
LIBERO_HF_REPO = "https://huggingface.co/datasets/openvla/modified_libero_rlds"
LIBERO_DEST_NAME = "modified_libero_rlds"


def download_libero(out_dir: Path, dry_run: bool) -> None:
    """Clone modified_libero_rlds from HuggingFace via git-lfs (~10GB)."""
    dest = out_dir / LIBERO_DEST_NAME
    if dest.exists():
        print(f"Already exists: {dest}")
        print(f"Set DATA_ROOT_DIR={dest} and DATASET_NAME=libero_spatial_no_noops in your config.")
        return

    if not shutil.which("git-lfs") and not dry_run:
        print(
            "ERROR: git-lfs not found. Install it first:\n"
            "  Ubuntu/Lambda: sudo apt-get install -y git-lfs\n"
            "  macOS:         brew install git-lfs",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading LIBERO from HuggingFace (~10GB) → {dest}")
    print(f"  git lfs install && git clone {LIBERO_HF_REPO} {dest}")
    if not dry_run:
        subprocess.run(["git", "lfs", "install"], check=True)
        subprocess.run(["git", "clone", LIBERO_HF_REPO, str(dest)], check=True)
    print(
        f"Done → {dest}\n"
        "Set DATA_ROOT_DIR=datasets/modified_libero_rlds and\n"
        "    DATASET_NAME=libero_spatial_no_noops  (or _object, _goal, _10) in your config."
    )


def download_bridge(out_dir: Path, dry_run: bool) -> None:
    """Download Bridge V2 via wget — exactly as OpenVLA README instructs."""
    dest = out_dir / "bridge_orig"
    if dest.exists():
        print(f"Already exists: {dest}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget", "-r", "-nH", "--cut-dirs=4",
        "--reject=index.html*",
        "-P", str(out_dir),
        BRIDGE_URL,
    ]
    print(f"Downloading Bridge V2 (wget):\n  {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)
        raw = out_dir / "bridge_dataset"
        if raw.exists():
            raw.rename(dest)
    print(f"Done → {dest}\nSet dataset_name=bridge_orig in your config.")


def download_oxe(
    dataset: str,
    out_dir: Path,
    n_workers: int,
    skip_preprocess: bool,
    dry_run: bool,
) -> None:
    """Download any OXE dataset via gsutil + preprocess with rlds_dataset_mod."""
    if not shutil.which("gsutil"):
        print("ERROR: gsutil not found. Install: gcloud components install gsutil", file=sys.stderr)
        sys.exit(1)

    raw_dest = out_dir / dataset
    if not raw_dest.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["gsutil", "-m", "cp", "-r", f"{GCS_BUCKET}/{dataset}", str(out_dir)]
        print(f"Downloading {dataset}:\n  {' '.join(cmd)}")
        if not dry_run:
            subprocess.run(cmd, check=True)

    if skip_preprocess:
        print(f"Skipping preprocess. Use dataset_name={dataset} in config.")
        return

    if not RLDS_MOD_SCRIPT.exists():
        print(
            f"ERROR: rlds_dataset_mod not found at {RLDS_MOD_SCRIPT}.\n"
            "Run: make submodule-init",
            file=sys.stderr,
        )
        sys.exit(1)

    processed = out_dir / f"{dataset}_processed"
    cmd = [
        sys.executable, str(RLDS_MOD_SCRIPT),
        f"--dataset={dataset}", f"--data_dir={out_dir}",
        f"--target_dir={processed}", "--mods=resize_and_jpeg_encode",
        f"--n_workers={n_workers}",
    ]
    print(f"Preprocessing:\n  {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)
    print(f"Done → {processed}\nSet dataset_name={dataset}_processed in config.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OXE datasets for OpenVLA fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        help=(
            "'libero' (HuggingFace git-lfs, ~10GB), "
            "'bridge_orig' (wget, ~200GB), "
            "or any OXE name like fractal20220817_data (gsutil)"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: datasets/modified_libero_rlds for libero, datasets/open-x-embodiment for others)",
    )
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.dataset == "libero":
        out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "datasets"
        download_libero(out_dir, args.dry_run)
    elif args.dataset == "bridge_orig":
        out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "datasets" / "open-x-embodiment"
        download_bridge(out_dir, args.dry_run)
    else:
        out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "datasets" / "open-x-embodiment"
        download_oxe(
            args.dataset, out_dir,
            args.n_workers, args.skip_preprocess, args.dry_run,
        )


if __name__ == "__main__":
    main()
