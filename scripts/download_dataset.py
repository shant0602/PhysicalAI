#!/usr/bin/env python3
"""
Download OXE datasets for OpenVLA fine-tuning.

Follows the exact download methods documented in the OpenVLA README:
  https://github.com/openvla/openvla#finetuning-openvla

Bridge V2 (bridge_orig):
  Uses wget from Berkeley RAIL — as shown in OpenVLA's README.

Other OXE datasets (fractal20220817_data, kuka, taco_play, etc.):
  Uses gsutil from gs://gresearch/robotics/ + preprocessing via rlds_dataset_mod.

Prerequisites for OXE datasets:
  gcloud components install gsutil
  make submodule-init   (clones third_party/rlds_dataset_mod)

Usage:
  python scripts/download_dataset.py bridge_orig --out_dir datasets/open-x-embodiment
  python scripts/download_dataset.py fractal20220817_data
  python scripts/download_dataset.py bridge_orig --dry_run
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
    version: str,
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
        cmd = ["gsutil", "-m", "cp", "-r", f"{GCS_BUCKET}/{dataset}/{version}", str(out_dir)]
        print(f"Downloading {dataset}:\n  {' '.join(cmd)}")
        if not dry_run:
            subprocess.run(cmd, check=True)

    if skip_preprocess:
        print(f"Skipping preprocess. Use dataset_name={dataset} in config.")
        return

    if not RLDS_MOD_SCRIPT.exists():
        print(
            f"WARNING: rlds_dataset_mod not found at {RLDS_MOD_SCRIPT}.\n"
            "Run: make submodule-init",
            file=sys.stderr,
        )
        return

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
        help="'bridge_orig' (wget) or any OXE name like fractal20220817_data (gsutil)",
    )
    parser.add_argument("--out_dir", default="datasets/open-x-embodiment")
    parser.add_argument(
        "--version",
        default="0.1.0",
        help="TFDS version for OXE gsutil downloads (default: 0.1.0)",
    )
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.dataset == "bridge_orig":
        download_bridge(out_dir, args.dry_run)
    else:
        download_oxe(
            args.dataset, args.version, out_dir,
            args.n_workers, args.skip_preprocess, args.dry_run,
        )


if __name__ == "__main__":
    main()
