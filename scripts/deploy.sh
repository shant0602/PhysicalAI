#!/usr/bin/env bash
# Wrapper around OpenVLA's REST API deployment script.
# Usage: bash scripts/deploy.sh --openvla_path openvla/openvla-7b [--port 8000]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_SCRIPT="$REPO_ROOT/third_party/openvla/vla-scripts/deploy.py"

if [ ! -f "$DEPLOY_SCRIPT" ]; then
    echo "ERROR: OpenVLA deploy script not found. Run: make submodule-init"
    exit 1
fi

exec python "$DEPLOY_SCRIPT" "$@"
