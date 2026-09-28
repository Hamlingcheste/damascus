#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/experiment/main.yaml}"
shift || true
python -m damascus.anvil proof --config "${CONFIG}" "$@"
