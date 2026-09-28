#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/experiment/main.yaml}"
python -m damascus.anvil draw --config "${CONFIG}"
