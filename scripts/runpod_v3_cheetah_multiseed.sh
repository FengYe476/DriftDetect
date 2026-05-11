#!/usr/bin/env bash
set -Eeuo pipefail

# Backward-compatible Cheetah wrapper.
#
# Usage:
#   bash scripts/runpod_v3_cheetah_multiseed.sh 43

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/runpod_v3_cheetah_multiseed.sh SEED"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/runpod_v3_multiseed.sh" "$1" cheetah
