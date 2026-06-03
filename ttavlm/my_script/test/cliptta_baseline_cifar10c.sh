#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CLIPTTA baseline on CIFAR-10-C (gaussian_noise, severity 5)
# Ready to run on a vast.ai GPU instance.
#
# Usage:
#   bash ttavlm/my_script/test/run_cliptta_baseline_cifar10c.sh
#
# Edit DATAROOT / SAVE_ROOT below to match your instance before running.
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- Paths (edit these) ---------------------------------------------------
DATAROOT="/workspace/datasets"          # where CIFAR-10-C is stored
SAVE_ROOT="/workspace/cliptta_results"  # where results/weights are written
# Repo root = three levels up from this script (ttavlm/my_script/test/ -> repo root)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"   # ensure `python -m ttavlm.main` can import the package

# ---- Experiment config ----------------------------------------------------
BASE_MODEL="clip-ViT-B/16"
DATASET="cifar10c"
SHIFT_TYPE="gaussian_noise"
SEVERITY=5
BATCH_SIZE=128

mkdir -p "${SAVE_ROOT}"

python -m ttavlm.main \
    --env vast \
    --exp_name cliptta_baseline_cifar10c \
    --root "${ROOT}" \
    --dataroot "${DATAROOT}" \
    --save_root "${SAVE_ROOT}" \
    --base_model_name "${BASE_MODEL}" \
    --dataset "${DATASET}" \
    --shift_type "${SHIFT_TYPE}" \
    --severity "${SEVERITY}" \
    --batch_size "${BATCH_SIZE}" \
    --adaptation cliptta \
    --closed_set \
    --steps 1 \
    --lr 1e-3 \
    --optimizer_type adam \
    --beta_tta 1.0 \
    --beta_reg 0.1 \
    --display_progress
