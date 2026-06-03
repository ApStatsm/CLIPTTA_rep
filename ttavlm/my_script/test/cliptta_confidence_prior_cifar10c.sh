#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CLIPTTA + Confidence Prior on CIFAR-10-C (gaussian_noise, severity 5).
#
# Same dynamic class-prior method (cliptta_prior) as run_cliptta_prior_*, but
# the prior is updated only from *confident* samples: a high --prior_min_conf
# threshold filters the batch before the (gain-weighted) prior update.
# Compare against run_cliptta_prior_cifar10c.sh (threshold = 0.0) to isolate
# the effect of confidence filtering.
#
# Usage:
#   bash ttavlm/my_script/test/run_cliptta_confidence_prior_cifar10c.sh
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

# ---- Dynamic prior hyper-parameters --------------------------------------
PRIOR_ALPHA=0.9
PRIOR_STRENGTH=1.0
PRIOR_MIN_GAIN=0.0
PRIOR_MIN_CONF=0.9   # confidence threshold: only confident samples update the prior

mkdir -p "${SAVE_ROOT}"

python -m ttavlm.main \
    --env vast \
    --exp_name cliptta_confidence_prior_cifar10c \
    --root "${ROOT}" \
    --dataroot "${DATAROOT}" \
    --save_root "${SAVE_ROOT}" \
    --base_model_name "${BASE_MODEL}" \
    --dataset "${DATASET}" \
    --shift_type "${SHIFT_TYPE}" \
    --severity "${SEVERITY}" \
    --batch_size "${BATCH_SIZE}" \
    --adaptation cliptta_prior \
    --closed_set \
    --steps 1 \
    --lr 1e-3 \
    --optimizer_type adam \
    --beta_tta 1.0 \
    --beta_reg 0.1 \
    --use_dynamic_prior \
    --prior_alpha "${PRIOR_ALPHA}" \
    --prior_strength "${PRIOR_STRENGTH}" \
    --prior_min_gain "${PRIOR_MIN_GAIN}" \
    --prior_min_conf "${PRIOR_MIN_CONF}" \
    --display_progress
