#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

CONFIG="${CONFIG:-configs/gw.yaml}"
SPLIT="${SPLIT:-100%_beijing}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
SAVE_PATH="${SAVE_PATH:-exp/gw/supervised/${SPLIT}}"

LABELED_IDS="splits/gw/${SPLIT}/labeled.txt"

if [[ ! -f "$LABELED_IDS" ]]; then
  echo "Missing labeled split file: ${LABELED_IDS}" >&2
  exit 1
fi

mkdir -p "$SAVE_PATH"

torchrun \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  supervised.py \
  --config="$CONFIG" \
  --labeled-id-path="$LABELED_IDS" \
  --save-path="$SAVE_PATH" \
  2>&1 | tee "$SAVE_PATH/out.log"
