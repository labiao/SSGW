#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

CONFIG="${CONFIG:-configs/gw.yaml}"
SPLIT="${SPLIT:-10%_beijing}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
SAVE_PATH="${SAVE_PATH:-exp/gw/ssgw/${SPLIT}}"

LABELED_IDS="splits/gw/${SPLIT}/labeled.txt"
UNLABELED_IDS="splits/gw/${SPLIT}/unlabeled.txt"

if [[ ! -f "$LABELED_IDS" || ! -f "$UNLABELED_IDS" ]]; then
  echo "Missing labeled or unlabeled split files under splits/gw/${SPLIT}" >&2
  exit 1
fi

mkdir -p "$SAVE_PATH"

torchrun \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  ssgw.py \
  --config="$CONFIG" \
  --labeled-id-path="$LABELED_IDS" \
  --unlabeled-id-path="$UNLABELED_IDS" \
  --save-path="$SAVE_PATH" \
  2>&1 | tee "$SAVE_PATH/out.log"
