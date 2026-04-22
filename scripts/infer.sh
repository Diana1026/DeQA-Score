#!/bin/bash

# This script uses bash arrays. If invoked via `sh scripts/infer.sh ...`,
# re-exec under bash so it still works (some systems link `sh` to `dash`).
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=./:$PYTHONPATH
shift

BACKBONE="${BACKBONE:-mplug}"
MODEL_PATH="${MODEL_PATH:-./checkpoints/deqa_lora_2023}"
MODEL_BASE="${MODEL_BASE:-zhiyuanyou/DeQA-Score-Mix3}"
PREPROCESSOR_PATH="${PREPROCESSOR_PATH:-./preprocessor/}"
SAVE_DIR="${SAVE_DIR:-results/deqa_lora/}"
META_PATHS="${META_PATHS:-./data/AIGCIQA2023/metas/val_alignment.json}"
DEVICE="${DEVICE:-cuda:0}"
FUSE_ALPHA="${FUSE_ALPHA:-1.0}"
PYTHON_BIN="python"
if [ -x "./.venv/bin/python" ]; then
  PYTHON_BIN="./.venv/bin/python"
fi

ARGS=(
  src/evaluate/iqa_eval_vqa_current.py
  --backbone "$BACKBONE"
  --model-path "$MODEL_PATH"
  --save-dir "$SAVE_DIR"
  --meta-paths "$META_PATHS"
  --device "$DEVICE"
  --fuse-alpha "$FUSE_ALPHA"
)

if [ -n "$PREPROCESSOR_PATH" ]; then
  ARGS+=(--preprocessor-path "$PREPROCESSOR_PATH")
fi

if [ "$BACKBONE" = "mplug" ] && [ -n "$MODEL_BASE" ]; then
  ARGS+=(--model-base "$MODEL_BASE")
fi

"$PYTHON_BIN" "${ARGS[@]}" "$@"

# Examples:
# BACKBONE=mplug MODEL_PATH=./checkpoints/test MODEL_BASE=zhiyuanyou/DeQA-Score-Mix3 \
# SAVE_DIR=results/test META_PATHS=./data/AGIQA3K/metas/val_alignment.json sh scripts/infer.sh 0
#
# BACKBONE=minicpm_v25 MODEL_PATH=openbmb/MiniCPM-Llama3-V-2_5 MODEL_BASE= \
# PREPROCESSOR_PATH=openbmb/MiniCPM-Llama3-V-2_5 SAVE_DIR=results/minicpm \
# META_PATHS=./data/AIGCIQA2023/metas/val_alignment.json sh scripts/infer.sh 0
