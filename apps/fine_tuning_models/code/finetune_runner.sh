#!/usr/bin/env bash
# Internal runner — called by tmux_finetune.sh. Do not run directly.
set -euo pipefail

PYTHON="/srv/project/speech/.venv/bin/python3"
SCRIPT="/srv/project/speech/apps/fine_tuning_models/code/finetune_lora.py"

export OMP_NUM_THREADS=112
export MKL_NUM_THREADS=112

echo "======================================================"
echo "  LoRA Fine-Tuning  (CPU · sequential)"
echo "  $(date)"
echo "======================================================"

echo ""
echo ">>> [1/2] qwen3_8b  ($(date))"
"${PYTHON}" "${SCRIPT}" --model qwen3_8b
echo "<<< [1/2] qwen3_8b  DONE  ($(date))"

echo ""
echo ">>> [2/2] llama31_8b  ($(date))"
"${PYTHON}" "${SCRIPT}" --model llama31_8b
echo "<<< [2/2] llama31_8b  DONE  ($(date))"

echo ""
echo "======================================================"
echo "  ALL DONE  ($(date))"
echo "======================================================"
