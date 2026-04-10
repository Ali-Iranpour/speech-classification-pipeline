#!/usr/bin/env bash
# =============================================================================
# tmux_finetune.sh
# =============================================================================
# Runs LoRA fine-tuning for qwen3_8b and llama31_8b sequentially in tmux.
#
# Usage:
#   bash /srv/project/speech/apps/fine_tuning_models/code/tmux_finetune.sh
#
# Monitor:
#   tmux attach -t finetune     (Ctrl-b d to detach, Ctrl-b n/p for windows)
#
# Output:
#   /srv/project/speech/models/fine_tuned/qwen3_8b/lora_adapter/
#   /srv/project/speech/models/fine_tuned/llama31_8b/lora_adapter/
# =============================================================================

set -euo pipefail

SESSION="finetune"
RUNNER="/srv/project/speech/apps/fine_tuning_models/code/finetune_runner.sh"

chmod +x "${RUNNER}"

# Kill existing session if any
tmux kill-session -t "${SESSION}" 2>/dev/null || true

# Window 0: sequential runner (executes finetune_runner.sh as a real file)
tmux new-session -d -s "${SESSION}" -n "runner" -x 220 -y 50
tmux send-keys -t "${SESSION}:runner" "bash ${RUNNER}" Enter

# Window 1: output watcher
tmux new-window -t "${SESSION}" -n "logs"
tmux send-keys -t "${SESSION}:logs" \
  "watch -n 30 'ls -lth /srv/project/speech/models/fine_tuned/*/lora_adapter/ 2>/dev/null | head -20'" Enter

tmux select-window -t "${SESSION}:runner"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  tmux session 'finetune' started                     ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  window 0 'runner' — sequential LoRA fine-tuning     ║"
echo "║  window 1 'logs'   — adapter output watcher          ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  1. Qwen/Qwen3-8B              → qwen3_8b            ║"
echo "║  2. meta-llama/Llama-3.1-8B   → llama31_8b          ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Method  : LoRA (r=16, alpha=32, dropout=0.05)       ║"
echo "║  Epochs  : 3                                         ║"
echo "║  Batch   : 4 × accum 4 = 16 effective               ║"
echo "║  LR      : 1e-5 cosine                               ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Ctrl-b d  detach · Ctrl-b n/p  switch windows      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

tmux attach -t "${SESSION}"
