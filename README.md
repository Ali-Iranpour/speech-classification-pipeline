# Fine-Tuning Models — DW Speech Classification

LoRA fine-tuning of 8B instruction models to classify political speech paragraphs
on a 1–5 collusion scale. Uses HuggingFace `transformers` + `peft` + `trl`.

---

## Directory Structure

```
apps/fine_tuning_models/
│
├── README.md                          ← this file
│
├── code/                              ← all runnable scripts
│   ├── finetune_lora.py               ← main fine-tuning script (LoRA / SFTTrainer)
│   ├── finetune_runner.sh             ← sequential runner: runs both models one after another
│   ├── tmux_finetune.sh               ← launches tmux session and calls finetune_runner.sh
│   ├── local_llm_classify.py          ← inference with local GGUF models via llama-server
│   ├── Fine_tune.ipynb                ← notebook (OpenAI fine-tuning experiments)
│   └── Batch.ipynb                    ← notebook (OpenAI batch API experiments)
│
├── input/                             ← raw data for fine-tuning
│   ├── speech_classification_cleaned_dataset.xlsx
│   ├── speech_classification_results_*.xlsx
│   ├── manual_review_checkmarks_5000_non_key.xlsx
│   └── ft_sample/                     ← stratified training samples at various sizes
│       ├── speech_sample_100_stratified.xlsx
│       ├── speech_sample_200_stratified.xlsx
│       ├── speech_sample_200_with_reasonings.xlsx
│       ├── speech_sample_300_stratified.xlsx
│       └── speech_sample_400_stratified.xlsx
│
├── output/
│   ├── ft_sample_jason/               ← JSONL files ready for fine-tuning
│   │   └── finetune_dw_ordinal_scale_sample300/   ← ACTIVE training data (300 samples)
│   │       ├── train_open.jsonl       ← 240 training examples (chat format)
│   │       ├── val_open.jsonl         ← 60 validation examples (chat format)
│   │       ├── train_openai.jsonl     ← OpenAI format variant
│   │       ├── val_openai.jsonl
│   │       ├── train_gemini.jsonl     ← Gemini format variant
│   │       ├── val_gemini.jsonl
│   │       ├── finetune_job_log_openai.json   ← log from prior OpenAI fine-tune run
│   │       └── README.md
│   └── batch_results/                 ← OpenAI batch API output files
│       ├── batch_input_reasoning_yes.jsonl
│       └── batch_input_score_only.jsonl
│
└── temp/                              ← auto-generated run logs (gitignored)


models/fine_tuned/                     ← saved LoRA adapters (outside app dir)
│
├── qwen3_8b/
│   └── lora_adapter/                  ← COMPLETE ✓
│       ├── adapter_model.safetensors  ← LoRA weights (167 MB)
│       ├── adapter_config.json        ← LoRA config (r=16, alpha=32, base model ID)
│       ├── tokenizer.json             ← tokenizer (11 MB)
│       ├── tokenizer_config.json
│       ├── chat_template.jinja        ← Qwen3 chat template
│       ├── README.md                  ← auto-generated PEFT model card
│       └── checkpoint-45/             ← final training checkpoint (step 45 = epoch 3)
│           ├── adapter_model.safetensors
│           ├── adapter_config.json
│           ├── optimizer.pt           ← optimizer state (resume training from here)
│           ├── scheduler.pt           ← LR scheduler state
│           ├── trainer_state.json     ← full training log (loss, accuracy per step)
│           ├── training_args.bin
│           ├── rng_state.pth
│           ├── tokenizer.json
│           └── tokenizer_config.json
│
└── llama31_8b/
    └── lora_adapter/                  ← PENDING (gated repo — needs HF token)
```

---

## Scripts

### `finetune_lora.py` — Main Training Script

Handles the full fine-tuning pipeline for one model at a time.

```
--model qwen3_8b     →  Qwen/Qwen3-8B
--model llama31_8b   →  meta-llama/Llama-3.1-8B-Instruct  (gated — needs HF_TOKEN)
                        unsloth/Meta-Llama-3.1-8B-Instruct (ungated mirror)
```

**Pipeline:**
1. Loads `train_open.jsonl` + `val_open.jsonl` as HuggingFace `Dataset`
2. Downloads model from HuggingFace Hub into `models/hf_cache/`
3. Applies LoRA adapter via `peft.get_peft_model()`
4. Trains with `trl.SFTTrainer` using the `messages` column + model chat template
5. Saves adapter to `models/fine_tuned/<model_key>/lora_adapter/`

**Key design decisions:**
- `pad_token = eos_token` — fixes missing pad token on LLaMA/Qwen
- `padding_side = "right"` — required for causal LM training
- `use_cache = False` — disabled during training (re-enable for inference)
- `torch_dtype = bfloat16` — halves memory; CPU does not run bf16 compute ops
- `dataset_text_field = None` — tells SFTTrainer to use the `messages` column

---

### `finetune_runner.sh` — Sequential Runner

Runs both models back-to-back in a single shell session. Sets CPU thread counts
and calls `finetune_lora.py --model` for each model in order.

```bash
bash finetune_runner.sh
```

Edit this file to skip a model (comment out one block) or add models.

---

### `tmux_finetune.sh` — tmux Launcher

Creates a detachable tmux session so training survives SSH disconnections.

```bash
bash apps/fine_tuning_models/code/tmux_finetune.sh
```

| Window | Name      | Contents                       |
|--------|-----------|--------------------------------|
| 0      | `runner`  | Sequential training progress   |
| 1      | `logs`    | Live adapter output watcher    |

**tmux controls:**  `Ctrl-b d` detach  ·  `Ctrl-b n/p` next/prev window

---

### `local_llm_classify.py` — GGUF Inference Script

Separate from fine-tuning. Runs the **original quantized GGUF models** via
`llama-server` for large-scale inference (not fine-tuning). See its own
docstring for usage.

---

## Hyperparameters

| Parameter                     | Value        | Notes                                  |
|-------------------------------|--------------|----------------------------------------|
| `lora_r`                      | 16           | LoRA rank                              |
| `lora_alpha`                  | 32           | Scaling = alpha/r = 2×                 |
| `lora_dropout`                | 0.05         | Regularisation                         |
| `target_modules`              | q/k/v/o/gate/up/down_proj | All attention + FFN layers  |
| `num_train_epochs`            | 3            |                                        |
| `per_device_train_batch_size` | 4            |                                        |
| `gradient_accumulation_steps` | 4            | Effective batch = 16                   |
| `learning_rate`               | 1e-5         | Cosine decay with 10 warmup steps      |
| `weight_decay`                | 0.01         |                                        |
| `max_length`                  | 2048 tokens  | Covers full system+user+assistant      |
| `optimizer`                   | adamw_torch  |                                        |
| `device`                      | CPU          | 112 cores, 503 GB RAM                  |

---

## Training Results — Qwen3-8B (completed)

| Step | Epoch | Train Loss | Val Loss | Token Accuracy |
|------|-------|-----------|----------|----------------|
| 10   | 0.67  | 2.900     | —        | 47.0 %         |
| 20   | 1.33  | 2.797     | —        | 47.2 %         |
| 30   | 2.00  | 2.649     | —        | 47.9 %         |
| 40   | 2.67  | 2.562     | —        | 48.3 %         |
| 45   | 3.00  | —         | **2.566**| **48.4 %**     |

Total training steps: 45 (240 samples / batch 4 / accum 4 = 15 steps/epoch × 3)  
Total tokens seen: ~339K train · ~382K val

---

## Data Format

Each JSONL line follows the standard chat format:

```json
{
  "messages": [
    {"role": "system",    "content": "You will analyze a passage ..."},
    {"role": "user",      "content": "The speaker is affiliated with the Republican party.\n\nPassage: \"...\""},
    {"role": "assistant", "content": "{\"score\": 3}"}
  ]
}
```

**Files:**
- `train_open.jsonl` — 240 examples (80 % stratified split)
- `val_open.jsonl`   — 60 examples  (20 % stratified split)

---

## Loading a Fine-Tuned Adapter for Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_id  = "Qwen/Qwen3-8B"
adapter_path   = "/srv/project/speech/models/fine_tuned/qwen3_8b/lora_adapter"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
model     = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16)
model     = PeftModel.from_pretrained(model, adapter_path)
model.eval()
```

---

## Running Fine-Tuning

**Full run (both models):**
```bash
bash /srv/project/speech/apps/fine_tuning_models/code/tmux_finetune.sh
```

**Single model:**
```bash
export OMP_NUM_THREADS=112
/srv/project/speech/.venv/bin/python3 \
    /srv/project/speech/apps/fine_tuning_models/code/finetune_lora.py \
    --model qwen3_8b
```

**LLaMA (gated repo — set token first):**
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
export OMP_NUM_THREADS=112
/srv/project/speech/.venv/bin/python3 \
    /srv/project/speech/apps/fine_tuning_models/code/finetune_lora.py \
    --model llama31_8b
```
