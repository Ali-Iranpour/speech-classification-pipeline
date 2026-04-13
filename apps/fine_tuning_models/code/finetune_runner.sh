#!/usr/bin/env python3
"""
finetune_lora.py
================
LoRA fine-tuning of 8B instruction models for DW speech classification.

Method  : LoRA via PEFT + SFTTrainer (TRL)
Models  : Qwen/Qwen3-8B-Instruct  |  unsloth/Meta-Llama-3.1-8B-Instruct
Data    : JSONL chat-format (messages array), 300 samples
Device  : CPU (no GPU available on this server)

FIXES applied vs. original:
  [1] DataCollatorForCompletionOnlyLM added — loss is now computed ONLY on the
      assistant turn (the score digit), not the full sequence.
  [2] Qwen3 model switched from base ("Qwen/Qwen3-8B") to instruct variant
      ("Qwen/Qwen3-8B-Instruct") so the chat template matches inference.
  [3] Learning rate raised from 1e-5 → 2e-4 (standard LoRA recommendation;
      1e-5 produced near-zero gradient updates across only ~45 steps).
  [4] Model loaded in float32 (consistent with bf16=False on CPU; avoids
      dtype mismatch between bf16 weights and float32 optimizer states).

Usage:
    .venv/bin/python3 apps/fine_tuning_models/code/finetune_lora.py --model qwen3_8b
    .venv/bin/python3 apps/fine_tuning_models/code/finetune_lora.py --model llama31_8b

Outputs:
    /srv/project/speech/models/fine_tuned/<model_key>/lora_adapter/
        adapter_config.json
        adapter_model.safetensors
        (+ tokenizer files)

To run inference with the fine-tuned adapter later:
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(model, adapter_path)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM  # FIX [1]

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════
ROOT       = Path("/srv/project/speech")
DATA_DIR   = ROOT / "apps/fine_tuning_models/output/ft_sample_jason/finetune_dw_ordinal_scale_sample300"
TRAIN_FILE = DATA_DIR / "train_open.jsonl"
VAL_FILE   = DATA_DIR / "val_open.jsonl"
OUT_BASE   = ROOT / "models/fine_tuned"

HF_CACHE   = ROOT / "models/hf_cache"          # local HuggingFace model cache

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_REGISTRY: dict[str, str] = {
    # FIX [2]: was "Qwen/Qwen3-8B" (base model — no chat template, no instruction
    #          following). Must match the instruct variant used at inference time.
    "qwen3_8b":   "Qwen/Qwen3-8B-Instruct",
    "llama31_8b": "unsloth/Meta-Llama-3.1-8B-Instruct",
}

# LoRA target modules per model family
LORA_TARGETS: dict[str, list[str]] = {
    "qwen3_8b":   ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    "llama31_8b": ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
}

# Response template that marks the START of the assistant turn.
# The loss mask will set all tokens BEFORE this marker to -100.
# These must exactly match what tokenizer.apply_chat_template() produces.
# Run verify_response_template() below if unsure.
RESPONSE_TEMPLATES: dict[str, str] = {
    "qwen3_8b":   "<|im_start|>assistant\n",
    "llama31_8b": "<|start_header_id|>assistant<|end_header_id|>\n\n",
}

# ═══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
# LoRA
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05

# Training
NUM_EPOCHS                  = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4      # effective batch = 4 × 4 = 16
# FIX [3]: was 1e-5 — far too low for LoRA on ~45 total gradient steps.
# At 1e-5 the adapter receives near-zero updates, leaving the model
# essentially at its random LoRA initialisation after all 3 epochs.
LEARNING_RATE               = 2e-4
WARMUP_STEPS                = 5      # ~10% of ~45 steps is plenty
LR_SCHEDULER                = "cosine"
WEIGHT_DECAY                = 0.01
MAX_SEQ_LENGTH              = 2048
SAVE_STEPS                  = 50
LOGGING_STEPS               = 10


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE VERIFICATION UTILITY
# ═══════════════════════════════════════════════════════════════════════════════
def verify_response_template(tokenizer: AutoTokenizer, model_key: str, sample: dict) -> None:
    """
    Print the fully rendered chat template for one training example so you can
    confirm the RESPONSE_TEMPLATES string appears verbatim in the output.

    Call this before training if you change models or the JSONL format.
    """
    rendered = tokenizer.apply_chat_template(
        sample["messages"], tokenize=False, add_generation_prompt=False
    )
    template = RESPONSE_TEMPLATES[model_key]
    print("\n" + "─" * 60)
    print("  TEMPLATE VERIFICATION")
    print("─" * 60)
    print(rendered)
    print("─" * 60)
    if template in rendered:
        print(f"  ✓  Response template found: {repr(template)}")
    else:
        print(f"  ✗  WARNING: response template NOT found in rendered output!")
        print(f"     Template : {repr(template)}")
        print(f"     Update RESPONSE_TEMPLATES['{model_key}'] to match the")
        print(f"     exact string that appears before the score digit above.")
    print("─" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_jsonl(path: Path) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_hf_dataset(train_path: Path, val_path: Path) -> tuple[Dataset, Dataset]:
    """Load JSONL files and return HuggingFace Datasets."""
    train_raw = load_jsonl(train_path)
    val_raw   = load_jsonl(val_path)

    # Each record is {"messages": [...]} — SFTTrainer handles the chat template
    train_ds = Dataset.from_list(train_raw)
    val_ds   = Dataset.from_list(val_raw)

    return train_ds, val_ds


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL + TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════════
def load_model_and_tokenizer(
    model_id: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:

    print(f"\n  Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=str(HF_CACHE),
        trust_remote_code=True,
    )

    # Ensure pad token is set (required for batched training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"   # pad on the right for causal LM

    print(f"  Loading model: {model_id}")
    # FIX [4]: was torch.bfloat16 — causes dtype mismatch on CPU because
    # bf16=False in SFTConfig means optimizer states are float32 while weights
    # are bf16. Using float32 throughout keeps everything consistent on CPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=str(HF_CACHE),
        torch_dtype=torch.float32,      # FIX [4]: was torch.bfloat16
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False          # disable KV cache during training
    model.enable_input_require_grads()      # required for gradient checkpointing

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# LORA ADAPTER
# ═══════════════════════════════════════════════════════════════════════════════
def apply_lora(model: AutoModelForCausalLM, model_key: str) -> AutoModelForCausalLM:
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS[model_key],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
def train(model_key: str) -> None:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    model_id  = MODEL_REGISTRY[model_key]
    out_dir   = OUT_BASE / model_key / "lora_adapter"
    out_dir.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  LoRA Fine-Tuning: {model_key}")
    print(f"  HF model ID     : {model_id}")
    print(f"  Output          : {out_dir}")
    print(f"  Epochs          : {NUM_EPOCHS}")
    print(f"  Batch size      : {PER_DEVICE_TRAIN_BATCH_SIZE}  (accum={GRADIENT_ACCUMULATION_STEPS})")
    print(f"  LR              : {LEARNING_RATE}")
    print(f"  LoRA r/alpha    : {LORA_R}/{LORA_ALPHA}  dropout={LORA_DROPOUT}")
    print(f"  Response tmpl   : {repr(RESPONSE_TEMPLATES[model_key])}")
    print(f"{'=' * 60}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, val_ds = build_hf_dataset(TRAIN_FILE, VAL_FILE)
    print(f"\n  Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    # ── Model + tokenizer ──────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(model_id)
    model = apply_lora(model, model_key)

    # ── Verify response template against a real training example ──────────────
    # This prints the rendered chat and confirms the template string is found.
    # If it prints a WARNING, update RESPONSE_TEMPLATES before continuing.
    verify_response_template(tokenizer, model_key, train_ds[0])

    # ── FIX [1]: Completion-only data collator ─────────────────────────────────
    # Without this, loss is computed over the ENTIRE sequence (system prompt +
    # user message + assistant response). The model then wastes ~2000 tokens of
    # gradient signal re-learning the prompt text, and the one or two tokens
    # that carry the actual score label are effectively invisible.
    # This collator sets all token labels before the response_template to -100,
    # so gradients flow ONLY from the assistant's score output.
    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATES[model_key],
        tokenizer=tokenizer,
    )

    # ── SFT training config ───────────────────────────────────────────────────
    sft_cfg = SFTConfig(
        output_dir=str(out_dir),

        # Epochs & batching
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        # Learning rate schedule
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,

        # Sequence length
        max_length=MAX_SEQ_LENGTH,

        # Data format: SFTTrainer uses the "messages" column + chat template
        dataset_text_field=None,
        dataset_kwargs={"skip_prepare_dataset": False},

        # Checkpointing
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Evaluation
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,

        # Logging
        logging_steps=LOGGING_STEPS,
        report_to="none",

        # Precision — CPU cannot run bf16 training ops; float32 throughout
        bf16=False,
        fp16=False,
        optim="adamw_torch",
        gradient_checkpointing=False,   # not well supported on CPU

        # Reproducibility
        seed=42,
        data_seed=42,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,         # FIX [1]: completion-only loss masking
    )

    print("\n  Starting training ...")
    trainer.train()

    # ── Save final adapter ────────────────────────────────────────────────────
    print(f"\n  Saving LoRA adapter → {out_dir}")
    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    print(f"\n{'=' * 60}")
    print(f"  DONE: {model_key}")
    print(f"  Adapter saved to: {out_dir}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune 8B models for DW speech classification"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model key to fine-tune",
    )
    args = parser.parse_args()

    # Set torch thread count to match available CPUs
    n_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 16))
    torch.set_num_threads(n_threads)
    print(f"  PyTorch threads: {n_threads}")

    train(args.model)


if __name__ == "__main__":
    main()