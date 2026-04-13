#!/usr/bin/env python3
"""
Fine-Tuned LoRA — DW Speech Classification
===========================================
Runs fine-tuned LoRA adapters (via HuggingFace transformers + PEFT)
on political speech paragraphs and scores them for collusion (1-5 Likert scale).

These are LoRA adapters trained on top of 8B base models — they cannot use
llama-server (which requires GGUF). This script loads base model + adapter
directly via transformers/peft and runs CPU inference.

Models:
    - llama31_8b_ft  (LoRA on Meta-Llama-3.1-8B-Instruct)  → DW_llama31_8b_ft_score
    - qwen3_8b_ft    (LoRA on Qwen3-8B)                     → DW_qwen3_8b_ft_score

Usage:
    /srv/project/speech/.venv/bin/python3 \\
        /srv/project/speech/apps/fine_tuning_models/code/ft_lora_classify.py

Notes:
    - Models are loaded sequentially (one at a time) to manage RAM.
    - Each model is fully unloaded before the next is loaded.
    - Resume-safe: skips rows that already have a score in the output file.
    - Saves progress to Excel every BATCH_SIZE rows.
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════
ROOT    = Path("/srv/project/speech")
HF_CACHE = ROOT / "models/hf_cache"

APP_DIR     = ROOT / "apps/fine_tuning_models"
INPUT_FILE  = APP_DIR / "output/speech_classification_results_local_llm.xlsx"
OUTPUT_FILE = APP_DIR / "output/speech_classification_results_ft_lora_2.xlsx"

RUN_TS  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
LOG_DIR = APP_DIR / "temp" / f"ft_lora_{RUN_TS}"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
BATCH_SIZE       = 50     # save to disk every N rows
RESUME_IF_EXISTS = True   # skip rows that already have a score
DEBUG_TEST_ONLY  = False  # set True to run only DEBUG_N rows
DEBUG_N          = 5
N_THREADS        = int(os.environ.get("OMP_NUM_THREADS", min(os.cpu_count() or 16, 64)))

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIG
#   score_col must be different from the base model columns in local_llm_classify.py
#   Base model columns use e.g. "DW_llama31_8b_score"
#   Fine-tuned columns use              "DW_llama31_8b_ft_score"  ← _ft_ suffix
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_CFG: Dict[str, Dict] = {

    "llama31_8b_ft": {
        "enabled":       True,
        "base_model_id": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "adapter_path":  ROOT / "models/fine_tuned/llama31_8b/lora_adapter",
        "score_col":     "DW_llama31_8b_ft_score",   # unique — base uses DW_llama31_8b_score
        "max_new_tokens": 64,                          # only need a short JSON reply
        "no_think":      False,                        # not a Qwen3 model
    },

    "qwen3_8b_ft": {
        "enabled":       True,
        "base_model_id": "Qwen/Qwen3-8B",
        "adapter_path":  ROOT / "models/fine_tuned/qwen3_8b/lora_adapter",
        "score_col":     "DW_qwen3_8b_ft_score",      # unique — base uses DW_qwen3_8b_score
        "max_new_tokens": 64,
        "no_think":      True,   # suppress Qwen3 chain-of-thought for direct JSON output
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS  (identical to local_llm_classify.py for consistency)
# ═══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """
You will analyze a passage from a political speech to determine whether it suggests collusion by powerful actors against the American people.

Collusion refers to secretive coordination to manipulate events or undermine public interests of United States.
Only consider collusion relevant if it concerns the American public, U.S. institutions, or U.S. political, economic, or social interests.

Evaluate the passage based on the following scale:
1. The passage does not mention or imply any collusion.
2. The passage mentions powerful actors but does not suggest collusion, or it explicitly rejects the idea of collusion.
3. The passage refers to powerful actors and could potentially suggest collusion, but the implication is weak or ambiguous.
4. The passage strongly implies that powerful actors are engaged in collusion to manipulate events or undermine public interests.
5. The passage explicitly asserts that powerful actors are engaged in collusion to secretly influence events or act against the public interest.

### Additional Considerations:
- Consider that references to powerful actors or coded language may subtly imply collusion, even if it is not explicitly stated.
- If the paragraph uses phrases or rhetorical cues often associated with conspiracy or secrecy, treat these as indicators of collusion.
- If the speaker alludes to hidden motives or behind-the-scenes coordination, assume this suggests collusion.
- If the passage contains a direct quote or secondary source, only assign it to the speaker if they appear to endorse or align with that material. Analyze tone, intention, and how statements relate to each other to answer whether the speaker believes in collusion or criticizes someone else claiming it.
- Consider collusion in a strict sense, meaning that there are multiple powerful actors for whom coordination is costly.
- Consider that if a speaker quotes someone from a different political party, they may be disagreeing with or criticizing the quoted content. Be attentive to party dynamics in such cases.

Return VALID JSON ONLY in EXACTLY this format (no markdown, no extra text):
{"score": <integer between 1 and 5 inclusive>}
""".strip()

USER_TEMPLATE = """
The speaker is affiliated with the {party} party.
If the paragraph quotes someone from a different political party, the speaker might disagree with the quoted content.

Passage:
"{passage}"\
"""

# ═══════════════════════════════════════════════════════════════════════════════
# RUN STATS
# ═══════════════════════════════════════════════════════════════════════════════
RUN_STATS: Dict[str, Counter] = {
    name: Counter(ok=0, failed=0, skipped=0) for name in MODEL_CFG
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS  (shared with local_llm_classify.py logic)
# ═══════════════════════════════════════════════════════════════════════════════
def all_score_cols() -> List[str]:
    return [v["score_col"] for v in MODEL_CFG.values()]


def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in all_score_cols():
        if c not in df.columns:
            df[c] = ""
    return df


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(str(path), index=False)
    else:
        df.to_excel(str(path), index=False)


def resume_merge(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Merge previously scored rows back into the dataframe."""
    if not out_path.exists():
        return ensure_cols(df)
    old = ensure_cols(
        pd.read_excel(str(out_path), dtype=str, keep_default_na=False)
    )
    df = ensure_cols(df)
    key = next(
        (k for k in ("q_id", "ParagraphID") if k in df.columns and k in old.columns),
        None,
    )
    if key is None:
        return df
    for d in (df, old):
        d[key] = d[key].astype(str).str.strip()
    old = old.sort_values(key).drop_duplicates(key, keep="last")
    old = old[[c for c in [key, *all_score_cols()] if c in old.columns]]
    df = df.merge(old, on=key, how="left", suffixes=("", "_saved"))
    for c in all_score_cols():
        sc = f"{c}_saved"
        if sc in df.columns:
            df[c] = df[c].where(
                df[c].astype(str).str.strip() != "", df[sc].fillna("")
            )
            df.drop(columns=[sc], inplace=True)
    return df


def row_done(df: pd.DataFrame, i: int, col: str) -> bool:
    return str(df.at[i, col]).strip() != ""


def build_messages(row: pd.Series) -> List[Dict]:
    user_msg = USER_TEMPLATE.format(
        party=(str(row.get("party", "")) or "UNKNOWN").strip(),
        passage=str(row.get("paragraph_clean", "")).strip(),
    )
    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_msg},
    ]


def extract_score(text: str) -> Optional[int]:
    """Extract an integer 1-5 from model output text."""
    if not text:
        return None
    t = str(text)
    # Strip Qwen3 thinking tags
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()
    t = re.sub(r"```json|```", "", t).strip()

    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            s = obj.get("score")
            if isinstance(s, str):
                s = int(s.strip())
            if isinstance(s, int) and 1 <= s <= 5:
                return s
    except Exception:
        pass

    m = re.search(r'"score"\s*:\s*([1-5])', t)
    if m:
        return int(m.group(1))

    m = re.search(r'\b([1-5])\b', t)
    if m:
        return int(m.group(1))

    return None


def save_log(model_name: str, row_idx: int, prompt: str,
             raw: str, status: str) -> None:
    obj = {
        "model":     model_name,
        "row_index": row_idx,
        "status":    status,
        "prompt":    prompt,
        "output":    raw,
    }
    fname = f"{model_name}_row{row_idx:05d}_{status}.json"
    with open(LOG_DIR / fname, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOAD / UNLOAD
# ═══════════════════════════════════════════════════════════════════════════════
def load_ft_model(model_name: str, cfg: Dict) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model + LoRA adapter. Returns (model, tokenizer)."""
    base_id      = cfg["base_model_id"]
    adapter_path = str(cfg["adapter_path"])

    print(f"\n  Loading tokenizer: {base_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_id,
        cache_dir=str(HF_CACHE),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  Loading base model: {base_id}  (bfloat16, CPU)")
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        cache_dir=str(HF_CACHE),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = True   # enable KV cache for inference

    print(f"  Applying LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    print(f"  Model ready: {model_name}")
    return model, tokenizer


def unload_model(model: AutoModelForCausalLM) -> None:
    """Free model memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  Model unloaded.")


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE — one row
# ═══════════════════════════════════════════════════════════════════════════════
def score_one_row(model_name: str, row: pd.Series, row_idx: int,
                  cfg: Dict,
                  model: AutoModelForCausalLM,
                  tokenizer: AutoTokenizer) -> Optional[int]:
    """Tokenize one row, generate, parse score. Returns int 1-5 or None."""
    messages  = build_messages(row)
    no_think  = cfg.get("no_think", False)

    # Build the prompt string via the tokenizer's chat template.
    # For Qwen3: pass enable_thinking=False to suppress <think> blocks.
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **({"enable_thinking": False} if no_think else {}),
        )
    except TypeError:
        # Older tokenizer that doesn't support enable_thinking
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    for attempt in range(3):
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=cfg["max_new_tokens"],
                    max_length=None,          # suppress conflict with generation_config
                    do_sample=False,          # greedy → deterministic
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode only the newly generated tokens
            new_ids = output_ids[0][input_ids.shape[-1]:]
            text    = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

            s = extract_score(text)
            if s is not None:
                return s

            save_log(model_name, row_idx, prompt, text, "bad_output")
            tqdm.write(f"  [{model_name}] row {row_idx} attempt {attempt+1}: "
                       f"unparseable output: {text[:120]!r}")

        except Exception as exc:
            tqdm.write(f"  [{model_name}] row {row_idx} attempt {attempt+1}: {exc}")
            save_log(model_name, row_idx, prompt, str(exc), "exception")
            time.sleep(2)

    save_log(model_name, row_idx, prompt, "", "failed")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER — one model over all rows
# ═══════════════════════════════════════════════════════════════════════════════
def run_one_model(df: pd.DataFrame, model_name: str, cfg: Dict,
                  indices: List[int], out_path: Path) -> None:
    col  = cfg["score_col"]
    todo = [i for i in indices if not row_done(df, i, col)]

    if not todo:
        print(f"  {model_name}: all {len(indices)} rows already scored — skipping.")
        RUN_STATS[model_name]["skipped"] = len(indices)
        return

    print(f"  {model_name}: {len(todo)} rows to score "
          f"({len(indices) - len(todo)} already done)")

    model, tokenizer = load_ft_model(model_name, cfg)

    try:
        pbar = tqdm(todo, desc=model_name, dynamic_ncols=True)
        for count, i in enumerate(pbar, 1):
            s = score_one_row(model_name, df.iloc[i], i, cfg, model, tokenizer)
            df.at[i, col] = "" if s is None else str(s)

            if s is None:
                RUN_STATS[model_name]["failed"] += 1
            else:
                RUN_STATS[model_name]["ok"] += 1

            if count % BATCH_SIZE == 0:
                save_df(df, out_path)
                tqdm.write(f"  [checkpoint] batch {count}/{len(todo)} saved")

    except KeyboardInterrupt:
        tqdm.write(f"\n  Interrupted — saving progress ...")
    finally:
        unload_model(model)
        save_df(df, out_path)

    st = RUN_STATS[model_name]
    print(f"\n  {'=' * 50}")
    print(f"  DONE: {model_name}")
    print(f"    ok     = {st['ok']}")
    print(f"    failed = {st['failed']}")
    print(f"  {'=' * 50}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── Validate ──────────────────────────────────────────────────────────────
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    active = {k: v for k, v in MODEL_CFG.items() if v.get("enabled", True)}
    if not active:
        raise ValueError("No models enabled in MODEL_CFG")

    for name, cfg in active.items():
        if not Path(cfg["adapter_path"]).exists():
            raise FileNotFoundError(
                f"LoRA adapter not found: {cfg['adapter_path']}\n"
                "  Run finetune_lora.py first."
            )

    # ── Torch threads ─────────────────────────────────────────────────────────
    torch.set_num_threads(N_THREADS)
    print(f"  PyTorch threads: {N_THREADS}")

    # ── Output path ───────────────────────────────────────────────────────────
    out_path = (
        OUTPUT_FILE.with_stem(OUTPUT_FILE.stem + "_TEST")
        if DEBUG_TEST_ONLY
        else OUTPUT_FILE
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_excel(str(INPUT_FILE), dtype=str, keep_default_na=False)
    df = ensure_cols(df)

    if RESUME_IF_EXISTS and out_path.exists():
        df = resume_merge(df, out_path)

    for c in ("paragraph_clean", "party"):
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c!r}")

    n       = len(df)
    indices = list(range(min(DEBUG_N, n))) if DEBUG_TEST_ONLY else list(range(n))

    # ── Print run info ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  FINE-TUNED LoRA CLASSIFICATION")
    print(f"{'=' * 60}")
    print(f"  Input   : {INPUT_FILE}")
    print(f"  Output  : {out_path}")
    print(f"  Rows    : {n:,} total, {len(indices):,} to process")
    print(f"  Models  : {list(active.keys())}")
    print(f"  Test    : {DEBUG_TEST_ONLY}")
    print(f"  Logs    : {LOG_DIR}")
    print(f"{'=' * 60}")
    print()
    print("  Column name mapping (ft vs base):")
    for name, cfg in active.items():
        print(f"    {cfg['score_col']:35s}  ←  fine-tuned")
    print(f"{'=' * 60}")

    # ── Run each model sequentially ───────────────────────────────────────────
    for name, cfg in active.items():
        print(f"\n{'=' * 60}")
        print(f"  MODEL   : {name}")
        print(f"  Base    : {cfg['base_model_id']}")
        print(f"  Adapter : {Path(cfg['adapter_path']).name}")
        print(f"  Column  : {cfg['score_col']}")
        print(f"{'=' * 60}")
        run_one_model(df, name, cfg, indices, out_path)

    # ── Final save + summary ──────────────────────────────────────────────────
    save_df(df, out_path)

    print(f"\n{'=' * 60}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 60}")
    for name in active:
        st = RUN_STATS[name]
        print(f"  {name:25s}  ok={st['ok']:,}  failed={st['failed']:,}  "
              f"skipped={st['skipped']:,}")
    print(f"\n  Output saved: {out_path}")
    print(f"  Logs saved:   {LOG_DIR}")
    print(f"{'=' * 60}")
    print("  DONE")


if __name__ == "__main__":
    main()
