#!/usr/bin/env python3
"""
Local LLM — DW Speech Classification
=====================================
Runs local GGUF models via llama-server on political speech paragraphs
and scores them for collusion (1-5 Likert scale).

Models:
    - llama-3.1-8b  (Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf)
    - qwen3-8b      (Qwen3-8B-Q4_K_M.gguf)

Usage:
    /srv/project/speech/.venv/bin/python3 \\
        /srv/project/speech/apps/fine_tuning_models/code/local_llm_classify.py

How it works:
    1. For each enabled model, starts a llama-server subprocess
    2. Sends each row via the OpenAI-compatible /v1/chat/completions endpoint
    3. Parses the integer score (1-5) from the response
    4. Saves progress to Excel after every BATCH_SIZE rows
    5. Shuts down the server and moves to the next model
    6. Resume-safe: skips rows that already have a score in the output file
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════
ROOT         = Path("/srv/project/speech")
LLAMA_SERVER = ROOT / "infra/llama.cpp/build/bin/llama-server"
LLAMA_LIB    = str(ROOT / "infra/llama.cpp/build/bin")

APP_DIR     = ROOT / "apps/fine_tuning_models"
INPUT_FILE  = APP_DIR / "output/speech_classification_results_base_multi_models.xlsx"
OUTPUT_FILE = APP_DIR / "output/speech_classification_results_local_llm.xlsx"

RUN_TS  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
LOG_DIR = APP_DIR / "temp" / f"local_llm_{RUN_TS}"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8081            # change if port is busy
BATCH_SIZE  = 50              # save to disk every N rows

RESUME_IF_EXISTS = True       # skip rows that already have a score
DEBUG_TEST_ONLY  = False      # set True to run on only DEBUG_N rows
DEBUG_N          = 5

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIG
#   - Add or remove models here
#   - Each model needs: enabled, model_path, score_col, max_tokens
#   - n_threads: match your CPU core count (run `nproc` in terminal)
#   - ctx_size:  context window; 4096 is safe for most models
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_CFG: Dict[str, Dict] = {

    # ── 8B models ─────────────────────────────────────────────────────────────
    # Fast baselines. Useful for sanity checks and cost-free iteration.
    # Run these first on DEBUG_N=10 to validate the pipeline before 70B runs.

    "qwen3_8b": {
        "enabled":    False,
        "model_path": ROOT / "models/installed/qwen3-8b/Qwen3-8B-Q4_K_M.gguf",
        "score_col":  "DW_qwen3_8b_score",
        "max_tokens": 6144,       # reasoning ON → needs headroom
        "n_threads":  64,
        "ctx_size":   4096,
        "no_think":   False,      # keep reasoning for publication quality
    },

    "llama31_8b": {
        "enabled":    False,
        "model_path": ROOT / "models/installed/llama-3.1-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "score_col":  "DW_llama31_8b_score",
        "max_tokens": 1024,       # no thinking mode, but room for chain-of-thought
        "n_threads":  64,
        "ctx_size":   4096,
        "no_think":   False,      # not a Qwen3 model, field is ignored
    },

    # ── 32B models ────────────────────────────────────────────────────────────
    # The sweet spot for your server: strong quality, reasonable speed.
    # Expect ~3-4x slower than 8B models per row.
    # Qwen3-32B with reasoning ON is likely your best local model overall.

    "qwen25_32b": {
        "enabled":    True,
        "model_path": ROOT / "models/installed/qwen2.5-32b/Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        "score_col":  "DW_qwen25_32b_score",
        "max_tokens": 512,        # no thinking mode → direct JSON
        "n_threads":  64,
        "ctx_size":   4096,
        "no_think":   False,      # not a Qwen3 model, field is ignored
    },

    "qwen3_32b": {
        "enabled":    True,
        "model_path": ROOT / "models/installed/qwen3-32b/Qwen3-32B-Q4_K_M.gguf",
        "score_col":  "DW_qwen3_32b_score",
        "max_tokens": 8192,       # 32B thinks more verbosely — give it full room
        "n_threads":  64,
        "ctx_size":   4096,
        "no_think":   True,      # reasoning ON — this is your strongest local model
    },

    # ── 70B models ────────────────────────────────────────────────────────────
    # Near GPT-4.1 quality. Expect ~8-10x slower than 8B per row.
    # On 800K rows, run only after validating on DEBUG_N=50 first.
    # Both are split files — model_path points to the first part only.

    "qwen25_72b": {
        "enabled":    True,      # ← enable after 32B runs look good
        "model_path": ROOT / "models/installed/qwen2.5-72b/Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        "score_col":  "DW_qwen25_72b_score",
        "max_tokens": 1000,        # no thinking mode
        "n_threads":  64,        # use most of your 200 cores for large models
        "ctx_size":   4096,
        "no_think":   False,      # not a Qwen3 model, field is ignored
    },

    "llama33_70b": {
        "enabled":    True,      # ← enable after 32B runs look good
        "model_path": ROOT / "models/installed/llama-3.3-70b/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "score_col":  "DW_llama33_70b_score",
        "max_tokens": 1000,        # no thinking mode
        "n_threads":  64,
        "ctx_size":   4096,
        "no_think":   False,      # not a Qwen3 model, field is ignored
    },
    "gemma4_26b": {
        "enabled":    False,
        "model_path": ROOT / "models/installed/gemma4-26b/google_gemma-4-26B-A4B-it-Q4_K_M.gguf",
        "score_col":  "DW_gemma4_26b_score",
        "max_tokens": 512,
        "n_threads":  64,
        "ctx_size":   4096,
        "thinking":   False,   # ← was no_think: True (wrong key, did nothing)
    },

    "gemma4_31b": {
        "enabled":    False,
        "model_path": ROOT / "models/installed/gemma4-31b/google_gemma-4-31B-it-Q4_K_M.gguf",
        "score_col":  "DW_gemma4_31b_score",
        "max_tokens": 512,
        "n_threads":  64,
        "ctx_size":   4096,
        "thinking":   False,   # ← same fix
    },

    # ── NEW: Mistral Small 3.2 24B ────────────────────────────────────────────
    # Newer than 3.1. Use this OR 3.1, not both — they are the same size/family.
    "mistral_small_32_24b": {
        "enabled":    True,
        "model_path": ROOT / "models/installed/mistral-small-3.2-24b/mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
        "score_col":  "DW_mistral_small32_24b_score",
        "max_tokens": 512,
        "n_threads":  64,
        "ctx_size":   4096,
        "no_think":   False,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS  (same as the OpenAI version)
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
# HELPERS
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


def build_user_msg(row: pd.Series) -> str:
    return USER_TEMPLATE.format(
        party=(str(row.get("party", "")) or "UNKNOWN").strip(),
        passage=str(row.get("paragraph_clean", "")).strip(),
    )


def extract_score(text: str) -> Optional[int]:
    """Extract an integer 1-5 from model output text."""
    if not text:
        return None
    t = str(text)
    # Strip Qwen3 thinking tags: <think>...</think>
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()
    t = re.sub(r"```json|```", "", t).strip()

    # Try JSON parse
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

    # Regex: {"score": N}
    m = re.search(r'"score"\s*:\s*([1-5])', t)
    if m:
        return int(m.group(1))

    # Last resort: first standalone digit 1-5
    m = re.search(r'\b([1-5])\b', t)
    if m:
        return int(m.group(1))

    return None


def save_log(model_name: str, row_idx: int, user_msg: str,
             raw: str, status: str) -> None:
    """Save per-row inference log to JSON for debugging."""
    obj = {
        "model":     model_name,
        "row_index": row_idx,
        "status":    status,
        "prompt":    user_msg,
        "output":    raw,
    }
    fname = f"{model_name}_row{row_idx:05d}_{status}.json"
    with open(LOG_DIR / fname, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# LLAMA-SERVER MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════
def start_server(model_name: str, cfg: Dict) -> subprocess.Popen:
    """Start a llama-server subprocess and wait until it's healthy."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LLAMA_LIB

    cmd = [
        str(LLAMA_SERVER),
        "-m",     str(cfg["model_path"]),
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "-ngl",   "0",
        "-t",     str(cfg.get("n_threads", 56)),
        "-c",     str(cfg.get("ctx_size", 4096)),
        "--jinja",        # ← forces correct chat template for Gemma 4 + Mistral
        "--log-disable",
    ]

    print(f"\n  Starting llama-server for {model_name} ...")
    print(f"  Command: {' '.join(cmd[:6])} ...")
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for the /health endpoint to return 200
    url = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
    deadline = time.time() + 180  # 3 minute timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=3).status_code == 200:
                print(f"  Server ready for {model_name}")
                return proc
        except Exception:
            pass
        time.sleep(3)

    proc.terminate()
    proc.wait(timeout=15)
    raise RuntimeError(f"llama-server for {model_name} did not start within 3 minutes")


def stop_server(proc: subprocess.Popen) -> None:
    """Gracefully shut down the llama-server subprocess."""
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("  Server stopped.")


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE — one row
# ═══════════════════════════════════════════════════════════════════════════════
def score_one_row(model_name: str, row: pd.Series, row_idx: int,
                  cfg: Dict) -> Optional[int]:
    """Send one row to llama-server and return the score (1-5) or None."""
    user_msg = build_user_msg(row)
    # Qwen3 thinking mode: prefix /no_think so the model replies directly
    if cfg.get("no_think"):
        user_msg = "/no_think\n" + user_msg
    payload = {
        "model":       "local",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "max_tokens":  cfg["max_tokens"],
        "temperature": 0,
        "thinking":    cfg.get("thinking", False),   # ← disables Gemma 4 thinking
    }
    url = f"http://{SERVER_HOST}:{SERVER_PORT}/v1/chat/completions"

    for attempt in range(5):
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            msg  = data.get("choices", [{}])[0].get("message", {})
            text = (msg.get("content", "") or "")
            # Qwen3 thinking mode: actual answer in content, reasoning in
            # reasoning_content. If content is empty, fall back to reasoning.
            if not text.strip():
                text = (msg.get("reasoning_content", "") or "")
            if not text.strip():
                # Log the full raw API response for debugging
                raw_dump = json.dumps(data, indent=2, ensure_ascii=False)[:2000]
                tqdm.write(f"  [{model_name}] row {row_idx} attempt {attempt+1}: "
                           f"empty content. Raw response: {raw_dump[:300]}")
                save_log(model_name, row_idx, user_msg, raw_dump, "empty_response")
                time.sleep(3)
                continue
            s = extract_score(text)
            if s is not None:
                return s
            # Score not parseable — log and retry
            save_log(model_name, row_idx, user_msg, text, "bad_output")
        except Exception as exc:
            tqdm.write(f"  [{model_name}] row {row_idx} attempt {attempt+1}: {exc}")
            time.sleep(min(60, 3 * (2 ** attempt)))
            continue

    save_log(model_name, row_idx, user_msg, "", "failed")
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

    print(f"  {model_name}: {len(todo)} rows to score ({len(indices) - len(todo)} already done)")

    proc = start_server(model_name, cfg)

    try:
        pbar = tqdm(todo, desc=model_name, dynamic_ncols=True)
        for count, i in enumerate(pbar, 1):
            s = score_one_row(model_name, df.iloc[i], i, cfg)
            df.at[i, col] = "" if s is None else str(s)

            if s is None:
                RUN_STATS[model_name]["failed"] += 1
            else:
                RUN_STATS[model_name]["ok"] += 1

            # Checkpoint save
            if count % BATCH_SIZE == 0:
                save_df(df, out_path)
                tqdm.write(f"  [checkpoint] batch {count}/{len(todo)} saved")

    except KeyboardInterrupt:
        tqdm.write(f"\n  Interrupted — saving progress ...")
    finally:
        stop_server(proc)
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
    # ── Validate inputs ───────────────────────────────────────────────────────
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    if not LLAMA_SERVER.exists():
        raise FileNotFoundError(
            f"llama-server not found at {LLAMA_SERVER}\n"
            "  Build it first: see infra/llama.cpp/README.md"
        )

    active = {k: v for k, v in MODEL_CFG.items() if v.get("enabled", True)}
    if not active:
        raise ValueError("No models enabled in MODEL_CFG")

    for name, cfg in active.items():
        if not Path(cfg["model_path"]).exists():
            raise FileNotFoundError(f"Model not found: {cfg['model_path']}")

    # ── Choose output path ────────────────────────────────────────────────────
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
    print(f"  LOCAL LLM CLASSIFICATION")
    print(f"{'=' * 60}")
    print(f"  Input   : {INPUT_FILE}")
    print(f"  Output  : {out_path}")
    print(f"  Rows    : {n:,} total, {len(indices):,} to process")
    print(f"  Models  : {list(active.keys())}")
    print(f"  Test    : {DEBUG_TEST_ONLY}")
    print(f"  Logs    : {LOG_DIR}")
    print(f"{'=' * 60}")

    # ── Run each model sequentially ───────────────────────────────────────────
    for name, cfg in active.items():
        print(f"\n{'=' * 60}")
        print(f"  MODEL: {name}")
        print(f"  File : {Path(cfg['model_path']).name}")
        print(f"{'=' * 60}")
        run_one_model(df, name, cfg, indices, out_path)

    # ── Final save + summary ──────────────────────────────────────────────────
    save_df(df, out_path)

    print(f"\n{'=' * 60}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 60}")
    for name in active:
        st = RUN_STATS[name]
        total = st["ok"] + st["failed"] + st["skipped"]
        print(f"  {name:20s}  ok={st['ok']:,}  failed={st['failed']:,}  skipped={st['skipped']:,}")
    print(f"\n  Output saved: {out_path}")
    print(f"  Logs saved:   {LOG_DIR}")
    print(f"{'=' * 60}")
    print("  DONE")


if __name__ == "__main__":
    main()
