#!/usr/bin/env python3
"""
Local LLM — DW Speech Classification  (optimised build)
=========================================================
Changes vs original
--------------------
1. max_tokens      reduced to 32 for all direct models (task only needs {"score":3})
2. --parallel 4    llama-server handles 4 concurrent requests via continuous batching
3. Async requests  asyncio + aiohttp keeps 4 requests in-flight at all times per server
4. 2 parallel      two server instances on ports 8081/8082, each gets half the rows
   server instances
5. ctx_size 1024   short paragraphs don't need 4096; halves KV-cache memory per slot
6. --flash-attn on explicitly passed when supported (flag now takes on|off|auto arg)
   --mlock        already on by default in this build; not passed explicitly

Core budget: 2 servers × 55 threads = 110 cores  (within shared-server limit of 100-120)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
import requests as _req
from tqdm import tqdm

# ===============================================================================
# PATHS
# ===============================================================================
ROOT         = Path("/srv/project/speech")
LLAMA_SERVER = ROOT / "infra/llama.cpp/build/bin/llama-server"
LLAMA_LIB    = str(ROOT / "infra/llama.cpp/build/bin")

APP_DIR     = ROOT / "apps/fine_tuning_models"
INPUT_FILE  = APP_DIR / "output/speech_classification_results_base_multi_models.xlsx"
OUTPUT_FILE = APP_DIR / "output/speech_classification_results_local_llm.xlsx"

RUN_TS  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
LOG_DIR = APP_DIR / "temp" / f"local_llm_{RUN_TS}"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ===============================================================================
# SETTINGS
# ===============================================================================
SERVER_PORTS           = [8081, 8082]   # one port per parallel server instance
N_PARALLEL_SERVERS     = 2             # kept at 2 to stay within core budget
THREADS_PER_SERVER     = 55            # 2 x 55 = 110 cores total
CONCURRENCY_PER_SERVER = 4             # async requests in-flight per server
                                        # must match --parallel below
CTX_SIZE               = 1024          # short paragraphs; halves KV-cache vs 4096
BATCH_SIZE             = 100           # checkpoint save every N rows (per shard)

RESUME_IF_EXISTS = True
DEBUG_TEST_ONLY  = False
DEBUG_N          = 10

# ===============================================================================
# MODEL CONFIG  — only the three production models enabled
# ===============================================================================
MODEL_CFG: Dict[str, Dict] = {

    "mistral_small_32_24b": {
        "enabled":    True,
        "model_path": ROOT / "models/installed/mistral-small-3.2-24b/mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
        "score_col":  "DW_mistral_small32_24b_score",
        "max_tokens": 32,       # only needs {"score":3}
        "no_think":   False,
    },

    "qwen25_72b": {
        "enabled":    True,
        "model_path": ROOT / "models/installed/qwen2.5-72b/Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        "score_col":  "DW_qwen25_72b_score",
        "max_tokens": 32,
        "no_think":   False,
    },

    "llama33_70b": {
        "enabled":    True,
        "model_path": ROOT / "models/installed/llama-3.3-70b/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "score_col":  "DW_llama33_70b_score",
        "max_tokens": 32,
        "no_think":   False,
    },

    # === NEW MODELS (recommended) ===
    "qwen3_5_35b_a3b": {   # ← This is the TOP pick (MoE, very fast on CPU)
        "enabled": True,
        "model_path": ROOT / "models/installed/qwen3.5-35b-a3b/Qwen3.5-35B-A3B-Q4_K_M.gguf",
        "score_col": "DW_qwen3_5_35b_a3b_score",
        "max_tokens": 32,
        "no_think": False,
    },
    "qwen3_5_27b": {
        "enabled": True,
        "model_path": ROOT / "models/installed/qwen3.5-27b/Qwen3.5-27B-Q4_K_M.gguf",
        "score_col": "DW_qwen3_5_27b_score",
        "max_tokens": 32,
        "no_think": False,
    },
    "ministral3_14b": {
        "enabled": True,
        "model_path": ROOT / "models/installed/ministral3-14b/Ministral-3-14B-Instruct-2512-Q4_K_M.gguf",
        "score_col": "DW_ministral3_14b_score",
        "max_tokens": 32,
        "no_think": False,
    },
}

# ===============================================================================
# PROMPTS
# ===============================================================================
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

# ===============================================================================
# RUN STATS
# ===============================================================================
RUN_STATS: Dict[str, Counter] = {
    name: Counter(ok=0, failed=0, skipped=0) for name in MODEL_CFG
}

# ===============================================================================
# CAPABILITY CHECK  — flash-attn and mlock
# ===============================================================================
def check_server_capabilities() -> Dict[str, bool]:
    """
    Run `llama-server --help` and check which optional flags are available.
    Returns {"flash_attn": bool, "mlock": bool}

    flash-attn: in this build the flag takes an argument (on|off|auto).
                We search for '-fa,' (with comma) to avoid matching model
                names like 'GLM-4.7-Flash' that appear in the help examples.
                Default in this build is 'auto', but we pass 'on' explicitly
                so there is no ambiguity.

    mlock:      already enabled by default in this build per the --help output
                ("reduce pageouts if not using mlock (default: enabled)").
                We detect and report it, but do NOT pass the flag again since
                it is already on — passing it twice can cause a parse error.
    """
    caps = {"flash_attn": False, "mlock": False}
    try:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = LLAMA_LIB
        result = subprocess.run(
            [str(LLAMA_SERVER), "--help"],
            capture_output=True, text=True, timeout=15, env=env,
        )
        help_text = result.stdout + result.stderr
        # '-fa,' with trailing comma matches the help line:
        #   -fa,   --flash-attn [on|off|auto]  ...
        # and avoids false positives on model names containing 'Flash'.
        caps["flash_attn"] = "-fa," in help_text or "--flash-attn" in help_text
        caps["mlock"]      = "--mlock" in help_text
    except Exception as exc:
        print(f"  [capability check] could not run --help: {exc}")
    return caps


# ===============================================================================
# HELPERS
# ===============================================================================
def all_score_cols() -> List[str]:
    return [v["score_col"] for v in MODEL_CFG.values()]


def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in all_score_cols():
        if c not in df.columns:
            df[c] = ""
    return df


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(str(path), index=False)


def resume_merge(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
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
            df[c] = df[c].where(df[c].astype(str).str.strip() != "", df[sc].fillna(""))
            df.drop(columns=[sc], inplace=True)
    return df


def row_done(df: pd.DataFrame, i: int, col: str) -> bool:
    return str(df.at[i, col]).strip() != ""


def build_user_msg(row: pd.Series, no_think: bool = False) -> str:
    msg = USER_TEMPLATE.format(
        party=(str(row.get("party", "")) or "UNKNOWN").strip(),
        passage=str(row.get("paragraph_clean", "")).strip(),
    )
    return ("/no_think\n" + msg) if no_think else msg


def extract_score(text: str) -> Optional[int]:
    if not text:
        return None
    t = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL).strip()
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


def save_log(model_name: str, row_idx: int, user_msg: str,
             raw: str, status: str) -> None:
    obj = {"model": model_name, "row_index": row_idx,
           "status": status, "prompt": user_msg, "output": raw}
    fname = f"{model_name}_row{row_idx:05d}_{status}.json"
    with open(LOG_DIR / fname, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ===============================================================================
# LLAMA-SERVER MANAGEMENT
# ===============================================================================
def start_server(model_name: str, cfg: Dict, port: int,
                 caps: Dict[str, bool]) -> subprocess.Popen:
    """Start llama-server on the given port and wait until healthy."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LLAMA_LIB

    cmd = [
        str(LLAMA_SERVER),
        "-m",         str(cfg["model_path"]),
        "--host",     "127.0.0.1",
        "--port",     str(port),
        "-ngl",       "0",                            # CPU-only (no GPU)
        "-t",         str(THREADS_PER_SERVER),
        # Total context = ctx_size x parallel slots, allocated at server start.
        # e.g. 1024 tokens x 4 slots = 4096 tokens of KV cache total per server.
        "-c",         str(CTX_SIZE * CONCURRENCY_PER_SERVER),
        "--parallel", str(CONCURRENCY_PER_SERVER),
        "--jinja",
        "--log-disable",
    ]

    # flash-attn: flag now takes an argument in this build (on|off|auto).
    # We pass 'on' explicitly rather than relying on 'auto'.
    if caps.get("flash_attn"):
        cmd.extend(["--flash-attn", "on"])

    # mlock: already on by default in this build — do NOT pass the flag again.
    # Passing it twice causes a parse error on some llama.cpp versions.

    print(f"\n  Starting llama-server [{model_name}] on port {port}")
    print(f"  Threads : {THREADS_PER_SERVER}  |  Parallel slots : {CONCURRENCY_PER_SERVER}"
          f"  |  flash-attn : {caps.get('flash_attn')}  |  mlock : default-on")

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    url      = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + 300   # 5 min timeout — 70B models load slowly on CPU
    while time.time() < deadline:
        try:
            if _req.get(url, timeout=3).status_code == 200:
                print(f"  Server ready [{model_name}] on port {port}")
                return proc
        except Exception:
            pass
        time.sleep(3)

    proc.terminate()
    proc.wait(timeout=15)
    raise RuntimeError(
        f"llama-server [{model_name}] port {port} did not become healthy within 5 min"
    )


def stop_server(proc: subprocess.Popen, port: int) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    print(f"  Server on port {port} stopped.")


# ===============================================================================
# ASYNC INFERENCE  — one shard (subset of rows) against one server port
# ===============================================================================
async def score_row_async(
    session:    aiohttp.ClientSession,
    model_name: str,
    row:        pd.Series,
    row_idx:    int,
    cfg:        Dict,
    port:       int,
) -> Tuple[int, Optional[int]]:
    """Send one row to llama-server; returns (row_idx, score|None)."""
    user_msg = build_user_msg(row, no_think=cfg.get("no_think", False))
    payload = {
        "model":       "local",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "max_tokens":  cfg["max_tokens"],
        "temperature": 0,
    }
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    for attempt in range(5):
        try:
            async with session.post(url, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=180)) as resp:
                data = await resp.json()
                msg  = data.get("choices", [{}])[0].get("message", {})
                text = (msg.get("content", "") or "").strip()
                if not text:
                    text = (msg.get("reasoning_content", "") or "").strip()
                if not text:
                    raw = json.dumps(data)[:300]
                    save_log(model_name, row_idx, user_msg, raw, "empty_response")
                    await asyncio.sleep(3)
                    continue
                s = extract_score(text)
                if s is not None:
                    return row_idx, s
                save_log(model_name, row_idx, user_msg, text, "bad_output")
        except Exception as exc:
            wait = min(60, 3 * (2 ** attempt))
            print(f"  [{model_name}:port{port}] row {row_idx} "
                  f"attempt {attempt + 1}: {exc} (retry in {wait}s)")
            await asyncio.sleep(wait)

    save_log(model_name, row_idx, user_msg, "", "failed")
    return row_idx, None


async def run_shard_async(
    df:         pd.DataFrame,
    model_name: str,
    cfg:        Dict,
    shard:      List[int],
    out_path:   Path,
    port:       int,
    alock:      asyncio.Lock,
) -> None:
    """
    Score all rows in `shard` against the server on `port`.
    Semaphore keeps exactly CONCURRENCY_PER_SERVER requests in-flight.
    Checkpoints to disk every BATCH_SIZE completions (guarded by alock).
    """
    col   = cfg["score_col"]
    todo  = [i for i in shard if not row_done(df, i, col)]
    if not todo:
        return

    sem   = asyncio.Semaphore(CONCURRENCY_PER_SERVER)
    done  = 0
    stats = RUN_STATS[model_name]

    async with aiohttp.ClientSession() as session:

        async def bounded(i: int) -> None:
            nonlocal done
            async with sem:
                row_idx, s = await score_row_async(
                    session, model_name, df.iloc[i], i, cfg, port
                )
            async with alock:
                df.at[row_idx, col] = "" if s is None else str(s)
                stats["failed" if s is None else "ok"] += 1
                done += 1
                if done % BATCH_SIZE == 0:
                    save_df(df, out_path)

        pbar  = tqdm(total=len(todo), desc=f"{model_name}:port{port}",
                     dynamic_ncols=True)
        tasks = [asyncio.create_task(bounded(i)) for i in todo]
        for coro in asyncio.as_completed(tasks):
            await coro
            pbar.update(1)
        pbar.close()


# ===============================================================================
# RUNNER — one model, two parallel shards
# ===============================================================================
def run_one_model(df: pd.DataFrame, model_name: str, cfg: Dict,
                  indices: List[int], out_path: Path,
                  caps: Dict[str, bool]) -> None:
    col     = cfg["score_col"]
    todo    = [i for i in indices if not row_done(df, i, col)]
    skipped = len(indices) - len(todo)
    RUN_STATS[model_name]["skipped"] = skipped

    if not todo:
        print(f"  {model_name}: all {len(indices)} rows already scored — skipping.")
        return

    print(f"\n  {model_name}: {len(todo)} rows to score "
          f"({skipped} already done, split across {N_PARALLEL_SERVERS} servers)")

    # Round-robin split: shard 0 gets rows [0, 2, 4, ...], shard 1 gets [1, 3, 5, ...]
    # This interleaving means both shards checkpoint at roughly the same time.
    shards = [todo[i::N_PARALLEL_SERVERS] for i in range(N_PARALLEL_SERVERS)]

    procs: Dict[int, subprocess.Popen] = {}
    try:
        # Load both model instances in sequence (each takes several minutes for 70B).
        # They share the same GGUF file on disk so no extra storage is needed.
        for port in SERVER_PORTS:
            procs[port] = start_server(model_name, cfg, port, caps)

        def run_shard_in_thread(port: int, shard: List[int]) -> None:
            """Each shard runs in its own thread with its own asyncio event loop."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            alock = asyncio.Lock()
            try:
                loop.run_until_complete(
                    run_shard_async(df, model_name, cfg, shard, out_path, port, alock)
                )
            finally:
                loop.close()

        with ThreadPoolExecutor(max_workers=N_PARALLEL_SERVERS) as pool:
            futures = [
                pool.submit(run_shard_in_thread, SERVER_PORTS[i], shards[i])
                for i in range(N_PARALLEL_SERVERS)
            ]
            for f in futures:
                f.result()   # re-raises any exception from the thread

    except KeyboardInterrupt:
        print(f"\n  Interrupted during {model_name} — saving progress ...")
    finally:
        for port, proc in procs.items():
            stop_server(proc, port)
        save_df(df, out_path)

    st = RUN_STATS[model_name]
    print(f"\n  {'=' * 50}")
    print(f"  DONE: {model_name}")
    print(f"    ok={st['ok']:,}  failed={st['failed']:,}  skipped={st['skipped']:,}")
    print(f"  {'=' * 50}")


# ===============================================================================
# MAIN
# ===============================================================================
def main() -> None:
    # ── Validate ──────────────────────────────────────────────────────────────
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FILE}")
    if not LLAMA_SERVER.exists():
        raise FileNotFoundError(f"llama-server not found: {LLAMA_SERVER}")

    active = {k: v for k, v in MODEL_CFG.items() if v.get("enabled")}
    if not active:
        raise ValueError("No models enabled in MODEL_CFG")
    for name, cfg in active.items():
        if not Path(cfg["model_path"]).exists():
            raise FileNotFoundError(f"Model file not found: {cfg['model_path']}")

    # ── Capability probe ───────────────────────────────────────────────────────
    print("\n  Checking llama-server capabilities ...")
    caps = check_server_capabilities()
    print(f"  flash-attn : {'YES — --flash-attn on will be passed' if caps['flash_attn'] else 'not found in --help (skipped)'}")
    print(f"  mlock      : {'present in --help — already on by default, no flag needed' if caps['mlock'] else 'not found in --help'}")

    # ── Output path ────────────────────────────────────────────────────────────
    out_path = (
        OUTPUT_FILE.with_stem(OUTPUT_FILE.stem + "_TEST")
        if DEBUG_TEST_ONLY else OUTPUT_FILE
    )

    # ── Load & resume ──────────────────────────────────────────────────────────
    df = pd.read_excel(str(INPUT_FILE), dtype=str, keep_default_na=False)
    df = ensure_cols(df)
    if RESUME_IF_EXISTS and out_path.exists():
        df = resume_merge(df, out_path)
    for c in ("paragraph_clean", "party"):
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c!r}")

    n       = len(df)
    indices = list(range(min(DEBUG_N, n))) if DEBUG_TEST_ONLY else list(range(n))

    # ── Print run summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  LOCAL LLM CLASSIFICATION  (optimised)")
    print(f"{'=' * 60}")
    print(f"  Input    : {INPUT_FILE.name}")
    print(f"  Output   : {out_path.name}")
    print(f"  Rows     : {n:,} total  |  {len(indices):,} to process")
    print(f"  Models   : {list(active.keys())}")
    print(f"  Servers  : {N_PARALLEL_SERVERS} x ports {SERVER_PORTS}")
    print(f"  Cores    : {N_PARALLEL_SERVERS} x {THREADS_PER_SERVER} = "
          f"{N_PARALLEL_SERVERS * THREADS_PER_SERVER} threads")
    print(f"  Async    : {CONCURRENCY_PER_SERVER} req/server x {N_PARALLEL_SERVERS} = "
          f"{CONCURRENCY_PER_SERVER * N_PARALLEL_SERVERS} in-flight total")
    print(f"  ctx_size : {CTX_SIZE} tokens x {CONCURRENCY_PER_SERVER} slots = "
          f"{CTX_SIZE * CONCURRENCY_PER_SERVER} total KV tokens per server")
    print(f"  Test     : {DEBUG_TEST_ONLY}")
    print(f"{'=' * 60}")

    # ── Run each model sequentially ────────────────────────────────────────────
    for name, cfg in active.items():
        print(f"\n{'=' * 60}  MODEL: {name}")
        run_one_model(df, name, cfg, indices, out_path, caps)

    # ── Final summary ──────────────────────────────────────────────────────────
    save_df(df, out_path)
    print(f"\n{'=' * 60}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 60}")
    for name in active:
        st = RUN_STATS[name]
        print(f"  {name:30s}  ok={st['ok']:,}  failed={st['failed']:,}  skipped={st['skipped']:,}")
    print(f"\n  Output : {out_path}")
    print(f"  Logs   : {LOG_DIR}")
    print(f"{'=' * 60}")
    print("  DONE")


if __name__ == "__main__":
    main()