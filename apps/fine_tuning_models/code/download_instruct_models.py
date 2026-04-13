#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

INSTRUCT_DIR = Path("/srv/project/speech/models/instruct")

MODEL_REGISTRY: dict[str, dict] = {
    "llama31_8b": {
        "repo_id":   "meta-llama/Llama-3.1-8B-Instruct",
        "local_dir": INSTRUCT_DIR / "Llama-3.1-8B-Instruct",
        "gated":     True,
        "size_gb":   16,
    },
}

IGNORE_PATTERNS = [
    "*.msgpack", "*.h5", "flax_model*", "tf_model*",
    "rust_model*", "onnx/*", "original/*",
]


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
def fmt_seconds(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s" if m else f"{s}s"


def check_disk_space_for_model(path: Path, required_gb: float):
    import shutil
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < required_gb:
        raise RuntimeError(
            f"Not enough disk space: {free_gb:.1f} GB free, need ~{required_gb} GB"
        )
    print(f"  ✓ Disk OK: {free_gb:.1f} GB free (needs ~{required_gb} GB)")


def check_token(token: str | None, model_key: str) -> str | None:
    resolved = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if MODEL_REGISTRY[model_key]["gated"] and not resolved:
        raise RuntimeError(
            f"{model_key} is gated. Provide HF token (env or --token)."
        )
    return resolved


def is_model_complete(local_dir: Path) -> bool:
    # Better check: ensure safetensors exist
    return any(local_dir.glob("*.safetensors"))


# ═══════════════════════════════════════════════════════════════════
# DOWNLOAD
# ═══════════════════════════════════════════════════════════════════
def download_model(model_key: str, token: str | None, dry_run: bool):
    from huggingface_hub import snapshot_download, list_repo_files

    cfg = MODEL_REGISTRY[model_key]
    repo_id = cfg["repo_id"]
    local_dir = cfg["local_dir"]
    size_gb = cfg["size_gb"]

    print(f"\n=== {model_key} ===")
    print(f"Repo: {repo_id}")
    print(f"Dest: {local_dir}")

    if dry_run:
        print("[DRY RUN] Listing files...")
        files = list_repo_files(repo_id, token=token)
        print(f"Total files: {len(files)}")
        return

    local_dir.mkdir(parents=True, exist_ok=True)

    # Better completeness check
    if is_model_complete(local_dir):
        print("  ✓ Already downloaded (safetensors found), skipping.")
        return

    token = check_token(token, model_key)

    # Disk check per model
    check_disk_space_for_model(INSTRUCT_DIR, size_gb + 2)

    print("  ↓ Downloading with retry...")

    t0 = time.time()

    # Retry logic
    for attempt in range(3):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                ignore_patterns=IGNORE_PATTERNS,
                token=token,
                resume_download=True,
            )
            break
        except Exception as e:
            print(f"  ⚠ Attempt {attempt+1} failed: {e}")
            if attempt == 2:
                raise
            time.sleep(10)

    print(f"  ✓ Done in {fmt_seconds(time.time() - t0)}")


# ═══════════════════════════════════════════════════════════════════
# VERIFY
# ═══════════════════════════════════════════════════════════════════
def verify_model(model_key: str):
    from transformers import AutoTokenizer

    cfg = MODEL_REGISTRY[model_key]
    local_dir = cfg["local_dir"]

    print(f"Verifying {model_key}...")

    tokenizer = AutoTokenizer.from_pretrained(str(local_dir), trust_remote_code=True)

    dummy = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "ok"},
    ]

    rendered = tokenizer.apply_chat_template(dummy, tokenize=False)

    if "assistant" in rendered:
        print("  ✓ Template OK")
    else:
        print("  ⚠ Template might be wrong")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--token")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    keys = [args.model] if args.model else list(MODEL_REGISTRY.keys())

    INSTRUCT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading: {keys}")
    print(f"Parallel downloads enabled\n")

    # ✅ Parallel downloads
    with ThreadPoolExecutor(max_workers=min(2, len(keys))) as executor:
        futures = [
            executor.submit(download_model, key, args.token, args.dry_run)
            for key in keys
        ]
        for f in futures:
            f.result()

    # Verify after all downloads
    if not args.dry_run and not args.skip_verify:
        for key in keys:
            verify_model(key)

    print("\n✅ ALL DONE\n")


if __name__ == "__main__":
    main()