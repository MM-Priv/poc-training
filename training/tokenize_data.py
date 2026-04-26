"""
Tokenize FineWeb-Edu (sample-10BT) from a local HuggingFace cache into uint32 binaries.

Run after download_data.py has populated --cache_dir. Tokenization is the slow step
(~2–4 hours for 10BT on a login node); if interrupted, simply rerun — it restarts from
scratch but the parquet files are already cached so no network traffic is needed.

Output (in --output_dir):
    train.bin   -- uint32 flat token array (~9.9B tokens, ~40 GB)
    val.bin     -- uint32 flat token array (~100M tokens, ~400 MB)
    meta.json   -- dataset provenance and token counts

Requirements:
    HF_TOKEN env var — needed only to pull the Llama 3.1 tokenizer (gated model).
    FineWeb-Edu itself is public; no token required for the dataset.

Usage:
    HF_TOKEN=hf_... python tokenize_data.py \\
        --cache_dir /mnt/data/poc-training/data/hf_cache \\
        --output_dir /mnt/data/poc-training/data

    # Smoke test with 50 000 docs (uses streaming; no full cache needed)
    HF_TOKEN=hf_... python tokenize_data.py \\
        --cache_dir /mnt/data/poc-training/data/hf_cache \\
        --output_dir /tmp/test --limit_docs 50000
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


DATASET_REPO = "HuggingFaceFW/fineweb-edu"
DATASET_NAME = "sample-10BT"
TOKENIZER_ID = "meta-llama/Meta-Llama-3.1-8B"
VAL_TOKENS   = 100_000_000   # 100M tokens held out for validation
SHARD_TOKENS = 500_000_000   # flush to disk every 500M tokens to cap RAM use


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir",   required=True,
                   help="HuggingFace datasets cache populated by download_data.py")
    p.add_argument("--output_dir",  required=True,
                   help="Where to write train.bin, val.bin, meta.json")
    p.add_argument("--limit_docs",  type=int, default=None,
                   help="Stop after N docs (smoke test; uses streaming)")
    p.add_argument("--hf_token",    default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace token for gated tokenizer repo")
    return p.parse_args()


def load_tokenizer(hf_token, cache_dir):
    print(f"Loading tokenizer: {TOKENIZER_ID}")
    if not hf_token:
        print("WARNING: HF_TOKEN not set — will fail for gated repos.")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, token=hf_token, cache_dir=cache_dir)
    print(f"  vocab_size={tok.vocab_size}  eos_token_id={tok.eos_token_id}")
    return tok


def tokenize_and_write(dataset, tokenizer, out_dir: Path, limit_docs: int | None):
    print("\nTokenising …")
    eos_id = tokenizer.eos_token_id

    tmp_path   = out_dir / "all_tokens.bin.tmp"
    train_path = out_dir / "train.bin"
    val_path   = out_dir / "val.bin"

    buf: list[int] = []
    total_tokens = 0
    n_docs = 0
    t0 = time.perf_counter()

    def flush(buf, f):
        arr = np.array(buf, dtype=np.uint32)
        arr.tofile(f)
        return len(arr)

    with open(tmp_path, "wb") as f_tmp:
        for example in tqdm(dataset, desc="tokenising", unit="doc"):
            text = example.get("text", "")
            if text:
                ids = tokenizer.encode(text, add_special_tokens=False)
                buf.extend(ids)
                buf.append(eos_id)

            n_docs += 1

            if n_docs % 100_000 == 0:
                elapsed = time.perf_counter() - t0
                rate = (total_tokens + len(buf)) / elapsed / 1e6
                print(f"  docs={n_docs:,}  tokens={(total_tokens+len(buf))/1e9:.2f}B  "
                      f"{rate:.1f}M tok/s", flush=True)

            if len(buf) >= SHARD_TOKENS:
                total_tokens += flush(buf, f_tmp)
                buf = []

            if limit_docs and n_docs >= limit_docs:
                print(f"Stopping at {n_docs:,} docs (--limit_docs)")
                break

        if buf:
            total_tokens += flush(buf, f_tmp)

    elapsed = time.perf_counter() - t0
    print(f"\nTokenised {n_docs:,} docs | {total_tokens/1e9:.2f}B tokens | {elapsed/60:.1f} min")

    val_size   = min(VAL_TOKENS, total_tokens // 10)
    train_size = total_tokens - val_size
    print(f"Split → train={train_size/1e9:.2f}B  val={val_size/1e6:.0f}M tokens")

    all_data = np.memmap(tmp_path, dtype=np.uint32, mode="r")
    for path, start, end in [
        (train_path, 0,          train_size),
        (val_path,   train_size, total_tokens),
    ]:
        chunk = np.array(all_data[start:end], dtype=np.uint32)
        chunk.tofile(path)
        print(f"  {path.name}  {len(chunk):,} tokens  {chunk.nbytes/1e9:.2f} GB")

    tmp_path.unlink()
    return n_docs, train_size, val_size


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.hf_token, args.cache_dir)

    if args.limit_docs:
        print(f"Smoke-test mode: streaming {args.limit_docs:,} docs")
        dataset = load_dataset(
            DATASET_REPO, name=DATASET_NAME, split="train",
            streaming=True, cache_dir=args.cache_dir,
        )
    else:
        print("Loading dataset from cache …")
        dataset = load_dataset(
            DATASET_REPO, name=DATASET_NAME, split="train",
            streaming=False, cache_dir=args.cache_dir,
        )

    n_docs, train_tokens, val_tokens = tokenize_and_write(
        dataset, tokenizer, out_dir, args.limit_docs,
    )

    meta = {
        "dataset":      f"{DATASET_REPO}/{DATASET_NAME}",
        "tokenizer":    TOKENIZER_ID,
        "vocab_size":   tokenizer.vocab_size,
        "eos_token_id": tokenizer.eos_token_id,
        "dtype":        "uint32",
        "n_docs":       n_docs,
        "train_tokens": train_tokens,
        "val_tokens":   val_tokens,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\nmeta.json written. Tokenization complete.")


if __name__ == "__main__":
    main()
