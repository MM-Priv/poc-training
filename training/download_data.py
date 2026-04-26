"""
Download FineWeb-Edu (sample-10BT) parquet files to a local cache.

Run this once from the login node. It writes ~20 GB of parquet files to --cache_dir
and prints a confirmation when done. Network errors will raise immediately — just
rerun; the HuggingFace datasets library resumes partial downloads automatically.

Once complete, run tokenize_data.py to convert the cached parquets to token binaries.

Usage:
    python download_data.py --cache_dir /mnt/data/poc-training/data/hf_cache

    # Smoke test — streaming, no full download
    python download_data.py --cache_dir /mnt/data/poc-training/data/hf_cache --limit_docs 50000
"""

import argparse

from datasets import load_dataset


DATASET_REPO = "HuggingFaceFW/fineweb-edu"
DATASET_NAME = "sample-10BT"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True,
                   help="Directory where parquet files will be stored (~20 GB)")
    p.add_argument("--limit_docs", type=int, default=None,
                   help="Download only N docs via streaming (smoke test)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.limit_docs:
        print(f"Smoke-test mode: streaming {args.limit_docs:,} docs (no full download)")
        ds = load_dataset(
            DATASET_REPO, name=DATASET_NAME, split="train",
            streaming=True, cache_dir=args.cache_dir,
        )
        count = 0
        for _ in ds:
            count += 1
            if count >= args.limit_docs:
                break
        print(f"Streamed {count:,} documents. Cache at: {args.cache_dir}")
        return

    print(f"Downloading {DATASET_REPO}/{DATASET_NAME} to {args.cache_dir} …")
    ds = load_dataset(
        DATASET_REPO, name=DATASET_NAME, split="train",
        streaming=False, cache_dir=args.cache_dir,
    )
    print(f"Done. {len(ds):,} documents cached at: {args.cache_dir}")


if __name__ == "__main__":
    main()
