#!/usr/bin/env python3
"""Parse nccl-tests all_reduce_perf output and write a result JSON fragment."""

import argparse, json, os, re, sys

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to captured nccl-tests stdout")
parser.add_argument("--output-dir", required=True)
parser.add_argument("--threshold", type=float, default=350, help="Avg bus bandwidth threshold in GB/s")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

try:
    text = open(args.input).read()
except OSError:
    text = ""

avg_bw = next(
    (line.split()[-1] for line in text.splitlines() if "Avg bus bandwidth" in line),
    None,
)
data_rows = [l for l in text.splitlines() if re.match(r"^\s+\d", l)]
# Find the out-of-place busbw column index from the header.
# Header has two busbw columns: in-place and out-of-place. We want the last one.
busbw_col = 7  # default for nccl-tests 2.x
for line in text.splitlines():
    if "#" in line and "busbw" in line:
        cols = line.replace("#", " ").split()
        for i, col in enumerate(cols):
            if col == "busbw":
                busbw_col = i  # keep updating; last match is out-of-place
        break
peak_bw = data_rows[-1].split()[busbw_col] if data_rows else "0"

status = "PASS"
message = ""
if avg_bw is None:
    status = "FAIL"
    message = "Could not parse bandwidth from nccl-tests output"
elif float(avg_bw) < args.threshold:
    status = "FAIL"
    message = f"Avg bus bandwidth {avg_bw} GB/s below threshold {args.threshold} GB/s"

with open(f"{args.output_dir}/02_nccl_allreduce.json", "w") as f:
    json.dump({
        "check": "nccl_allreduce",
        "status": status,
        "metrics": {
            "avg_bus_bw_GBs": float(avg_bw or 0),
            "peak_bus_bw_GBs": float(peak_bw),
            "threshold_GBs": args.threshold,
        },
        "messages": [message],
    }, f, indent=2)

print(f"[{status}] NCCL All-Reduce: avg {avg_bw or 'N/A'} GB/s (threshold: {args.threshold} GB/s)")
if message:
    print(f"       {message}")
sys.exit(0 if status == "PASS" else 1)
