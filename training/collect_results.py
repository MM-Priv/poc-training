"""Walk training/results/*/ and produce a structured result.json per run
plus a top-level summary.{json,csv} for the final report.

Reads:
  <run_dir>/run_meta.json   — algorithm + parallelism (written by each sbatch)
  <run_dir>/train.log       — TorchTitan stdout tee'd output
    OR  logs/exp1_ddp_<jobid>.err  — raw train.py stderr (for exp1)

Writes:
  <run_dir>/result.json     — steps + summary for that run
  training/results/summary.json
  training/results/summary.csv

Usage:
  python training/collect_results.py [--results_dir path]
"""

import argparse
import csv
import json
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")  # strip color codes

# TorchTitan step line:
#   step: N  loss: X  grad_norm: Y  memory: NGiB(%)  tps: N  tflops: N  mfu: N%
TT_STEP = re.compile(
    r"step:\s*(\d+)\s+loss:\s*([-\d.]+)\s+grad_norm:\s*([-\d.]+)\s+"
    r"memory:\s*([\d.]+)GiB\(([\d.]+)%\)\s+"
    r"tps:\s*([\d,]+)\s+tflops:\s*([\d.]+)\s+mfu:\s*([\d.]+)%"
)

# Raw train.py step line:
#   step=N/M  loss=X  ppl=Y  lr=Z  tok/s=NK  tok/s/GPU=N  MFU=N%  mem=NGB
DDP_STEP = re.compile(
    r"step=\s*(\d+)/\d+\s+loss=([-\d.]+)\s+ppl=[\d.]+\s+lr=[-\d.e]+\s+"
    r"tok/s=([\d.]+)K\s+tok/s/GPU=(\d+)\s+MFU=([\d.]+)%\s+mem=([\d.]+)GB"
)


def parse_torchtitan_log(log_path: Path) -> list[dict]:
    """Extract one metrics record per step (dedup across ranks; drop PP sentinel losses)."""
    rows: dict[int, dict] = {}
    for raw in log_path.read_text(errors="ignore").splitlines():
        line = ANSI.sub("", raw)
        m = TT_STEP.search(line)
        if not m:
            continue
        step = int(m.group(1))
        loss = float(m.group(2))
        # Prefer a real loss over PP's -1 sentinel from non-last stages.
        if step in rows and loss < 0 and rows[step]["loss"] >= 0:
            continue
        rows[step] = dict(
            step=step,
            loss=loss,
            grad_norm=float(m.group(3)),
            memory_gib=float(m.group(4)),
            memory_pct=float(m.group(5)),
            tps=int(m.group(6).replace(",", "")),
            tflops_per_gpu=float(m.group(7)),
            mfu_pct=float(m.group(8)),
        )
    return [rows[k] for k in sorted(rows)]


def parse_raw_ddp_log(log_path: Path, num_params: int) -> list[dict]:
    rows = []
    for raw in log_path.read_text(errors="ignore").splitlines():
        line = ANSI.sub("", raw)
        m = DDP_STEP.search(line)
        if not m:
            continue
        tps = int(m.group(4))
        # 8N flops/token includes the activation-checkpoint forward recompute.
        achieved_tflops = tps * 8 * num_params / 1e12
        rows.append(dict(
            step=int(m.group(1)),
            loss=float(m.group(2)),
            tps_per_gpu=tps,
            tflops_per_gpu=round(achieved_tflops, 2),
            mfu_pct=float(m.group(5)),
            memory_gib=float(m.group(6)),
        ))
    return rows


def summarize(steps: list[dict]) -> dict:
    if not steps:
        return {"completed": False, "n_steps": 0}
    peak_mem = max(s.get("memory_gib", 0.0) for s in steps)
    # Skip step 1 (warmup / JIT) for throughput averaging.
    bulk = [s for s in steps if s["step"] > 1] or steps
    mean_tflops = (sum(s.get("tflops_per_gpu", 0.0) for s in bulk) / len(bulk)
                   if "tflops_per_gpu" in bulk[0] else 0.0)

    # Measured tok/s/GPU.
    # Raw DDP logs `tps_per_gpu` directly; TorchTitan logs `tps`, which is already
    # per-GPU (verified against MFU × peak_TFLOPS / FLOPs_per_token across layouts).
    if "tps_per_gpu" in bulk[0]:
        mean_tps_per_gpu = sum(s["tps_per_gpu"] for s in bulk) / len(bulk)
    elif "tps" in bulk[0]:
        mean_tps_per_gpu = sum(s["tps"] for s in bulk) / len(bulk)
    else:
        mean_tps_per_gpu = 0.0
    return {
        "completed": True,
        "n_steps_logged": len(steps),
        "final_step": steps[-1]["step"],
        "mean_mfu_pct": round(sum(s.get("mfu_pct", 0.0) for s in bulk) / len(bulk), 2),
        "peak_memory_gib": round(peak_mem, 2),
        "mean_tflops_per_gpu": round(mean_tflops, 2) if mean_tflops else None,
        # Derived uniformly via 8× rule — compares cleanly across DDP/FSDP/TP/PP.
        "mean_tokens_per_sec_per_gpu": round(mean_tps_per_gpu, 0) if mean_tps_per_gpu else None,
    }


def process_run(run_dir: Path) -> dict | None:
    meta_path = run_dir / "run_meta.json"
    train_log = run_dir / "train.log"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    elif run_dir.name.startswith("exp_ddp_"):
        # Raw train.py — run_meta.json doesn't exist; synthesize minimal meta
        job_id = run_dir.name.split("_")[-1]
        meta = dict(
            experiment="exp1_ddp", algorithm="DDP (raw PyTorch, train.py)",
            job_id=job_id, model_flavor="8B",
            parallelism=dict(dp_replicate=16, dp_shard=1, tp=1, pp=1),
        )
    else:
        return None

    # exp_ddp_* runs come from train.py (raw DDP); anything else is TorchTitan.
    is_raw_ddp = run_dir.name.startswith("exp_ddp_")
    if is_raw_ddp and train_log.exists():
        # Read actual param count from train.py's config.json instead of hardcoding.
        cfg = json.loads((run_dir / "config.json").read_text())
        num_params = int(cfg.get("model_params_B", 7.5) * 1e9)
        steps = parse_raw_ddp_log(train_log, num_params)
    elif train_log.exists():
        steps = parse_torchtitan_log(train_log)
    else:
        steps = []

    result = {**meta, "steps": steps, "summary": summarize(steps)}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="training/results")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)

    all_runs: list[dict] = []
    for run in sorted(results_dir.iterdir()):
        if not run.is_dir():
            continue
        r = process_run(run)
        if r is not None:
            all_runs.append(r)
            s = r["summary"]
            print(f"{r['experiment']:30s}  mfu={s.get('mean_mfu_pct', 'n/a'):>5}%  "
                  f"mem={s.get('peak_memory_gib', 'n/a'):>6} GiB  "
                  f"tflops/GPU={s.get('mean_tflops_per_gpu', 'n/a')}  "
                  f"tok/s/GPU={s.get('mean_tokens_per_sec_per_gpu', 'n/a')}")

    # Top-level summary
    (results_dir / "summary.json").write_text(
        json.dumps([{**r["summary"],
                     "experiment": r["experiment"],
                     "algorithm": r["algorithm"],
                     "model_flavor": r["model_flavor"],
                     "parallelism": r["parallelism"],
                     "job_id": r.get("job_id")}
                    for r in all_runs], indent=2)
    )
    with (results_dir / "summary.csv").open("w", newline="") as f:
        cols = ["experiment", "model_flavor", "algorithm", "job_id",
                "mean_mfu_pct", "peak_memory_gib",
                "mean_tflops_per_gpu", "mean_tokens_per_sec_per_gpu", "completed"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_runs:
            w.writerow({**r, **r["summary"]})

    print(f"\n{len(all_runs)} runs → {results_dir}/summary.{{json,csv}}")


if __name__ == "__main__":
    main()
