"""
Shared utilities: logging, checkpointing, MFU, metrics output.
"""

import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


# H200 SXM dense BF16 Tensor Core peak (2:4 sparsity figure 1979e12 doesn't apply to training).
H200_BF16_PEAK_FLOPS = 989e12


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(rank: int, log_dir: str) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    if rank == 0 and not logger.handlers:
        fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(Path(log_dir) / "train.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Run directory + metrics writer
# ---------------------------------------------------------------------------

def make_run_dir(results_base: str, experiment: str) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", f"local_{int(time.time())}")
    run_dir = Path(results_base) / f"{experiment}_{job_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class MetricsWriter:
    """Writes metrics to both a JSONL file and TensorBoard (rank 0 only)."""

    def __init__(self, run_dir: Path):
        self.jsonl_path = run_dir / "metrics.jsonl"
        self.tb = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    def write(self, step: int, **kwargs):
        row = {"step": step, **{k: round(v, 6) if isinstance(v, float) else v for k, v in kwargs.items()}}
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                self.tb.add_scalar(k, v, step)

    def close(self):
        self.tb.close()


# ---------------------------------------------------------------------------
# MFU
# ---------------------------------------------------------------------------

def compute_mfu(tok_per_sec_per_gpu: float, num_params: int, with_ac: bool = True) -> float:
    # 6N flops/token without AC; 8N with AC (extra forward recompute).
    flops_per_token = 8 * num_params if with_ac else 6 * num_params
    return (tok_per_sec_per_gpu * flops_per_token) / H200_BF16_PEAK_FLOPS


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_schedule_with_warmup(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float = 0.1,
) -> float:
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


# Checkpointing — DDP only. TorchTitan experiments use its own sharded
# checkpoint subsystem; these helpers exist for train.py (exp1) and the
# test_resume.sbatch validation job.

def save_checkpoint(ckpt_dir: str, model, optimizer, scheduler, step: int, rank: int) -> None:
    save_dir = Path(ckpt_dir) / f"step_{step:07d}"
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "model":     model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step":      step,
        }
        torch.save(state, save_dir / "checkpoint.pt")
        (Path(ckpt_dir) / "latest").write_text(str(save_dir))
    dist.barrier()


def load_checkpoint(ckpt_path: str, model, optimizer, scheduler, rank: int) -> int:
    state = torch.load(Path(ckpt_path) / "checkpoint.pt", map_location="cpu", weights_only=False)
    model.module.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    dist.barrier()
    return state["step"]


def find_latest_checkpoint(ckpt_dir: str) -> str | None:
    latest = Path(ckpt_dir) / "latest"
    return latest.read_text().strip() if latest.exists() else None

