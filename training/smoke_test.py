"""
Distributed training smoke test — no data files required.

Tests multi-node NCCL communication, forward/backward pass, and gradient sync
using a small transformer with synthetic random data.

Metrics reported every log_interval steps:
  - loss (should hover near log(vocab_size) ≈ 10.4 for random labels)
  - tok/s total and per GPU
  - MFU (Model FLOP Utilization) vs H200 BF16 peak
  - peak GPU memory

Usage:
    # Single node (quick sanity check)
    torchrun --nproc_per_node=8 training/smoke_test.py

    # Multi-node via sbatch
    sbatch training/slurm/smoke_test.sbatch
"""

import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Config (edit here or override via env vars)
# ---------------------------------------------------------------------------

VOCAB_SIZE    = 32_000
SEQ_LEN       = 2_048
BATCH_SIZE    = 4       # per GPU
HIDDEN        = 1_024
NUM_LAYERS    = 8
NUM_HEADS     = 8
FFN_DIM       = 4_096
MAX_STEPS     = 200
LOG_INTERVAL  = 10

# H200 SXM dense BF16 peak (no sparsity)
H200_BF16_TFLOPS = 989  # dense BF16 peak (no 2:4 sparsity)

# ---------------------------------------------------------------------------
# Toy model
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(HIDDEN)
        self.wq    = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.wk    = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.wv    = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.wo    = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.norm2 = nn.LayerNorm(HIDDEN)
        self.ffn   = nn.Sequential(
            nn.Linear(HIDDEN, FFN_DIM),
            nn.GELU(),
            nn.Linear(FFN_DIM, HIDDEN),
        )

    def forward(self, x):
        B, T, C = x.shape
        head_dim = C // NUM_HEADS
        h = self.norm1(x)
        q = self.wq(h).view(B, T, NUM_HEADS, head_dim).transpose(1, 2)
        k = self.wk(h).view(B, T, NUM_HEADS, head_dim).transpose(1, 2)
        v = self.wv(h).view(B, T, NUM_HEADS, head_dim).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.wo(a)
        x = x + self.ffn(self.norm2(x))
        return x


class ToyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed   = nn.Embedding(VOCAB_SIZE, HIDDEN)
        self.pos     = nn.Embedding(SEQ_LEN, HIDDEN)
        self.blocks  = nn.Sequential(*[Block() for _ in range(NUM_LAYERS)])
        self.norm    = nn.LayerNorm(HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

    def forward(self, x, targets=None):
        B, T = x.shape
        pos   = torch.arange(T, device=x.device)
        h     = self.embed(x) + self.pos(pos)
        h     = self.blocks(h)
        h     = self.norm(h)
        logits = self.lm_head(h)
        loss  = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss

    def num_parameters(self):
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total


# ---------------------------------------------------------------------------
# Synthetic dataset — no files needed
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    def __init__(self, size=100_000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # deterministic per index so each rank gets different data
        g = torch.Generator()
        g.manual_seed(idx)
        tokens = torch.randint(0, VOCAB_SIZE, (SEQ_LEN + 1,), generator=g)
        return tokens[:-1], tokens[1:]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mfu(tok_per_sec_per_gpu: float, n_params: int) -> float:
    """Model FLOP Utilization: achieved vs H200 BF16 peak."""
    flops_per_token = 6 * n_params
    achieved = tok_per_sec_per_gpu * flops_per_token
    peak     = H200_BF16_TFLOPS * 1e12
    return achieved / peak


def make_run_dir(base: str) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", f"local_{int(time.time())}")
    run_dir = Path(base) / f"smoke_test_{job_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- distributed init ---
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    is_rank0 = rank == 0

    # --- run directory + writers (rank 0 only) ---
    run_dir = make_run_dir("/mnt/data/poc-training/training/results")
    tb_writer   = SummaryWriter(log_dir=run_dir / "tensorboard") if is_rank0 else None
    jsonl_path  = run_dir / "metrics.jsonl"

    if is_rank0:
        # Write run config once
        config = dict(
            job_id=os.environ.get("SLURM_JOB_ID"),
            world_size=world_size,
            vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
            hidden=HIDDEN, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
            max_steps=MAX_STEPS,
        )
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # --- model ---
    torch.manual_seed(42)
    model = ToyTransformer().to(device)
    model = DDP(model, device_ids=[local_rank])

    n_params = model.module.num_parameters()
    if is_rank0:
        print(f"{'='*60}")
        print(f"Smoke test — {world_size} GPUs  ({world_size // 8} nodes × 8)")
        print(f"Model: {n_params/1e6:.0f}M params")
        print(f"Batch: {BATCH_SIZE} seq/GPU × {SEQ_LEN} tokens = "
              f"{BATCH_SIZE * SEQ_LEN * world_size:,} tokens/step")
        print(f"{'='*60}", flush=True)

    # --- data ---
    dataset = SyntheticDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                         num_workers=2, pin_memory=True, drop_last=True)

    # --- optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)

    tokens_per_step = BATCH_SIZE * SEQ_LEN * world_size
    data_iter = iter(loader)
    step = 0

    torch.cuda.reset_peak_memory_stats(device)

    while step < MAX_STEPS:
        try:
            x, y = next(data_iter)
        except StopIteration:
            sampler.set_epoch(step)
            data_iter = iter(loader)
            x, y = next(data_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        t0 = time.perf_counter()

        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        elapsed = time.perf_counter() - t0
        step += 1

        if step % LOG_INTERVAL == 0:
            # gather loss from all ranks
            loss_val = loss.detach()
            dist.all_reduce(loss_val, op=dist.ReduceOp.AVG)

            tok_s_total   = tokens_per_step / elapsed
            tok_s_per_gpu = tok_s_total / world_size
            mfu_val       = mfu(tok_s_per_gpu, n_params)
            mem_gb        = torch.cuda.max_memory_allocated(device) / 1e9

            if is_rank0:
                row = dict(
                    step=step,
                    loss=round(loss_val.item(), 4),
                    tok_per_sec=round(tok_s_total),
                    tok_per_sec_per_gpu=round(tok_s_per_gpu),
                    mfu_pct=round(mfu_val * 100, 2),
                    peak_mem_gb=round(mem_gb, 3),
                    step_time_s=round(elapsed, 3),
                )
                # stdout
                print(
                    f"step {step:>4d}/{MAX_STEPS}  "
                    f"loss={row['loss']:.3f}  "
                    f"tok/s={tok_s_total/1e3:.1f}K  "
                    f"tok/s/GPU={tok_s_per_gpu:.0f}  "
                    f"MFU={mfu_val*100:.1f}%  "
                    f"mem={mem_gb:.2f}GB",
                    flush=True,
                )
                # JSONL — one record per log step
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                # TensorBoard
                for k, v in row.items():
                    if k != "step":
                        tb_writer.add_scalar(k, v, step)

    if is_rank0:
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"\nDone. Peak GPU memory: {peak_mem:.2f} GB")
        print(f"Results written to: {run_dir}")
        tb_writer.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
