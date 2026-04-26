"""
Experiment 1: Llama 3.1 8B DDP baseline — raw PyTorch DistributedDataParallel.

Used as the reference point against which TorchTitan's sharded strategies
(exp2–exp4c) are compared. Writes metrics.jsonl + tensorboard/ under
training/results/exp_ddp_<SLURM_JOB_ID>/.
"""

import argparse
import contextlib
import datetime
import os
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import LlamaConfig, LlamaForCausalLM

from dataset import SyntheticDataset, build_dataloaders
from utils import (
    MetricsWriter, compute_mfu, cosine_schedule_with_warmup,
    find_latest_checkpoint, get_logger, load_checkpoint, make_run_dir, save_checkpoint,
)


LLAMA_8B_CONFIG = dict(
    hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
    num_key_value_heads=8, intermediate_size=14336,
    max_position_embeddings=8192, vocab_size=128256,
    rope_theta=500_000.0, rms_norm_eps=1e-5,
    tie_word_embeddings=True,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--model_path", default=None, help="Local dir or HF model ID")
    p.add_argument("--resume", default=None, metavar="CKPT_DIR",
                   help="Checkpoint directory to resume from")
    p.add_argument("--synthetic", action="store_true", help="Use random tokens instead of train.bin")
    return p.parse_args()


def setup_distributed():
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=3600),
    )
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def build_model(cfg: dict, local_rank: int, model_path=None):
    device = torch.device(f"cuda:{local_rank}")
    llama_cfg = LlamaConfig(**{**LLAMA_8B_CONFIG, **cfg.get("model_overrides", {})})

    if model_path:
        model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to(device)
    else:
        torch.manual_seed(42)
        model = LlamaForCausalLM(llama_cfg).to(dtype=torch.bfloat16).to(device)

    # Activation checkpointing: params(15GB) + optimizer(60GB) + grads(30GB) + activations
    # would exceed 141GB H200 without it.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = DDP(model, device_ids=[local_rank])
    return model, llama_cfg


@torch.no_grad()
def evaluate(model, val_loader, local_rank: int, max_batches: int = 20) -> float:
    model.eval()
    total = torch.zeros(1, device=f"cuda:{local_rank}")
    count = 0
    for x, y in val_loader:
        if count >= max_batches:
            break
        x = x.to(f"cuda:{local_rank}", non_blocking=True)
        y = y.to(f"cuda:{local_rank}", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=x, labels=y)
        total += out.loss
        count += 1
    dist.all_reduce(total, op=dist.ReduceOp.AVG)
    model.train()
    return (total / max(count, 1)).item()


def train():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rank, local_rank, world_size = setup_distributed()
    is_rank0 = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    results_base = cfg.get("results_dir", "/mnt/data/poc-training/training/results")
    run_dir = make_run_dir(results_base, "exp_ddp") if is_rank0 else None
    run_dir_str = str(run_dir) if is_rank0 else ""
    obj = [run_dir_str]
    dist.broadcast_object_list(obj, src=0)
    run_dir = Path(obj[0])
    run_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    ckpt_dir = str(run_dir / "checkpoints")
    log_dir  = str(run_dir)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    logger  = get_logger(rank, log_dir)
    metrics = MetricsWriter(run_dir) if is_rank0 else None

    model, llama_cfg = build_model(cfg, local_rank, args.model_path)

    seen, num_params = set(), 0
    src = model.module if hasattr(model, "module") else model
    for p in src.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            num_params += p.numel()

    if is_rank0:
        size_b = num_params / 1e9
        logger.info(f"Strategy: ddp  |  Model: {size_b:.1f}B params  |  world_size={world_size}")
        import json
        run_config = dict(
            strategy="ddp", world_size=world_size,
            model_params_B=round(size_b, 2),
            model_path=args.model_path, **cfg,
        )
        (run_dir / "config.json").write_text(json.dumps(run_config, indent=2, default=str))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["training"]["lr"],
        betas        = (0.9, 0.95),
        weight_decay = cfg["training"]["weight_decay"],
        fused        = True,
    )
    total_steps  = cfg["training"]["max_steps"]
    warmup_steps = cfg["training"]["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=partial(cosine_schedule_with_warmup,
                          warmup_steps=warmup_steps, total_steps=total_steps, min_ratio=0.1),
    )

    if args.synthetic:
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
        seq_len    = cfg["model"]["seq_len"]
        batch_size = cfg["training"]["batch_size"]
        train_ds   = SyntheticDataset(seq_len)
        val_ds     = SyntheticDataset(seq_len, size=1000)
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)
        train_loader  = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                                   num_workers=2, pin_memory=True, drop_last=True)
        val_loader    = DataLoader(val_ds,   batch_size=batch_size, sampler=val_sampler,
                                   num_workers=2, pin_memory=True, drop_last=True)
        if is_rank0:
            logger.info("Using synthetic data (--synthetic flag)")
    else:
        train_loader, val_loader, train_sampler = build_dataloaders(
            data_path   = cfg["data"]["path"],
            seq_len     = cfg["model"]["seq_len"],
            batch_size  = cfg["training"]["batch_size"],
            rank        = rank,
            world_size  = world_size,
            num_workers = cfg["data"].get("num_workers", 4),
        )

    accum_steps   = cfg["training"]["grad_accum_steps"]
    clip_norm     = cfg["training"]["grad_clip"]
    val_interval  = cfg["training"]["val_interval"]
    log_interval  = cfg["training"]["log_interval"]
    ckpt_interval = cfg["training"]["checkpoint_interval"]
    save_ckpts    = cfg["training"].get("save_checkpoints", False)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, scheduler, rank)
        logger.info(f"Resumed from {args.resume} (step={start_step})")
    elif cfg["training"].get("auto_resume"):
        latest = find_latest_checkpoint(ckpt_dir)
        if latest:
            start_step = load_checkpoint(latest, model, optimizer, scheduler, rank)
            logger.info(f"Auto-resumed from {latest} (step={start_step})")

    tokens_per_step = world_size * accum_steps * cfg["training"]["batch_size"] * cfg["model"]["seq_len"]
    logger.info(f"tokens/step={tokens_per_step:,}  grad_accum={accum_steps}  max_steps={total_steps}")

    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    step      = start_step
    data_iter = iter(train_loader)

    while step < total_steps:
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss_accum = torch.zeros(1, device=device)

        for micro in range(accum_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                train_sampler.set_epoch(step)
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            is_last = micro == accum_steps - 1
            sync_ctx = contextlib.nullcontext() if is_last else model.no_sync()

            with sync_ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=x, labels=y)
                (out.loss / accum_steps).backward()

            loss_accum += out.loss.detach() / accum_steps

        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm).item()

        optimizer.step()
        scheduler.step()
        step += 1

        elapsed = time.perf_counter() - t0
        tok_s_total   = tokens_per_step / elapsed
        tok_s_per_gpu = tok_s_total / world_size
        mfu_val       = compute_mfu(tok_s_per_gpu, num_params, with_ac=True)
        peak_mem_gb   = torch.cuda.max_memory_allocated(device) / 1e9
        lr            = scheduler.get_last_lr()[0]
        loss_val      = loss_accum.item()

        if step % log_interval == 0 and is_rank0:
            import math
            logger.info(
                f"step={step:>5d}/{total_steps}  loss={loss_val:.4f}  "
                f"ppl={math.exp(min(loss_val, 20)):.1f}  lr={lr:.2e}  "
                f"tok/s={tok_s_total/1e3:.1f}K  tok/s/GPU={tok_s_per_gpu:.0f}  "
                f"MFU={mfu_val*100:.1f}%  mem={peak_mem_gb:.1f}GB"
            )
            metrics.write(
                step,
                loss=loss_val,
                perplexity=math.exp(min(loss_val, 20)),
                lr=lr,
                grad_norm=grad_norm,
                tok_per_sec=round(tok_s_total),
                tok_per_sec_per_gpu=round(tok_s_per_gpu),
                mfu_pct=round(mfu_val * 100, 2),
                peak_mem_gb=round(peak_mem_gb, 2),
                step_time_s=round(elapsed, 3),
            )

        if step % val_interval == 0:
            val_loss = evaluate(model, val_loader, local_rank)
            if is_rank0:
                import math
                logger.info(f"[VAL] step={step}  val_loss={val_loss:.4f}  ppl={math.exp(min(val_loss, 20)):.1f}")
                metrics.write(step, val_loss=val_loss, val_perplexity=math.exp(min(val_loss, 20)))

        if save_ckpts and step % ckpt_interval == 0:
            save_checkpoint(ckpt_dir, model, optimizer, scheduler, step, rank)
            logger.info(f"Checkpoint saved at step {step}")

    if save_ckpts:
        save_checkpoint(ckpt_dir, model, optimizer, scheduler, step, rank)

    if is_rank0:
        logger.info(f"Training complete. Results: {run_dir}")
        metrics.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    train()
