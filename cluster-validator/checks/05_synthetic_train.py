#!/usr/bin/env python3
"""
Synthetic distributed training + inference check.
Uses a small Llama-architecture model with random weights — no download needed.

Launch with:
  torchrun --standalone --nproc_per_node=8 checks/05_synthetic_train.py
  torchrun --nnodes=N --nproc_per_node=8 --rdzv_backend=c10d ... 05_synthetic_train.py
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import LlamaConfig, LlamaForCausalLM


TINY_LLAMA_CONFIG = LlamaConfig(
    hidden_size=1024,
    intermediate_size=2048,
    num_hidden_layers=8,
    num_attention_heads=16,
    num_key_value_heads=4,
    max_position_embeddings=4096,
    vocab_size=32000,
    rms_norm_eps=1e-5,
    rope_theta=500000.0,
)

WARMUP_STEPS = 5
MEASURE_STEPS = 10
BATCH_SIZE = 2
SEQ_LEN = 1024
TRAIN_TOKENS_PER_GPU_THRESHOLD = 2_000


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def build_model(local_rank):
    model = LlamaForCausalLM(TINY_LLAMA_CONFIG).to(f"cuda:{local_rank}")
    model = DDP(model, device_ids=[local_rank])
    return model


def synthetic_batch(local_rank):
    return torch.randint(
        0, TINY_LLAMA_CONFIG.vocab_size,
        (BATCH_SIZE, SEQ_LEN),
        device=f"cuda:{local_rank}",
    )


def run_training(model, local_rank):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    world_size = dist.get_world_size()
    tokens_per_step = BATCH_SIZE * SEQ_LEN * world_size

    for _ in range(WARMUP_STEPS):
        ids = synthetic_batch(local_rank)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()

    for _ in range(MEASURE_STEPS):
        ids = synthetic_batch(local_rank)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - t0

    total_tokens = tokens_per_step * MEASURE_STEPS
    tps_total = total_tokens / elapsed
    tps_per_gpu = tps_total / world_size
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return tps_total, tps_per_gpu, peak_mem


def run_inference(model, local_rank):
    model.eval()
    world_size = dist.get_world_size()
    tokens_per_step = BATCH_SIZE * SEQ_LEN * world_size

    for _ in range(WARMUP_STEPS):
        ids = synthetic_batch(local_rank)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _ = model(input_ids=ids)

    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()

    for _ in range(MEASURE_STEPS):
        ids = synthetic_batch(local_rank)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _ = model(input_ids=ids)

    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - t0

    total_tokens = tokens_per_step * MEASURE_STEPS
    tps_total = total_tokens / elapsed
    tps_per_gpu = tps_total / world_size
    return tps_total, tps_per_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default=os.environ.get("SCRATCH_DIR", "/tmp/validator_results"))
    args = parser.parse_args()

    local_rank, rank, world_size = setup_distributed()

    if rank == 0:
        param_count = sum(p.numel() for p in LlamaForCausalLM(TINY_LLAMA_CONFIG).parameters())
        print(f"Synthetic model: {param_count/1e6:.0f}M params, {world_size} ranks, "
              f"batch={BATCH_SIZE}, seq={SEQ_LEN}")

    model = build_model(local_rank)

    status = "PASS"
    messages = []
    train_tps = train_tps_per_gpu = peak_mem = infer_tps = infer_tps_per_gpu = 0.0

    try:
        train_tps, train_tps_per_gpu, peak_mem = run_training(model, local_rank)
        infer_tps, infer_tps_per_gpu = run_inference(model, local_rank)
    except Exception as e:
        status = "FAIL"
        messages.append(f"Exception: {e}")

    if status == "PASS" and train_tps_per_gpu < TRAIN_TOKENS_PER_GPU_THRESHOLD:
        status = "FAIL"
        messages.append(
            f"Training throughput {train_tps_per_gpu:.0f} tok/s/GPU "
            f"below threshold {TRAIN_TOKENS_PER_GPU_THRESHOLD}"
        )

    if rank == 0:
        os.makedirs(args.result_dir, exist_ok=True)
        param_count = sum(p.numel() for p in LlamaForCausalLM(TINY_LLAMA_CONFIG).parameters())
        result = {
            "check": "synthetic_training",
            "status": status,
            "metrics": {
                "world_size": world_size,
                "model_params_M": round(param_count / 1e6),
                "batch_size": BATCH_SIZE,
                "seq_len": SEQ_LEN,
                "train_tokens_per_sec_total": round(train_tps),
                "train_tokens_per_sec_per_gpu": round(train_tps_per_gpu),
                "infer_tokens_per_sec_total": round(infer_tps),
                "infer_tokens_per_sec_per_gpu": round(infer_tps_per_gpu),
                "peak_memory_per_rank_GiB": round(peak_mem, 2),
                "threshold_train_tok_per_sec_per_gpu": TRAIN_TOKENS_PER_GPU_THRESHOLD,
            },
            "messages": messages,
        }
        with open(os.path.join(args.result_dir, "05_synthetic_train.json"), "w") as f:
            json.dump(result, f, indent=2)

        print(
            f"[{status}] Synthetic Training: "
            f"train {train_tps_per_gpu:.0f} tok/s/GPU, "
            f"infer {infer_tps_per_gpu:.0f} tok/s/GPU, "
            f"peak mem {peak_mem:.1f} GiB/rank"
        )
        for msg in messages:
            print(f"       {msg}")

    dist.destroy_process_group()

    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
