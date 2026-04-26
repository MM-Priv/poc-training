# Training

End-to-end multi-node LLM pre-training on the Nebius H200 cluster, exercising six
distribution strategies across Llama 3.1 8B and 70B. Each experiment is a thin sbatch
wrapper over TorchTitan (except `exp1`, which uses a hand-rolled raw-PyTorch DDP
loop as baseline). Metrics are parsed into structured JSON/CSV per run.

## Experiments

| # | Strategy | 8B | 70B | Parallelism |
|---|---|---|---|---|
| 1  | DDP (raw PyTorch)           | ✅ | —   | `dp_replicate=16` |
| 2  | Tensor Parallel             | ✅ | ✅ | `tp=8, dp_shard=2` |
| 3  | Tensor + Pipeline           | ✅ | ✅ | `tp=8, pp=2, dp=1` |
| 4a | FSDP FULL_SHARD (ZeRO-3)    | ✅ | ✅ | `dp_shard=16` |
| 4b | FSDP HYBRID_SHARD           | ✅ | ✅ (OOM) | `dp_replicate=2, dp_shard=8` |
| 4c | FSDP SHARD_GRAD_OP (ZeRO-2) | ✅ | ✅ (OOM) | `dp_shard=16, reshard_after_forward=never` |

Plus optimization variants:
- `exp4a_*_70b_fp8` — FSDP FULL + `torch.compile` + Float8 (production ceiling)
- `exp3_*_70b_mb8` — TP+PP with 8 microbatches (reduces pipeline bubble)
- `exp2_*_70b_bs4` — TP with `local_batch_size=4` (amortizes per-layer collectives)

And validation jobs:
- `smoke_test.sbatch` — toy-transformer NCCL smoke test (cluster bring-up)
- `test_resume.sbatch` — verifies checkpoint/resume round-trip

## Prerequisites

One-time setup on the head node:

```bash
bash setup.sh
```

This runs `uv sync` (installs PyTorch, TorchTitan, TorchAO, HuggingFace) and stages
the Llama 3.1 tokenizer under `assets/hf/llama-3.1-tokenizer/` (reuses the cached
Llama 3.1 8B snapshot — 8B and 70B share the same tokenizer).

Then tokenize FineWeb-Edu once (≈2–4 h). Both steps need `HF_TOKEN` — the
FineWeb-Edu dataset is public but the Llama 3.1 tokenizer used for tokenization
is gated:

```bash
export HF_TOKEN=hf_...

HF_TOKEN=$HF_TOKEN python training/download_data.py \
    --cache_dir /mnt/data/poc-training/data/hf_cache

HF_TOKEN=$HF_TOKEN python training/tokenize_data.py \
    --cache_dir /mnt/data/poc-training/data/hf_cache \
    --output_dir /mnt/data/poc-training/data
```

Produces `train.bin` (9.6B uint32 tokens), `val.bin` (100M tokens), `meta.json`.

## Running experiments

Each sbatch is self-contained; all six 8B experiments use `llama3_8b_base.toml`
as the shared TorchTitan config, and 70B uses `llama3_70b_base.toml`:

```bash
# Single experiment
sbatch training/slurm/exp4a_torchtitan_fsdp_8b.sbatch

# Full 8B matrix (queued sequentially by slurm --exclusive)
for f in training/slurm/exp{1_ddp,2_torchtitan_tp8,3_torchtitan_tp8_pp2,4a_torchtitan_fsdp,4b_torchtitan_hsdp,4c_torchtitan_fsdp_sgo}_8b.sbatch; do
  sbatch $f
done

# 70B matrix (5 experiments — 4b and 4c expected to OOM, flagged in sbatch comments)
for f in training/slurm/exp{2_torchtitan_tp8,3_torchtitan_tp8_pp2,4a_torchtitan_fsdp,4b_torchtitan_hsdp,4c_torchtitan_fsdp_sgo}_70b.sbatch; do
  sbatch $f
done
```

Per-run artifacts land under `/mnt/data/poc-training/training/results/<exp>_<jobid>/`:

- `run_meta.json` — algorithm + parallelism params (written by the sbatch)
- `train.log`     — full stdout with TorchTitan's per-step metrics line
- `tb/`           — TensorBoard event files

## Collecting results

After runs complete, aggregate:

```bash
python training/collect_results.py
```

Produces per run `result.json` and a cluster-wide `summary.{json,csv}`:

```
training/results/
├── exp1_ddp_232/
│   ├── run_meta.json
│   ├── train.log
│   ├── result.json       # steps + summary
│   └── tb/
├── exp2_tp8_70b_238/
│   ...
├── summary.csv           # one row per run
├── summary.json
└── report.md             # client-facing PoC writeup
```

The aggregate `summary.csv` has columns: `experiment, model_flavor, algorithm, job_id, mean_mfu_pct, peak_memory_gib, mean_tflops_per_gpu, mean_tokens_per_sec_per_gpu, completed`.

## Configuration

| File | Purpose |
|---|---|
| `configs/llama3_8b_base.toml`  | TorchTitan base config for 8B experiments |
| `configs/llama3_70b_base.toml` | TorchTitan base config for 70B experiments |
| `configs/exp1_ddp_8b.yaml`     | YAML config for the raw-PyTorch DDP run |
| `titan_launcher.py`            | TorchTitan entry-point shim that registers FineWeb-Edu in the dataset registry before delegating to `torchtitan.train` |

Per-experiment parallelism (TP/PP/DP degrees, FSDP mode) is set via
`--parallelism.*` CLI flags in each sbatch — the base TOMLs only carry
model/optimizer/training defaults.

## Checkpoint / resume

Checkpoint saving is off by default in all experiments (`save_checkpoints: false`
for DDP; `checkpoint.enable=false` for TorchTitan). To verify the resume path:

```bash
sbatch training/slurm/test_resume.sbatch
```

Runs two phases in one job: trains 10 steps and saves a full checkpoint, then
loads it and continues to step 20. The test passes if step 11 onwards logs a
loss close to phase 1's step-10 loss (confirms model + optimizer + LR state all
reloaded), not a random-init loss ~12.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `KeyError: 'float8'` at launch | `torchao` not installed, or `--model.converters` value is `float8` instead of `quantize.linear.float8` |
| OOM at step 1 on 70B | Expected for `exp4b_*_70b` (HSDP) and `exp4c_*_70b` (SGO); annotated in the sbatch headers |
| `Missing key in checkpoint state_dict: dataloader.dp_rank_*` | Streaming dataset can't save per-rank cursor state — switch to the FineWeb-Edu `titan_launcher.py` registration (non-streaming) or pass `--checkpoint.exclude_from_loading dataloader` |
| Large result directories | DDP saves full model+optimizer checkpoints (~40 GB); disable with `save_checkpoints: false` in the yaml, or prune `training/results/*/checkpoints/` between runs |
