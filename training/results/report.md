# Multi-node LLM training PoC — Nebius H200 cluster

## Executive summary

We set up an end-to-end, multi-node pre-training stack on a 2-node × 8×H200
Nebius cluster and ran Llama 3.1 (8B and 70B) under six distribution
strategies each. The goal was to answer three questions for the client:

1. **Can we run multi-node LLM training on this platform with reasonable
   efficiency?** Yes. **63% MFU measured** for 70B FSDP FULL with
   torch.compile + FP8 (1,396 tok/s/GPU, 89 GB/GPU). Baseline BF16-only run
   gives 40% MFU — the gap is the "production optimization headroom" on the
   same hardware.
2. **What should the client's initial setup look like on 512 H100s?**
   The *strategy* recommendation is framework-agnostic: FSDP FULL for
   anything up to ~200B; TP + PP composed on top for 200B+. The choice of
   *training framework* is the client's — any of the major options (HF
   Accelerate, DeepSpeed, Megatron-Core/NeMo, raw PyTorch, TorchTitan) can
   realize these strategies. We picked TorchTitan for the PoC because it
   made switching between all six strategies a one-line CLI change,
   minimizing PoC engineering. The trade-offs that would motivate a
   different choice are mostly about model architecture — notably, custom
   non-Llama architectures may be better served by Megatron-Core.
3. **Which strategies scale to 100B+?** Of the six tested, **FSDP FULL_SHARD**
   is the safest default; **TP=8 × PP=2 × FSDP-DP** is the growth path when
   the model outgrows FSDP alone.

## What was built

A 2-node × 8-GPU Nebius H200 cluster (soperator SLURM, NVLink intra-node,
InfiniBand inter-node) running:

- **Data pipeline** — FineWeb-Edu (10BT sample) downloaded + tokenized to
  uint32 binaries with the Llama 3.1 tokenizer (9.6B train / 100M val tokens).
- **Training framework** — TorchTitan (chosen for the PoC because it
  exposes all six parallelism strategies as CLI flags on one launcher, so
  the PoC itself could be built with minimal per-strategy engineering).
  The client is free to use any framework that supports the same
  strategies — the measured results here are properties of the *hardware
  and strategy*, not of TorchTitan specifically. A ~15-line launcher
  (`training/titan_launcher.py`) registers the local HF cache as a
  TorchTitan dataset.
- **Six strategies**, each a thin sbatch wrapper that overrides
  `--parallelism.*` CLI flags:

| # | Strategy | Parallelism config |
|---|---|---|
| 1 | DDP (raw PyTorch baseline) | dp_replicate=16 |
| 2 | Tensor Parallel | tp=8, dp_shard=2 |
| 3 | Tensor + Pipeline | tp=8, pp=2, dp=1 |
| 4a | FSDP FULL_SHARD (ZeRO-3) | dp_shard=16 |
| 4b | FSDP HYBRID_SHARD | dp_replicate=2, dp_shard=8 |
| 4c | FSDP SHARD_GRAD_OP (ZeRO-2) | dp_shard=16, reshard_after_forward=never |

Plus a resume-flag validation job (`test_resume.sbatch`) that verifies
checkpoint/restart works correctly, and a toy-transformer smoke test
(`smoke_test.sbatch`) for cluster bring-up.

Supporting tooling:
- `training/collect_results.py` — parses training logs into per-run
  `result.json` and aggregate `summary.{json,csv}`
- Ephemeral-storage caches configured (`TRITON_CACHE_DIR` etc.) on `/tmp`
  to avoid concurrent-write conflicts on the shared filesystem
- `setup.sh` stages the Llama 3.1 tokenizer to a shared asset path and
  installs all deps via `uv sync`

## Measured training efficiency

MFU is computed against H200's **989 TFLOPS BF16 dense peak** (the
no-sparsity figure that applies to real training). All runs use full
activation checkpointing, BF16 mixed precision, seq_len=4096, and 50 steps
after ~12-step warmup.

### 8B (Llama 3.1 8B, all strategies fit)

Each FSDP variant measured at two batch sizes: `local_batch=1` (initial
fair-comparison baseline) and `local_batch=4` (matmul-friendly configuration
that exploits the ample memory headroom at 8B).

| # | Strategy | MFU @ bs=1 | MFU @ bs=4 | Peak Mem @ bs=4 | tok/s/GPU @ bs=4 |
|---|---|---|---|---|---|
| 1 | DDP | 48.0% | — | 83.7 GB | 7,906 |
| 4c | FSDP SGO (ZeRO-2) | 38.9% | **44.6%** | 54.9 GB | **8,559** |
| 4a | FSDP FULL (ZeRO-3) | 36.6% | 43.7% | 42.2 GB | 8,399 |
| 4b | FSDP HYBRID | 36.4% | 43.6% | 53.7 GB | 8,370 |
| 2 | Tensor Parallel | 4.6% | — | 10.5 GB | 111 |
| 3 | TP + PP | 1.9% | — | 9.6 GB | 22 |

### 70B (Llama 3.1 70B, memory pressure exposes strategy limits)

| # | Strategy | Config | MFU | Peak Mem/GPU | tok/s/GPU | Outcome |
|---|---|---|---|---|---|---|
| **4a** | **FSDP FULL** | bs=1 | **40.2%** | 90.9 GB | **886** | ✅ |
| 2 | Tensor Parallel | bs=1 | 16.3% | 71.6 GB | 45 | ✅ |
| 2 | Tensor Parallel | **bs=4** | **31.0%** | 77.2 GB | **85** | ✅ **bs=4 nearly doubles MFU** |
| 3 | TP + PP | μbatches=2 | 2.2% | 69.9 GB | 3 | ✅ |
| 3 | TP + PP | μbatches=8 | 2.9% | 70.1 GB | 4 | ✅ marginal gain |
| 4b | FSDP HYBRID | bs=1 | (14.6%) | 137.4 GB peak | 321 | ⚠️ OOM mid-run |
| 4c | FSDP SGO | bs=1 | — | — | — | ❌ OOM at init |
| 1 | DDP | bs=1 | — | — | — | ❌ N/A (140 GB params alone exceed budget) |
| **4a** | **FSDP + compile + FP8** | bs=1 | **63.4%** | 89.3 GB | **1,396** | ✅ **production-grade ceiling** |

## Interpretation

Each strategy has hyperparameters (batch size, microbatch count) that must
be sized to the cluster and the model. Per-strategy MFU numbers are only
meaningful once those are set sensibly — the bs=1 baselines in the table
are fair for cross-strategy comparison, but correspond to *naively default*
settings rather than production configurations.

**Memory wall at 70B.** DDP at 70B is infeasible (140 GB BF16 params alone
exceed an H200). At 8B it's the fastest option (48% MFU) because the full
model fits per-GPU and comms are one all-reduce per step. This is the
central reason to use sharded strategies at 70B+.

**FSDP FULL_SHARD is the workhorse for 70B.** 40% MFU, 91 GB/GPU, within
TorchTitan's published 33–42% range. It's the only FSDP variant that
survives at 70B on our 2-node topology — HYBRID's 8-way shard group leaves
too much state per GPU (OOM), and SGO keeps params gathered after forward
(the full 140 GB). Both failure modes are pure memory arithmetic, not
runtime surprises.

**TP and TP+PP at 8B look underwhelming (4.6% and 1.9% MFU) because they're
designed for bigger models and bigger batches.** At 70B with the right
configuration (`local_batch_size=4`) TP-alone reaches **31% MFU** — in the
same league as FSDP FULL. With `local_batch_size=1` TP would give 16% —
that's not a property of the strategy, it's a mis-set hyperparameter. The
rule of thumb: TP's per-layer collectives are amortized across the batch,
so the batch must grow to keep them in the background of compute.

**TP+PP is bubble-limited with the default TorchTitan 1F1B schedule.**
Going from 2 → 8 microbatches only moved MFU 2.2% → 2.9%. Realizing the
theoretical bubble reduction needs interleaved or zero-bubble schedules,
which are experimental. For the client the practical takeaway is: use
TP+PP when PP is unavoidable (200B+) and plan to tune the schedule on top
of TorchTitan's default.

**torch.compile + FP8 is the production ceiling: 63% MFU on 70B FSDP FULL.**
That's a 1.58× throughput improvement over the BF16 baseline (398 → 627
TFLOPS/GPU, 886 → 1,396 tok/s/GPU) with *lower* per-GPU memory because FP8
weights are smaller. 63% MFU is above TorchTitan's published 54% (Together
AI, 64×H100, BF16) and at the current practical ceiling for 70B on H200.

| Config | MFU | TFLOPS/GPU | tok/s/GPU | Mem/GPU |
|---|---|---|---|---|
| BF16 baseline | 40.2% | 398 | 886 | 91 GB |
| BF16 + compile + FP8 | **63.4%** | **627** | **1,396** | 89 GB |

### tok/s/GPU as the practical budgeting metric

tok/s/GPU maps directly to training-time and cost.

**Baseline (bs=1, no compile/FP8)** — our measured 70B FSDP FULL:

- 886 tok/s/GPU × 16 GPUs = 14,176 tok/s total
- 1T tokens → **~820 hours (~34 days)** on 16 H200s
- Linear-scaled to 512 H100s: ~25 days for 1T tokens. Real efficiency at
  512 GPUs is typically 70–80% of linear due to cross-rail InfiniBand
  contention — budget accordingly.

**Measured production ceiling (bs=1, compile+FP8, 63% MFU)** — now verified
on-cluster, not projected:

- **1,396 tok/s/GPU × 16 GPUs = 22,336 tok/s total**
- 1T tokens → **~12.5 days** on 16 H200s
- Linear-scaled to 512 H100s: **~9.5 days** for 1T tokens (at ~75% scaling
  efficiency: ~12.5 days)

The 1.58× throughput gap between BF16 baseline and compile+FP8 ceiling is
the "software optimization budget" that compile + `torchao` float8 provide
on the same hardware.

### What this means for scaling to 512 GPUs

The strategy selection doesn't change. At 512 H100s (64 nodes × 8):

- **FSDP FULL** with `dp_shard=512` remains the simplest and most efficient
  option up to ~200B parameters. Cross-node all-gather/reduce-scatter
  becomes more sensitive to IB topology — HSDP (shard within node, replicate
  across) starts to pay off when each shard group holds a full copy of
  optimizer state comfortably, which for Llama 70B means shard groups of 16+
  GPUs (i.e. DGX systems with NVSwitch, or ≥2 nodes per shard group).

- **TP=8 × PP=N × FSDP-DP** becomes the canonical layout for 200B+ models.
  TP=8 stays intra-node (NVLink); PP ships activations across IB in
  send-recv (cheap relative to all-reduce); FSDP shards what's left across
  the DP dimension.

Both are one-line TOML + sbatch CLI changes in TorchTitan. No training-code
modifications are required to switch between them.

### Does the 512 H100 cluster comfortably handle a 405B-class model?

**Yes, memory-wise — easily.** A typical 3D layout `TP=8 × PP=8 × FSDP-DP=8`
puts each GPU at about `1/(8×8) = 1/64` of each layer's parameters, then
FSDP-shards the remaining state 8-way within its pipeline stage. Static
per-GPU state for Llama 3.1 405B in BF16 + FP32 AdamW comes out to **~10 GB
of sharded params+grads+optimizer** — well within an 80 GB H100 budget even
before activation-checkpointing savings. 405B fits with substantial headroom
for longer sequences or bigger batches.

**Training time is the realistic constraint, not memory.** Extrapolating
the PoC's measured 63% MFU (with compile + FP8) to 512 H100s:

- Aggregate throughput: 512 × ~625 TFLOPS = **~320 PFLOPS**
- FLOPs required for 405B at 15T tokens (Llama 3.1 scale): 3.8 × 10²⁵ FLOPs
- Wall-clock: **~3.5–4 years**. Meta used 16,000 H100s for ~2 months for
  exactly this reason.

For realistic **startup-scale training budgets**, the same cluster handles:

| Target tokens | Wall-clock on 512 H100s (405B, 63% MFU) |
|---|---|
| 100B (small continued-pretraining) | ~10 days |
| 500B (substantial continued-pretraining) | ~50 days |
| 1T (meaningful from-scratch attempt) | ~3.5 months |
| 3T | ~10.5 months |
| 15T (Llama-3.1-class full pretraining) | ~3.8 years (impractical) |

**Takeaway:** the 512-GPU reservation is the right size for fine-tuning,
continued-pretraining, or training a 70B–200B model from scratch. Full
from-scratch pretraining of a 405B-class model on Llama's full token budget
is not in scope for this cluster size — that class of run requires
thousands of GPUs. If the client's product assumes a 405B base model, the
realistic path is **LoRA/QLoRA fine-tuning** of an open-weights checkpoint
(fits easily even on 16 GPUs; see operational breakdown in the main report).

## Operational readiness demonstrated

Beyond the throughput numbers, the PoC validated four operational concerns
that matter for a real training run:

| Concern | What was demonstrated |
|---|---|
| **Multi-node NCCL health** | Smoke test passes (200 steps across 16 GPUs, 2 nodes over IB) |
| **Checkpoint/resume** | `test_resume.sbatch` verifies phase-2 resumes from phase-1's step-10 checkpoint; loss continues cleanly |
| **Data pipeline at scale** | 9.6B tokens pre-tokenized as uint32 `.bin`; `StatefulDataLoader` resumes mid-epoch |
| **Deterministic failure modes** | Expected OOMs at 70B for HSDP and SGO were pre-flagged in sbatch comments; they failed *exactly* where and why predicted |

## Recommendations for the client's production setup

1. **Framework: client's choice.** The parallelism strategies measured here
   are standard PyTorch primitives (FSDP, TP via `parallelize_module`, PP
   via `torch.distributed.pipelining`) and work in any framework that
   exposes them — HF Accelerate, DeepSpeed, Megatron-Core/NeMo, or direct
   PyTorch. We used **TorchTitan** for the PoC because it let us flip
   between all six strategies via CLI flags, which minimized PoC
   engineering; but the client should pick the framework that best matches
   their model architecture and team familiarity. Notably, TorchTitan is
   Llama-optimized and needs integration work (~1 week) for custom
   non-Llama decoder transformers, and more for exotic architectures where
   Megatron-Core is the stronger fit.

2. **Default strategy: FSDP FULL_SHARD (ZeRO-3)**. One knob, 40% MFU at 70B,
   scales to any cluster size. Use this unless the model outgrows it.

3. **Growth path: TP=8 × PP=N × FSDP-DP for 200B+**. Keep TP=8 within-node
   over NVLink; add PP only when per-GPU memory is exhausted even with FSDP.

4. **Avoid at 70B on 2-node topology:** FSDP HYBRID_SHARD (needs larger
   shard groups), FSDP SHARD_GRAD_OP (keeps params gathered), DDP (fits
   nothing past ~15B).

5. **Storage layout:** ephemeral caches (`/tmp/*`) on node-local SSD for
   compiler artifacts; shared filesystem for tokenized data, checkpoints,
   and TensorBoard events. This PoC's `/mnt/data/poc-training/training/`
   layout is directly transferable.

6. **Observability:** every experiment writes `run_meta.json` (algorithm +
   parallelism params), `train.log` (per-step metrics), and TensorBoard
   events. `collect_results.py` aggregates into a single
   `summary.{json,csv}` ready for the efficiency report.

## Deliverables

| Artifact | Location |
|---|---|
| Six sbatch files × 8B + 5 × 70B | `training/slurm/exp{1,2,3,4a,4b,4c}_*_{8b,70b}.sbatch` |
| TorchTitan launcher | `training/titan_launcher.py` |
| Raw-PyTorch DDP reference | `training/train.py` |
| Checkpoint resume validation | `training/slurm/test_resume.sbatch` |
| Data prep scripts | `training/download_data.py`, `training/tokenize_data.py` |
| Results aggregator | `training/collect_results.py` |
| Per-run metrics | `training/results/<run>/result.json` |
| Aggregate table | `training/results/summary.{json,csv}` |

All runs reproducible via `bash sync.sh && ssh <cluster> 'sbatch <script>'`.
