# Multi-node LLM training PoC — Nebius H200 cluster

## Executive summary

We set up an end-to-end, multi-node pre-training stack on a 2-node × 8×H200
Nebius cluster and ran Llama 3.1 (8B and 70B) under six distribution
strategies. Three findings stand out:

1. **The platform delivers production-grade efficiency.** With `torch.compile`
   and FP8 enabled, 70B FSDP FULL reaches **63.4% MFU** (Model FLOPs
   Utilization) at 1,396 tok/s/GPU. The plain BF16 baseline lands at 40.2% MFU,
   so the gap between the two represents the software-optimization headroom
   available on the same hardware.
2. **DDP is the highest-MFU choice as long as the model fits per-GPU.** At 8B
   it reaches 48% MFU and beats every FSDP variant we measured. FSDP only
   pays off when memory forces it. At 70B, FSDP FULL_SHARD is the workhorse;
   HYBRID and SGO need shard groups too large for a 2-node topology.
3. **For models above 100B parameters, the growth path is `TP=8 × PP=N ×
   FSDP-DP`.** Tensor parallelism stays intra-node over NVLink, pipeline
   parallelism ships activations cheaply across InfiniBand, and FSDP shards
   whatever state is left.

The choice of training framework is independent of the strategy. We used
TorchTitan because it exposes all six strategies as CLI flags on a single
launcher, which minimised PoC engineering. The same numbers are reachable from
HF Accelerate, DeepSpeed, Megatron-Core / NeMo, or raw PyTorch — they are
properties of the hardware and the strategy, not of TorchTitan.

## What was built

The cluster is a 2-node × 8-GPU H200 system running soperator Slurm, with
NVLink intra-node and InfiniBand inter-node. The pre-training stack consists
of:

- **A data pipeline** that downloads the FineWeb-Edu 10BT sample and tokenises
  it with the Llama 3.1 tokenizer into uint32 binaries (9.6B train and 100M
  validation tokens).
- **Six distribution strategies**, each implemented as a thin sbatch wrapper.
  Five run via TorchTitan with overridden `--parallelism.*` CLI flags;
  experiment 1 (DDP) uses a raw-PyTorch reference loop in `training/train.py`
  so the customer has a minimal example without a framework dependency.

| # | Strategy | Parallelism config |
|---|---|---|
| 1 | DDP (raw PyTorch baseline) | dp_replicate=16 |
| 2 | Tensor Parallel | tp=8, dp_shard=2 |
| 3 | Tensor + Pipeline | tp=8, pp=2, dp=1 |
| 4a | FSDP FULL_SHARD (ZeRO-3) | dp_shard=16 |
| 4b | FSDP HYBRID_SHARD | dp_replicate=2, dp_shard=8 |
| 4c | FSDP SHARD_GRAD_OP (ZeRO-2) | dp_shard=16, reshard_after_forward=never |

Two further sbatch jobs validate the cluster itself: `smoke_test.sbatch` is a
toy-transformer NCCL bring-up, and `test_resume.sbatch` verifies that
checkpoint and resume round-trip correctly.

Supporting tooling: `collect_results.py` parses per-run logs into a
`result.json` per run plus an aggregate `summary.{json,csv}`. Compiler and
kernel caches (`TRITON_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`,
`PYTORCH_KERNEL_CACHE_PATH`, `CUDA_CACHE_PATH`) are pinned to node-local
`/tmp` to avoid concurrent-write conflicts on the shared filesystem. The
top-level `setup.sh` stages the Llama 3.1 tokenizer to a shared asset path
and installs all dependencies via `uv sync`.

## Measured training efficiency

MFU is computed against H200's 989 TFLOPS BF16 dense peak (the no-sparsity
figure that applies to real training). All runs use full activation
checkpointing, BF16 mixed precision, sequence length 4096, and 50 measured
steps after a ~12-step warmup.

### 8B — every strategy fits per-GPU

We measured each FSDP variant at two batch sizes: `local_batch=1` provides a
fair cross-strategy baseline, and `local_batch=4` exploits the spare memory
that 8B leaves on an H200. DDP, TP, and TP+PP were measured only at bs=1: DDP
already saturates compute at bs=1, and TP / TP+PP at 8B are bandwidth-floor
cases that don't benefit from larger batches.

| # | Strategy | MFU bs=1 | MFU bs=4 | Mem (config) | tok/s/GPU (config) |
|---|---|---|---|---|---|
| 1 | DDP | **48.0%** | — | 83.7 GB (bs=1) | 7,906 (bs=1) |
| 4c | FSDP SGO | 38.9% | **44.6%** | 54.9 GB (bs=4) | **8,559** (bs=4) |
| 4a | FSDP FULL | 36.6% | 43.7% | 42.2 GB (bs=4) | 8,399 (bs=4) |
| 4b | FSDP HYBRID | 36.4% | 43.6% | 53.7 GB (bs=4) | 8,370 (bs=4) |
| 2 | TP | 4.6% | — | 10.5 GB (bs=1) | 886 (bs=1) |
| 3 | TP + PP | 1.9% | — | 9.6 GB (bs=1) | 357 (bs=1) |

### 70B — memory pressure exposes strategy limits

| # | Strategy | Config | MFU | Mem/GPU | tok/s/GPU | Outcome |
|---|---|---|---|---|---|---|
| **4a** | **FSDP FULL** | bs=1 | **40.2%** | 90.9 GB | **886** | ✅ |
| 2 | TP | bs=1 | 16.3% | 71.6 GB | 358 | ✅ |
| 2 | TP | bs=4 | **31.0%** | 77.2 GB | **683** | ✅ bs=4 nearly doubles MFU |
| 3 | TP + PP | μb=2 | 2.2% | 69.9 GB | 49 | ⚠️ bubble-limited |
| 3 | TP + PP | μb=8 | 2.9% | 70.1 GB | 65 | ⚠️ marginal gain over μb=2 |
| 4b | FSDP HYBRID | bs=1 | 14.6%* | 137.4 GB peak | 321* | ❌ OOM mid-run |
| 4c | FSDP SGO | bs=1 | — | — | — | ❌ OOM at init |
| 1 | DDP | bs=1 | — | — | — | ❌ N/A (140 GB params alone) |
| **4a+** | **FSDP FULL + compile + FP8** | bs=1 | **63.4%** | 89.3 GB | **1,396** | ✅ tuned ceiling |

\* HYBRID values are partial-run figures captured before the OOM and are not
directly comparable to the other rows.

## Interpretation

Per-strategy MFU only becomes meaningful once the batch size and microbatch
count have been sized to the cluster. The bs=1 numbers are fair for
cross-strategy comparison, but they correspond to default settings rather
than to a production configuration.

**The memory wall at 70B.** Running DDP at 70B is infeasible because 140 GB
of BF16 parameters alone already exceed an H200. At 8B, DDP is the fastest
option (48% MFU) because the full model fits per-GPU and PyTorch's gradient
all-reduce is overlapped with backward. Above 8B, sharded strategies become
mandatory.

**FSDP FULL_SHARD is the workhorse for 70B.** It delivers 40% MFU at 91 GB/GPU
and is the only FSDP variant that survives at 70B on our 2-node topology.
FSDP HYBRID's 8-way shard group leaves too much state on each GPU; FSDP SGO
keeps parameters gathered after the forward pass and never frees the full
140 GB. Both failures are pure memory arithmetic, not runtime surprises.

**TP and TP+PP look weak at 8B (4.6% and 1.9% MFU) because they are designed
for bigger models and bigger batches.** At 70B with `local_batch_size=4`,
TP-alone reaches 31% MFU — comparable to FSDP FULL. The rule of thumb is that
TP's per-layer collectives are amortised across the batch, so the batch must
grow large enough to keep them hidden behind compute.

**TP+PP is bubble-limited under TorchTitan's default 1F1B schedule.**
Increasing the number of microbatches from 2 to 8 only moved MFU from 2.2% to
2.9%. Realising the theoretical bubble reduction requires interleaved or
zero-bubble schedules, which are still experimental in TorchTitan. The
practical recommendation for the customer is to use TP+PP only when pipeline
parallelism is unavoidable (200B+) and to plan for schedule tuning on top.

**`torch.compile` together with FP8 is the tuned ceiling.** At 63.4% MFU on
70B FSDP FULL, this is a 1.58× throughput improvement over the BF16 baseline
(398 → 627 TFLOPS/GPU, 886 → 1,396 tok/s/GPU) — and at *lower* per-GPU memory,
because FP8 weights are smaller. The number sits above TorchTitan's published
reference of 54.5% MFU (Llama 3 70B on 64 H100s, BF16; see [TorchTitan
paper, arXiv:2410.06511](https://arxiv.org/abs/2410.06511)) and is at the
well-tuned ceiling for 70B on H200.

| Config | MFU | TFLOPS/GPU | tok/s/GPU | Mem/GPU |
|---|---|---|---|---|
| BF16 baseline | 40.2% | 398 | 886 | 91 GB |
| BF16 + compile + FP8 | **63.4%** | **627** | **1,396** | 89 GB |

## Scaling guidance for 512 H100s

The strategy selection does not change as the cluster grows. FSDP FULL with
`dp_shard=512` remains the simplest and most efficient option up to roughly
200B parameters. HSDP only starts to pay off when each shard group is large
enough to hold a full optimizer-state copy — for a Llama 70B that means at
least 16 GPUs per shard group, which corresponds to ≥2 nodes with
topology-aware placement.

Above 200B parameters the canonical layout becomes `TP=8 × PP=N × FSDP-DP`.
TP=8 stays inside one node over NVLink, PP ships activations across IB in
inexpensive send-recv operations, and FSDP shards whatever state remains
across the data-parallel dimension. Switching between any of these layouts is
a one-line CLI change in TorchTitan and requires no modifications to the
training code.

### Does 512 H100s comfortably handle a 405B-class model?

**Memory is not the constraint.** A 3D layout `TP=8 × PP=8 × FSDP-DP=8`
shards a 405B model along with its gradients and FP32 AdamW state down to
roughly 10 GB/GPU of resident state, which fits well within an 80 GB H100
even before activation-checkpointing savings.

**Wall-clock is the constraint.** At 63% MFU the aggregate throughput on 512
H100s is roughly 320 PFLOPS. For 405B at Llama-3.1 token budgets:

| Target tokens | Wall-clock (405B, 63% MFU, 512 H100) |
|---|---|
| 100B (small continued-pretraining) | ~10 days |
| 500B (substantial continued-pretraining) | ~50 days |
| 1T (meaningful from-scratch attempt) | ~3.5 months |
| 3T | ~10.5 months |
| 15T (Llama-3.1-class full pretraining) | ~3.8 years (impractical) |

A full 15T-token pretraining of a 405B-class model requires thousands of GPUs —
Meta used roughly 16,000 H100s for two months for exactly this reason. If the
customer's product needs a model in the order of 405B, the realistic path on
512 GPUs is LoRA or QLoRA fine-tuning of an open-weights checkpoint, which fits
even on 16 GPUs.

## Operational readiness

Beyond the throughput numbers, the PoC validated four operational concerns
that matter for a real training run:

| Concern | What was demonstrated |
|---|---|
| Multi-node NCCL health | Smoke test ran 200 steps across 16 GPUs and 2 nodes over IB |
| Checkpoint and resume | `test_resume.sbatch` verifies that phase-2 picks up cleanly from phase-1's step-10 checkpoint and that the loss continues without a discontinuity |
| Data pipeline at scale | 9.6B tokens were pre-tokenised as uint32 `.bin` files; TorchTitan's `StatefulDataLoader` resumes mid-epoch |
| Deterministic failure modes | The 70B HYBRID and SGO OOMs were pre-flagged in sbatch comments and failed exactly where and why predicted |

## Recommendations for the customer's setup

1. **Framework: customer's choice.** Every strategy measured here is built on
   standard PyTorch primitives — FSDP, `parallelize_module`, and
   `torch.distributed.pipelining` — so any framework that exposes them will
   reach the same numbers. We picked TorchTitan to flip strategies via CLI
   flags during the PoC; the customer should pick whatever best matches their
   model architecture and team familiarity. TorchTitan is Llama-optimised and
   needs roughly a week of integration work for custom non-Llama decoder
   transformers; exotic architectures are often a better fit for
   Megatron-Core.
2. **Default strategy: FSDP FULL_SHARD.** A single knob, 40% MFU at 70B
   (63% with compile and FP8), and it scales to any cluster size. Use this
   unless the model outgrows it.
3. **Growth path: `TP=8 × PP=N × FSDP-DP` for 200B+.** Keep TP=8 within a
   node, and add pipeline parallelism only when per-GPU memory is exhausted
   even with FSDP.
4. **Storage layout.** Keep ephemeral compiler and kernel caches on
   node-local `/tmp`, and put tokenised data, checkpoints, and TensorBoard
   events on the shared filesystem. The `/mnt/data/poc-training/training/`
   layout used here is directly transferable.
5. **Observability.** Every run writes a `result.json` (config plus a metrics
   summary), a `train.log` (per-step metrics), and TensorBoard events.
   `collect_results.py` aggregates them into `summary.{json,csv}`, which is
   ready to drop into an efficiency report.

## Artifacts

| Artifact | Location |
|---|---|
| Sbatch files (6 × 8B + 5 × 70B + 3 variants `_bs4`/`_mb8`/`_fp8`) | `training/slurm/` |
| TorchTitan launcher | `training/titan_launcher.py` |
| Raw-PyTorch DDP reference | `training/train.py` |
| Checkpoint-resume validation | `training/slurm/test_resume.sbatch` |
| Data prep | `training/download_data.py`, `training/tokenize_data.py` |
| Results aggregator | `training/collect_results.py` |
| Aggregate table (in repo) | `training/results/summary.{json,csv}` |
| Per-run metrics (cluster only) | `/mnt/data/poc-training/training/results/<run>/result.json` |

Every run is reproducible via `bash sync.sh && ssh <cluster> 'sbatch <script>'`.
