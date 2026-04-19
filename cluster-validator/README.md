# Cluster Validator

This tool runs 6 checks that cover the full stack a training or inference workload
depends on: GPU health, GPU-to-GPU communication bandwidth (NCCL), InfiniBand fabric,
node-to-node network throughput, a live distributed training step, and storage I/O.
Each check produces a structured result, and the tool exits non-zero if anything fails.

## Checks

| # | Check | Passes When |
|---|---|---|
| 1 | GPU Health | All GPUs visible, no ECC errors, temperature < 85°C |
| 2 | NCCL All-Reduce | Avg bus bandwidth ≥ 350 GB/s (single-node) / ≥ 200 GB/s (multi-node) |
| 3 | InfiniBand | All ports Active |
| 4 | Node-to-Node Bandwidth | TCP throughput ≥ 10 Gbit/s (multi-node only) |
| 5 | Synthetic Training | DDP training + inference completes without errors |
| 6 | Storage I/O | Write ≥ 300 MB/s, read ≥ 500 MB/s on `/mnt/data` |

Thresholds can be overridden via environment variables (see individual scripts in `checks/`).

## Single-Node (Docker)

Runs all checks on one node via `run.sh` as the container entrypoint. Check 4
(node-to-node bandwidth) is skipped — it requires two nodes and cannot run from a
single container entrypoint. The NCCL all-reduce check (check 2) covers GPU
interconnect bandwidth and runs fine single-node via mpirun.

```bash
docker build -t poc-validator .

docker run --gpus all --rm \
  -v /mnt/data:/mnt/data \
  poc-validator
```

Or via Slurm (1 node, container entrypoint):

```bash
sbatch slurm/single_node_validate_cluster.sbatch
```

Results are written to `/mnt/data/poc-training/cluster-validator/results/<timestamp>/`.

## Multi-Node (Slurm + Docker)

Runs all 6 checks across 2 nodes (16 GPUs) inside the same container image. Every
`srun` step executes inside the container — no cluster-side Python or tools needed.
Edit the `#SBATCH` headers to adjust node count.

```bash
sbatch slurm/multi_node_validate_cluster.sbatch
```

Results are written to `/mnt/data/poc-training/cluster-validator/results/validate_<job_id>/`.

## Output

Both modes produce two files in the results directory:

- `result.md` — summary table with PASS/FAIL per check
- `result.json` — full report with metrics for each check
