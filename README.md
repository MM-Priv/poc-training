# Multi-node LLM training PoC

## Repo layout

```
.
├── cluster-validator/   Docker container with 6 cluster-health checks
├── training/            Training scripts, configs, and results
│   ├── slurm/           sbatch files for each experiment
│   ├── configs/         TorchTitan TOML + raw-PyTorch YAML
│   └── results/         report.md + summary.{json,csv} (in repo)
├── infra/               Terraform for cluster bring-up
├── setup.sh             one-time setup (uv sync, stage tokenizer)
└── sync.sh              rsync local repo → cluster
```


## Where to start

| If you want… | Open |
|---|---|
| The summary | [`training/results/report.md`](training/results/report.md) |
| Cluster-validator details | [`cluster-validator/README.md`](cluster-validator/README.md) |
| Training-script details | [`training/README.md`](training/README.md) |

## Reproducing on a Soperator cluster

```bash
# One-time setup on the login node
bash setup.sh

# Tokenize FineWeb-Edu (~3 h)
HF_TOKEN=hf_... .venv/bin/python training/download_data.py  --cache_dir /mnt/data/poc-training/data/hf_cache
HF_TOKEN=hf_... .venv/bin/python training/tokenize_data.py  --cache_dir /mnt/data/poc-training/data/hf_cache --output_dir /mnt/data/poc-training/data

# Validate the cluster
sbatch cluster-validator/slurm/multi_node_validate_cluster.sbatch

# Run an experiment
sbatch training/slurm/exp4a_torchtitan_fsdp_70b.sbatch

# Aggregate results
python training/collect_results.py --results_dir training/results
```

Every experiment writes `result.json` (config + metrics summary),
`train.log`, and TensorBoard events under `training/results/<run>/`.
`collect_results.py` aggregates them into `training/results/summary.{json,csv}`.
