#!/bin/bash
# Experiment 4a: FSDP FULL_SHARD (ZeRO-3) via TorchTitan — Llama 3.1 70B.
# Shard params+grads+optimizer across all 16 GPUs. Every grad sync crosses IB.
#
# Submit: sbatch training/slurm/exp4a_torchtitan_fsdp.sbatch

#SBATCH --job-name=exp4a_tt_fsdp_70b
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/poc-training/training/logs/exp4a_tt_fsdp_70b_%j.out
#SBATCH --error=/mnt/data/poc-training/training/logs/exp4a_tt_fsdp_70b_%j.err

set -euo pipefail
VENV=/mnt/data/poc-training/.venv
BASE=/mnt/data/poc-training

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
HEAD_NODE="${NODES[0]}"
HEAD_IP=$(srun --ntasks=1 --nodelist="$HEAD_NODE" hostname -I | \
          awk '{for(i=1;i<=NF;i++) if($i !~ /^169\./ && $i !~ /^127\./) {print $i; exit}}')

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8       # align BLAS thread pools with OMP (prevents oversubscription)
export OPENBLAS_NUM_THREADS=8
export SLURM_CPU_BIND=cores    # pin ranks to physical cores
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=20
export NCCL_SOCKET_IFNAME=eth0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export TRITON_CACHE_DIR=/tmp/triton_cache
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
export CUDA_CACHE_PATH=/tmp/cuda_cache
export PYTORCH_KERNEL_CACHE_PATH=/tmp/pytorch_kernel_cache

mkdir -p /mnt/data/poc-training/training/logs

EXP_NAME=exp4a_fsdp_full_70b
ALGO="FSDP FULL_SHARD (ZeRO-3)"
DP_REPLICATE=1; DP_SHARD=16; TP=1; PP=1
RUN_DIR="/mnt/data/poc-training/training/results/${EXP_NAME}_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
cat > "$RUN_DIR/run_meta.json" <<EOF
{"experiment": "$EXP_NAME", "algorithm": "$ALGO", "job_id": "$SLURM_JOB_ID",
 "model_flavor": "70B", "nodes": 2, "gpus_per_node": 8,
 "parallelism": {"dp_replicate": $DP_REPLICATE, "dp_shard": $DP_SHARD, "tp": $TP, "pp": $PP,
                 "reshard_after_forward": "default"}}
EOF

echo "Job $SLURM_JOB_ID | nodes=$SLURM_JOB_NODELIST | head_ip=$HEAD_IP | run_dir=$RUN_DIR"

srun --ntasks=2 --ntasks-per-node=1 \
  "$VENV/bin/torchrun" \
  --nnodes         2 \
  --nproc_per_node 8 \
  --rdzv_id        "$SLURM_JOB_ID" \
  --rdzv_backend   c10d \
  --rdzv_endpoint  "$HEAD_IP:29500" \
  --max_restarts   0 \
  --role           rank \
  --tee            3 \
  "$BASE/training/titan_launcher.py" \
  --job.config_file "$BASE/training/configs/llama3_70b_base.toml" \
  --job.dump_folder "$RUN_DIR" \
  --comm.train_timeout_seconds 300 \
  --parallelism.data_parallel_replicate_degree $DP_REPLICATE \
  --parallelism.data_parallel_shard_degree     $DP_SHARD \
  --parallelism.tensor_parallel_degree         $TP \
  --parallelism.pipeline_parallel_degree       $PP \
  2>&1 | tee "$RUN_DIR/train.log"

echo "Job finished at $(date)"
