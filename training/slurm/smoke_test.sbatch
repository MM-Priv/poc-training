#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --partition=main
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/poc-training/training/logs/smoke_test_%j.out
#SBATCH --error=/mnt/data/poc-training/training/logs/smoke_test_%j.err

set -euo pipefail

TORCHRUN=/mnt/data/poc-training/.venv/bin/torchrun
SCRIPT=/mnt/data/poc-training/training/smoke_test.py

# Resolve head node IP (filter link-local and loopback)
NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
HEAD_NODE="${NODES[0]}"
HEAD_IP=$(srun --ntasks=1 --nodelist="$HEAD_NODE" hostname -I | \
          awk '{for(i=1;i<=NF;i++) if($i !~ /^169\./ && $i !~ /^127\./) {print $i; exit}}')

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8       # align BLAS thread pools with OMP (prevents oversubscription)
export OPENBLAS_NUM_THREADS=8
export SLURM_CPU_BIND=cores    # pin ranks to physical cores
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=20
export NCCL_SOCKET_IFNAME=eth0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Keep compiler/kernel caches on local ephemeral storage to avoid concurrent-write
# conflicts on the shared filesystem when multiple nodes compile simultaneously.
export TRITON_CACHE_DIR=/tmp/triton_cache
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
export CUDA_CACHE_PATH=/tmp/cuda_cache
export PYTORCH_KERNEL_CACHE_PATH=/tmp/pytorch_kernel_cache

mkdir -p /mnt/data/poc-training/training/logs

echo "Nodes: $SLURM_JOB_NODELIST"
echo "Head IP: $HEAD_IP"

srun --ntasks=2 --ntasks-per-node=1 \
  "$TORCHRUN" \
  --nnodes=2 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${HEAD_IP}:29500" \
  --rdzv_id="${SLURM_JOB_ID}" \
  --max_restarts=0 \
  --role=rank \
  --tee=3 \
  "$SCRIPT"
