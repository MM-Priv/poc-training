#!/bin/bash
# Experiment 1: DDP — Llama 3.1 8B baseline (raw PyTorch DDP via train.py).
# Each GPU holds a full model copy; gradients all-reduced after every backward.
# Standalone entry point — all other experiments route through TorchTitan.
#
# Submit: sbatch training/slurm/exp1_ddp.sbatch

#SBATCH --job-name=exp1_ddp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/poc-training/training/logs/exp1_ddp_%j.out
#SBATCH --error=/mnt/data/poc-training/training/logs/exp1_ddp_%j.err

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
export SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1

export TRITON_CACHE_DIR=/tmp/triton_cache
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
export CUDA_CACHE_PATH=/tmp/cuda_cache
export PYTORCH_KERNEL_CACHE_PATH=/tmp/pytorch_kernel_cache

mkdir -p /mnt/data/poc-training/training/logs

echo "Job $SLURM_JOB_ID | nodes=$SLURM_JOB_NODELIST | head_ip=$HEAD_IP"

srun --ntasks=2 --ntasks-per-node=1 \
  "$VENV/bin/torchrun" \
  --nnodes=2 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${HEAD_IP}:29500" \
  --rdzv_id="${SLURM_JOB_ID}" \
  --max_restarts=0 \
  --role=rank \
  --tee=3 \
  "$BASE/training/train.py" \
  --config "$BASE/training/configs/exp1_ddp_8b.yaml"

echo "Job finished at $(date)"
