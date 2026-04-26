#!/bin/bash
# Quick resume-flag check — Llama 3.1 8B, FSDP FULL.
# Phase 1: train 10 steps, save a checkpoint.
# Phase 2: resume from that checkpoint, train to step 20.
# Pass condition: phase 2 starts at step ~11 with a loss close to where phase 1
# left off (not a random-init loss near 12).
#
# Submit: sbatch training/slurm/test_resume.sbatch

#SBATCH --job-name=test_resume
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/poc-training/training/logs/test_resume_%j.out
#SBATCH --error=/mnt/data/poc-training/training/logs/test_resume_%j.err

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

RUN_DIR="/mnt/data/poc-training/training/results/test_resume_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
CKPT_DIR="$RUN_DIR/checkpoint"

launch() {
  srun --ntasks=2 --ntasks-per-node=1 \
    "$VENV/bin/torchrun" \
    --nnodes         2 \
    --nproc_per_node 8 \
    --rdzv_id        "${SLURM_JOB_ID}_$1" \
    --rdzv_backend   c10d \
    --rdzv_endpoint  "$HEAD_IP:29500" \
  --max_restarts   0 \
  --role           rank \
  --tee            3 \
    "$BASE/training/titan_launcher.py" \
    --job.config_file "$BASE/training/configs/llama3_8b_base.toml" \
    --job.dump_folder "$RUN_DIR" \
  --comm.train_timeout_seconds 300 \
    --parallelism.data_parallel_shard_degree 16 \
    --checkpoint.enable \
    --checkpoint.folder              "$CKPT_DIR" \
    --checkpoint.interval            10 \
    --checkpoint.no_last_save_model_only \
    "${@:2}"
}

echo "=== Phase 1: train 10 steps + save a full checkpoint at step 10 ==="
launch phase1 --training.steps 10

echo "=== Phase 2: resume from checkpoint, train to step 20 ==="
# With the stateful fineweb_edu loader, dataloader state is saved & reloaded
# cleanly — no exclude_from_loading workaround needed.
launch phase2 --training.steps 20

echo "Job finished at $(date)"
