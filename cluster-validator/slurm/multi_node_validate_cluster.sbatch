#!/bin/bash
#SBATCH --job-name=validate-docker
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --partition=main
#SBATCH --output=/mnt/data/poc-training/cluster-validator/logs/validate_docker_%j.out
#SBATCH --error=/mnt/data/poc-training/cluster-validator/logs/validate_docker_%j.err

set -euo pipefail

CHECKS=/validator/checks
REPORT=/validator/report
RESULT_DIR="/mnt/data/poc-training/cluster-validator/results/validate_${SLURM_JOB_ID}"

mkdir -p "$RESULT_DIR" /mnt/data/poc-training/cluster-validator/logs

# Container image used per srun step — script body runs on the host.
CONTAINER_IMAGE=ghcr.io/blonck/poc-validator:v4
CONTAINER_MOUNTS=/mnt/data:/mnt/data
SRUN_CONTAINER="--container-image=$CONTAINER_IMAGE --container-mounts=$CONTAINER_MOUNTS"
# env SCRATCH_DIR=... ensures the NFS path overrides the container's default ENV
SRUN_ENV="env SCRATCH_DIR=$RESULT_DIR"

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_NODE_IP=$(srun --ntasks=1 --ntasks-per-node=1 --nodelist="$HEAD_NODE" hostname -I | \
               awk '{for(i=1;i<=NF;i++) if($i !~ /^169\./ && $i !~ /^127\./) {print $i; exit}}')

echo "======================================================"
echo " Cluster Validator (Docker)"
echo " Image:  $CONTAINER_IMAGE"
echo " Nodes:  $SLURM_NODELIST   Head IP: $HEAD_NODE_IP"
echo " Results: $RESULT_DIR"
echo "======================================================"
echo ""

OVERALL_EXIT=0
fail() { OVERALL_EXIT=1; }

# 01 GPU Health
echo "--- GPU Health ---"
srun --ntasks=2 --ntasks-per-node=1 $SRUN_CONTAINER \
    $SRUN_ENV python3 "$CHECKS/01_gpu_health.py" || fail
echo ""

# 02 NCCL All-Reduce (threshold lowered for cross-node IB vs single-node NVLink)
echo "--- NCCL All-Reduce ---"
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=20

NCCL_OUT_FILE="$RESULT_DIR/nccl_raw.txt"
srun --ntasks=16 --ntasks-per-node=8 --mpi=pmix $SRUN_CONTAINER \
    /usr/local/bin/all_reduce_perf_mpi -b 512M -e 8G -f 2 -g 1 2>&1 | tee "$NCCL_OUT_FILE" || fail
srun --ntasks=1 $SRUN_CONTAINER \
    $SRUN_ENV python3 "$CHECKS/02_nccl_parse.py" \
    --input "$NCCL_OUT_FILE" \
    --output-dir "$RESULT_DIR" \
    --threshold 200 || fail
echo ""

# 03 InfiniBand
echo "--- InfiniBand ---"
srun --ntasks=2 --ntasks-per-node=1 $SRUN_CONTAINER \
    $SRUN_ENV python3 "$CHECKS/03_infiniband.py" || fail
echo ""

# 04 Network Bandwidth
echo "--- Network Bandwidth ---"
export IPERF3_SERVER_IP="$HEAD_NODE_IP"
srun --ntasks=2 --ntasks-per-node=1 $SRUN_CONTAINER \
    $SRUN_ENV bash "$CHECKS/04_network_bandwidth.sh" || fail
echo ""

# 05 Synthetic Training
echo "--- Synthetic Training ---"
export NCCL_SOCKET_IFNAME=eth0

srun --ntasks=2 --ntasks-per-node=1 $SRUN_CONTAINER \
    $SRUN_ENV torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${HEAD_NODE_IP}:29500" \
    --rdzv_id="${SLURM_JOB_ID}" \
    "$CHECKS/05_synthetic_train.py" \
    --result-dir "$RESULT_DIR" || fail
echo ""

# 06 Storage I/O
echo "--- Storage I/O ---"
srun --ntasks=1 --ntasks-per-node=1 --nodelist="${HEAD_NODE}" $SRUN_CONTAINER \
    $SRUN_ENV bash "$CHECKS/06_storage_io.sh" || fail
echo ""

# Report
echo "--- Generating Report ---"
srun --ntasks=1 $SRUN_CONTAINER \
    $SRUN_ENV python3 "$REPORT/report.py" \
    --input  "$RESULT_DIR" \
    --output "$RESULT_DIR" \
    --nodes  "$SLURM_JOB_NUM_NODES" \
    --gpus-per-node 8

exit $OVERALL_EXIT
