#!/bin/bash
#SBATCH --job-name=nccl-p2p-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --partition=main
#SBATCH --output=nccl_p2p_%j.out
#SBATCH --exclusive

export NCCL_DEBUG=INFO

srun --mpi=pmix /usr/bin/all_reduce_perf_mpi -b 8 -e 1G -f 2 -g 1
