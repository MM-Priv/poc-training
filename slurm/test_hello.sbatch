#!/bin/bash
#SBATCH --job-name=hello-cluster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --partition=main
#SBATCH --output=hello_cluster_%j.out

srun bash /mnt/data/poc-training/tests/hello_cluster.sh
