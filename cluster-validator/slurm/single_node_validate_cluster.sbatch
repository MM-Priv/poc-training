#!/bin/bash
#SBATCH --job-name=validate-docker
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --partition=main
#SBATCH --output=/mnt/data/poc-training/cluster-validator/logs/validate_docker_%j.out
#SBATCH --error=/mnt/data/poc-training/cluster-validator/logs/validate_docker_%j.err
#SBATCH --container-image=ghcr.io/blonck/poc-validator:v4
#SBATCH --container-mounts=/mnt/data:/mnt/data
#SBATCH --container-entrypoint
