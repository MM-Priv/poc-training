#!/bin/bash
# NCCL all-reduce benchmark across all GPUs on all nodes

echo "=== Node: $(hostname) ==="
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Run NCCL all-reduce perf test across all ranks
# -b: min message size, -e: max message size, -f: step factor, -g: GPUs per thread
/usr/bin/all_reduce_perf -b 8 -e 1G -f 2 -g 1
