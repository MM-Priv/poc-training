#!/bin/bash
# Simple test: print hostname, GPU info, and NCCL topology on each node

echo "=== Node: $(hostname) ==="
echo "Date: $(date)"
echo "User: $(whoami)"
echo "GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "  No GPUs found"
echo ""
