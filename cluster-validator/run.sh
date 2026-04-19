#!/bin/bash
# Docker entrypoint — runs all checks and writes a report.
#
# Usage:
#   docker run --gpus all --rm \
#     -v /mnt/data:/mnt/data \
#     poc-validator:latest

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH_DIR="${SCRATCH_DIR:-/tmp/validator_results}"
if [ -w "/mnt/data" ]; then
    RESULT_DIR="${RESULT_DIR:-/mnt/data/poc-training/cluster-validator/results/$(date +%Y%m%d_%H%M%S)}"
else
    RESULT_DIR="${RESULT_DIR:-/tmp/validator_results_$(date +%Y%m%d_%H%M%S)}"
fi
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}"
NCCL_BIN="${NCCL_BIN:-}"
for _candidate in /usr/local/bin/all_reduce_perf_mpi \
                  /usr/local/bin/all_reduce_perf \
                  /opt/nccl-tests/build/all_reduce_perf_mpi \
                  /opt/nccl-tests/build/all_reduce_perf \
                  /usr/bin/all_reduce_perf_mpi \
                  /usr/bin/all_reduce_perf; do
    [ -x "$_candidate" ] && { NCCL_BIN="$_candidate"; break; }
done
MPIRUN="${MPIRUN:-$(which mpirun 2>/dev/null || echo mpirun)}"

if [ -z "$NCCL_BIN" ]; then
    echo "ERROR: No all_reduce_perf binary found. Cannot run NCCL check."
    exit 1
fi

export SCRATCH_DIR
mkdir -p "$SCRATCH_DIR"

echo "======================================================"
echo " Cluster Validator"
echo " GPUs: $NUM_GPUS   Results: $SCRATCH_DIR"
echo "======================================================"
echo ""

OVERALL_EXIT=0

run_check() {
    local name="$1"; shift
    echo "--- $name ---"
    if ! "$@"; then
        OVERALL_EXIT=1
    fi
    echo ""
}

run_check "GPU Health" python3 "$SCRIPT_DIR/checks/01_gpu_health.py"

echo "--- NCCL All-Reduce ---"
NCCL_OUT_FILE="$SCRATCH_DIR/nccl_raw.txt"
"$MPIRUN" --allow-run-as-root -np "$NUM_GPUS" "$NCCL_BIN" \
    -b 512M -e 8G -f 2 -g 1 2>&1 | tee "$NCCL_OUT_FILE" || true
python3 "$SCRIPT_DIR/checks/02_nccl_parse.py" \
    --input "$NCCL_OUT_FILE" \
    --output-dir "$SCRATCH_DIR" \
    --threshold "${NCCL_BW_THRESHOLD_GBS:-350}" || OVERALL_EXIT=1
echo ""

run_check "InfiniBand" python3 "$SCRIPT_DIR/checks/03_infiniband.py"

echo "--- Network Bandwidth ---"
echo "[SKIP] Network Bandwidth: single-node mode, skipped"
cat > "$SCRATCH_DIR/04_network_bandwidth.json" <<EOF
{"check": "network_bandwidth", "status": "SKIP", "metrics": {}, "messages": ["Single-node mode — skipped"]}
EOF
echo ""

run_check "Synthetic Training" \
    torchrun --standalone --nproc_per_node="$NUM_GPUS" \
    "$SCRIPT_DIR/checks/05_synthetic_train.py" \
    --result-dir "$SCRATCH_DIR"

run_check "Storage I/O" bash "$SCRIPT_DIR/checks/06_storage_io.sh"

echo "--- Generating Report ---"
python3 "$SCRIPT_DIR/report/report.py" \
    --input "$SCRATCH_DIR" \
    --output "$RESULT_DIR" \
    --gpus-per-node "$NUM_GPUS" \
    --nodes 1

exit $OVERALL_EXIT
