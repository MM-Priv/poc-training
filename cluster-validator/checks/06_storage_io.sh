#!/bin/bash
# Storage I/O throughput via dd with direct I/O (bypasses page cache).

set -euo pipefail

SCRATCH_DIR="${SCRATCH_DIR:-/tmp/validator_results}"
mkdir -p "$SCRATCH_DIR"

STORAGE_PATH="${STORAGE_PATH:-/mnt/data}"
TEST_FILE="$STORAGE_PATH/validator_io_test_$$"
SIZE_MB=2048

WRITE_THRESHOLD_MBS="${STORAGE_WRITE_THRESHOLD_MBS:-300}"
READ_THRESHOLD_MBS="${STORAGE_READ_THRESHOLD_MBS:-500}"

NODE=$(hostname)
STATUS="PASS"
MESSAGES=()

cleanup() { rm -f "$TEST_FILE"; }
trap cleanup EXIT

echo "Storage I/O test on $NODE: writing ${SIZE_MB}MB to $TEST_FILE"

WRITE_OUT=$(LC_ALL=C dd if=/dev/zero of="$TEST_FILE" bs=1M count="$SIZE_MB" oflag=direct 2>&1)
WRITE_MBS=$(echo "$WRITE_OUT" | grep -oP '[0-9.]+ [MG]B/s' | tail -1 | awk '{
    val=$1; unit=$2
    if (unit == "GB/s") val = val * 1024
    printf "%.0f", val
}')

READ_OUT=$(LC_ALL=C dd if="$TEST_FILE" of=/dev/null bs=1M count="$SIZE_MB" iflag=direct 2>&1)
READ_MBS=$(echo "$READ_OUT" | grep -oP '[0-9.]+ [MG]B/s' | tail -1 | awk '{
    val=$1; unit=$2
    if (unit == "GB/s") val = val * 1024
    printf "%.0f", val
}')

if [ -z "$WRITE_MBS" ] || [ "$WRITE_MBS" -eq 0 ] 2>/dev/null; then
    STATUS="FAIL"; MESSAGES+=("Could not measure write throughput"); WRITE_MBS=0
elif [ "$WRITE_MBS" -lt "$WRITE_THRESHOLD_MBS" ]; then
    STATUS="FAIL"; MESSAGES+=("Write ${WRITE_MBS} MB/s below threshold ${WRITE_THRESHOLD_MBS} MB/s")
fi

if [ -z "$READ_MBS" ] || [ "$READ_MBS" -eq 0 ] 2>/dev/null; then
    STATUS="FAIL"; MESSAGES+=("Could not measure read throughput"); READ_MBS=0
elif [ "$READ_MBS" -lt "$READ_THRESHOLD_MBS" ]; then
    STATUS="FAIL"; MESSAGES+=("Read ${READ_MBS} MB/s below threshold ${READ_THRESHOLD_MBS} MB/s")
fi

MSGS_JSON=$(printf '"%s",' "${MESSAGES[@]:-}" | sed 's/,$//' | awk '{print "["$0"]"}')
[ ${#MESSAGES[@]} -eq 0 ] && MSGS_JSON="[]"

cat > "$SCRATCH_DIR/06_storage_io.json" <<EOF
{
  "check": "storage_io",
  "node": "$NODE",
  "status": "$STATUS",
  "metrics": {
    "path": "$STORAGE_PATH",
    "test_size_MB": $SIZE_MB,
    "write_MBs": $WRITE_MBS,
    "read_MBs": $READ_MBS,
    "write_threshold_MBs": $WRITE_THRESHOLD_MBS,
    "read_threshold_MBs":  $READ_THRESHOLD_MBS
  },
  "messages": $MSGS_JSON
}
EOF

echo "[${STATUS}] Storage I/O on $NODE: write ${WRITE_MBS} MB/s, read ${READ_MBS} MB/s"
for msg in "${MESSAGES[@]:-}"; do
    [ -n "$msg" ] && echo "       $msg"
done

[ "$STATUS" = "PASS" ]
