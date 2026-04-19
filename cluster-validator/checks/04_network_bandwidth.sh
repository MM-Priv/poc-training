#!/bin/bash
# Node-to-Node Network Bandwidth via iperf3.
# Node 0 acts as server, node 1 as client.
# Run with: srun --ntasks=2 --ntasks-per-node=1 bash checks/04_network_bandwidth.sh

set -euo pipefail

SCRATCH_DIR="${SCRATCH_DIR:-/tmp/validator_results}"
IPERF3_PORT="${IPERF3_PORT:-15201}"
IPERF3_DURATION="${IPERF3_DURATION:-10}"
PASS_THRESHOLD_GBITS="${NET_BW_THRESHOLD_GBITS:-10}"

NODE_RANK="${SLURM_NODEID:-0}"
NODE=$(hostname)
READY_FILE="$SCRATCH_DIR/.iperf3_ready_${SLURM_JOB_ID:-0}"
SERVER_IP_FILE="$SCRATCH_DIR/.iperf3_server_ip_${SLURM_JOB_ID:-0}"

mkdir -p "$SCRATCH_DIR"

if ! command -v iperf3 &>/dev/null; then
    if [ "$NODE_RANK" = "0" ]; then
        cat > "$SCRATCH_DIR/04_network_bandwidth.json" <<EOF
{"check":"network_bandwidth","status":"SKIP","metrics":{},"messages":["iperf3 not found"]}
EOF
        echo "[SKIP] Network Bandwidth: iperf3 not installed"
    fi
    exit 0
fi

# First non-link-local IP
MY_IP=$(hostname -I | awk '{for(i=1;i<=NF;i++) if($i !~ /^169\./ && $i !~ /^127\./) {print $i; exit}}')
SERVER_IP="${IPERF3_SERVER_IP:-}"

if [ "$NODE_RANK" = "0" ]; then
    echo "$MY_IP" > "$SERVER_IP_FILE"
    iperf3 -s -p "$IPERF3_PORT" &
    IPERF_PID=$!
    # Wait for iperf3 to start listening before signaling ready
    for _i in $(seq 1 20); do
        ss -tln | grep -q ":${IPERF3_PORT}" && break
        sleep 0.5
    done
    echo "ready" > "$READY_FILE"
    echo "iperf3 server on $NODE ($MY_IP:$IPERF3_PORT)"

    WAIT=0
    while [ -f "$READY_FILE" ] && [ $WAIT -lt 60 ]; do
        sleep 1; WAIT=$((WAIT+1))
    done
    kill "$IPERF_PID" 2>/dev/null || true
    rm -f "$SERVER_IP_FILE"
else
    WAIT=0
    while [ ! -f "$READY_FILE" ] && [ $WAIT -lt 30 ]; do
        sleep 1; WAIT=$((WAIT+1))
    done

    if [ ! -f "$READY_FILE" ]; then
        cat > "$SCRATCH_DIR/04_network_bandwidth.json" <<EOF
{"check":"network_bandwidth","status":"FAIL","metrics":{},"messages":["Server did not start within 30s"]}
EOF
        echo "[FAIL] Network Bandwidth: server did not start"
        exit 1
    fi

    if [ -z "$SERVER_IP" ] && [ -f "$SERVER_IP_FILE" ]; then
        SERVER_IP=$(cat "$SERVER_IP_FILE")
    fi

    echo "iperf3 client on $NODE → $SERVER_IP:$IPERF3_PORT"
    IPERF_JSON=$(iperf3 -c "$SERVER_IP" -p "$IPERF3_PORT" -P 8 \
        -t "$IPERF3_DURATION" -J 2>&1) || true

    rm -f "$READY_FILE"

    BW_BITS=$(echo "$IPERF_JSON" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    bw = data.get('end', {}).get('sum_sent', {}).get('bits_per_second', 0)
    print(f'{bw:.0f}')
except Exception:
    print('0')
" 2>/dev/null || echo "0")

    BW_GBITS=$(awk "BEGIN{printf \"%.2f\", $BW_BITS / 1000000000}")

    STATUS="PASS"
    MESSAGE=""
    if awk "BEGIN{exit !($BW_GBITS < $PASS_THRESHOLD_GBITS)}"; then
        STATUS="FAIL"
        MESSAGE="Bandwidth ${BW_GBITS} Gbit/s below threshold ${PASS_THRESHOLD_GBITS} Gbit/s"
    fi

    cat > "$SCRATCH_DIR/04_network_bandwidth.json" <<EOF
{
  "check": "network_bandwidth",
  "status": "$STATUS",
  "metrics": {
    "bandwidth_Gbits": $BW_GBITS,
    "threshold_Gbits": $PASS_THRESHOLD_GBITS,
    "parallel_streams": 8,
    "duration_s": $IPERF3_DURATION,
    "server_ip": "$SERVER_IP",
    "client_node": "$NODE"
  },
  "messages": ["$MESSAGE"]
}
EOF
    echo "[${STATUS}] Network Bandwidth: ${BW_GBITS} Gbit/s (threshold: ${PASS_THRESHOLD_GBITS} Gbit/s)"
    [ -n "$MESSAGE" ] && echo "       $MESSAGE"
    [ "$STATUS" = "PASS" ]
fi
