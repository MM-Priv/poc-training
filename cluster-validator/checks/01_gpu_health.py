#!/usr/bin/env python3
import json, os, subprocess, sys

SCRATCH_DIR = os.environ.get("SCRATCH_DIR", "/tmp/validator_results")
EXPECTED_GPUS = int(os.environ.get("EXPECTED_GPUS_PER_NODE", "8"))
TEMP_THRESHOLD = 85
os.makedirs(SCRATCH_DIR, exist_ok=True)

node = subprocess.check_output("hostname").decode().strip()
out = subprocess.run(
    ["nvidia-smi",
     "--query-gpu=index,name,memory.total,temperature.gpu,ecc.errors.uncorrected.volatile.total",
     "--format=csv,noheader"],
    capture_output=True, text=True,
).stdout.strip()
rows = [r.strip() for r in out.splitlines() if r.strip()]

status = "PASS"
messages = []
gpus = []

if len(rows) != EXPECTED_GPUS:
    status = "FAIL"
    messages.append(f"Expected {EXPECTED_GPUS} GPUs, found {len(rows)}")

for row in rows:
    idx, name, mem, temp, ecc = [f.strip() for f in row.split(",", 4)]
    gpu_ok = True
    if ecc not in ("0", "[N/A]"):
        status = "FAIL"; gpu_ok = False
        messages.append(f"GPU {idx}: ECC uncorrectable errors: {ecc}")
    try:
        if int(temp) >= TEMP_THRESHOLD:
            status = "FAIL"; gpu_ok = False
            messages.append(f"GPU {idx}: temperature {temp}°C >= {TEMP_THRESHOLD}°C")
    except ValueError:
        pass
    gpus.append({"index": int(idx), "name": name, "memory": mem,
                 "temperature_c": temp, "ecc_uncorrectable": ecc,
                 "status": "PASS" if gpu_ok else "FAIL"})

with open(f"{SCRATCH_DIR}/01_gpu_health_{node}.json", "w") as f:
    json.dump({"check": "gpu_health", "node": node, "status": status,
               "metrics": {"gpu_count": len(rows), "gpus": gpus},
               "messages": messages}, f, indent=2)

print(f"[{status}] GPU Health on {node}: {len(rows)} GPUs found")
for msg in messages:
    print(f"       {msg}")

sys.exit(0 if status == "PASS" else 1)
