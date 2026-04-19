#!/usr/bin/env python3
import json, os, re, shutil, subprocess, sys

SCRATCH_DIR = os.environ.get("SCRATCH_DIR", "/tmp/validator_results")
os.makedirs(SCRATCH_DIR, exist_ok=True)
node = subprocess.check_output("hostname").decode().strip()

if not shutil.which("ibstat"):
    with open(f"{SCRATCH_DIR}/03_infiniband_{node}.json", "w") as f:
        json.dump({"check": "infiniband", "node": node, "status": "SKIP",
                   "metrics": {"ports": []},
                   "messages": ["ibstat not found — skipping IB check"]}, f, indent=2)
    print(f"[SKIP] InfiniBand: ibstat not found on {node}")
    sys.exit(0)

output = subprocess.run(["ibstat"], capture_output=True, text=True).stdout
status = "PASS"
messages = []
ports = []
ca, cur = None, {}

for line in output.splitlines():
    if m := re.match(r"^CA '(.+)'", line):
        ca = m.group(1)
    elif m := re.match(r"^\s+Port (\d+):", line):
        if cur:
            ok = cur.get("state") == "Active"
            if not ok:
                status = "FAIL"
                messages.append(f"{cur['ca']} port {cur['port']}: state {cur['state']} (expected Active)")
            ports.append({**cur, "status": "PASS" if ok else "FAIL"})
        cur = {"ca": ca, "port": int(m.group(1)), "state": None, "physical_state": None, "rate": None}
    elif cur:
        for key, pat in [("state", r"State: (.+)"),
                         ("physical_state", r"Physical state: (.+)"),
                         ("rate", r"Rate: (.+)")]:
            if m := re.search(pat, line):
                cur[key] = m.group(1).strip()

if cur:
    ok = cur.get("state") == "Active"
    if not ok:
        status = "FAIL"
        messages.append(f"{cur['ca']} port {cur['port']}: state {cur['state']} (expected Active)")
    ports.append({**cur, "status": "PASS" if ok else "FAIL"})

if not ports:
    status = "FAIL"
    messages.append("No IB ports found")

with open(f"{SCRATCH_DIR}/03_infiniband_{node}.json", "w") as f:
    json.dump({"check": "infiniband", "node": node, "status": status,
               "metrics": {"ports": ports}, "messages": messages}, f, indent=2)

print(f"[{status}] InfiniBand on {node}: {len(ports)} port(s) found")
for msg in messages:
    print(f"       {msg}")
sys.exit(0 if status == "PASS" else 1)
