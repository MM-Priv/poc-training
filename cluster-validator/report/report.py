#!/usr/bin/env python3

import argparse
import json
import os
import glob
from datetime import datetime, timezone


STATUS_ICON = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}

CHECK_LABELS = {
    "gpu_health":          "GPU Health",
    "nccl_allreduce":      "NCCL All-Reduce Bandwidth",
    "infiniband":          "InfiniBand Link Health",
    "network_bandwidth":   "Node-to-Node Network Bandwidth",
    "synthetic_training":  "Synthetic Distributed Training",
    "storage_io":          "Storage I/O Throughput",
}


def load_fragments(input_dir: str) -> list[dict]:
    fragments = []
    for path in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        if path.endswith("result.json"):
            continue
        with open(path) as f:
            try:
                fragments.append(json.load(f))
            except json.JSONDecodeError as e:
                print(f"Warning: could not parse {path}: {e}")
    return fragments


def overall_status(fragments: list[dict]) -> str:
    statuses = [f["status"] for f in fragments]
    if "FAIL" in statuses:
        return "FAIL"
    if all(s == "SKIP" for s in statuses):
        return "SKIP"
    return "PASS"


def format_metrics(check_name: str, metrics: dict) -> str:
    if check_name == "gpu_health":
        return f"{metrics.get('gpu_count', '?')} GPUs detected"
    if check_name == "nccl_allreduce":
        avg = metrics.get("avg_bus_bw_GBs", "?")
        peak = metrics.get("peak_bus_bw_GBs", "?")
        thr = metrics.get("threshold_GBs", "?")
        return f"avg {avg} GB/s, peak {peak} GB/s (threshold: {thr} GB/s)"
    if check_name == "infiniband":
        ports = metrics.get("ports", [])
        active = sum(1 for p in ports if p.get("state") == "Active")
        return f"{active}/{len(ports)} ports Active"
    if check_name == "network_bandwidth":
        bw = metrics.get("bandwidth_Gbits", "?")
        thr = metrics.get("threshold_Gbits", "?")
        return f"{bw} Gbit/s (threshold: {thr} Gbit/s)"
    if check_name == "synthetic_training":
        train = metrics.get("train_tokens_per_sec_per_gpu", "?")
        infer = metrics.get("infer_tokens_per_sec_per_gpu", "?")
        mem = metrics.get("peak_memory_per_rank_GiB", "?")
        return f"train {train} tok/s/GPU, infer {infer} tok/s/GPU, {mem} GiB/rank"
    if check_name == "storage_io":
        w = metrics.get("write_MBs", "?")
        r = metrics.get("read_MBs", "?")
        return f"write {w} MB/s, read {r} MB/s"
    return str(metrics)


def merge_metrics(check_name: str, existing: dict, new: dict) -> dict:
    """Merge metrics from multiple nodes for the same check."""
    if check_name == "gpu_health":
        merged = dict(existing)
        merged["gpu_count"] = existing.get("gpu_count", 0) + new.get("gpu_count", 0)
        merged["gpus"] = existing.get("gpus", []) + new.get("gpus", [])
        return merged
    if check_name == "infiniband":
        merged = dict(existing)
        merged["ports"] = existing.get("ports", []) + new.get("ports", [])
        return merged
    return existing


def build_report(fragments: list[dict], cluster_info: dict) -> dict:
    checks = {}
    for f in fragments:
        name = f["check"]
        # Merge multi-node fragments for the same check (keep worst status)
        if name in checks:
            existing = checks[name]
            if f["status"] == "FAIL" or (f["status"] != "PASS" and existing["status"] == "PASS"):
                checks[name]["status"] = f["status"]
            checks[name].setdefault("nodes", []).append(f.get("node", "unknown"))
            existing["messages"].extend(f.get("messages", []))
            existing["metrics"] = merge_metrics(name, existing["metrics"], f.get("metrics", {}))
        else:
            checks[name] = {
                "label": CHECK_LABELS.get(name, name),
                "status": f["status"],
                "metrics": f.get("metrics", {}),
                "messages": [m for m in f.get("messages", []) if m],
                "nodes": [f["node"]] if "node" in f else [],
            }

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cluster": cluster_info,
        "overall_status": overall_status(fragments),
        "checks": checks,
    }


def render_markdown(report: dict) -> str:
    ts = report["generated_at"]
    overall = report["overall_status"]
    icon = STATUS_ICON.get(overall, "?")
    cluster = report.get("cluster", {})

    lines = [
        "# Cluster Validation Report",
        "",
        f"**Generated:** {ts}  ",
        f"**Cluster:** {cluster.get('nodes', '?')} nodes x {cluster.get('gpus_per_node', '?')} GPUs  ",
        f"**Overall:** {icon} **{overall}**",
        "",
        "---",
        "",
        "## Results",
        "",
        "| Check | Status | Details |",
        "|---|---|---|",
    ]

    for name, check in report["checks"].items():
        status = check["status"]
        icon_s = STATUS_ICON.get(status, "?")
        label = check["label"]
        detail = format_metrics(name, check.get("metrics", {}))
        lines.append(f"| {label} | {icon_s} {status} | {detail} |")

    lines += ["", "---", "", "## Details", ""]

    for name, check in report["checks"].items():
        status = check["status"]
        icon_s = STATUS_ICON.get(status, "?")
        lines.append(f"### {icon_s} {check['label']}")
        metrics = check.get("metrics", {})
        if metrics:
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(metrics, indent=2))
            lines.append("```")
        msgs = [m for m in check.get("messages", []) if m]
        if msgs:
            lines.append("")
            for msg in msgs:
                lines.append(f"> WARNING: {msg}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.environ.get("SCRATCH_DIR", "/tmp/validator_results"),
                        help="Directory containing per-check JSON fragments")
    parser.add_argument("--output", required=True,
                        help="Output directory for result.json and result.md")
    parser.add_argument("--nodes", type=int, default=int(os.environ.get("SLURM_JOB_NUM_NODES", 1)))
    parser.add_argument("--gpus-per-node", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    fragments = load_fragments(args.input)
    if not fragments:
        print(f"No result fragments found in {args.input}")
        raise SystemExit(1)

    cluster_info = {
        "nodes": args.nodes,
        "gpus_per_node": args.gpus_per_node,
        "total_gpus": args.nodes * args.gpus_per_node,
    }

    report = build_report(fragments, cluster_info)

    json_path = os.path.join(args.output, "result.json")
    md_path = os.path.join(args.output, "result.md")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(md_path, "w") as f:
        f.write(render_markdown(report))

    overall = report["overall_status"]
    icon = STATUS_ICON.get(overall, "?")
    print(f"\nOverall: {icon} {overall}")
    print(f"   report:  {md_path}")
    print(f"   json:    {json_path}")

    if overall == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
