# Cluster Validation Report

**Generated:** 2026-04-19T09:57:04.062252+00:00  
**Cluster:** 2 nodes x 8 GPUs  
**Overall:** [PASS] **PASS**

---

## Results

| Check | Status | Details |
|---|---|---|
| GPU Health | [PASS] PASS | 16 GPUs detected |
| NCCL All-Reduce Bandwidth | [PASS] PASS | avg 299.146 GB/s, peak 300.98 GB/s (threshold: 200.0 GB/s) |
| InfiniBand Link Health | [PASS] PASS | 16/16 ports Active |
| Node-to-Node Network Bandwidth | [PASS] PASS | 80.66 Gbit/s (threshold: 10 Gbit/s) |
| Synthetic Distributed Training | [PASS] PASS | train 16141 tok/s/GPU, infer 198853 tok/s/GPU, 3.79 GiB/rank |
| Storage I/O Throughput | [PASS] PASS | write 623 MB/s, read 821 MB/s |

---

## Details

### [PASS] GPU Health

```json
{
  "gpu_count": 16,
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA H200",
      "memory": "143771 MiB",
      "temperature_c": "33",
      "ecc_uncorrectable": "0",
      "status": "PASS"
    },
    {
      "index": 1,
      "name": "NVIDIA H200",
      "memory": "143771 MiB",
      "temperature_c": "27",
      "ecc_uncorrectable": "0",
      "status": "PASS"
    },
    {
      "index": 2,
      "name": "NVIDIA H200",
      "memory": "143771 MiB",
      "temperature_c": "29",
      "ecc_uncorrectable": "0",
      "status": "PASS"
    },
    {
      "index": 3,
      "name": "NVIDIA H200",
      "memory": "143771 MiB",
      "temperature_c": "28",
      "ecc_uncorrectable": "0",
      "status": "PASS"
    },
    {
      "index": 4,
      "name": "NVIDIA H200",
      "memory": "143771 MiB",
      "temperature_c": "29",
      "ecc_uncorrectable": "0",
      "status": "PASS"
    },
    {
      "index": 5,
      "name": "NVIDIA H200",
      "memory": "143771 MiB",
      "temperature_c": "27",
      "ecc_uncorrectable": "0",
      "status": "PASS"
    },
    {
      "index": 6,
      "name": "NVIDIA H200",
      "memory": "143771 MiB",
      "temperature_c": "28",
      "ecc_uncorrectable": "0",
      "status": "PASS"
    },
    {
      "index": 7,
      "name": "NVIDIA H200",
      "memory": "143771 MiB",
      "temperature_c": "26",
      "ecc_uncorrectable": "0",
      "status": "PASS"
    },
    {
      "index": 0,
      "name": "NVIDIA H200",
      "memory": "143771 MiB",
      "temperature_c": "29",
      "ecc_uncorrectable": "0",
      "status": "PASS"
    },
    {
      "index": 1,
      "name": "NVIDIA H200",
      "memory": "143771 MiB",
      "temperature_c": "27",
      "ecc_uncorrectable": "0",
      "status": "PASS"
    },
    {
      "index": 2,
      "name": "NVIDIA H200",
      "memory": "143771 MiB",
      "temperature_c": "29",
      "ecc_uncorrectable": "0",
      "status": "PASS"
    },
    {
      "index": 3,
      "name": "NVIDIA H200",
      "memory": "143771 MiB",
      "temperature_c": "28",
      "ecc_uncorrectable": "0",
      "status": "PASS"
    },
    {
      "index": 4,
      "name": "NVIDIA H200",
      "memory": "143771 MiB",
      "temperature_c": "29",
      "ecc_uncorrectable": "0",
      "status": "PASS"
    },
    {
      "index": 5,
      "name": "NVIDIA H200",
      "memory": "143771 MiB",
      "temperature_c": "28",
      "ecc_uncorrectable": "0",
      "status": "PASS"
    },
    {
      "index": 6,
      "name": "NVIDIA H200",
      "memory": "143771 MiB",
      "temperature_c": "29",
      "ecc_uncorrectable": "0",
      "status": "PASS"
    },
    {
      "index": 7,
      "name": "NVIDIA H200",
      "memory": "143771 MiB",
      "temperature_c": "27",
      "ecc_uncorrectable": "0",
      "status": "PASS"
    }
  ]
}
```

### [PASS] NCCL All-Reduce Bandwidth

```json
{
  "avg_bus_bw_GBs": 299.146,
  "peak_bus_bw_GBs": 300.98,
  "threshold_GBs": 200.0
}
```

### [PASS] InfiniBand Link Health

```json
{
  "ports": [
    {
      "ca": "mlx5_0",
      "port": 1,
      "state": "Active",
      "physical_state": "LinkUp",
      "rate": "400",
      "status": "PASS"
    },
    {
      "ca": "mlx5_1",
      "port": 1,
      "state": "Active",
      "physical_state": "LinkUp",
      "rate": "400",
      "status": "PASS"
    },
    {
      "ca": "mlx5_2",
      "port": 1,
      "state": "Active",
      "physical_state": "LinkUp",
      "rate": "400",
      "status": "PASS"
    },
    {
      "ca": "mlx5_3",
      "port": 1,
      "state": "Active",
      "physical_state": "LinkUp",
      "rate": "400",
      "status": "PASS"
    },
    {
      "ca": "mlx5_4",
      "port": 1,
      "state": "Active",
      "physical_state": "LinkUp",
      "rate": "400",
      "status": "PASS"
    },
    {
      "ca": "mlx5_5",
      "port": 1,
      "state": "Active",
      "physical_state": "LinkUp",
      "rate": "400",
      "status": "PASS"
    },
    {
      "ca": "mlx5_6",
      "port": 1,
      "state": "Active",
      "physical_state": "LinkUp",
      "rate": "400",
      "status": "PASS"
    },
    {
      "ca": "mlx5_7",
      "port": 1,
      "state": "Active",
      "physical_state": "LinkUp",
      "rate": "400",
      "status": "PASS"
    },
    {
      "ca": "mlx5_0",
      "port": 1,
      "state": "Active",
      "physical_state": "LinkUp",
      "rate": "400",
      "status": "PASS"
    },
    {
      "ca": "mlx5_1",
      "port": 1,
      "state": "Active",
      "physical_state": "LinkUp",
      "rate": "400",
      "status": "PASS"
    },
    {
      "ca": "mlx5_2",
      "port": 1,
      "state": "Active",
      "physical_state": "LinkUp",
      "rate": "400",
      "status": "PASS"
    },
    {
      "ca": "mlx5_3",
      "port": 1,
      "state": "Active",
      "physical_state": "LinkUp",
      "rate": "400",
      "status": "PASS"
    },
    {
      "ca": "mlx5_4",
      "port": 1,
      "state": "Active",
      "physical_state": "LinkUp",
      "rate": "400",
      "status": "PASS"
    },
    {
      "ca": "mlx5_5",
      "port": 1,
      "state": "Active",
      "physical_state": "LinkUp",
      "rate": "400",
      "status": "PASS"
    },
    {
      "ca": "mlx5_6",
      "port": 1,
      "state": "Active",
      "physical_state": "LinkUp",
      "rate": "400",
      "status": "PASS"
    },
    {
      "ca": "mlx5_7",
      "port": 1,
      "state": "Active",
      "physical_state": "LinkUp",
      "rate": "400",
      "status": "PASS"
    }
  ]
}
```

### [PASS] Node-to-Node Network Bandwidth

```json
{
  "bandwidth_Gbits": 80.66,
  "threshold_Gbits": 10,
  "parallel_streams": 8,
  "duration_s": 10,
  "server_ip": "10.26.67.231",
  "client_node": "worker-1"
}
```

### [PASS] Synthetic Distributed Training

```json
{
  "world_size": 16,
  "model_params_M": 137,
  "batch_size": 2,
  "seq_len": 1024,
  "train_tokens_per_sec_total": 258262,
  "train_tokens_per_sec_per_gpu": 16141,
  "infer_tokens_per_sec_total": 3181640,
  "infer_tokens_per_sec_per_gpu": 198853,
  "peak_memory_per_rank_GiB": 3.79,
  "threshold_train_tok_per_sec_per_gpu": 2000
}
```

### [PASS] Storage I/O Throughput

```json
{
  "path": "/mnt/data",
  "test_size_MB": 2048,
  "write_MBs": 623,
  "read_MBs": 821,
  "write_threshold_MBs": 300,
  "read_threshold_MBs": 500
}
```
