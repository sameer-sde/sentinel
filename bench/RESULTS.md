# k6 Load Test Results

Run on MacBook Air (Apple Silicon, ARM64). Sentinel server local, no Docker.
Each test: 45s ramp to 200 VUs.

## Summary

| Workload                  | RPS    | p50    | p95    | Errors | Total req |
|---------------------------|--------|--------|--------|--------|-----------|
| Cache-hot (all repeats)   | 71,483 | 0.83ms | 3.88ms | 0      | 3,216,755 |
| Mixed (80% repeat, 20%)   | 53,636 | 1.05ms | 5.46ms | 0      | 2,413,779 |
| Baseline (per-iter varied)| 48,203 | 1.52ms | 5.20ms | 0      | 2,169,214 |

Total: **7.8M requests across three tests, zero errors.**

## Interpretation

- **Cache-hot** is the upper bound — sustained when production traffic repeats
  (idempotency retries, dashboard polling). Sub-millisecond median.
- **Mixed** is the realistic operating point — 80% repeated traffic, 20%
  unique. Within 27% of the cache-hot ceiling.
- **Baseline** forces the cache to be largely ineffective (every iteration
  perturbs feature[0]); still 48k RPS thanks to the batcher.

Throughput is bounded by the ONNX session's serialized batch execution.
At higher load a multi-session pool would scale linearly until ONNX
saturates a CPU.

## Reproducing

```bash
# Start the server
cd ../cmd/server && ./server &

# Run all three tests
cd ../bench
k6 run 01_baseline.js
k6 run 02_cache_hot.js
k6 run 03_mixed.js
```
