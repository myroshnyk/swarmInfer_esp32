# Reference Paper Runs

This directory contains the raw serial logs and derived results from the exact
1,000-image experiment suite that generated every table and numerical claim in
the IEEE Access manuscript *"SwarmInfer: Intra-Layer Tensor-Parallel
Distributed Inference on ESP32-S3 Microcontroller Clusters via ESP-NOW"*.

Shipping these artifacts lets readers verify all statistical claims
(Wilson CIs, McNemar tests, per-class accuracy, per-layer timing) without
needing access to a six-board ESP32-S3 testbed.

## Files

### Raw serial captures (gzipped)

| File | Uncompressed | What it is |
|------|--------------|------------|
| `lite_n1.log.gz` | ~160 KB | Single-node FatCNN-Lite over 1,000 CIFAR-10 test images |
| `fatcnn_n2.log.gz` | ~1.0 MB | Distributed FatCNN, N=2 workers, 1,000 images |
| `fatcnn_n4.log.gz` | ~1.3 MB | Distributed FatCNN, N=4 workers, 1,000 images |

Each file is the raw output of `idf.py monitor` captured by
`scripts/capture_serial.py`. Each file contains 1,000 `CSV,...` data lines
plus the ESP log prefixes, boot banners, and per-layer `ESP_LOGI` traces.

Captured 2026-04-20 on a 6× ESP32-S3 N16R8 testbed (coordinator + up to 4
workers); see the board registry in `docs/SwarmInfer_Port_Mapping.md`.

### Derived results

| File | What it is |
|------|------------|
| `summary_table.tex` | LaTeX source of the main summary table (Table I in paper) |
| `per_layer_table.tex` | LaTeX source of the per-layer latency table |
| `scalability_table.tex` | LaTeX source of the scalability table |
| `accuracy_table.tex` | LaTeX source of the per-class accuracy table |
| `stats.json` | Machine-readable summary: mean/std/p95/p99 latency and Wilson 95% CI per config |
| `mcnemar_lite_vs_n4.json` | McNemar paired-accuracy test FatCNN-Lite vs FatCNN N=4 (the main claim) |
| `mcnemar_lite_vs_n2.json` | McNemar FatCNN-Lite vs FatCNN N=2 |
| `mcnemar_n2_vs_n4.json` | McNemar FatCNN N=2 vs FatCNN N=4 (bit-identical → should show no significant difference) |

## Reproducing the tables from these logs

```bash
# From the repo root:
gunzip -k logs/reference_paper_runs/*.log.gz
python scripts/analyze_logs.py \
    logs/reference_paper_runs/lite_n1.log \
    logs/reference_paper_runs/fatcnn_n2.log \
    logs/reference_paper_runs/fatcnn_n4.log

python scripts/mcnemar.py \
    --log-a logs/reference_paper_runs/lite_n1.log \
    --log-b logs/reference_paper_runs/fatcnn_n4.log \
    --out results/mcnemar_lite_vs_n4.json
```

The generated `results/*.tex` and `results/stats.json` should be **bit-identical**
to the reference copies in this directory. If not, please file an issue.

## Paper numbers reproduced by these logs

| Claim | Value from these logs |
|-------|----------------------|
| FatCNN-Lite accuracy (744/1000) | 74.4% |
| FatCNN N=4 accuracy (791/1000) | 79.1% |
| FatCNN N=2 accuracy | 79.1% (bit-identical to N=4) |
| Wilson 95% CI, FatCNN-Lite | [71.6%, 77.0%] |
| Wilson 95% CI, FatCNN N=4 | [76.5%, 81.5%] |
| CI overlap | 0.54 pp |
| McNemar χ²_cc (Lite vs N=4) | 10.96 |
| McNemar p (chi-square, cc) | 9.29 × 10⁻⁴ |
| McNemar p (exact binomial) | 8.82 × 10⁻⁴ |
| Discordant pairs b, c | 120, 73 |
| Mean latency, FatCNN-Lite | 1,897 ms |
| Mean latency, FatCNN N=2 | 3,653 ms |
| Mean latency, FatCNN N=4 | 2,115 ms |
| N=2 → N=4 speedup | 1.73× |
