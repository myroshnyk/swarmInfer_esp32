# SwarmInfer: Intra-Layer Parallel Distributed Inference on ESP32-S3 Microcontroller Clusters via ESP-NOW

Distributed neural network inference across multiple ESP32-S3 microcontrollers using ESP-NOW protocol. Each worker computes a subset of output channels (intra-layer tensor parallelism), enabling models that don't fit in a single MCU's SRAM.

## Hardware Requirements

| Component | Qty | Notes |
|---|---|---|
| ESP32-S3-DevKitC N16R8 | 5-6 | 1 coordinator + 4 workers (+ 1 spare) |
| USB-A to USB-C data cables | 6 | Must be data cables, not charge-only |
| Powered USB hub | 1 | 7+ ports, powered (12V/2A recommended) |
| Computer (macOS/Linux) | 1 | For building and flashing |

## Software Requirements

- **ESP-IDF v6.0** — [Installation guide](https://docs.espressif.com/projects/esp-idf/en/v6.0/esp32s3/get-started/)
- **Python 3.11+** with TensorFlow, NumPy (`pip install tensorflow numpy`)
- **Git** (for cloning this repository)

## Repository Structure

```
swarmInfer_esp32/
├── common/                       # Shared libraries (used by every firmware project)
│   ├── tensor_ops.c/h            # INT8 tensor ops (conv2d, relu, maxpool, gap, dense)
│   ├── swarm_protocol.h          # ESP-NOW packet protocol definitions
│   ├── ina219.c/h                # INA219 high-side power-sensor driver
│   └── power_meter.c/h           # On-device power sampling (behind SWARM_POWER_MEASURE)
├── models/                       # ML + analysis pipeline
│   ├── train_fatcnn.py           # Train FatCNN (CIFAR-10, 408K params)
│   ├── train_fatcnn_lite.py      # Train FatCNN-Lite baseline (103K params)
│   ├── fix_quantize.py           # INT8 quantization + C weight export
│   ├── partition_n.py            # Even partition of weights for N workers
│   ├── partition_nscale.py       # Uneven (non-divisible) shards for N=3/5
│   ├── export_batch.py           # Export CIFAR-10 test images for batch testing
│   ├── verify_int8.py            # Python INT8 simulation (bit-exact verification)
│   ├── verify_prediction.py      # Quick float32 prediction check
│   ├── power_analyze.py          # Derive the measured power/energy table (E8)
│   ├── quant_ablation.py         # float32 / INT8 / per-channel ablation (E12)
│   ├── sparsity_dist.py          # Per-layer/per-image sparsity distribution (E13)
│   ├── seed_sweep.py             # 3-seed training-variance sweep (E14)
│   ├── analyze_nscale.py         # N=3/5 uneven-shard scaling analysis (E9)
│   ├── analyze_rf_sweep.py       # Degraded-link robustness analysis (E10)
│   ├── capture_*.py              # Serial-capture helpers for the experiments
│   └── mobilenet/                # Scaled-MobileNet 96x96 toolchain (E11)
├── firmware/                     # ESP-IDF projects
│   ├── swarm_coordinator/        # Distributed coordinator (broadcast → gather → dense)
│   ├── swarm_worker/             # Distributed worker (receive → compute → send)
│   ├── single_inference/         # Single-node baseline (FatCNN-Lite)
│   ├── swarm_{coordinator,worker}_nscale/  # Uneven shards, N=3/5 (E9)
│   ├── swarm_{coordinator,worker}_rel/     # Reliability-layer prototype (E10)
│   ├── {single_inference,swarm_coordinator,swarm_worker}_mbnet/  # MobileNet (E11)
│   ├── sparse_bench/             # On-device encode/decode benchmark (E13)
│   ├── espnow_benchmark/         # ESP-NOW latency benchmark (ping-pong)
│   ├── multi_{coordinator,worker}/         # Multi-peer broadcast test
│   └── gather_benchmark_{coordinator,worker}/  # Communication-overhead benchmark
├── scripts/                      # Build & experiment automation
│   ├── setup_weights.sh          # Generate and partition weights
│   ├── flash_{workers,coordinator}.sh      # Flash boards
│   ├── run_experiment.sh         # Run a complete experiment
│   ├── analyze_logs.py           # Accuracy/latency/Wilson-CI from serial logs
│   ├── mcnemar.py                # Paired McNemar test on two configs
│   ├── capture_serial.py         # Serial capture utility
│   └── power_estimation.py       # DEPRECATED datasheet estimate (superseded by power_analyze.py)
├── results/                      # Derived results (json/md/csv) backing every table
├── logs/                         # Raw serial captures (large ones gzipped)
│   ├── reference_paper_runs/     # The 1,000-image runs that reproduce the main tables
│   ├── power_runs/               # INA219 power captures (E8)
│   ├── instrumented_runs/        # Compute/transmit + sparsification ablation (gzipped)
│   └── *.log                     # nscale, MobileNet, sparse-bench captures
└── docs/                         # Documentation (incl. power_measurement_procedure.md)
```

## Quick Start

### 1. Setup Environment

```bash
# ESP-IDF
source /path/to/esp-idf/export.sh

# Python (for model training)
pip install tensorflow numpy
# or: conda activate swarm-ml
```

### 2. Board Setup

Connect all ESP32-S3 boards to the USB hub. Identify each board's USB port:

```bash
ls /dev/cu.usbmodem*  # macOS
ls /dev/ttyACM*       # Linux
```

Update the port assignments in `scripts/flash_workers.sh` and `scripts/flash_coordinator.sh`.

### 3. Initialize ESP-IDF Targets (one-time only)

```bash
cd firmware/swarm_worker && idf.py set-target esp32s3 && cd ../..
cd firmware/swarm_coordinator && idf.py set-target esp32s3 && cd ../..
cd firmware/single_inference && idf.py set-target esp32s3 && cd ../..
# Repeat for other firmware projects as needed
```

### 4. Train Model and Generate Weights

```bash
cd models

# Train FatCNN (~5 minutes on GPU). NOTE: the original training runs did not
# fix a random seed, so retraining produces a model near—but not bit-identical
# to—the one used in the paper. The exact per-table numbers in the paper are
# reproducible bit-identically from the committed raw serial logs
# (logs/reference_paper_runs/) via the analysis scripts, independent of
# retraining or TF/hardware version differences.
python train_fatcnn.py
# Output: fatcnn_float32.keras (~77% CIFAR-10 accuracy on the full 10k test set)

# Train FatCNN-Lite baseline
python train_fatcnn_lite.py
# Output: fatcnn_lite_float32.keras (~74% accuracy on the full 10k test set)

# Training-time variance is characterized by a deterministic, fixed-seed sweep
# (seeds 0-2) — see results/seed_sweep.json (gap 2.95 pp, 95% CI [1.76, 4.13]):
python seed_sweep.py

# Quantize + export weights for N=4 workers
python fix_quantize.py
mv c_weights c_weights_n4

# Partition for N=4
python partition_n.py 4
# Output: c_weights_n4/ with worker_0..3_weights.h, coordinator_weights.h

# Export test images
python export_batch.py
cp test_images_batch.h c_weights_n4/

# Symlink
ln -s c_weights_n4 c_weights

cd ..
```

Or use the automated script:
```bash
scripts/setup_weights.sh 4
```

### 5. Verify Quantization (optional)

```bash
cd models
python verify_int8.py
# Should print: Prediction matches Python INT8 simulation
```

## Experiments

### E1: Single-Node Baseline

```bash
cd firmware/single_inference
idf.py build
idf.py -p /dev/cu.usbmodem212301 flash   # Use your coordinator port
idf.py -p /dev/cu.usbmodem212301 monitor
```

**Expected output:** `Prediction: cat (class 3), Latency: ~1,890 ms`

### E2: ESP-NOW Point-to-Point Benchmark

```bash
cd firmware/espnow_benchmark
idf.py build
# Flash to Board 0, then run with Board 1 also running the benchmark firmware
idf.py -p /dev/cu.usbmodem212301 flash
idf.py -p /dev/cu.usbmodem212301 monitor
```

**Expected output:** RTT table for payload sizes 10-240 bytes, ~81 KB/s max throughput.

### E3: Multi-Peer Communication Test

```bash
# Flash workers
cd firmware/multi_worker
idf.py build
# Flash to boards 1-4

# Flash coordinator
cd firmware/multi_coordinator
idf.py build
idf.py -p /dev/cu.usbmodem212301 flash
idf.py -p /dev/cu.usbmodem212301 monitor
```

**Expected output:** 200/200 broadcast success, ~5.3 ms all-reply latency.

### E4: Distributed Inference (N=4 Workers)

```bash
# Ensure weights are set up for N=4
scripts/setup_weights.sh 4

# Flash workers (uses SWARM_WORKER_ID env var — no manual sed needed)
cd firmware/swarm_worker
SWARM_WORKER_ID=0 idf.py build && idf.py -p /dev/cu.usbmodem212401 flash
SWARM_WORKER_ID=1 idf.py build && idf.py -p /dev/cu.usbmodem212201 flash
SWARM_WORKER_ID=2 idf.py build && idf.py -p /dev/cu.usbmodem2121401 flash
SWARM_WORKER_ID=3 idf.py build && idf.py -p /dev/cu.usbmodem2121301 flash

# Flash coordinator
cd ../swarm_coordinator
SWARM_N_WORKERS=4 idf.py build && idf.py -p /dev/cu.usbmodem212301 flash

# Monitor
idf.py -p /dev/cu.usbmodem212301 monitor
```

**Expected output:**
- Batch accuracy: 791/1000 = 79.1%
- Average latency: ~2,115 ms (with sparse encoding)
- 0 timeouts

Or use the automated script:
```bash
scripts/run_experiment.sh 4
```

### E5: Distributed Inference (N=2 Workers)

```bash
# IMPORTANT: Disconnect boards 3 and 4 from USB hub first!

scripts/setup_weights.sh 2

cd firmware/swarm_worker
SWARM_WORKER_ID=0 idf.py build && idf.py -p /dev/cu.usbmodem212401 flash
SWARM_WORKER_ID=1 idf.py build && idf.py -p /dev/cu.usbmodem212201 flash

cd ../swarm_coordinator
SWARM_N_WORKERS=2 idf.py build && idf.py -p /dev/cu.usbmodem212301 flash
idf.py -p /dev/cu.usbmodem212301 monitor
```

**Expected output:** 791/1000 = 79.1% accuracy, ~3,653 ms avg latency.

**IMPORTANT:** Unused worker boards must be disconnected to prevent WiFi interference.

### E6: Gather Communication Benchmark

Measures pure communication overhead without compute.

```bash
# Flash benchmark worker to all 4 boards
cd firmware/gather_benchmark_worker
SWARM_WORKER_ID=0 idf.py build && idf.py -p /dev/cu.usbmodem212401 flash
SWARM_WORKER_ID=1 idf.py build && idf.py -p /dev/cu.usbmodem212201 flash
SWARM_WORKER_ID=2 idf.py build && idf.py -p /dev/cu.usbmodem2121401 flash
SWARM_WORKER_ID=3 idf.py build && idf.py -p /dev/cu.usbmodem2121301 flash

# Flash + run coordinator
cd ../gather_benchmark_coordinator
SWARM_N_WORKERS=4 idf.py build && idf.py -p /dev/cu.usbmodem212301 flash
idf.py -p /dev/cu.usbmodem212301 monitor
```

Then repeat with `SWARM_N_WORKERS=1` (only worker 0 connected) for baseline.

**Expected output:** Timing table for 64-4096 byte payloads, N=1 vs N=4.

### E7: CPU Frequency Experiment (240 MHz)

Modify `sdkconfig` in both `firmware/swarm_worker/` and `firmware/swarm_coordinator/`:

```
CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_160=n
CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y
CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ=240
CONFIG_ESP32S3_DEFAULT_CPU_FREQ_160=n
CONFIG_ESP32S3_DEFAULT_CPU_FREQ_240=y
CONFIG_ESP32S3_DEFAULT_CPU_FREQ_MHZ=240
```

Then `idf.py fullclean && idf.py build` for both, flash all boards, and run E4.

**Expected output:** ~1,623 ms avg latency (−23% vs 160 MHz).

### E8: Power & Energy (measured, INA219)

Active power and energy per inference are measured directly with an INA219
high-side sensor (0.1 Ω shunt on the 5 V rail). Because USB power bypasses the
shunt, each board is battery-powered through the sensor; see
[`docs/power_measurement_procedure.md`](docs/power_measurement_procedure.md)
for the full capture procedure. Then derive the table:

```bash
python models/power_analyze.py \
    --single logs/power_runs/single.log \
    --coord  logs/power_runs/coord.log \
    --pwrw   logs/power_runs/coord_workers_run.log
# -> results/power.json : single-node 0.37 W / 0.71 J, N=4 cluster 3.40 W / 7.20 J (~10x energy)
```

The legacy `scripts/power_estimation.py` is a **deprecated** datasheet estimate,
superseded by this direct measurement.

### E9–E14: Resubmission experiments (v1.2.0)

The revised paper adds five experiments; each ships its scripts, raw logs, and
derived results so the corresponding table/figure reproduces without hardware:

| # | Experiment | Run / analyze | Data |
|---|---|---|---|
| E9  | Uneven shards + N=5 scaling | `firmware/swarm_*_nscale`, `models/partition_nscale.py`, `models/analyze_nscale.py` | `results/nscale_scaling.json`, `logs/fatcnn_nscale_n{3,5}.log` |
| E10 | Robustness under a degraded link | `firmware/swarm_*_rel`, `models/analyze_rf_sweep.py` | `results/rf_sweep.{json,md}` |
| E11 | Scaled MobileNet 96×96 | `firmware/*_mbnet`, `models/mobilenet/` | `results/mbnet_r2_11.{json,md}`, `logs/mbnet_distributed.log` |
| E12 | Quantization ablation (per-tensor vs per-channel, ranges, saturation) | `models/quant_ablation.py` | `results/quant_ablation.{json,md}` |
| E13 | Sparsity distribution + encode/decode overhead | `firmware/sparse_bench`, `models/sparsity_dist.py` | `results/sparsity_dist.{json,md}`, `results/sparse_bench.csv` |
| E14 | Training-time variance (3 seeds) | `models/seed_sweep.py` | `results/seed_sweep.json` |

## Key Results

Evaluated on 1,000 CIFAR-10 test images per configuration.

| Experiment | Result |
|---|---|
| Single-node FatCNN-Lite (103K params) | 1,897 ms, 74.4% accuracy (744/1000) |
| **Distributed FatCNN N=4 (408K params)** | **2,115 ms, 79.1% accuracy (791/1000)** |
| Distributed FatCNN N=2 | 3,653 ms, 79.1% accuracy (identical predictions to N=4) |
| Scalability N=2→N=4→N=5 | 1.73× → 1.99× speedup (1,838 ms at N=5) |
| Accuracy difference (Lite vs N=4) | McNemar χ²=10.96, p<10⁻³; 3-seed gap 2.95 pp, 95% CI [1.76, 4.13] |
| Bitmap sparsification | −5.9% latency (lossless, isolated on/off ablation) |
| Measured power / energy | single 0.37 W / 0.71 J; N=4 cluster 3.40 W / 7.20 J (~10×) |
| Robustness (one worker behind a wall) | released firmware ~14× latency tail; reliability-layer prototype 30/30 bit-exact |
| Scaled MobileNet 96×96 (1.09M params) | float 89.3% / INT8 89.2%, communication-bound, 670 KB/inference |
| 240 MHz vs 160 MHz | −23% latency |
| ESP-NOW throughput | 81.3 KB/s unicast, 0% loss (1 m bench, no contention) |

## Reproducing the Paper's Numbers Without Hardware

Every table and statistical claim in the paper can be regenerated from the
raw serial captures shipped in [`logs/reference_paper_runs/`](logs/reference_paper_runs/)
(three gzipped log files, one per configuration, ~300 KB total). These are
the exact 1,000-image runs that produced the numbers cited in the
manuscript.

```bash
gunzip -k logs/reference_paper_runs/*.log.gz
python scripts/analyze_logs.py \
    logs/reference_paper_runs/lite_n1.log \
    logs/reference_paper_runs/fatcnn_n2.log \
    logs/reference_paper_runs/fatcnn_n4.log
# Regenerates: results/summary_table.tex, per_layer_table.tex,
#              accuracy_table.tex, scalability_table.tex, stats.json
# Prints: accuracy, Wilson 95% CIs, per-layer latency, CI overlap analysis

python scripts/mcnemar.py \
    --log-a logs/reference_paper_runs/lite_n1.log \
    --log-b logs/reference_paper_runs/fatcnn_n4.log \
    --out results/mcnemar_lite_vs_n4.json
# Prints: χ² = 10.96, p = 9.3×10⁻⁴ (paper's main statistical claim)
```

The `logs/reference_paper_runs/README.md` file lists every paper number
reproduced by these artifacts. Running the full hardware experiment suite
(requires 4 ESP32-S3 + 1 coordinator board, ~6 hours wall-clock time) is
only necessary to collect *new* data — for verifying the paper's existing
claims, the shipped logs are sufficient.

The worker firmware now emits `CSV_SPARSE` telemetry lines after each
convolutional layer (`layer, worker_id, result_size, zero_count,
sparsity_ppm`); `analyze_logs.py` aggregates these into the per-layer
activation-sparsity values cited in Section III. Because sparsity depends
on trained-weight values, the shipped reference logs predate this
instrumentation; to reproduce the ~50% / ~67% sparsity figures from §III
you need to re-run the distributed firmware on hardware.

## Technical Details

### Quantization
- INT8 asymmetric per-tensor: `real_value = (quantized - zero_point) × scale`
- Accumulation in INT32, requantization via fixed-point multiplier
- Bias quantization: `bias_q = round(bias_float / (input_scale × kernel_scale))`

### Protocol
- ESP-NOW, 240-byte packets (8B header + 232B data)
- Broadcast for input distribution, unicast for result gathering
- WORKER_READY handshake between layers
- Auto-trigger compute on last chunk received

### Memory Layout
- Tensors: channels-last `[H, W, C]`
- Weights: `[C_out, kH, kW, C_in]` (transposed from TensorFlow)
- All large arrays heap-allocated (stack >4KB causes crash on ESP32)

## License

MIT

## Citation

If you use this code, please cite:

```bibtex
@article{myroshnyk2026swarminfer,
  title={SwarmInfer: Intra-Layer Parallel Distributed Inference on ESP32 Microcontroller Clusters via ESP-NOW},
  author={Myroshnyk, Yurii},
  journal={IEEE Access},
  year={2026}
}
```
