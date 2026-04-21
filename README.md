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
pub/
├── common/                     # Shared libraries
│   ├── tensor_ops.c/h         # INT8 tensor operations (conv2d, relu, maxpool, gap, dense)
│   └── swarm_protocol.h       # ESP-NOW packet protocol definitions
├── models/                     # ML pipeline
│   ├── train_fatcnn.py        # Train FatCNN (CIFAR-10, 408K params)
│   ├── train_fatcnn_lite.py   # Train FatCNN-Lite baseline (103K params)
│   ├── fix_quantize.py        # INT8 quantization + C weight export
│   ├── partition_n.py         # Partition weights for N workers
│   ├── export_batch.py        # Export CIFAR-10 test images for batch testing
│   ├── verify_int8.py         # Python INT8 simulation (bit-exact verification)
│   └── verify_prediction.py   # Quick float32 prediction check
├── firmware/                   # ESP-IDF projects
│   ├── swarm_coordinator/     # Distributed coordinator (broadcast → gather → dense)
│   ├── swarm_worker/          # Distributed worker (receive → compute → send)
│   ├── single_inference/      # Single-node baseline (FatCNN-Lite)
│   ├── espnow_benchmark/      # ESP-NOW latency benchmark (ping-pong)
│   ├── multi_coordinator/     # Multi-peer broadcast test
│   ├── multi_worker/          # Multi-peer worker
│   ├── gather_benchmark_coordinator/  # Communication overhead benchmark
│   └── gather_benchmark_worker/       # Communication overhead worker
├── scripts/                    # Build & experiment automation
│   ├── setup_weights.sh       # Generate and partition weights
│   ├── flash_workers.sh       # Flash all worker boards
│   ├── flash_coordinator.sh   # Flash coordinator board
│   ├── run_experiment.sh      # Run complete experiment
│   └── power_estimation.py    # Datasheet-based energy estimation
└── docs/                       # Documentation
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

# Train FatCNN (takes ~5 minutes on GPU; SEED=42 is set for reproducibility
# of the training procedure, though TF/hardware version differences can still
# perturb the final weights)
python train_fatcnn.py
# Output: fatcnn_float32.keras (~77% CIFAR-10 accuracy on the full 10k test set)

# Train FatCNN-Lite baseline
python train_fatcnn_lite.py
# Output: fatcnn_lite_float32.keras (~74% accuracy on the full 10k test set)

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

### E8: Power Estimation

```bash
python scripts/power_estimation.py
```

No hardware needed — uses datasheet values + measured timing.

## Key Results

Evaluated on 1,000 CIFAR-10 test images per configuration.

| Experiment | Result |
|---|---|
| Single-node FatCNN-Lite (103K params) | 1,897 ms, 74.4% accuracy (744/1000) |
| **Distributed FatCNN N=4 (408K params)** | **2,115 ms, 79.1% accuracy (791/1000)** |
| Distributed FatCNN N=2 | 3,653 ms, 79.1% accuracy (identical predictions to N=4) |
| Scalability N=2→N=4 | 1.73× speedup |
| Accuracy difference (Lite vs N=4) | McNemar χ²=10.96, p<10⁻³ |
| Bitmap sparsification | −5.5% latency (lossless) |
| 240 MHz vs 160 MHz | −23% latency |
| ESP-NOW throughput | 81.3 KB/s unicast, 0% loss |

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
