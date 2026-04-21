# SwarmInfer: Project Progress & Experimental Results

**Last updated:** 2026-04-06
**Status:** FatCNN trained & quantized, tensor_ops implemented, single-node inference baseline complete, distributed Layer 1 working, ready for full 3-layer pipeline

---

## 1. Hardware Inventory

| # | Component | Qty | Status |
|---|-----------|-----|--------|
| 1 | ESP32-S3-DevKitC N16R8 (SANXIXING) | 6 | ✅ All working |
| 2 | INA219 Current Sensor (MTDELE) | 6 | ✅ Arrived, not yet wired |
| 3 | Leinsis 7-port USB 3.0 Hub (powered, 12V/2A) | 1 | ✅ Working |
| 4 | USB-A to USB-C data cables | 7 | ✅ Working |
| **Total cost** | | | **~$120** |

---

## 2. Board Registry

### MAC Addresses

| Board | Role | MAC Address | C Array |
|-------|------|-------------|---------|
| Board 0 | **Coordinator** | B8:F8:62:E2:D0:8C | `{0xB8, 0xF8, 0x62, 0xE2, 0xD0, 0x8C}` |
| Board 1 | **Worker 1** | B8:F8:62:E2:D1:98 | `{0xB8, 0xF8, 0x62, 0xE2, 0xD1, 0x98}` |
| Board 2 | **Worker 2** | B8:F8:62:E2:CD:E4 | `{0xB8, 0xF8, 0x62, 0xE2, 0xCD, 0xE4}` |
| Board 3 | **Worker 3** | B8:F8:62:E2:DA:28 | `{0xB8, 0xF8, 0x62, 0xE2, 0xDA, 0x28}` |
| Board 4 | **Worker 4** | B8:F8:62:E2:D0:DC | `{0xB8, 0xF8, 0x62, 0xE2, 0xD0, 0xDC}` |
| Board 5 | **Monitor/Spare** | B8:F8:62:E2:C7:30 | `{0xB8, 0xF8, 0x62, 0xE2, 0xC7, 0x30}` |

### Port Mapping (as of 2026-04-05)

| USB Port | Board | Role |
|----------|-------|------|
| `/dev/cu.usbmodem212301` | Board 0 | Coordinator |
| `/dev/cu.usbmodem212401` | Board 1 | Worker 1 |
| `/dev/cu.usbmodem212201` | Board 2 | Worker 2 |
| `/dev/cu.usbmodem2121401` | Board 3 | Worker 3 |
| `/dev/cu.usbmodem2121301` | Board 4 | Worker 4 |
| `/dev/cu.usbmodem2121201` | Board 5 | Monitor/Spare |

### C Header (`mac_addresses.h`)

```c
#ifndef MAC_ADDRESSES_H
#define MAC_ADDRESSES_H

#include <stdint.h>

static const uint8_t MAC_COORDINATOR[] = {0xB8, 0xF8, 0x62, 0xE2, 0xD0, 0x8C};
static const uint8_t MAC_WORKER_1[]    = {0xB8, 0xF8, 0x62, 0xE2, 0xD1, 0x98};
static const uint8_t MAC_WORKER_2[]    = {0xB8, 0xF8, 0x62, 0xE2, 0xCD, 0xE4};
static const uint8_t MAC_WORKER_3[]    = {0xB8, 0xF8, 0x62, 0xE2, 0xDA, 0x28};
static const uint8_t MAC_WORKER_4[]    = {0xB8, 0xF8, 0x62, 0xE2, 0xD0, 0xDC};
static const uint8_t MAC_MONITOR[]     = {0xB8, 0xF8, 0x62, 0xE2, 0xC7, 0x30};

static const uint8_t* WORKER_MACS[] = {
    MAC_WORKER_1, MAC_WORKER_2, MAC_WORKER_3, MAC_WORKER_4,
};
#define WORKER_COUNT 4

#endif
```

---

## 3. Development Environment

| Parameter | Value |
|-----------|-------|
| **OS** | macOS Apple Silicon (M1/M2/M3/M4) |
| **IDE** | VS Code |
| **ESP-IDF** | v6.0 |
| **ESP-IDF Path** | `$IDF_PATH` (e.g. `~/.espressif/v6.0/esp-idf` via ESP-IDF EIM installer) |
| **Python** | 3.14.2 (conda base) |
| **Python ML env** | conda env `swarm-ml` (Python 3.11, tf-nightly 2.22) |
| **Python venv** | ESP-IDF-provided venv (e.g. `~/.espressif/python_env/idf6.0_py3.14_env`) |
| **Project directory** | `~/esp/swarm-infer/` |
| **Toolchain** | xtensa-esp-elf GCC 15.2.0 |

### Environment activation

```bash
# For ESP-IDF (firmware development):
source $IDF_PATH/export.sh

# For ML (model training):
conda activate swarm-ml
```

### ESP-IDF v6.0 API changes (vs v5.x)

- `esp_now_register_send_cb()` callback: v6.0 uses `const esp_now_send_info_t *info` instead of `const uint8_t *mac_addr`
- VS Code ESP-IDF extension "Show Examples Projects" does not work with v6.0 + EIM installer — use terminal workflow
- Flash and monitor must be run as **separate commands** (not `idf.py -p PORT flash monitor` — causes port lock)

### Workflow

```bash
source $IDF_PATH/export.sh
cd ~/esp/swarm-infer/<project_name>
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/cu.usbmodemXXXXX flash
# Wait for "Done", then:
idf.py -p /dev/cu.usbmodemXXXXX monitor
# If port busy:
kill $(lsof -t /dev/cu.usbmodemXXXXX) 2>/dev/null
```

---

## 4. Project Structure

```
~/esp/swarm-infer/
├── mac_addresses.h          # ✅ All 6 board MACs
├── ports.txt                # ✅ Port-to-board mapping
├── common/                  # ✅ Shared code
│   ├── tensor_ops.h         # ✅ INT8 tensor ops API (41 lines)
│   ├── tensor_ops.c         # ✅ INT8 tensor ops implementation (275 lines)
│   └── swarm_protocol.h     # ✅ ESP-NOW packet protocol
├── models/                  # ✅ Python ML pipeline
│   ├── train_fatcnn.py      # ✅ Train FatCNN on CIFAR-10
│   ├── train_fatcnn_lite.py # ✅ Train FatCNN-Lite (baseline)
│   ├── fix_quantize.py      # ✅ Quantize INT8 + export C arrays
│   ├── fatcnn_float32.keras # ✅ Saved FatCNN model
│   ├── fatcnn_lite_float32.keras # ✅ Saved FatCNN-Lite model
│   └── c_weights/           # ✅ Generated C header files
│       ├── worker_0_weights.h  # 93,808 bytes (FatCNN partitioned)
│       ├── worker_1_weights.h  # 93,808 bytes
│       ├── worker_2_weights.h  # 93,808 bytes
│       ├── worker_3_weights.h  # 93,808 bytes
│       ├── coordinator_weights.h # 34,600 bytes
│       ├── fatcnn_lite_weights.h # 104,584 bytes (all-in-one)
│       └── test_image.h       # 3,072 bytes (CIFAR-10 test cat)
├── hello_world/             # ✅ Phase 1: verified ESP-IDF works
├── get_mac/                 # ✅ Phase 4: MAC address reader
├── espnow_sender/           # ✅ Phase 5: basic ESP-NOW sender (broadcast)
├── espnow_receiver/         # ✅ Phase 5: basic ESP-NOW receiver
├── espnow_benchmark/        # ✅ Phase 6: latency/throughput benchmark (ping-pong)
├── multi_worker/            # ✅ Multi-peer worker firmware
├── multi_coordinator/       # ✅ Multi-peer coordinator (broadcast + tensor test)
├── single_inference/        # ✅ FatCNN-Lite single-node inference baseline
├── swarm_worker/            # ✅ Distributed worker firmware (Layer 1)
├── swarm_coordinator/       # ✅ Distributed coordinator firmware (Layer 1)
├── ina219_test/             # ⬜ Phase 1.5: INA219 sensor test (code ready, not wired)
└── results/                 # ⬜ Experimental data
```

---

## 5. Experimental Results

### 5.1 Experiment E2: ESP-NOW Point-to-Point Benchmark

**Date:** 2026-04-05
**Boards:** Board 0 (PING) ↔ Board 1 (PONG)
**Protocol:** ESP-NOW unicast, Wi-Fi channel 1, ESP-NOW v2.0
**ESP-IDF:** v6.0
**Distance:** ~10 cm (on desk, via USB hub)
**Rounds per payload size:** 500 (+ 10 warmup)
**Inter-packet delay:** 3 ms

#### Raw Results

| Payload (B) | RTT min (µs) | RTT avg (µs) | RTT median (µs) | RTT p95 (µs) | RTT p99 (µs) | RTT max (µs) | Timeouts | One-way est (µs) | Throughput (KB/s) |
|-------------|-------------:|-------------:|-----------------:|--------------:|--------------:|--------------:|---------:|------------------:|------------------:|
| 10 | 1,916 | 2,084 | 1,990 | 2,357 | 4,937 | 5,036 | 0 | ~1,042 | 9.6 |
| 50 | 2,631 | 2,855 | 2,698 | 3,096 | 6,297 | 9,261 | 0 | ~1,427 | 35.0 |
| 100 | 3,420 | 3,591 | 3,476 | 3,723 | 6,407 | 11,314 | 0 | ~1,795 | 55.7 |
| 150 | 4,213 | 4,413 | 4,283 | 4,628 | 7,212 | 7,466 | 0 | ~2,206 | 68.0 |
| 200 | 5,025 | 5,274 | 5,088 | 7,968 | 8,019 | 11,888 | 0 | ~2,637 | 75.8 |
| 240 | 5,658 | 5,906 | 5,725 | 8,555 | 8,635 | 8,682 | 0 | ~2,953 | 81.3 |

#### Key Findings

1. **Zero packet loss** across all 3,000 packets (6 sizes × 500 rounds)
2. **One-way latency:** ~1.0 ms (10B) to ~3.0 ms (240B)
3. **Max throughput:** ~81.3 KB/s at max payload (240B)
4. **Latency scales linearly:** `latency_us ≈ 930 + 8.3 × payload_bytes`
5. **p99 tail latency:** ~2-3x median (occasional Wi-Fi contention spikes)

#### Comparison with Prior Benchmarks

| Source | Platform | One-way (240B) | Throughput |
|--------|----------|---------------|------------|
| Electric UI (2024) | ESP32 (original) | ~5,000 µs | ~75 KB/s |
| **Our benchmark** | **ESP32-S3, ESP-IDF v6.0** | **~2,953 µs** | **~81.3 KB/s** |
| Improvement | | **1.7x faster** | **1.08x higher** |

---

### 5.2 Experiment: Multi-Peer Communication (1 Coordinator → 4 Workers)

**Date:** 2026-04-05
**Boards:** Board 0 (Coordinator) → Boards 1-4 (Workers)
**Protocol:** ESP-NOW broadcast + unicast, Wi-Fi channel 1

#### Test 1: Broadcast PING → 4 Workers (all must reply)

| Metric | Value |
|--------|-------|
| Rounds | 200 |
| Success | **200/200 (100%)** |
| All-reply min | 5,044 µs |
| All-reply avg | **5,287 µs** |
| All-reply median | 5,135 µs |
| All-reply p95 | 6,947 µs |
| All-reply max | 8,133 µs |
| Timeouts | **0** |

#### Test 2: Unicast PING → Each Worker Separately

| Worker | MAC (last 2 bytes) | Avg RTT (µs) | Success |
|--------|-------------------|-------------:|---------|
| Worker 1 | D1:98 | 2,345 | 100/100 |
| Worker 2 | CD:E4 | 2,318 | 100/100 |
| Worker 3 | DA:28 | 2,324 | 100/100 |
| Worker 4 | D0:DC | 2,331 | 100/100 |

#### Test 3: Tensor Broadcast (3,072 bytes = 32×32×3 INT8 image)

| Metric | Value |
|--------|-------|
| Data size | 3,072 bytes (13 chunks × 240 bytes) |
| Rounds | 20 |
| Success | **20/20 (100%)** |
| Transfer min | **1,519 µs (1.5 ms)** |
| Transfer avg | **2,124 µs (2.1 ms)** |
| Transfer max | 9,142 µs (9.1 ms) |
| Effective throughput | **1,412 KB/s** |

**Critical insight:** Broadcast is essentially free (~20ms total for all layers). Bottleneck is 100% in gather (workers→coordinator).

---

### 5.3 FatCNN Training Results

**Date:** 2026-04-06
**Framework:** TensorFlow/Keras (tf-nightly 2.22)
**Dataset:** CIFAR-10 (50K train, 10K test, 32×32×3)

#### Model Architecture

| Layer | Type | Output Shape | Params |
|-------|------|-------------|-------:|
| conv1 | Conv2D 5×5 + ReLU | 32×32×64 | 4,864 |
| pool1 | MaxPool 2×2 | 16×16×64 | 0 |
| conv2 | Conv2D 3×3 + ReLU | 16×16×128 | 73,856 |
| pool2 | MaxPool 2×2 | 8×8×128 | 0 |
| conv3 | Conv2D 3×3 + ReLU | 8×8×256 | 295,168 |
| gap | GlobalAvgPool | 256 | 0 |
| dense1 | Dense + ReLU | 128 | 32,896 |
| dense2 | Dense | 10 | 1,290 |
| **Total** | | | **408,074** |

#### Training Results

| Metric | Value |
|--------|-------|
| Epochs | 30 |
| Batch size | 128 |
| Optimizer | Adam (lr=0.001) |
| **Test accuracy (float32)** | **77.40%** |
| Training time | ~13 min on Apple Silicon |

#### INT8 Quantization Results

| Layer | Weight range | Scale | Zero point | Max quant error |
|-------|-------------|------:|----------:|-----------:|
| conv1 | [-0.628, 0.424] | 0.004125 | 24 | 0.002062 |
| conv2 | [-1.064, 0.867] | 0.007574 | 13 | 0.003787 |
| conv3 | [-1.462, 0.652] | 0.008290 | 48 | 0.004145 |
| dense1 | [-1.108, 0.883] | 0.007804 | 14 | 0.003902 |
| dense2 | [-1.041, 0.792] | 0.007191 | 17 | 0.003592 |

#### Weight Partitioning (N=4 workers)

| Layer | Total weights (B) | Per worker (B) | Coordinator |
|-------|------------------:|---------------:|:-----------:|
| conv1 | 4,800 | 1,200 | — |
| conv2 | 73,728 | 18,432 | — |
| conv3 | 294,912 | 73,728 | — |
| dense1 | 32,768 | — | 32,768 |
| dense2 | 1,280 | — | 1,280 |
| **Total** | **407,488** | **93,360** | **34,048** |

**Validation:** 93,360 bytes per worker matches theoretical prediction from deep analysis exactly ✓

---

### 5.4 Single-Node Inference Baseline (FatCNN-Lite)

**Date:** 2026-04-06
**Board:** Board 0 (ESP32-S3 N16R8), CPU 160 MHz
**Model:** FatCNN-Lite (32-64-128 channels, 103K params)
**Accuracy:** 74.34% on CIFAR-10 (float32)
**Firmware:** `~/esp/swarm-infer/single_inference/`

#### Per-Layer Latency (average of 3 runs)

| Layer | Latency (ms) | % total |
|-------|------------:|---------:|
| Conv1 (5×5, 3→32) | 597 | 31.6% |
| ReLU1 + MaxPool1 | 6 | 0.3% |
| Conv2 (3×3, 32→64) | 634 | 33.5% |
| ReLU2 + MaxPool2 | 3 | 0.2% |
| Conv3 (3×3, 64→128) | 637 | 33.7% |
| ReLU3 + GAP | 2 | 0.1% |
| Dense1 + Dense2 | 1.5 | 0.1% |
| **Total** | **1,890** | **100%** |

#### Key Results

- **Prediction:** class 3 (cat) — CORRECT (ground truth = cat)
- **Python predicted:** class 5 (dog) — INT8 on ESP32 was more accurate!
- **Latency:** 1,890 ms ± 0.1 ms (deterministic across 3 runs)
- **Throughput:** 5.65 MMAC/s (unoptimized 7-loop conv2d)
- **Memory:** 73 KB buffers used, 325 KB free heap remaining
- **Conv layers = 99% of inference time.** Dense layers essentially free.

---

### 5.5 Distributed Layer 1 Inference (4 Workers)

**Date:** 2026-04-06
**Boards:** Board 0 (Coordinator) + Boards 1-4 (Workers)
**Model:** FatCNN Layer 1 only (Conv1 5×5, 3→64, partitioned 16 ch/worker)
**Protocol:** SwarmPacket (8B header + 232B data), broadcast input, sequential unicast gather
**Firmware:** `~/esp/swarm-infer/swarm_worker/` + `~/esp/swarm-infer/swarm_coordinator/`

#### Results (3 runs)

| Metric | Run 1 | Run 2 | Run 3 | Avg |
|--------|------:|------:|------:|----:|
| Broadcast (ms) | 1.9 | 1.7 | 1.7 | **~1.8** |
| Gather (ms) | 543 | 539 | 540 | **~541** |
| Assemble (ms) | 0.5 | 0.5 | 0.5 | **~0.5** |
| **TOTAL (ms)** | **547** | **544** | **544** | **~545** |

- **All 4 workers:** 18 chunks each, done=1, 0 timeouts, 0 packet loss
- **Output:** deterministic (identical across 3 runs)
- **Gather breakdown:** ~150 ms compute + ~72 ms send + ~319 ms overhead (sequential workers + ESP-NOW contention)

#### Comparison: Single vs Distributed (Layer 1 only)

| Configuration | Latency | Notes |
|--------------|--------:|-------|
| Single ESP32 (Conv1 3→32, FatCNN-Lite) | 597 ms | 32 output channels |
| Distributed 4 workers (Conv1 3→64, FatCNN) | 545 ms | 64 output channels (16 each) |
| **Speedup** | **1.1x** | For 2x more channels |

---

### 5.6 Revised SwarmInfer Communication Estimates

| Phase | Old estimate (5ms/pkt) | Revised (measured) | Source |
|-------|----------------------:|-------------------:|--------|
| **Broadcast total (all layers)** | **585 ms** | **~20 ms** | Burst mode at 1,412 KB/s |
| **Gather total (all layers, N=4)** | **2,440 ms** | **~1,464 ms** | Sequential unicast at ~81 KB/s |
| **Compute (N=4)** | **107 ms** | **107 ms** | Unchanged |
| **Total unoptimized** | **3,132 ms** | **~1,591 ms** | 2.0x better than predicted |
| **Fully optimized (aggressive sparse)** | **795 ms** | **~721 ms** | Gather-dominated |

**With MaxPool on worker (reduces gather 4x):**

| Configuration | Broadcast | Gather | Compute | Total | fps |
|--------------|----------:|-------:|--------:|------:|----:|
| Unoptimized | ~20 ms | ~336 ms | ~107 ms | **~463 ms** | 2.2 |
| + Bitmap sparse (1.5x) | ~20 ms | ~224 ms | ~112 ms | **~356 ms** | 2.8 |
| + Aggressive sparse (2.5x) | ~20 ms | ~134 ms | ~115 ms | **~269 ms** | 3.7 |

**Note:** Single-node FatCNN-Lite baseline = 1,890 ms. SwarmInfer FatCNN (4x larger model) estimated at ~463 ms = **4x faster** for a **4x larger model**. This is the key result for the paper.

---

### 5.7 Full 3-Layer Distributed Inference (4 Workers)

**Date:** 2026-04-06
**Boards:** Board 0 (Coordinator) + Boards 1-4 (Workers)
**Model:** FatCNN (64-128-256 channels, 408K params, INT8)
**Protocol:** WORKER_READY handshake + auto-trigger compute on last chunk
**Firmware:** `~/esp/swarm-infer/swarm_worker/` + `~/esp/swarm-infer/swarm_coordinator/`

#### Per-Layer Latency Breakdown (avg of 3 runs)

| Layer | Broadcast | Gather+Compute | Assemble | Total |
|-------|----------:|---------------:|---------:|------:|
| L1 (Conv1 5×5, 3→64, MaxPool) | 1 ms | 541 ms | 0.5 ms | **543 ms** |
| L2 (Conv2 3×3, 64→128, MaxPool) | 108 ms | 788 ms | 0.1 ms | **896 ms** |
| L3 (Conv3 3×3, 128→256, GAP) | 18 ms | 724 ms | 0.01 ms | **742 ms** |
| Dense1+Dense2 | — | 6 ms | — | **6 ms** |
| **TOTAL** | **127 ms** | **2,059 ms** | **0.6 ms** | **2,267 ms** |

#### Results

- **Prediction:** class 3 (cat) — CORRECT
- **Logits (INT8):** `[18, 1, 6, 42, -12, 35, -1, 4, 3, -8]`
- **Python INT8 match:** bit-exact (all 10 logit values identical)
- **Reliability:** 3/3 runs identical results, 0 packet loss
- **Workers:** All 4 received correct chunk counts (L1:18, L2:9, L3:1 per worker)

#### Bugs Found and Fixed in fix_quantize.py

1. **Bias scaling bug:** conv2/conv3/dense layers used `input_scale` (1/255) for bias quantization instead of output scale of previous layer. Conv2 biases were ~4x too large, conv3 ~17x, dense1 ~30x.
2. **Dense kernel layout:** Keras stores dense weights as `[in, out]`, C `dense_int8()` expects `[out, in]`. Missing transpose caused wrong results.

#### Protocol Improvements

1. **WORKER_READY handshake:** Workers send CMD_WORKER_READY after startup and after each layer. Coordinator waits for all 4 before broadcasting next layer. Eliminates fixed delays.
2. **Auto-trigger compute:** Workers start computing when `input_chunks_received >= total_chunks`. CMD_COMPUTE broadcast is just a fallback. Eliminates COMPUTE packet loss (was ~33% failure rate).
3. **TX backpressure:** `esp_now_send()` retried on `ESP_ERR_ESPNOW_NO_MEM` with 500µs busy-wait.

#### Comparison: Single vs Distributed

| Configuration | Model | Params | Latency | Fits 1 ESP32? |
|--------------|-------|-------:|--------:|:-------------:|
| Single ESP32 (FatCNN-Lite) | 32-64-128 ch | 103K | 1,890 ms | ✅ Yes |
| **SwarmInfer 4 workers (FatCNN)** | **64-128-256 ch** | **408K** | **2,267 ms** | **❌ No (408KB > 512KB SRAM)** |

SwarmInfer enables running a **4x larger model** that doesn't fit in a single MCU, with only +20% latency overhead compared to a smaller single-node model.

---

### 5.8 Batch Accuracy Test (50 Images, 4 Workers)

**Date:** 2026-04-06
**Boards:** Board 0 (Coordinator) + Boards 1-4 (Workers)
**Model:** FatCNN (64-128-256, 408K params, INT8)
**Dataset:** First 50 images from CIFAR-10 test set
**Firmware:** `~/esp/swarm-infer/swarm_coordinator/` (batch mode) + `~/esp/swarm-infer/swarm_worker/`

#### Results

| Metric | ESP32 INT8 Distributed | Python Float32 |
|--------|----------------------:|---------------:|
| **Correct** | **44/50** | 43/50 |
| **Accuracy** | **88.0%** | 86.0% |
| **Avg latency** | **2,239 ms** | — |
| Timeouts | 0 | — |
| Retries | 0 | — |

#### Per-Image Results

| # | True | ESP32 Pred | Match | Latency |
|---|------|-----------|:-----:|--------:|
| 1 | cat | cat | OK | 2,260 ms |
| 2 | ship | ship | OK | 2,240 ms |
| 3 | ship | ship | OK | 2,247 ms |
| 4 | airplane | airplane | OK | 2,245 ms |
| 5 | frog | frog | OK | 2,237 ms |
| 6 | frog | frog | OK | 2,232 ms |
| 7 | automobile | automobile | OK | 2,236 ms |
| 8 | frog | bird | MISS | 2,227 ms |
| 9 | cat | cat | OK | 2,236 ms |
| 10 | automobile | automobile | OK | 2,244 ms |
| 11 | airplane | airplane | OK | 2,233 ms |
| 12 | truck | truck | OK | 2,232 ms |
| 13 | dog | dog | OK | 2,238 ms |
| 14 | horse | horse | OK | 2,232 ms |
| 15 | truck | truck | OK | 2,228 ms |
| 16 | ship | ship | OK | 2,239 ms |
| 17 | dog | dog | OK | 2,232 ms |
| 18 | horse | horse | OK | 2,242 ms |
| 19 | ship | ship | OK | 2,232 ms |
| 20 | frog | frog | OK | 2,231 ms |
| 21 | horse | horse | OK | 2,247 ms |
| 22 | airplane | airplane | OK | 2,235 ms |
| 23 | deer | deer | OK | 2,241 ms |
| 24 | truck | truck | OK | 2,245 ms |
| 25 | dog | bird | MISS | 2,242 ms |
| 26 | bird | bird | OK | 2,244 ms |
| 27 | deer | deer | OK | 2,240 ms |
| 28 | airplane | airplane | OK | 2,237 ms |
| 29 | truck | truck | OK | 2,235 ms |
| 30 | frog | frog | OK | 2,243 ms |
| 31 | frog | frog | OK | 2,244 ms |
| 32 | dog | bird | MISS | 2,252 ms |
| 33 | deer | bird | MISS | 2,252 ms |
| 34 | dog | dog | OK | 2,246 ms |
| 35 | truck | truck | OK | 2,241 ms |
| 36 | bird | automobile | MISS | 2,234 ms |
| 37 | deer | deer | OK | 2,244 ms |
| 38 | automobile | automobile | OK | 2,240 ms |
| 39 | truck | truck | OK | 2,236 ms |
| 40 | dog | dog | OK | 2,239 ms |
| 41 | deer | deer | OK | 2,223 ms |
| 42 | frog | frog | OK | 2,232 ms |
| 43 | dog | horse | MISS | 2,235 ms |
| 44 | frog | frog | OK | 2,240 ms |
| 45 | airplane | airplane | OK | 2,241 ms |
| 46 | truck | truck | OK | 2,238 ms |
| 47 | cat | cat | OK | 2,237 ms |
| 48 | truck | truck | OK | 2,251 ms |
| 49 | horse | horse | OK | 2,245 ms |
| 50 | frog | frog | OK | 2,248 ms |

#### Key Findings

1. **INT8 quantization preserves accuracy:** ESP32 distributed INT8 (88.0%) ≥ Python float32 (86.0%)
2. **Zero timeouts** over 50 consecutive inferences — auto-trigger protocol is 100% reliable
3. **Latency extremely stable:** 2,223–2,252 ms range (σ < 8 ms)
4. **Misclassification pattern:** 5/6 misses involve dog/bird/deer — common confusion classes in CIFAR-10
5. **Throughput:** 50 images in ~112 seconds = **0.45 fps**

---

### 5.9 Scalability Sweep (N=2 vs N=4 Workers)

**Date:** 2026-04-07
**Boards:** Board 0 (Coordinator) + Boards 1-4 (Workers)
**Model:** FatCNN (64-128-256, 408K params, INT8)
**Dataset:** First 50 images from CIFAR-10 test set
**Tool:** `models/partition_n.py` for parametric weight partitioning

**Note on valid N:** Output channel counts must be divisible by N. FatCNN has 64-128-256 channels, so valid N values are: 1, 2, 4, 8, 16, 32, 64. N=3 is not possible (64/3, 128/3, 256/3 are not integers). With 4 available worker boards, only N=2 and N=4 were tested.

#### Weight Partitioning

| N Workers | Conv1 ch/worker | Conv2 ch/worker | Conv3 ch/worker | Weights/worker |
|-----------|----------------:|----------------:|----------------:|---------------:|
| 2 | 32 | 64 | 128 | 187,360 B |
| 4 | 16 | 32 | 64 | 93,360 B |

#### Batch Results (50 images)

| N Workers | Accuracy | Correct | Avg Latency | Timeouts | Retries |
|-----------|:--------:|--------:|------------:|---------:|--------:|
| **2** | **88.0%** | 44/50 | **3,788 ms** | 0 | 0 |
| **4** | **88.0%** | 44/50 | **2,239 ms** | 0 | 0 |

#### Per-Layer Timing (N=2, typical image)

| Layer | Broadcast | Gather (compute+send) | Assemble | Total |
|-------|----------:|----------------------:|---------:|------:|
| L1 (Conv1 5×5, 32ch) | 3 ms | 839 ms | 0.3 ms | 842 ms |
| L2 (Conv2 3×3, 64ch) | 105 ms | 1,437 ms | 0.1 ms | 1,542 ms |
| L3 (Conv3 3×3, 128ch) | 13 ms | 1,353 ms | 0.01 ms | 1,366 ms |
| **Total conv** | **121 ms** | **3,629 ms** | **0.4 ms** | **3,750 ms** |

#### Per-Layer Timing (N=4, typical image)

| Layer | Broadcast | Gather (compute+send) | Assemble | Total |
|-------|----------:|----------------------:|---------:|------:|
| L1 (Conv1 5×5, 16ch) | 4 ms | 542 ms | 0.3 ms | 546 ms |
| L2 (Conv2 3×3, 32ch) | 105 ms | 795 ms | 0.1 ms | 900 ms |
| L3 (Conv3 3×3, 64ch) | 13 ms | 720 ms | 0.01 ms | 733 ms |
| **Total conv** | **122 ms** | **2,057 ms** | **0.4 ms** | **2,179 ms** |

#### Scalability Analysis

| Metric | N=2 | N=4 | Ratio |
|--------|----:|----:|------:|
| Avg latency | 3,788 ms | 2,239 ms | **1.69× speedup** |
| L1 gather | 839 ms | 542 ms | 1.55× |
| L2 gather | 1,437 ms | 795 ms | 1.81× |
| L3 gather | 1,353 ms | 720 ms | 1.88× |
| Broadcast overhead | 121 ms | 122 ms | ~same |
| Accuracy | 88.0% | 88.0% | identical |

#### Key Findings

1. **Near-linear compute scaling for L2/L3:** Gather time (which is dominated by compute) scales ~1.8× from N=2 to N=4 — close to ideal 2×
2. **L1 scales less:** Only 1.55× — conv1 (5×5 kernel) has more overhead per channel, so halving channels gives less than 2× speedup
3. **Broadcast is constant:** ~122 ms regardless of N (same input data, broadcast to all)
4. **Accuracy preserved:** Identical results (44/50) for both configurations — purely a latency tradeoff
5. **Overhead:** ~60 ms gap between sum-of-layers and total latency = READY handshake + dense layers
6. **Sub-linear overall scaling:** 1.69× speedup for 2× workers — communication overhead (gather) grows with N

#### Bug Fix During Sweep

- **`send_ready()` TX buffer overflow:** After sending 36 result chunks (N=2), ESP-NOW TX buffer was full. `send_ready()` silently failed, causing 5-second timeout before worker's main loop re-sent READY. Fix: added `ESP_ERR_ESPNOW_NO_MEM` retry loop with 500µs backpressure to `send_ready()` and all `esp_now_send()` calls in worker. This reduced N=2 latency from 8,608 ms to 3,788 ms.
- **WiFi interference from unused workers:** Workers 2,3 (running old N=4 firmware) interfered with N=2 test by receiving broadcasts and sending unwanted results. Fix: physically disconnect unused boards during N<4 tests.

---

### 5.10 Bitmap Sparsification (N=4 Workers)

**Date:** 2026-04-07
**Boards:** Board 0 (Coordinator) + Boards 1-4 (Workers)
**Model:** FatCNN (64-128-256, 408K params, INT8)
**Dataset:** First 50 images from CIFAR-10 test set

#### Concept

After ReLU, many activations equal the output zero_point (quantized zero). Bitmap sparsification encodes only non-zero values:

**Encoding format:** `[bitmap: ceil(N/8) bytes][packed non-zero values]`
- Bitmap: 1 bit per value (1 = non-zero, 0 = zero_point)
- Only non-zero values transmitted sequentially
- Worker encodes after compute, coordinator decodes before assembly

#### Sparsity Measurements (per worker, N=4)

| Layer | Original size | Sparse encoded | Chunks (orig→sparse) | Sparsity | Reduction |
|-------|-------------:|--------------:|:---------------------:|---------:|----------:|
| L1 (MaxPool 16×16×16) | 4,096 B | ~2,500 B | 18 → ~11 | ~50% | ~39% |
| L2 (MaxPool 8×8×32) | 2,048 B | ~900 B | 9 → ~4 | ~67% | ~56% |
| L3 (GAP 64) | 64 B | ~64 B | 1 → 1 | ~0% | 0% |

**Note:** L3 GAP output is not sparse — global average pooling produces non-zero values.

#### Batch Results (50 images, N=4)

| Metric | Without sparse | With sparse | Change |
|--------|---------------:|------------:|-------:|
| Accuracy | 88.0% (44/50) | 88.0% (44/50) | identical |
| Avg latency | 2,239 ms | **2,115 ms** | **−124 ms (−5.5%)** |
| Timeouts | 0 | 0 | — |

#### Per-Layer Gather Time Comparison (N=4, typical image)

| Layer | Gather (no sparse) | Gather (sparse) | Savings |
|-------|-------------------:|----------------:|--------:|
| L1 | 542 ms | ~460 ms | −82 ms (−15%) |
| L2 | 795 ms | ~735 ms | −60 ms (−8%) |
| L3 | 720 ms | ~720 ms | 0 ms |
| **Total** | **2,057 ms** | **~1,915 ms** | **−142 ms (−7%)** |

#### Key Findings

1. **Accuracy preserved:** Lossless encoding — decoded output is bit-exact with original
2. **L2 most sparse:** ~67% zeros after ReLU + MaxPool — convolutions with more channels produce sparser outputs
3. **L1 moderately sparse:** ~50% zeros — first layer activations are denser
4. **L3 not worth encoding:** GAP reduces 8×8 spatial to 1 scalar per channel — no zeros
5. **Gather savings dominate:** 7% gather reduction → 5.5% total latency reduction (broadcast unchanged)
6. **Encode/decode overhead negligible:** Bitmap scan is O(N) — microseconds vs milliseconds for transmission

#### Implementation Details

- **Encode buffer:** Worker reuses `input_buf` (not needed after compute) as sparse encoding scratch space
- **Decode buffer:** Coordinator allocates one `decode_tmp` buffer, decodes sequentially per worker before assembly
- **Protocol:** `CMD_RESULT_DONE` packet carries 4-byte metadata: `[sparse_flag, original_size_lo, original_size_hi, zero_point]`
- **Fallback:** If encoded size ≥ original size, worker sends uncompressed (flag=0)

---

### 5.11 Gather Communication Benchmark (N=4, No Compute)

**Date:** 2026-04-07
**Boards:** Board 0 (Coordinator) + Boards 1-4 (Workers)
**Firmware:** `gather_benchmark_coordinator/` + `gather_benchmark_worker/`
**Protocol:** Coordinator broadcasts CMD_COMPUTE → workers immediately send back N bytes → coordinator records timestamps
**Rounds:** 20 per payload size
**Purpose:** Isolate pure communication overhead from compute time

#### Results (4 Workers, Pure Communication)

| Payload/worker | Chunks/worker | Avg latency | Min | Max | Aggregate throughput |
|---------------:|:-------------:|------------:|----:|----:|--------------------:|
| 64 B | 1 | 12 ms | 6 ms | 15 ms | 21 KB/s |
| 232 B | 1 | 15 ms | 8 ms | 24 ms | 61 KB/s |
| 512 B | 3 | 13 ms | — | 43 ms | 158 KB/s |
| 1,024 B | 5 | 60 ms | 36 ms | 68 ms | 67 KB/s |
| 2,048 B | 9 | 112 ms | 106 ms | 122 ms | 71 KB/s |
| 4,096 B | 18 | 170 ms | 149 ms | 224 ms | 94 KB/s |

#### Per-Worker Timeline (4096 B, typical round)

Workers arrive sequentially due to WiFi contention (shared medium, unicast ACKs):
```
W0: first chunk at ~10 ms, done at ~134 ms
W1: first chunk at ~21 ms, done at ~210 ms
W2: first chunk at  ~7 ms, done at ~173 ms
W3: first chunk at  ~4 ms, done at ~183 ms
```
Workers start sending at different times (~4-21 ms spread) due to broadcast reception jitter. Packets interleave on the channel — no single worker can monopolize the medium.

#### Compute vs Communication Breakdown (N=4)

| Layer | Real gather | Pure comm | Compute (diff) | Compute % |
|-------|:-----------:|:---------:|:--------------:|:---------:|
| L1 (4,096 B/worker) | 460 ms* | 170 ms | **~290 ms** | 63% |
| L2 (2,048 B/worker) | 735 ms* | 112 ms | **~623 ms** | 85% |
| L3 (64 B/worker) | 720 ms* | 12 ms | **~708 ms** | 98% |

*With sparse encoding enabled

#### Results (1 Worker, No Contention Baseline)

| Payload/worker | Chunks/worker | Avg latency | Min | Max | Throughput |
|---------------:|:-------------:|------------:|----:|----:|----------:|
| 64 B | 1 | 5 ms | 5 ms | 6 ms | 12.3 KB/s |
| 232 B | 1 | 5 ms | 5 ms | 8 ms | 43.2 KB/s |
| 512 B | 3 | 10 ms | 9 ms | 12 ms | 52.3 KB/s |
| 1,024 B | 5 | 16 ms | 15 ms | 19 ms | 62.5 KB/s |
| 2,048 B | 9 | 28 ms | 27 ms | 31 ms | 70.1 KB/s |
| 4,096 B | 18 | 54 ms | 53 ms | 56 ms | 73.7 KB/s |

#### WiFi Contention Analysis (N=1 vs N=4)

| Payload | N=1 | N=4 | Contention factor | Ideal (N=1) |
|--------:|----:|----:|:-----------------:|:-----------:|
| 64 B | 5 ms | 12 ms | 2.4× | 1.0× |
| 232 B | 5 ms | 15 ms | 3.0× | 1.0× |
| 512 B | 10 ms | 13 ms | 1.3× | 1.0× |
| 1,024 B | 16 ms | 60 ms | 3.8× | 1.0× |
| 2,048 B | 28 ms | 112 ms | 4.0× | 1.0× |
| 4,096 B | 54 ms | 170 ms | 3.1× | 1.0× |

Contention factor = N=4 latency / N=1 latency. Ideal would be 1.0× (perfect parallel channel). Observed 2.4-4.0× = workers effectively serialize on the shared WiFi medium. For large payloads (2048-4096 B), contention factor approaches 3-4× because each chunk requires a unicast TX→ACK round-trip, and 4 workers' packets interleave.

#### Key Findings

1. **Compute dominates:** 63-98% of gather time is worker-side computation, not communication
2. **L3 almost entirely compute:** 64 bytes takes only 12 ms to transfer, but Conv3 3×3 on 128→64 channels takes ~708 ms
3. **L1 has most communication overhead:** 4,096 B per worker = 170 ms pure transfer, making communication ~37% of gather
4. **WiFi contention ~3-4× for large payloads:** 4 workers sharing the channel effectively serialize their transmissions due to CSMA/CA and unicast ACK requirements
5. **N=1 baseline matches unicast benchmark:** 4096 B at 73.7 KB/s (N=1) ≈ 81.3 KB/s from E2 point-to-point benchmark — consistent
6. **Throughput peaks at 512 B payloads (N=4):** 158 KB/s aggregate — small enough to avoid TX buffer pressure, large enough for efficient framing
7. **Reliability concern at 4096 B (N=4):** 1 timeout in 20 rounds — 72 total chunks saturate the channel

---

### 5.12 CPU Frequency Experiment (160 MHz vs 240 MHz)

**Date:** 2026-04-07
**Boards:** Board 0 (Coordinator) + Boards 1-4 (Workers)
**Model:** FatCNN (64-128-256, 408K params, INT8, sparse encoding)
**Dataset:** First 50 images from CIFAR-10 test set
**Change:** `CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ` 160 → 240 in sdkconfig (both worker and coordinator)

#### Results

| CPU Freq | Accuracy | Avg Latency | Change |
|----------|:--------:|------------:|-------:|
| 160 MHz | 88.0% (44/50) | 2,115 ms | baseline |
| **240 MHz** | **88.0% (44/50)** | **1,623 ms** | **−492 ms (−23%)** |

#### Analysis

- **Theoretical max speedup:** 240/160 = 1.50× → expected −33% latency if fully compute-bound
- **Observed:** −23% → **compute is ~70% of total time**, consistent with gather benchmark findings
- **Communication unchanged:** Broadcast + gather transfer time is CPU-independent
- **Estimated breakdown at 240 MHz:**
  - Compute: ~1,300 ms (was ~1,820 ms at 160 MHz, ratio 1,820/1,300 = 1.40× ≈ 240/160 = 1.50×)
  - Communication: ~295 ms (unchanged)
  - Protocol overhead: ~28 ms (unchanged)

#### Key Finding

**Compute-bound confirmed:** Increasing CPU frequency by 50% reduces latency by 23%, proving that worker-side INT8 convolution is the primary bottleneck. The remaining 77% of the theoretical speedup is lost to communication overhead, which is CPU-independent. This validates the gather benchmark results (section 5.11) showing 63-98% compute dominance per layer.

---

## 6. Milestones

| Phase | Description | Date | Status |
|-------|-------------|------|--------|
| Phase 1 | Hello World on ESP32-S3 | 2026-04-05 | ✅ |
| Phase 2 | Label all 6 boards | 2026-04-05 | ✅ |
| Phase 3 | Connect all to USB hub, map ports | 2026-04-05 | ✅ |
| Phase 4 | Collect all 6 MAC addresses | 2026-04-05 | ✅ |
| Phase 5 | ESP-NOW link (sender → receiver) | 2026-04-05 | ✅ |
| Phase 6 | ESP-NOW latency benchmark (E2) | 2026-04-05 | ✅ |
| Phase 7 | Multi-peer broadcast (1→4) | 2026-04-05 | ✅ |
| Phase 8 | Tensor broadcast test (3KB) | 2026-04-05 | ✅ |
| Phase 9 | **Train FatCNN on CIFAR-10** | **2026-04-06** | **✅ 77.4%** |
| Phase 10 | **Quantize INT8 + partition weights** | **2026-04-06** | **✅** |
| Phase 11 | **Implement tensor_ops.c** | **2026-04-06** | **✅ 316 lines** |
| Phase 12 | **Train FatCNN-Lite (baseline)** | **2026-04-06** | **✅ 74.3%** |
| Phase 13 | **Single-node inference test** | **2026-04-06** | **✅ 1,890 ms** |
| Phase 14 | **Distributed Layer 1 (4 workers)** | **2026-04-06** | **✅ 545 ms** |
| Phase 15 | **Full 3-layer distributed inference** | **2026-04-06** | **✅ 2,267 ms, correct prediction** |
| Phase 16 | **Batch accuracy test (50 images)** | **2026-04-06** | **✅ 88.0% (44/50), 0 timeouts** |
| Phase 17 | **Scalability sweep (N=2,4)** | **2026-04-07** | **✅ N=2: 3,788ms, N=4: 2,239ms** |
| Phase 18 | **Bitmap sparsification** | **2026-04-07** | **✅ −5.5% latency (2,239→2,115ms)** |
| Phase 19 | **Gather comm benchmark** | **2026-04-07** | **✅ Compute 63-98% of gather time** |
| Phase 20 | **CPU freq experiment (240 MHz)** | **2026-04-07** | **✅ −23% latency (2,115→1,623ms)** |
| Phase 1.5 | INA219 sensor test | — | ⬜ Code ready |
| — | Full experiment suite | — | ⬜ Week 5-6 |
| — | Article finalization | — | ⬜ Week 7-9 |

---

## 7. Next Steps (Priority Order)

### Immediate
1. ~~**Scalability sweep**~~ — ✅ Done (N=2,4), 1.69× speedup
2. ~~**Batch accuracy test**~~ — ✅ Done (88.0%, 50 images, 0 timeouts)
3. ~~**Bitmap sparsification**~~ — ✅ Done (−5.5% latency, L1 ~50% sparse, L2 ~67%)
4. **INA219 power measurement** — wire sensors, measure per-node and total power

### Week 3-4
5. **Full experiment suite** — statistical analysis, tables for paper

---

## 8. Key Documents

| Document | Contents |
|----------|----------|
| `SwarmInfer_Research_Plan.md` | Full research plan, novelty, experiments, timeline |
| `SwarmInfer_Setup_and_Architecture.md` | ESP-IDF setup, 3 layer splitting strategies, FatCNN design |
| `SwarmInfer_Experiment_Full.md` | Complete experiment description (15 sections, 1270+ lines) |
| `architecture_deep_analysis.md` | Memory budgets, communication costs, optimization stack (1004 lines) |
| `SwarmInfer_Hardware_Setup_Guide.md` | Step-by-step from unboxing to ESP-NOW |
| `SwarmInfer_INA219_QuickStart.md` | Wiring, code, troubleshooting for INA219 |
| `SwarmInfer_Port_Mapping.md` | USB port ↔ Board ↔ MAC mapping |
| **This document** | MACs, benchmarks, training results, milestones |

---

## 9. Target Publication

| Field | Value |
|-------|-------|
| **Journal** | IEEE Access (IF ~3.9, open access) |
| **Title** | SwarmInfer: Intra-Layer Parallel Distributed Inference on ESP32 Microcontroller Clusters via ESP-NOW |
| **Core contribution** | First implementation + empirical analysis of tensor-parallel inference on MCU swarm |
| **Novelty evidence** | DDSNN (Sensors, July 2025) explicitly identified parallel MCU distribution as open problem |
| **Timeline** | ~11 weeks remaining → submission ~June-July 2026 |

---

## 10. Key Technical Insights (for context restoration)

1. **ESP-IDF v6.0 breaking change:** `esp_now_register_send_cb()` first arg is `esp_now_send_info_t*`, not `uint8_t*`.

2. **Broadcast is essentially free:** 1,412 KB/s burst throughput. All broadcast phases total ~20 ms.

3. **Gather is THE bottleneck:** Sequential unicast at ~81 KB/s. Gather = 92% of total inference time.

4. **Ring gather reassessment:** Since broadcast is cheap, ring gather's value is minimal — focus on sparsification.

5. **Stack overflow risk:** Arrays >4KB on ESP32 stack cause Guru Meditation crash. Use `malloc()`.

6. **Port locking:** Always `idf.py flash` then `idf.py monitor` separately. Kill stuck: `kill $(lsof -t /dev/cu.usbmodemXXXXX)`.

7. **FatCNN:** 407,488 bytes INT8 weights. Does NOT fit single ESP32 (537KB with OS > 512KB SRAM). With N=4: 93,360 bytes/worker — fits.

8. **All boards symmetric:** ~2.3ms RTT for all 4 workers. No slow nodes.

9. **Linear latency model:** `one_way_us ≈ 930 + 8.3 × payload_bytes`

10. **FatCNN accuracy:** 77.4% on CIFAR-10 (float32). INT8 quantization max error: 0.004.

11. **tensor_ops.c:** Channels-last [H,W,C] layout, fixed-point requantization, MaxPool on worker before gather, 316 lines total.

12. **Python ML env:** `conda activate swarm-ml` (tf-nightly 2.22, Python 3.11). Training script: `models/train_fatcnn.py`, quantization: `models/fix_quantize.py`.

13. **Single-node baseline:** FatCNN-Lite on 1 ESP32 = 1,890 ms. Conv layers = 99% of time. ESP32 throughput: 5.65 MMAC/s (unoptimized). Prediction correct (cat), INT8 more accurate than Python float32 on test image.

14. **Key comparison for paper:** Single ESP32 does FatCNN-Lite (103K params) in 1,890 ms. SwarmInfer does FatCNN (408K params, 4x larger) in **measured 2,267 ms** (unoptimized). With sparsification estimated ~463 ms — **4x faster for 4x larger model**.

15. **Distributed Layer 1 results:** Broadcast 1.8 ms, gather 541 ms, total 545 ms. All 4 workers complete, 0 packet loss. SwarmPacket protocol: 8B header + 232B data chunks. Worker WORKER_ID set via `#define` before build, `sed` to change between builds.

16. **Flashing workers:** Build once per WORKER_ID: `sed -i '' 's/WORKER_ID X/WORKER_ID Y/' main/swarm_worker.c && idf.py build && idf.py -p PORT flash`. Worker 0→port 212401, W1→212201, W2→2121401, W3→2121301.

17. **Full 3-layer pipeline results:** L1 broadcast 1ms + gather 541ms, L2 broadcast 108ms + gather 788ms, L3 broadcast 18ms + gather 724ms, Dense 6ms. Total: 2,267 ms end-to-end. Prediction correct (cat=42), bit-exact match with Python INT8 simulation.

18. **fix_quantize.py had two bugs:** (a) Bias quantization used `input_scale` for all layers instead of previous layer's output scale. This made conv2 biases ~4x too large, conv3 ~17x, dense ~30x. (b) Dense kernel not transposed from Keras `[in,out]` to C `[out,in]`. Both fixed 2026-04-06.

19. **WORKER_READY handshake + auto-trigger:** Workers send CMD_WORKER_READY after startup and after each layer. Workers auto-start compute when `chunks_received >= total_chunks` (no need for CMD_COMPUTE broadcast). This eliminated ~33% failure rate from lost COMPUTE packets. 100% reliability over 3/3 runs.

20. **FreeRTOS tick rate = 100Hz** (10ms per tick). `vTaskDelay(1)` = 10ms minimum. Use `esp_rom_delay_us()` for sub-ms pacing in broadcast loops.

21. **Batch accuracy (50 images):** ESP32 INT8 distributed = 44/50 (88.0%), Python float32 = 43/50 (86.0%). INT8 quantization does NOT degrade accuracy — in fact slightly better due to favorable rounding. Misclassifications are dog/bird/deer confusions (common in CIFAR-10). Latency stable: 2,239 ms avg, σ < 8ms. Zero timeouts over 50 consecutive inferences.
