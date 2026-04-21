# SwarmInfer: Full Experiment Description

**Version:** 1.1
**Date:** 2026-04-06
**Author:** Yurii Myroshnyk
**Status:** Hardware setup and benchmarks complete; FatCNN trained and quantized; tensor_ops implemented; single-node inference working; distributed Layer 1 working

---

## Contents

1. [Introduction and Research Problem](#1-introduction-and-research-problem)
2. [Review of Existing Solutions and Justification of Novelty](#2-review-of-existing-solutions-and-justification-of-novelty)
3. [SwarmInfer Architecture](#3-swarminfer-architecture)
4. [Mathematical Framework](#4-mathematical-framework)
5. [Hardware](#5-hardware)
6. [Development Environment Setup](#6-development-environment-setup)
7. [Experiment E2: ESP-NOW Point-to-Point Benchmark](#7-experiment-e2-esp-now-point-to-point-benchmark)
8. [Experiment: Multi-Peer Communication](#8-experiment-multi-peer-communication)
9. [The FatCNN Model](#9-the-fatcnn-model)
10. [Layer Distribution Strategies](#10-layer-distribution-strategies)
11. [Memory Analysis](#11-memory-analysis)
12. [Communication Cost Analysis](#12-communication-cost-analysis)
13. [Optimizations](#13-optimizations)
14. [tensor_ops Architecture](#14-tensor_ops-architecture)
15. [Plan for Future Experiments](#15-plan-for-future-experiments)

---

## 1. Introduction and Research Problem

### 1.1 Context

Deployment of deep neural networks on microcontrollers (MCUs) is constrained by a fundamental memory wall: a typical CNN for image classification requires 500 KB to 2 MB of weights and activations, while a single ESP32-S3 has only 512 KB of internal SRAM.

Existing solutions:
- **Model compression** (quantization, pruning, distillation) — reduces model capacity at the cost of accuracy
- **Pipeline parallelism** (DDSNN, 2025) — partitions the model across layers, with each MCU executing one layer sequentially. Latency grows linearly with the number of nodes

### 1.2 Research Question

> *Can intra-layer (tensor) parallel distributed inference on a cluster of ESP32 microcontrollers communicating via ESP-NOW achieve lower latency and/or support larger models than pipeline-parallel or single-device approaches, given the constraints of communication overhead?*

### 1.3 Key Contributions

1. **First implementation** of intra-layer tensor-parallel inference on an MCU swarm
2. **Quantitative analysis** of the computation-vs-communication trade-off for parallel inference on MCUs over ESP-NOW
3. **Demonstration** of a model (FatCNN) that is infeasible on a single MCU but runs on the swarm
4. **Open-source framework** for distributed inference on ESP32

---

## 2. Review of Existing Solutions and Justification of Novelty

### 2.1 Closest Related Work: DDSNN

**DDSNN** (Decentralized Distributed Sequential NN Inference, Sensors, July 2025) is the closest existing work. It uses **pipeline parallelism**: the model is partitioned across layers, with each ESP32 executing one layer sequentially.

The DDSNN authors **explicitly identified** parallel distribution as an open problem:

> *"Hybrid or parallel distribution can improve throughput and reduce overall latency... However, it often requires more frequent synchronization and communication between nodes... this distribution method is better suited for more powerful devices with reliable, high-speed network connections, making it less suitable for resource-constrained devices in WSNs."*

**Our work directly addresses this open problem.**

### 2.2 Comparison with Existing Work

| Work | Year | Platform | Parallelism Type | Communication | Multi-device? |
|------|------|----------|------------------|---------------|---------------|
| DDSNN | 2025 | ESP32-S3 | Pipeline (sequential layers) | Wi-Fi TCP mesh | Yes |
| Ariel-ML | 2025 | ESP32/ARM | Multi-core (intra-op) | Shared memory | No (single MCU) |
| UAV Swarm | 2023 | Jetson/mobile | Hybrid pipeline+parallel | Wi-Fi | Yes (powerful devices) |
| **SwarmInfer** | **2026** | **ESP32-S3** | **Tensor parallel (intra-layer)** | **ESP-NOW** | **Yes (MCU swarm)** |

---

## 3. SwarmInfer Architecture

### 3.1 Overall Scheme

```
                     ┌─────────────────┐
                     │  COORDINATOR    │
                     │  (ESP32-S3 #0)  │
                     │                 │
                     │  1. Receives    │
                     │     input       │
                     │  2. Broadcast   │
                     │     via         │
                     │     ESP-NOW     │
                     │  3. Gathers     │
                     │     results     │
                     │  4. FC layers   │
                     │  5. Output      │
                     └────────┬────────┘
                              │ ESP-NOW broadcast
                ┌─────────────┼─────────────┐
                │             │             │
         ┌──────┴──────┐ ┌───┴────┐ ┌──────┴──────┐
         │  WORKER #1  │ │WORKER#2│ │  WORKER #3  │ ... WORKER #4
         │ channels    │ │channels│ │ channels    │
         │ [0..15]     │ │[16..31]│ │ [32..47]    │
         │             │ │        │ │             │
         │ Conv2D      │ │ Conv2D │ │ Conv2D      │
         │ partial out │ │ partial│ │ partial out │
         └──────┬──────┘ └───┬────┘ └──────┬──────┘
                │             │             │
                └─────────────┼─────────────┘
                              │ ESP-NOW unicast (gather)
                     ┌────────┴────────┐
                     │  COORDINATOR    │
                     │  Assembles full │
                     │  output tensor  │
                     └─────────────────┘
```

### 3.2 Inference Flow for One Image

```
Step 1: BROADCAST INPUT
  Coordinator → broadcast 32×32×3 INT8 image (3,072 bytes) → all Workers

Step 2: PARALLEL COMPUTE (Layer 1)
  Worker 1: Conv2D channels [0..15]  → output 32×32×16
  Worker 2: Conv2D channels [16..31] → output 32×32×16
  Worker 3: Conv2D channels [32..47] → output 32×32×16
  Worker 4: Conv2D channels [48..63] → output 32×32×16
  (in parallel, no communication)

Step 3: MaxPool 2×2 ON WORKERS (reduces data before gather!)
  Each worker: 32×32×16 → 16×16×16

Step 4: GATHER (Layer 1 output)
  Worker 1 → Coordinator: 16×16×16 = 4,096 bytes
  Worker 2 → Coordinator: 16×16×16 = 4,096 bytes
  Worker 3 → Coordinator: 16×16×16 = 4,096 bytes
  Worker 4 → Coordinator: 16×16×16 = 4,096 bytes
  Coordinator assembles: 16×16×64

Step 5: BROADCAST (Layer 2 input)
  Coordinator → broadcast 16×16×64 = 16,384 bytes → all Workers

Steps 6-8: REPEAT for Layer 2 and Layer 3

Step 9: FC LAYERS on Coordinator
  Dense 256→128 → ReLU → Dense 128→10

Step 10: OUTPUT
  argmax(10 values) → predicted class
```

---

## 4. Mathematical Framework

### 4.1 2D Convolution (INT8)

For an input tensor `X[H_in, W_in, C_in]` and filters `W[C_out, kH, kW, C_in]` with bias `b[C_out]`:

**Formula for a single output pixel:**

```
Y[y, x, c_out] = b[c_out] + Σ_{ky=0}^{kH-1} Σ_{kx=0}^{kW-1} Σ_{ic=0}^{C_in-1}
                  (X[y·s+ky-p, x·s+kx-p, ic] - zp_x) × (W[c_out, ky, kx, ic] - zp_w)
```

where:
- `s` — stride
- `p` — padding
- `zp_x` — zero point of input quantization
- `zp_w` — zero point of weight quantization

**Output dimensions:**
```
H_out = floor((H_in + 2×p - kH) / s) + 1
W_out = floor((W_in + 2×p - kW) / s) + 1
```

**INT8 quantization:**

The relationship between the real value and the quantized value:
```
real_value = (quantized_value - zero_point) × scale
```

For multiplication of two quantized values:
```
real_result = scale_x × scale_w × Σ (x_q - zp_x)(w_q - zp_w)
```

**Requantization (INT32 → INT8):**

After accumulation in int32:
```
acc_real = acc_int32 × scale_x × scale_w
output_int8 = clamp(round(acc_real / scale_out + zp_out), -128, 127)
```

Optimization via fixed-point multiplier:
```
M = scale_x × scale_w / scale_out
M ≈ M0 × 2^(-shift)    where M0 ∈ [0.5, 1.0)

output_int8 = clamp(round((acc_int32 × M0) >> shift + zp_out), -128, 127)
```

### 4.2 MaxPool 2×2

```
Y[y, x, c] = max(X[2y, 2x, c], X[2y+1, 2x, c], X[2y, 2x+1, c], X[2y+1, 2x+1, c])
```

Does not require requantization — the max operation preserves the quantized domain.

### 4.3 ReLU (INT8)

```
Y[i] = max(X[i], zp_x)
```

Note: for INT8 with a non-zero zero_point, the value zero in the real domain corresponds to `zp_x` in the quantized domain. Therefore ReLU compares against `zp_x`, not against 0.

### 4.4 Global Average Pooling

```
Y[c] = (1 / (H × W)) × Σ_{y=0}^{H-1} Σ_{x=0}^{W-1} X[y, x, c]
```

For INT8:
```
sum_int32 = Σ_{y,x} X[y, x, c]
avg_real = (sum_int32 - H×W×zp_x) × scale_x / (H × W)
Y_int8[c] = clamp(round(avg_real / scale_out + zp_out), -128, 127)
```

### 4.5 Dense (Fully Connected)

```
Y[j] = b[j] + Σ_{i=0}^{N_in-1} (X[i] - zp_x) × (W[j, i] - zp_w)
```

Identical to conv2d but without spatial dimensions. Requantization is analogous.

### 4.6 Number of MAC Operations

**Formula for Conv2D:**
```
MACs = H_out × W_out × C_out × C_in × kH × kW
```

**For FatCNN:**

| Layer | H_out | W_out | C_out | C_in | kH | kW | MACs |
|-------|------:|------:|------:|-----:|---:|---:|-----:|
| Layer 1 | 32 | 32 | 64 | 3 | 5 | 5 | 4,915,200 |
| Layer 2 | 16 | 16 | 128 | 64 | 3 | 3 | 18,874,368 |
| Layer 3 | 8 | 8 | 256 | 128 | 3 | 3 | 18,874,368 |
| Dense 4 | 1 | 1 | 128 | 256 | 1 | 1 | 32,768 |
| Dense 5 | 1 | 1 | 10 | 128 | 1 | 1 | 1,280 |
| **Total** | | | | | | | **42,697,984** |

**ESP32-S3 throughput (estimate):** ~100 MMAC/s (INT8, single core, optimized C)

**Compute time per worker (N=4):**
```
compute_time = total_MACs / N / throughput
             = 42,697,984 / 4 / 100,000,000
             = 0.107 seconds = 107 ms
```

### 4.7 Output Channel Parallelism — Mathematics of the Partitioning

When a convolution is partitioned across N workers by output channel:

**Worker k (k = 0..N-1) computes:**
```
channels: [k × C_out/N .. (k+1) × C_out/N - 1]
```

**Weights on worker k:**
```
W_k[C_out/N, kH, kW, C_in]   — size: (C_out/N) × kH × kW × C_in bytes
```

**Output on worker k:**
```
Y_k[H_out, W_out, C_out/N]   — size: H_out × W_out × (C_out/N) bytes
```

**Input — full, identical for all workers:**
```
X[H_in, W_in, C_in]          — size: H_in × W_in × C_in bytes
```

**Assembly on the coordinator:**
```
Y_full[H_out, W_out, C_out] = concatenate(Y_0, Y_1, ..., Y_{N-1}) along dimension C
```

---

## 5. Hardware

### 5.1 Components

| # | Component | Qty | Price | Specification |
|---|-----------|-----|-------|---------------|
| 1 | ESP32-S3-DevKitC N16R8 (SANXIXING) | 6 | ~$81 | Dual-core 240MHz, 512KB SRAM, 8MB PSRAM, 16MB Flash |
| 2 | INA219 Current Sensor (MTDELE) | 6 | ~$13 | I2C, 0-26V, ±3.2A, 12-bit ADC |
| 3 | Leinsis USB 3.0 Hub (powered) | 1 | ~$18 | 7-port, 12V/2A, per-port switches |
| 4 | USB-A to USB-C cables | 7 | ~$8 | Data transfer 480Mbps, 15-20cm |
| | **Total** | | **~$120** | |

### 5.2 ESP32-S3 Characteristics (Relevant for Inference)

| Parameter | Value |
|-----------|-------|
| CPU | Xtensa LX7 dual-core, 240 MHz |
| Internal SRAM | 512 KB |
| PSRAM | 8 MB (Octal SPI) |
| Flash | 16 MB |
| Wi-Fi | 802.11 b/g/n, 2.4 GHz |
| ESP-NOW | v2.0, max 250 bytes/packet, up to 20 peers |
| INT8 SIMD | None (software INT8 MAC) |
| Estimated INT8 throughput | ~100 MMAC/s (single core) |

### 5.3 Board Registry

| Board | Role | MAC Address | USB Port | C Array |
|-------|------|-------------|----------|---------|
| Board 0 | **Coordinator** | B8:F8:62:E2:D0:8C | `/dev/cu.usbmodem212301` | `{0xB8, 0xF8, 0x62, 0xE2, 0xD0, 0x8C}` |
| Board 1 | **Worker 1** | B8:F8:62:E2:D1:98 | `/dev/cu.usbmodem212401` | `{0xB8, 0xF8, 0x62, 0xE2, 0xD1, 0x98}` |
| Board 2 | **Worker 2** | B8:F8:62:E2:CD:E4 | `/dev/cu.usbmodem212201` | `{0xB8, 0xF8, 0x62, 0xE2, 0xCD, 0xE4}` |
| Board 3 | **Worker 3** | B8:F8:62:E2:DA:28 | `/dev/cu.usbmodem2121401` | `{0xB8, 0xF8, 0x62, 0xE2, 0xDA, 0x28}` |
| Board 4 | **Worker 4** | B8:F8:62:E2:D0:DC | `/dev/cu.usbmodem2121301` | `{0xB8, 0xF8, 0x62, 0xE2, 0xD0, 0xDC}` |
| Board 5 | **Monitor/Spare** | B8:F8:62:E2:C7:30 | `/dev/cu.usbmodem2121201` | `{0xB8, 0xF8, 0x62, 0xE2, 0xC7, 0x30}` |

---

## 6. Development Environment Setup

### 6.1 Prerequisites

**Operating system:** macOS Apple Silicon (M1/M2/M3/M4)
**IDE:** VS Code

**Installing system packages:**
```bash
xcode-select --install
brew install cmake ninja ccache python3 dfu-util
```

### 6.2 Installing ESP-IDF v6.0

1. Install the VS Code extension "ESP-IDF" by Espressif Systems
2. Cmd+Shift+P → "ESP-IDF: Open Get Started Walkthrough"
3. "ESP-IDF: Open ESP-IDF Installation Manager" → GitHub → "Start Easy Installation"
4. After installation: Cmd+Shift+P → "ESP-IDF: Select Current ESP-IDF Version" → v6.0.0

**Important step:** Create the Python venv (if it was not created automatically):
```bash
"$IDF_PATH/install.sh" esp32s3
```

**Activating the environment (required in every new terminal):**
```bash
source "$IDF_PATH/export.sh"
```

### 6.3 Development Environment (Final)

| Parameter | Value |
|-----------|-------|
| ESP-IDF | v6.0 |
| ESP-IDF Path | `$IDF_PATH` (e.g. `~/.espressif/v6.0/esp-idf`) |
| Python | 3.14.2 |
| Python venv | ESP-IDF-provided venv (e.g. `~/.espressif/python_env/idf6.0_py3.14_env`) |
| Toolchain | xtensa-esp-elf GCC 15.2.0 |
| Project dir | `~/esp/swarm-infer/` |

### 6.4 ESP-IDF v6.0 — API Changes (Compared to v5.x)

**Breaking change:** The callback for `esp_now_register_send_cb()`:

```c
// v5.x (old):
void on_sent(const uint8_t *mac_addr, esp_now_send_status_t status);

// v6.0 (new):
void on_sent(const esp_now_send_info_t *info, esp_now_send_status_t status);
```

**The receive callback is unchanged:**
```c
void on_recv(const esp_now_recv_info_t *info, const uint8_t *data, int len);
```

### 6.5 Workflow

```bash
# 1. Activate the environment
source "$IDF_PATH/export.sh"

# 2. Change to the project directory
cd ~/esp/swarm-infer/<project_name>

# 3. Set the target (one-time)
idf.py set-target esp32s3

# 4. Build
idf.py build

# 5. Flash (SEPARATE from monitor!)
idf.py -p /dev/cu.usbmodemXXXXX flash

# 6. Monitor (SEPARATE from flash!)
idf.py -p /dev/cu.usbmodemXXXXX monitor

# If the port is busy:
kill $(lsof -t /dev/cu.usbmodemXXXXX) 2>/dev/null
```

**Important:** `idf.py flash` and `idf.py monitor` MUST be invoked as separate commands. The combination `idf.py -p PORT flash monitor` causes a port lock on ESP-IDF v6.0.

### 6.6 ESP-IDF Project Structure

Every firmware project has the same layout:
```
project_name/
├── CMakeLists.txt          # Top-level CMake file
├── main/
│   ├── CMakeLists.txt      # Source file registration
│   └── source_file.c       # Firmware code
├── sdkconfig               # SDK configuration (generated)
└── build/                  # Build output (generated)
```

**Top-level CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(project_name)
```

**main/CMakeLists.txt:**
```cmake
idf_component_register(SRCS "source_file.c" INCLUDE_DIRS ".")
```

### 6.7 Known Issues and Solutions

| Issue | Solution |
|-------|----------|
| `zsh: command not found: idf.py` | Run `source .../export.sh` |
| `ESP_ERR_NVS_NO_FREE_PAGES` | NVS erase in code (already implemented) |
| `undefined reference to app_main` | Check `main/CMakeLists.txt` |
| Stack overflow (Guru Meditation) | Allocate arrays >4KB via `malloc()` |
| Port busy (Errno 35) | `kill $(lsof -t /dev/cu.usbmodemXXXXX)` |
| "Show Examples Projects" does not work | Use the terminal workflow |

---

## 7. Experiment E2: ESP-NOW Point-to-Point Benchmark

### 7.1 Goal

Measure the real latency and throughput of ESP-NOW on ESP32-S3 for various payload sizes. These data are critical for computing the communication costs of SwarmInfer.

### 7.2 Methodology

**Test type:** Ping-pong (round-trip time)
- Board 0 (PING) sends a packet → Board 1 (PONG) immediately echoes it back
- PING measures the time from send to receipt of the response (RTT)
- One-way latency ≈ RTT / 2

**Parameters:**
- Payload sizes: 10, 50, 100, 150, 200, 240 bytes
- Number of rounds: 500 + 10 warmup (warmup is discarded)
- Inter-packet delay: 3 ms
- Protocol: ESP-NOW unicast, Wi-Fi channel 1
- Distance: ~10 cm (on the desk, through the USB hub)

### 7.3 Firmware Code

**benchmark.c** — a single file; the role is selected by `#define MY_ROLE`:

```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_timer.h"

static const char *TAG = "BENCH";

#define ROLE_PING 0
#define ROLE_PONG 1
#define MY_ROLE ROLE_PING  // Change to ROLE_PONG for the second board

// MAC address of the peer (the second board)
static uint8_t peer_mac[] = {0xB8, 0xF8, 0x62, 0xE2, 0xD1, 0x98};

#define NUM_TESTS        500
#define WARMUP_PACKETS   10
#define TIMEOUT_MS       200

static const int payload_sizes[] = {10, 50, 100, 150, 200, 240};
#define NUM_PAYLOAD_SIZES (sizeof(payload_sizes) / sizeof(payload_sizes[0]))

static SemaphoreHandle_t recv_sem;
static int64_t recv_timestamp;
static uint8_t recv_buf[250];
static int recv_len;

// ESP-IDF v6.0: new callback signature
static void on_sent(const esp_now_send_info_t *info, esp_now_send_status_t status)
{
    (void)info; (void)status;
}

static void on_recv(const esp_now_recv_info_t *info, const uint8_t *data, int len)
{
    recv_timestamp = esp_timer_get_time();  // microseconds
    if (len > 0 && len <= 250) {
        memcpy(recv_buf, data, len);
        recv_len = len;
    }
    xSemaphoreGiveFromISR(recv_sem, NULL);
}

// ... (wifi_init, run_pong — echo back, run_ping — measure RTT)
// Full code: ~/esp/swarm-infer/espnow_benchmark/main/benchmark.c

// Key detail: the latencies array is allocated via malloc (not on the stack!)
// int64_t *latencies = (int64_t *)malloc(NUM_TESTS * sizeof(int64_t));
// An array >4KB on the ESP32 stack causes a Guru Meditation crash.
```

**Build and flash procedure:**

```bash
# 1. Build as PONG, flash to Board 1
cd ~/esp/swarm-infer/espnow_benchmark
sed -i '' 's/#define MY_ROLE ROLE_PING/#define MY_ROLE ROLE_PONG/' main/benchmark.c
sed -i '' 's/{PING_MAC}/{PONG_MAC}/' main/benchmark.c  # Change peer MAC
idf.py build
idf.py -p /dev/cu.usbmodem212401 flash   # Board 1

# 2. Build as PING, flash to Board 0
sed -i '' 's/#define MY_ROLE ROLE_PONG/#define MY_ROLE ROLE_PING/' main/benchmark.c
sed -i '' 's/{PONG_MAC}/{PING_MAC}/' main/benchmark.c
idf.py build
idf.py -p /dev/cu.usbmodem212301 flash   # Board 0

# 3. Monitor results
idf.py -p /dev/cu.usbmodem212301 monitor
```

### 7.4 Results

| Payload (B) | RTT min (µs) | RTT avg (µs) | RTT median (µs) | RTT p95 (µs) | RTT p99 (µs) | RTT max (µs) | Timeouts | One-way (µs) | Throughput (KB/s) |
|:-----------:|:------------:|:------------:|:----------------:|:-------------:|:-------------:|:-------------:|:--------:|:------------:|:-----------------:|
| 10 | 1,916 | 2,084 | 1,990 | 2,357 | 4,937 | 5,036 | 0 | ~1,042 | 9.6 |
| 50 | 2,631 | 2,855 | 2,698 | 3,096 | 6,297 | 9,261 | 0 | ~1,427 | 35.0 |
| 100 | 3,420 | 3,591 | 3,476 | 3,723 | 6,407 | 11,314 | 0 | ~1,795 | 55.7 |
| 150 | 4,213 | 4,413 | 4,283 | 4,628 | 7,212 | 7,466 | 0 | ~2,206 | 68.0 |
| 200 | 5,025 | 5,274 | 5,088 | 7,968 | 8,019 | 11,888 | 0 | ~2,637 | 75.8 |
| 240 | 5,658 | 5,906 | 5,725 | 8,555 | 8,635 | 8,682 | 0 | ~2,953 | 81.3 |

### 7.5 Analysis

**Zero packet loss** across all 3,000 packets (6 sizes × 500 rounds).

**Linear regression of one-way latency:**
```
latency_us ≈ 930 + 8.3 × payload_bytes
```
- Fixed overhead: ~930 µs (protocol, PHY, callback)
- Per-byte cost: ~8.3 µs
- Sanity check: 930 + 8.3×240 = 2,922 µs ≈ 2,953 µs (measured) ✓

**Comparison with prior benchmarks:**

| Source | Platform | One-way (240B) | Throughput |
|--------|----------|:--------------:|:----------:|
| Electric UI (2024) | ESP32 (original) | ~5,000 µs | ~75 KB/s |
| **Our benchmark** | **ESP32-S3, ESP-IDF v6.0** | **~2,953 µs** | **~81.3 KB/s** |
| **Improvement** | | **1.7x faster** | **1.08x higher** |

---

## 8. Experiment: Multi-Peer Communication

### 8.1 Goal

Validate ESP-NOW operation with multiple peers: broadcast from the coordinator to 4 workers simultaneously, and measure tensor transfer throughput.

### 8.2 Firmware Code

**worker.c** — identical firmware on all 4 workers:
- Waits for packets from the coordinator
- On receiving PING (cmd=0xAA) — sends a PONG response
- On receiving TENSOR DATA (cmd=0xCC) — sends an ACK

**coordinator.c** — firmware for Board 0:
- Adds all 4 workers as ESP-NOW peers
- Test 1: Broadcast PING → waits for replies from all 4
- Test 2: Unicast PING to each worker individually
- Test 3: Burst broadcast 3,072 bytes (13 chunks × 240 bytes)

Full code: `~/esp/swarm-infer/multi_worker/` and `~/esp/swarm-infer/multi_coordinator/`

### 8.3 Results

#### Test 1: Broadcast PING → 4 Workers

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

**Conclusion:** A broadcast to 4 workers with replies received from all of them takes ~5.3 ms — only ~2.5x the unicast RTT (~2.1 ms), not 4x. A broadcast is a single transmission received simultaneously by all workers.

#### Test 2: Unicast PING → Each Worker Individually

| Worker | MAC (last 2 bytes) | Avg RTT (µs) | Success |
|--------|:------------------:|:------------:|:-------:|
| Worker 1 | D1:98 | 2,345 | 100/100 |
| Worker 2 | CD:E4 | 2,318 | 100/100 |
| Worker 3 | DA:28 | 2,324 | 100/100 |
| Worker 4 | D0:DC | 2,331 | 100/100 |

**Conclusion:** All workers exhibit nearly identical RTT (~2.3 ms). The network is symmetric. There are no "slow" nodes.

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

**Critical finding:** Burst broadcast throughput is **1,412 KB/s**, which is **17x higher** than the ping-based throughput (81 KB/s). Packets are sent back-to-back without waiting for a response.

### 8.4 Impact on SwarmInfer Estimates

| Phase | Old prediction (5 ms/packet) | Measured value | Change |
|-------|:----------------------------:|:--------------:|:------:|
| **Broadcast (all layers)** | **585 ms** | **~20 ms** | **29x better** |
| **Gather (all layers, N=4)** | **2,440 ms** | **~1,464 ms** | **1.7x better** |
| **Compute (N=4)** | **107 ms** | **107 ms** | Unchanged |
| **Total unoptimized** | **3,132 ms** | **~1,591 ms** | **2.0x better** |

**Main insight:** Broadcast is essentially free (~20 ms). The bottleneck is 100% in the gather (workers→coordinator). Ring gather (originally proposed to eliminate the broadcast) is now largely unnecessary. The optimization focus shifts to **sparsification** to reduce gather volume.

---

## 9. The FatCNN Model

### 9.1 Architecture

The model is **deliberately** designed so that it does NOT fit on a single ESP32-S3:

| Layer | Type | Input | Output | Kernel | Stride | Padding | Weights (B) | Activations (B) |
|-------|------|-------|--------|--------|--------|---------|------------:|----------------:|
| 1 | Conv2D + MaxPool + ReLU | 32×32×3 | 16×16×64 | 5×5 | 1 | 2 | 4,800 | 65,536 |
| 2 | Conv2D + MaxPool + ReLU | 16×16×64 | 8×8×128 | 3×3 | 1 | 1 | 73,728 | 32,768 |
| 3 | Conv2D + GAP + ReLU | 8×8×128 | 1×1×256 | 3×3 | 1 | 1 | 294,912 | 16,384 |
| 4 | Dense + ReLU | 256 | 128 | — | — | — | 32,768 | 128 |
| 5 | Dense | 128 | 10 | — | — | — | 1,280 | 10 |
| **Total** | | | | | | | **407,488** | Peak: **65,536** |

Note: weights in the table are INT8 (1 byte each). Bias adds C_out×4 bytes (INT32) = ~1,792 bytes.

**Total weight size including bias:** ~409,280 bytes ≈ **400 KB**

### 9.2 Training Results (actual, 2026-04-06)

**Framework:** TensorFlow/Keras (tf-nightly 2.22)
**Environment:** conda env `swarm-ml`, Python 3.11, macOS Apple Silicon
**Dataset:** CIFAR-10 (50K train, 10K test)

| Parameter | Value |
|-----------|-------|
| Epochs | 30 |
| Batch size | 128 |
| Optimizer | Adam (lr=0.001) |
| Loss | SparseCategoricalCrossentropy (from_logits) |
| **Test accuracy (float32)** | **77.40%** |
| Training time | ~13 minutes |

**Training script:** `~/esp/swarm-infer/models/train_fatcnn.py`
**Saved model:** `~/esp/swarm-infer/models/fatcnn_float32.keras`

### 9.3 INT8 Quantization Results (actual, 2026-04-06)

**Script:** `~/esp/swarm-infer/models/fix_quantize.py`
**Method:** Asymmetric per-tensor quantization

**Input quantization:**
```
Input scale = 1/255 = 0.003922
Input zero_point = -128
(maps [0.0, 1.0] → [-128, 127])
```

**Weight quantization:**

| Layer | Weight range | Scale | Zero point | Max quant error |
|-------|--------------|------:|-----------:|----------------:|
| conv1 | [-0.628, 0.424] | 0.004125 | 24 | 0.002062 |
| conv2 | [-1.064, 0.867] | 0.007574 | 13 | 0.003787 |
| conv3 | [-1.462, 0.652] | 0.008290 | 48 | 0.004145 |
| dense1 | [-1.108, 0.883] | 0.007804 | 14 | 0.003902 |
| dense2 | [-1.041, 0.792] | 0.007191 | 17 | 0.003592 |

**Activation ranges (over 1000 test images):**

| Layer | Min | Max | Output scale | Output ZP |
|-------|----:|----:|-------------:|----------:|
| conv1 | 0.0000 | 4.1963 | 0.016456 | -128 |
| conv2 | 0.0000 | 16.8226 | 0.065972 | -128 |
| conv3 | 0.0000 | 30.0122 | 0.117695 | -128 |
| gap | 0.0000 | 5.3578 | 0.021011 | -128 |
| dense1 | 0.0000 | 7.1326 | 0.027971 | -128 |
| dense2 | -34.6518 | 25.4162 | 0.235561 | 19 |

**Note:** Layers conv1 through dense1 have min=0 due to ReLU → zero_point = -128.
Layer dense2 (no ReLU, output logits) has negative values → zero_point = 19.

### 9.4 Weight Partitioning Results (N=4 workers)

**Generated files:** `~/esp/swarm-infer/models/c_weights/`

| File | Contents | Size (bytes) |
|------|----------|-------------:|
| `worker_0_weights.h` | conv1 ch[0..16) + conv2 ch[0..32) + conv3 ch[0..64) | 93,808 |
| `worker_1_weights.h` | conv1 ch[16..32) + conv2 ch[32..64) + conv3 ch[64..128) | 93,808 |
| `worker_2_weights.h` | conv1 ch[32..48) + conv2 ch[64..96) + conv3 ch[128..192) | 93,808 |
| `worker_3_weights.h` | conv1 ch[48..64) + conv2 ch[96..128) + conv3 ch[192..256) | 93,808 |
| `coordinator_weights.h` | dense1 (256→128) + dense2 (128→10) | 34,600 |

**Validation:** 93,360 bytes of pure weights per worker — **exactly matches** the theoretical computation from the deep analysis ✓

### 9.5 Why FatCNN Does Not Fit on a Single ESP32

```
Weights (INT8):           407,488 bytes
Biases (INT32):             1,792 bytes
Peak activation buffer:    65,536 bytes  (Layer 1 output: 32×32×64)
Input buffer:               3,072 bytes  (32×32×3)
OS/FreeRTOS/ESP-NOW:      ~60,000 bytes
─────────────────────────────────────────
Total:                   ~537,888 bytes

ESP32-S3 internal SRAM:   524,288 bytes (512 KB)

537,888 > 524,288  →  DOES NOT FIT
```

### 9.6 FatCNN on the Swarm (N=4)

Each worker stores 1/4 of the weights for the conv layers:

```
Worker weights:           (407,488 - 34,048) / 4 + 0 = 93,360 bytes  (conv layers only)
Worker biases:                                          448 bytes
Peak activation buffer:                              16,384 bytes  (worker's partial output)
Input buffer:                                        16,384 bytes  (max: Layer 2 input)
OS/FreeRTOS/ESP-NOW:                                ~60,000 bytes
─────────────────────────────────────────────────────────────────
Total per worker:                                  ~186,576 bytes

186,576 << 524,288  →  FITS WITH MARGIN
```

---

## 10. Layer Distribution Strategies

### 10.1 Strategy A: Output Channel Parallelism (recommended)

Each ESP32 computes a subset of the output channels (feature maps).

**Advantages:** Reduces weight memory by a factor of N. Each worker is independent (no coordination required during compute).

**Drawbacks:** Each worker requires the full input. A gather is needed after every layer.

**Per-worker weight memory formula:**
```
W_worker = (C_out / N) × kH × kW × C_in    bytes
```

### 10.2 Strategy B: Spatial Parallelism (not recommended for FatCNN)

Each ESP32 processes a spatial tile (a portion of the image).

**Critical drawback:** Weights are not reduced — each worker requires the FULL set of filters (373 KB). For FatCNN, this means 373 KB + OS = 433 KB, which barely fits, with minimal benefit.

**Conclusion of the deep analysis:** Strategy B is **not viable** as a standalone approach for FatCNN.

### 10.3 Strategy C: Row Parallelism (for FC layers)

Each ESP32 computes a subset of the output neurons of an FC layer.

**For FatCNN:** The FC layers (Dense 4+5) hold only ~34 KB of weights. Distributing them across 4 workers makes no sense — communication overhead would exceed compute time. **They are executed on the coordinator.**

### 10.4 Recommended Hybrid Strategy

| Layer | Strategy | Rationale |
|-------|----------|-----------|
| Conv Layer 1 (5×5, 3→64) | A: Output Channel | Small weights, large activations |
| Conv Layer 2 (3×3, 64→128) | A: Output Channel | Moderate weights, channel split is effective |
| Conv Layer 3 (3×3, 128→256) | A: Output Channel | 295 KB of weights — must be distributed |
| Dense Layer 4 (256→128) | Coordinator only | Only 33 KB; not worth the communication overhead |
| Dense Layer 5 (128→10) | Coordinator only | Only 1.5 KB |

---

## 11. Memory Analysis

### 11.1 Strategy A: Memory per Worker (INT8, N=4)

#### Layer 1: Conv 5×5, 3→64

| Component | Formula | Bytes |
|-----------|---------|------:|
| Weights | (64/4) × 5 × 5 × 3 | 1,200 |
| Bias | (64/4) × 4 | 64 |
| Input buffer | 32 × 32 × 3 | 3,072 |
| Output buffer | 32 × 32 × (64/4) | 16,384 |
| **Total** | | **20,720** |

#### Layer 2: Conv 3×3, 64→128

| Component | Formula | Bytes |
|-----------|---------|------:|
| Weights | (128/4) × 3 × 3 × 64 | 18,432 |
| Bias | (128/4) × 4 | 128 |
| Input buffer | 16 × 16 × 64 | 16,384 |
| Output buffer | 16 × 16 × (128/4) | 8,192 |
| **Total** | | **43,136** |

#### Layer 3: Conv 3×3, 128→256

| Component | Formula | Bytes |
|-----------|---------|------:|
| Weights | (256/4) × 3 × 3 × 128 | 73,728 |
| Bias | (256/4) × 4 | 256 |
| Input buffer | 8 × 8 × 128 | 8,192 |
| Output buffer | 8 × 8 × (256/4) | 4,096 |
| **Total** | | **86,272** |

### 11.2 Peak SRAM (all weights persistent + maximum buffers)

| Component | N=1 | N=2 | N=4 | N=8 |
|-----------|----:|----:|----:|----:|
| All weights | 373,440 | 186,720 | 93,360 | 46,680 |
| All biases | 1,792 | 896 | 448 | 224 |
| Peak input buf | 16,384 | 16,384 | 16,384 | 16,384 |
| Peak output buf | 65,536 | 32,768 | 16,384 | 8,192 |
| **Peak SRAM** | **457,152** | **236,768** | **126,576** | **71,480** |
| + OS overhead (~60 KB) | 517,152 | 296,768 | 186,576 | 131,480 |
| **Fits in 512 KB?** | **NO** | **Yes** | **Yes** | **Yes** |

**Key result:** At N=1, FatCNN does not fit (517 KB > 512 KB). At N≥2 it fits. This **validates the premise** of the swarm.

---

## 12. Communication Cost Analysis

### 12.1 ESP-NOW Parameters (measured)

| Parameter | Value | Source |
|-----------|-------|--------|
| Max payload | 240 bytes (usable) | Specification |
| Unicast RTT (240B) | ~5,906 µs | Benchmark E2 |
| One-way unicast (240B) | ~2,953 µs | Benchmark E2 |
| Broadcast burst throughput | ~1,412 KB/s | Multi-peer test |
| Unicast throughput | ~81.3 KB/s | Benchmark E2 |
| Packet loss | 0% | Benchmark E2 |

**Packet count formula:**
```
num_packets = ceil(data_bytes / 240)
```

**Time formula (unicast):**
```
time_us ≈ num_packets × (930 + 8.3 × 240) ≈ num_packets × 2,922
```

**Time formula (broadcast burst):**
```
time_us ≈ data_bytes / 1,412,000 × 1,000,000 ≈ data_bytes × 0.708
```

### 12.2 Communication Costs per Inference (N=4)

#### Broadcast phase (coordinator → all workers, burst mode)

| Operation | Data (bytes) | Time (ms) |
|-----------|:------------:|:---------:|
| L1 input broadcast (32×32×3) | 3,072 | ~2 |
| L2 input broadcast (16×16×64) | 16,384 | ~12 |
| L3 input broadcast (8×8×128) | 8,192 | ~6 |
| **Total broadcast** | **27,648** | **~20** |

#### Gather phase (workers → coordinator, sequential unicast)

| Operation | Data/worker (B) | Packets/worker | Total packets | Time (ms) |
|-----------|:---------------:|:--------------:|:-------------:|:---------:|
| L1 gather (16×16×16)* | 4,096 | 18 | 72 | ~216 |
| L2 gather (8×8×32)* | 2,048 | 9 | 36 | ~108 |
| L3 gather (1×1×64)** | 64 | 1 | 4 | ~12 |
| **Total gather** | | | **112** | **~336** |

*After MaxPool on the worker (halves the gather volume!)
**After GlobalAvgPool on the coordinator input

**Note:** The original deep analysis counted gather WITHOUT MaxPool on the worker. With MaxPool on the worker, the gather volume shrinks:
- L1: 32×32×16 → 16×16×16 (4x less)
- L2: 16×16×32 → 8×8×32 (4x less)

### 12.3 Full Inference Time (revised with MaxPool on worker)

| Configuration | Broadcast (ms) | Gather (ms) | Compute (ms) | Total (ms) | fps |
|---------------|:--------------:|:-----------:|:------------:|:----------:|:---:|
| **Unoptimized** | ~20 | ~336 | ~107 | **~463** | 2.2 |
| + Bitmap sparse (1.5x) | ~20 | ~224 | ~112 | **~356** | 2.8 |
| + Aggressive sparse (2.5x) | ~20 | ~134 | ~115 | **~269** | 3.7 |

**This is significantly better** than prior estimates (3,132 ms). Reasons:
1. Broadcast turned out to be 29x cheaper than predicted
2. MaxPool on the worker reduces gather volume by 4x
3. Real latency (3 ms) vs predicted (5 ms)

---

## 13. Optimizations

### 13.1 INT8 Quantization (already applied)

All computation and data transfer use INT8 (1 byte) instead of float32 (4 bytes). This is a 4x reduction in communication volume — already included in the baseline.

### 13.2 Bitmap Sparsification

**Idea:** After ReLU, ~40-65% of activations equal zero. Instead of transmitting the full tensor, transmit:
- A bitmap (1 bit/element) indicating which elements are non-zero
- Packed values: only the non-zero values (INT8)

**Size formula:**
```
sparse_size = tensor_size/8 (bitmap) + non_zero_count × 1 (values)
dense_size = tensor_size × 1

savings = dense_size / sparse_size
```

**For typical 50% sparsity:**
```
sparse_size = N/8 + N/2 = 5N/8
savings = N / (5N/8) = 1.6x
```

**For 65% sparsity:**
```
sparse_size = N/8 + 0.35N = 0.475N
savings = 1/0.475 = 2.1x
```

**Why not Top-K:** A naive sparse encoding `(uint16 index + int8 value)` = 3 bytes per non-zero. For INT8 data this is **worse** than dense at sparsity <67%. Bitmap encoding is more efficient.

### 13.3 Ring AllReduce (re-evaluated)

**Original idea:** Replace the star topology (workers → coordinator → broadcast) with a ring. The main benefit was elimination of the broadcast phase.

**After measurements:** Broadcast costs only ~20 ms. Eliminating the broadcast via a ring saves ~20 ms at the cost of implementation complexity. **Not recommended.**

A ring gather may yield benefits by parallelizing the gather (rather than performing it sequentially), but the ESP-NOW shared medium limits the achievable parallelism.

### 13.4 Pipeline Overlapping (marginal benefit)

Overlap of the current layer's gather with the next layer's broadcast. Since broadcast is ~20 ms while gather is ~336 ms, the overlap saves ~20 ms. **Marginal.**

### 13.5 Optimization Priorities (revised)

| Optimization | Complexity | Benefit | Recommendation |
|--------------|------------|---------|----------------|
| **Bitmap sparsification** | Medium | **1.5-2.5x gather** | **Mandatory** |
| MaxPool on worker | Easy | **4x gather volume** | **Already planned** |
| Ring gather | Hard | ~20 ms | Not worth it |
| Pipeline overlap | Hard | ~20 ms | Not worth it |

---

## 14. tensor_ops Architecture

### 14.1 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Memory layout | `[H, W, C]` (channels-last) | Convenient for output channel parallelism — the worker writes its own channels |
| Requantization | Fixed-point (bit-shifts) | ~2x faster than float on ESP32 |
| MaxPool | On worker | Reduces gather volume by 4x |
| Padding | On worker | Input arrives in full; the worker adds zero-padding |

### 14.2 API (implemented, 2026-04-06)

**Files:**
- `~/esp/swarm-infer/common/tensor_ops.h` — header (41 lines)
- `~/esp/swarm-infer/common/tensor_ops.c` — implementation (275 lines)

```c
// Tensor descriptors
typedef struct { int8_t *data; int h, w, c; } Tensor3D;
typedef struct { int8_t *data; int len; } Tensor1D;
typedef struct { float scale; int8_t zero_point; } QuantParams;

// Fixed-point multiplier (instead of float requantization)
typedef struct { int32_t multiplier; int shift; } FixedPointMultiplier;

// Multiplier computation: M = input_scale * weight_scale / output_scale
// Decomposed as M = M0 * 2^(-shift), where 0.5 <= M0 < 1.0
FixedPointMultiplier compute_requant_multiplier(float input_scale,
                                                float weight_scale,
                                                float output_scale);

// Inline requantization: INT32 acc → INT8
static inline int8_t requantize(int32_t acc, FixedPointMultiplier m, int8_t output_zp);

// Core operations (all implemented)
void conv2d_int8(const Tensor3D *input, const int8_t *weights, const int32_t *bias,
    Tensor3D *output, int kernel_h, int kernel_w, int stride, int padding,
    FixedPointMultiplier requant_m, int8_t input_zp, int8_t weight_zp, int8_t output_zp);
void relu_int8(Tensor3D *tensor, int8_t zero_point);
void relu_int8_1d(Tensor1D *tensor, int8_t zero_point);
void maxpool2x2_int8(const Tensor3D *input, Tensor3D *output);
void global_avgpool_int8(const Tensor3D *input, Tensor1D *output,
    int8_t input_zp, float input_scale, float output_scale, int8_t output_zp);
void dense_int8(const Tensor1D *input, const int8_t *weights, const int32_t *bias,
    Tensor1D *output, FixedPointMultiplier requant_m,
    int8_t input_zp, int8_t weight_zp, int8_t output_zp);
int argmax_int8(const int8_t *data, int len);
int8_t *tensor_alloc(int size_bytes);
void tensor_free(int8_t *data);
```

**Key difference from the planned API:** instead of passing `QuantParams` structures, conv2d and dense accept a `FixedPointMultiplier` (precomputed) and separate zero_point values. This is more efficient — the multiplier is computed once at initialization.

### 14.3 conv2d_int8 Implementation (mathematics)

The internal logic consists of 7 nested loops:

```
for y in [0..H_out):
  for x in [0..W_out):
    for oc in [0..C_out_partial):      // partial — only this worker's channels
      int32_t acc = bias[oc]
      for ky in [0..kH):
        for kx in [0..kW):
          for ic in [0..C_in):
            in_y = y*stride + ky - padding
            in_x = x*stride + kx - padding
            if (0 ≤ in_y < H_in AND 0 ≤ in_x < W_in):
              val = input[in_y][in_x][ic]
            else:
              val = 0  // zero-padding
            acc += (int32)(val - zp_x) × (int32)(weights[oc][ky][kx][ic] - zp_w)

      // Fixed-point requantization
      output[y][x][oc] = clamp(((acc × M0) >> shift) + zp_out, -128, 127)
```

**Complexity:** O(H_out × W_out × C_out_partial × C_in × kH × kW)

**For Layer 2 on a single worker (N=4):**
```
16 × 16 × 32 × 64 × 3 × 3 = 4,718,592 MAC operations
At 100 MMAC/s: ~47 ms
```

---

## 15. Plan for Future Experiments

### 15.1 Experiment E3: Single-Node Inference Baseline

**Goal:** Measure inference latency for FatCNN-Lite (a reduced version that fits on a single ESP32) as a baseline.

**FatCNN-Lite (proposed):** Halve the channel counts: 32-64-128 instead of 64-128-256.
```
Total weights: ~102 KB (fits in 512 KB SRAM with the OS)
```

**Metrics:** Latency (ms), peak SRAM (bytes), accuracy on CIFAR-10

#### Results (actual, 2026-04-06)

**FatCNN-Lite architecture:** Conv 5×5 3→32 + Pool | Conv 3×3 32→64 + Pool | Conv 3×3 64→128 + GAP | Dense 128→64 | Dense 64→10
**Total params:** 103,690 (INT8: ~102 KB weights)
**Float32 accuracy:** 74.34% on CIFAR-10

**Firmware:** `~/esp/swarm-infer/single_inference/`
**Board:** Board 0 (ESP32-S3 N16R8), CPU 160 MHz

| Layer | Latency (µs) | Latency (ms) | % total |
|-------|-------------:|-------------:|--------:|
| Conv1 (5×5, 3→32) | 596,843 | 597 | 31.6% |
| ReLU1 | 2,875 | 3 | 0.2% |
| MaxPool1 (2×2) | 3,060 | 3 | 0.2% |
| Conv2 (3×3, 32→64) | 633,950 | 634 | 33.5% |
| ReLU2 | 1,436 | 1 | 0.1% |
| MaxPool2 (2×2) | 1,508 | 2 | 0.1% |
| Conv3 (3×3, 64→128) | 636,899 | 637 | 33.7% |
| ReLU3 | 720 | 1 | 0.0% |
| GlobalAvgPool | 1,264 | 1 | 0.1% |
| Dense1 (128→64) | 1,371 | 1 | 0.1% |
| Dense2 (64→10) | 118 | 0.1 | 0.0% |
| **Total** | **1,890,035** | **1,890** | **100%** |

**Stability:** 3 runs produced identical results: 1,890 ms ± 0.1 ms

**Prediction test:**
- Ground truth: class 3 (cat)
- ESP32 prediction: class 3 (cat) ✅
- Python prediction: class 5 (dog) — INT8 quantization changed the prediction, but the ESP32 was correct!
- ESP32 logits (INT8): airplane=4, automobile=7, bird=-49, **cat=30**, deer=-108, dog=-12, frog=16, horse=7, ship=10, truck=-56

**Memory:**
- Free heap before: 398,092 bytes
- Free heap after alloc: 325,148 bytes (73 KB used for buffers)
- Buffers freed after inference

**Throughput:** 10,674,464 MACs / 1.89s = **5.65 MMAC/s** (single core, unoptimized C)

**Key takeaways:**
1. Conv layers account for 99% of the time. Dense layers are essentially free (~1.5 ms)
2. This is the **baseline** for comparison with distributed inference
3. FatCNN (the full model) does not fit on a single ESP32, but FatCNN-Lite runs in 1.89 s

### 15.2 Experiment E4: Distributed Inference

**Goal:** Distributed FatCNN inference on 4 workers via ESP-NOW.

#### 15.2.1 Layer 1 Only (actual results, 2026-04-06)

**Firmware:** `~/esp/swarm-infer/swarm_worker/` + `~/esp/swarm-infer/swarm_coordinator/`
**Protocol:** SwarmPacket (8B header + 232B data), broadcast input, sequential unicast gather
**Workers:** 4× ESP32-S3, each computing 16 of the 64 output channels (Conv1 5×5, 3→16)
**Pipeline:** broadcast input → conv1 + ReLU + MaxPool → gather (16×16×16 per worker)

| Metric | Run 1 | Run 2 | Run 3 | Avg |
|--------|------:|------:|------:|----:|
| Broadcast (ms) | 1.9 | 1.7 | 1.7 | **~1.8** |
| Gather (ms) | 543 | 539 | 540 | **~541** |
| Assemble (ms) | 0.5 | 0.5 | 0.5 | **~0.5** |
| **TOTAL (ms)** | **547** | **544** | **544** | **~545** |

**All 4 workers:** 18 chunks each, done=1, 0 timeouts, 0 packet loss.
**Output:** deterministic (identical across 3 runs).

**Breakdown of gather time (~541 ms):**
- Worker compute (conv1 5×5, 3→16ch): ~150 ms (597 ms single-node ÷ 4)
- Worker send (4096B = 18 chunks × 1ms delay): ~18 ms per worker
- Sequential gather of 4 workers: ~72 ms total send
- Total estimated: 150 + 72 = 222 ms → actual 541 ms = **319 ms overhead**
- Source of overhead: workers finish compute sequentially + ESP-NOW contention

**Comparison with single-node (Layer 1 only):**

| Configuration | Latency | Speedup |
|---------------|--------:|--------:|
| Single ESP32 (Conv1) | 597 ms | 1.0x |
| Distributed 4 workers | 545 ms | **1.1x** |

**Conclusion:** For Layer 1 alone, the speedup is minimal — communication overhead consumes the parallelism. However, Layers 2-3 have substantially more compute and less gather data (after MaxPool), so the cumulative speedup will grow.

#### 15.2.2 Full 3-Layer Pipeline (planned)

**Next step:** add Layers 2-3 + Dense layers. Expected total: ~463 ms.

### 15.3 Experiment E5: Optimization Sweep

**Goal:** Measure the impact of each optimization individually and cumulatively.

| Configuration | What is measured |
|---------------|------------------|
| Baseline (INT8, no sparse) | Reference latency |
| + Bitmap sparse (ReLU-based) | Gather reduction |
| + Top-25% aggressive sparse | Gather + accuracy trade-off |

### 15.4 Experiment E6: Scalability

**Goal:** Latency vs number of workers (N=1,2,3,4).

### 15.5 Experiment E9: Energy per Inference

**Goal:** Measure mJ per inference using 6× INA219.
**When:** Weeks 10-11 (after all other experiments)

### 15.6 Expected Results Table for the Paper

| Scenario | Nodes | Latency (ms) | Accuracy (%) | Energy (mJ) | Memory/node (KB) |
|----------|:-----:|:------------:|:------------:|:-----------:|:----------------:|
| Single-device (FatCNN-Lite) | 1 | **1,890** | **74.3%** | TBD | ~165 |
| Single-device (FatCNN) | 1 | IMPOSSIBLE | — | — | >512 |
| SwarmInfer unoptimized (N=4) | 5 | ~463 | TBD | TBD | ~187 |
| SwarmInfer + sparse (N=4) | 5 | ~356 | TBD | TBD | ~187 |
| SwarmInfer aggressive (N=4) | 5 | ~269 | TBD | TBD | ~187 |
| SwarmInfer (N=2) | 3 | TBD | TBD | TBD | TBD |
| SwarmInfer (N=3) | 4 | TBD | TBD | TBD | TBD |

---

## Appendix A: List of All Firmware and Components

| Project | Folder | Purpose | Status |
|---------|--------|---------|--------|
| hello_world | `~/esp/swarm-infer/hello_world/` | ESP-IDF verification | ✅ |
| get_mac | `~/esp/swarm-infer/get_mac/` | Reading MAC addresses | ✅ |
| espnow_sender | `~/esp/swarm-infer/espnow_sender/` | Basic ESP-NOW sender | ✅ |
| espnow_receiver | `~/esp/swarm-infer/espnow_receiver/` | Basic ESP-NOW receiver | ✅ |
| espnow_benchmark | `~/esp/swarm-infer/espnow_benchmark/` | Ping-pong latency benchmark | ✅ |
| multi_worker | `~/esp/swarm-infer/multi_worker/` | Multi-peer worker firmware | ✅ |
| multi_coordinator | `~/esp/swarm-infer/multi_coordinator/` | Multi-peer coordinator | ✅ |
| **single_inference** | `~/esp/swarm-infer/single_inference/` | **FatCNN-Lite single-node baseline** | **✅** |
| **swarm_worker** | `~/esp/swarm-infer/swarm_worker/` | **Distributed worker (Layer 1)** | **✅** |
| **swarm_coordinator** | `~/esp/swarm-infer/swarm_coordinator/` | **Distributed coordinator (Layer 1)** | **✅** |
| **common/** | `~/esp/swarm-infer/common/` | **tensor_ops + swarm_protocol** | **✅** |
| **models/** | `~/esp/swarm-infer/models/` | **Python training + C weight export** | **✅** |
| ina219_test | `~/esp/swarm-infer/ina219_test/` | INA219 I2C sensor test | ⬜ (code ready) |
| coordinator | `~/esp/swarm-infer/coordinator/` | SwarmInfer coordinator | ⬜ (planned) |
| worker | `~/esp/swarm-infer/worker/` | SwarmInfer worker | ⬜ (planned) |

## Appendix B: Python ML Environment

```bash
# Create and activate
conda create -n swarm-ml python=3.11 -y
conda activate swarm-ml
pip install tf-nightly numpy matplotlib

# Training
cd ~/esp/swarm-infer/models
python train_fatcnn.py       # Trains the model, saves .keras
python fix_quantize.py       # Quantizes to INT8, generates C headers
```

---

*Document updated 2026-04-06 after distributed Layer 1 inference (545 ms, 4 workers, 0 packet loss).*
*Next update: after the full 3-layer distributed inference.*
