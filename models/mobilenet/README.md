# Scaled MobileNet — SwarmInfer generalization experiment (R2-11)

**Purpose.** Demonstrate that SwarmInfer's intra-layer partitioning generalizes
beyond the toy FatCNN to a real edge architecture (MobileNet, depthwise
separable convolutions) at higher input resolution (96×96), with at least one
layer whose INT8 weights exceed a single ESP32-S3's usable SRAM. Answers
Reviewer 2's "beyond CIFAR-10 / 32 px" concern.

**Isolation.** Everything lives under `models/mobilenet/` plus (later) separate
`*_mbnet` firmware projects and separate INT8 kernels. The validated FatCNN
pipeline, results, and firmware are **not touched** — MobileNet is purely
additive. The output-channel / per-channel partitioning theory is unchanged:
pointwise (1×1) convs split by output channel (as FatCNN does); depthwise convs
split per channel.

## Architecture (96×96×3 → 10 classes, ~1.09 M params)

| # | Layer | Output | Pointwise INT8 weights |
|---|-------|--------|------------------------|
| 0 | Conv 3×3 s2, 3→32 | 48×48×32 | — |
| b1 | DWsep 32→64 | 48×48×64 | 2 KB |
| b2 | DWsep s2 64→128 | 24×24×128 | 8 KB |
| b3 | DWsep 128→128 | 24×24×128 | 16 KB |
| b4 | DWsep s2 128→256 | 12×12×256 | 32 KB |
| b5 | DWsep 256→256 | 12×12×256 | 64 KB |
| b6 | DWsep s2 256→512 | 6×6×512 | 128 KB |
| b7 | DWsep 512→512 | 6×6×512 | 256 KB |
| **b8** | **DWsep s2 512→1024** | **3×3×1024** | **512 KB ← exceeds single-MCU SRAM** |
| gap | GlobalAvgPool → 1024 | 1024 | — |
| dense | 1024→10 | 10 | 10 KB |

Each `DWsep` = DepthwiseConv2D 3×3 (+BN+ReLU) → Conv2D 1×1 (+BN+ReLU).
**BatchNorm (momentum 0.9)** is used for stable from-scratch training and is
**folded into the preceding convolution** before INT8 quantization (Phase 2), so
on-device inference has no BN.

**Block 8's pointwise (512→1024) = 524,288 INT8 weights = 512 KB**, which does
not fit a single ESP32-S3 → distributed across 4 workers (128 KB each). This is
the concrete "a single layer exceeds SRAM" demonstration.

## Memory (on-device)
- Activations live in PSRAM (N16R8 = 8 MB). Largest activation: b1 output
  48×48×64 = 147 KB (INT8). Broadcast input per layer is small (≤ early conv).
- Pointwise weights are the large arrays → partitioned across workers.
- Depthwise weights are tiny (Cin×9) → not partitioned (kept whole per node).

## Phased plan
1. **Python model + training** (this dir) — `train_mbnet.py`. Float accuracy. ← in progress
2. **INT8 quantize + numpy reference + C-weight export** — fold BN, per-tensor
   INT8 for DW/PW, bit-exact numpy engine as the on-device target.
3. **INT8 C kernels** — `dw_conv_int8`, `pw_conv_int8` (1×1); host-validate
   bit-exact vs numpy. New file, does not touch `tensor_ops.c`.
4. **Partitioning** — PW by output channel, DW per channel.
5. **Firmware** — separate `swarm_worker_mbnet` / `swarm_coordinator_mbnet`.
6. **Cluster run** — bit-exact validation + accuracy/latency; write into paper +
   response (R2-11).

## Status
- Phase 1: pipeline validated (smoke 3 epochs/5k → 45.9%); full 40-epoch run
  launched (log `/tmp/mbnet_train.log`, out `mbnet_float32.keras`).
