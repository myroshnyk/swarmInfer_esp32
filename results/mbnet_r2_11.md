# R2-11: Scaled MobileNet distributed results

**Model:** scaled MobileNet-V1 (depthwise-separable), 96x96x3, 1,089,738 params
**SRAM-exceeding layer:** b8_pw 512->1024: 524,288 INT8 weight bytes (512 KB) > 512 KB SRAM; 128 KB/worker at N=4
**Distribution:** N=4 workers, 8 distributed pointwise rounds (output-channel partitioning); depthwise/stem/dense on coordinator.

## Accuracy
- Float (10k test, best-checkpoint): **89.34%**
- INT8 (10k test, per-tensor): **89.2%**

## On-device correctness
- **5/5 runs bit-exact** (mism=0), predicted class 3 == label 3.

## Latency (mean over 5 runs, ms)
| Phase | ms |
|---|---|
| Total | 9810.6 |
| Broadcast | 3973.6 |
| Gather (compute+transmit) | 4322.8 |
|   - worker pointwise compute | 1976.0 |
|   - transmit (derived) | 2346.8 |
| Local (conv0/depthwise/dense) | 1312 |
| Other | 201 |

Per-run totals (ms): [9770, 9804, 9770, 9852, 9857]

## Compute vs. communication (the regime shift R2-11 asks for)
| Component | ms | % |
|---|---|---|
| Compute (worker pw + local) | 3288.0 | 33.5% |
| Communication (broadcast + gather transmit) | 6320.4 | 64.4% |
| Other | 201 | 2.0% |

**Verdict: communication-bound** at 96x96 — contrast with FatCNN at 32x32 (73% compute, compute-bound).

## Communication volume (per inference, INT8 activations)
- Broadcast: 265.5 KB
- Gather: 405.0 KB
- **Total: 670.5 KB**
