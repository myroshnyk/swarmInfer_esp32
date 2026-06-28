# Quantization Ablation (R2-5)

Smoke run: **False**. Images evaluated per arch: **10000**. Activations per-tensor in all INT8 modes; weights per-tensor vs per-output-channel.

## Accuracy

| Arch | Subset | float32 | INT8 per-tensor | INT8 per-channel | Δ f32→pt | Δ pt→pc |
|------|--------|---------|-----------------|------------------|----------|---------|
| fatcnn | 1k | 78.5% | 79.2% | 78.8% | +0.7 pp | -0.4 pp |
| fatcnn | 10k | 77.4% | 77.0% | 77.0% | -0.4 pp | +0.0 pp |
| fatcnn_lite | 1k | 75.4% | 74.3% | 74.1% | -1.1 pp | -0.2 pp |
| fatcnn_lite | 10k | 74.3% | 74.0% | 74.1% | -0.3 pp | +0.1 pp |

## Per-layer ranges & accumulator saturation

**Accumulator saturation** = fraction of requantized INT8 outputs that clamp to the UPPER bound +127 (true requant/accumulator overflow). Clamps to the lower bound -128 are the post-ReLU zero point (activation sparsity), not harmful saturation, and are listed separately as *ReLU-zero %*.

| Arch | Layer | Act range | Weight range | Sat→+127 (pt) | Sat→+127 (pc) | ReLU-zero % |
|------|-------|-----------|--------------|---------------|---------------|-------------|
| fatcnn | conv1 | [0.00, 4.20] | [-0.628, 0.424] | 0.000% | 0.000% | 68.0% |
| fatcnn | conv2 | [0.00, 16.82] | [-1.064, 0.867] | 0.000% | 0.000% | 88.5% |
| fatcnn | conv3 | [0.00, 30.01] | [-1.462, 0.652] | 0.000% | 0.000% | 89.6% |
| fatcnn_lite | conv1 | [0.00, 3.44] | [-0.483, 0.363] | 0.001% | 0.001% | 56.6% |
| fatcnn_lite | conv2 | [0.00, 9.86] | [-1.506, 0.737] | 0.000% | 0.000% | 78.7% |
| fatcnn_lite | conv3 | [0.00, 24.84] | [-1.752, 0.771] | 0.000% | 0.000% | 82.0% |
