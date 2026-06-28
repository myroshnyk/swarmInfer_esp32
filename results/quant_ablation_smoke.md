# Quantization Ablation (R2-5)

Smoke run: **True**. Images evaluated per arch: **40**. Activations per-tensor in all INT8 modes; weights per-tensor vs per-output-channel.

## Accuracy

| Arch | Subset | float32 | INT8 per-tensor | INT8 per-channel | Δ f32→pt | Δ pt→pc |
|------|--------|---------|-----------------|------------------|----------|---------|
| fatcnn | 1k | 85.0% | 87.5% | 85.0% | +2.5 pp | -2.5 pp |
| fatcnn | 10k | 85.0% | 87.5% | 85.0% | +2.5 pp | -2.5 pp |
| fatcnn_lite | 1k | 82.5% | 80.0% | 80.0% | -2.5 pp | +0.0 pp |
| fatcnn_lite | 10k | 82.5% | 80.0% | 80.0% | -2.5 pp | +0.0 pp |

## Per-layer ranges & accumulator saturation

Saturation = fraction of requantized INT8 outputs clamped to ±127, over all evaluated images.

| Arch | Layer | Act range | Weight range | Sat (per-tensor) | Sat (per-channel) |
|------|-------|-----------|--------------|------------------|-------------------|
| fatcnn | conv1 | [0.00, 4.20] | [-0.628, 0.424] | 67.35% | 67.43% |
| fatcnn | conv2 | [0.00, 16.82] | [-1.064, 0.867] | 88.56% | 88.52% |
| fatcnn | conv3 | [0.00, 30.01] | [-1.462, 0.652] | 89.46% | 89.45% |
| fatcnn_lite | conv1 | [0.00, 3.44] | [-0.483, 0.363] | 56.46% | 56.60% |
| fatcnn_lite | conv2 | [0.00, 9.86] | [-1.506, 0.737] | 78.70% | 78.59% |
| fatcnn_lite | conv3 | [0.00, 24.84] | [-1.752, 0.771] | 82.17% | 82.17% |
