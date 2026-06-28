# R2-6 (A): activation sparsity distribution (post-ReLU gather payloads)

1000 CIFAR-10 test images. Sparsity = fraction of the transmitted tensor equal to the ReLU zero-point.

| Layer | Output elems | Mean sparsity | Std | Min | Max |
|---|---|---|---|---|---|
| Conv1 (pool) | 16384 | 54.0% | 5.6% | 39.2% | 71.6% |
| Conv2 (pool) | 8192 | 75.5% | 1.7% | 69.7% | 80.9% |
| Conv3 (GAP) | 256 | 41.6% | 5.7% | 31.2% | 69.5% |
