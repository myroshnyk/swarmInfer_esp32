/**
 * SwarmInfer: INT8 Tensor Operations for ESP32-S3
 * Memory layout: [H, W, C] (channels-last)
 * Quantization: real_value = (quantized_value - zero_point) * scale
 */
#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H
#include <stdint.h>
#include <stdbool.h>

typedef struct { int8_t *data; int h, w, c; } Tensor3D;
typedef struct { int8_t *data; int len; } Tensor1D;
typedef struct { float scale; int8_t zero_point; } QuantParams;
typedef struct { int32_t multiplier; int shift; } FixedPointMultiplier;

FixedPointMultiplier compute_requant_multiplier(float input_scale, float weight_scale, float output_scale);

static inline int8_t requantize(int32_t acc, FixedPointMultiplier m, int8_t output_zp) {
    int64_t product = (int64_t)acc * (int64_t)m.multiplier;
    int32_t shifted = (int32_t)((product + ((int64_t)1 << (m.shift - 1))) >> m.shift);
    int32_t result = shifted + (int32_t)output_zp;
    if (result < -128) result = -128;
    if (result > 127) result = 127;
    return (int8_t)result;
}

void conv2d_int8(const Tensor3D *input, const int8_t *weights, const int32_t *bias,
    Tensor3D *output, int kernel_h, int kernel_w, int stride, int padding,
    FixedPointMultiplier requant_m, int8_t input_zp, int8_t weight_zp, int8_t output_zp);
void relu_int8(Tensor3D *tensor, int8_t zero_point);
void relu_int8_1d(Tensor1D *tensor, int8_t zero_point);
void maxpool2x2_int8(const Tensor3D *input, Tensor3D *output);
void global_avgpool_int8(const Tensor3D *input, Tensor1D *output,
    int8_t input_zp, float input_scale, float output_scale, int8_t output_zp);
void dense_int8(const Tensor1D *input, const int8_t *weights, const int32_t *bias,
    Tensor1D *output, FixedPointMultiplier requant_m, int8_t input_zp, int8_t weight_zp, int8_t output_zp);
int argmax_int8(const int8_t *data, int len);
int8_t *tensor_alloc(int size_bytes);
void tensor_free(int8_t *data);

/* Bitmap sparsification: encode non-zero values (relative to zero_point).
 * Format: [bitmap: ceil(size/8) bytes][packed non-zero values]
 * Returns encoded size. out must have space for size + ceil(size/8) bytes. */
uint32_t sparse_encode(const int8_t *data, uint32_t size, int8_t *out, int8_t zero_point);

/* Decode sparse-encoded data back to full tensor.
 * src and dst must NOT overlap. */
void sparse_decode(const int8_t *src, int8_t *dst, uint32_t original_size, int8_t zero_point);

#endif
