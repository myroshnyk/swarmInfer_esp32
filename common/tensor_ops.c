/**
 * SwarmInfer: INT8 Tensor Operations Implementation
 *
 * Optimized for ESP32-S3 (no SIMD, software INT8 MAC).
 * All large arrays allocated via malloc (not stack).
 */

#include "tensor_ops.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================
// Utility: Fixed-point multiplier computation
// ============================================================

FixedPointMultiplier compute_requant_multiplier(float input_scale,
                                                  float weight_scale,
                                                  float output_scale)
{
    FixedPointMultiplier result;

    double real_multiplier = (double)input_scale * (double)weight_scale / (double)output_scale;

    // Decompose: real_multiplier = M0 * 2^(-shift)
    // where 0.5 <= M0 < 1.0
    int shift = 0;
    double m0 = real_multiplier;

    if (m0 == 0.0) {
        result.multiplier = 0;
        result.shift = 0;
        return result;
    }

    // Normalize m0 to [0.5, 1.0)
    while (m0 < 0.5) {
        m0 *= 2.0;
        shift++;
    }
    while (m0 >= 1.0) {
        m0 /= 2.0;
        shift--;
    }

    // Convert to Q31 fixed-point: M0 * 2^31
    result.multiplier = (int32_t)(m0 * (double)(1LL << 31));
    result.shift = 31 + shift;  // Total shift: 31 (from Q31) + decomposition shift

    return result;
}

// ============================================================
// Conv2D INT8
// ============================================================

void conv2d_int8(
    const Tensor3D *input,
    const int8_t *weights,
    const int32_t *bias,
    Tensor3D *output,
    int kernel_h, int kernel_w,
    int stride, int padding,
    FixedPointMultiplier requant_m,
    int8_t input_zp,
    int8_t weight_zp,
    int8_t output_zp)
{
    const int H_in = input->h;
    const int W_in = input->w;
    const int C_in = input->c;
    const int H_out = output->h;
    const int W_out = output->w;
    const int C_out = output->c;  // C_out_partial for this worker

    // Weight index: weights[oc * kH * kW * C_in + ky * kW * C_in + kx * C_in + ic]
    const int w_stride_oc = kernel_h * kernel_w * C_in;
    const int w_stride_ky = kernel_w * C_in;
    const int w_stride_kx = C_in;

    for (int oy = 0; oy < H_out; oy++) {
        for (int ox = 0; ox < W_out; ox++) {
            for (int oc = 0; oc < C_out; oc++) {

                int32_t acc = bias[oc];

                for (int ky = 0; ky < kernel_h; ky++) {
                    int iy = oy * stride + ky - padding;
                    if (iy < 0 || iy >= H_in) continue;  // zero-padding

                    for (int kx = 0; kx < kernel_w; kx++) {
                        int ix = ox * stride + kx - padding;
                        if (ix < 0 || ix >= W_in) continue;  // zero-padding

                        // Input index: input[iy * W_in * C_in + ix * C_in + ic]
                        const int8_t *in_ptr = &input->data[(iy * W_in + ix) * C_in];
                        // Weight index for this (oc, ky, kx, :)
                        const int8_t *w_ptr = &weights[oc * w_stride_oc +
                                                        ky * w_stride_ky +
                                                        kx * w_stride_kx];

                        for (int ic = 0; ic < C_in; ic++) {
                            int32_t input_val = (int32_t)in_ptr[ic] - (int32_t)input_zp;
                            int32_t weight_val = (int32_t)w_ptr[ic] - (int32_t)weight_zp;
                            acc += input_val * weight_val;
                        }
                    }
                }

                // Requantize INT32 → INT8
                output->data[(oy * W_out + ox) * C_out + oc] =
                    requantize(acc, requant_m, output_zp);
            }
        }
    }
}

// ============================================================
// ReLU INT8
// ============================================================

void relu_int8(Tensor3D *tensor, int8_t zero_point)
{
    int total = tensor->h * tensor->w * tensor->c;
    for (int i = 0; i < total; i++) {
        if (tensor->data[i] < zero_point) {
            tensor->data[i] = zero_point;
        }
    }
}

void relu_int8_1d(Tensor1D *tensor, int8_t zero_point)
{
    for (int i = 0; i < tensor->len; i++) {
        if (tensor->data[i] < zero_point) {
            tensor->data[i] = zero_point;
        }
    }
}

// ============================================================
// MaxPool 2x2 stride 2
// ============================================================

void maxpool2x2_int8(const Tensor3D *input, Tensor3D *output)
{
    const int C = input->c;
    const int H_out = output->h;  // = input->h / 2
    const int W_out = output->w;  // = input->w / 2
    const int W_in = input->w;

    for (int oy = 0; oy < H_out; oy++) {
        for (int ox = 0; ox < W_out; ox++) {
            int iy = oy * 2;
            int ix = ox * 2;

            for (int c = 0; c < C; c++) {
                // 4 input values in the 2x2 window
                int8_t v00 = input->data[(iy * W_in + ix) * C + c];
                int8_t v01 = input->data[(iy * W_in + ix + 1) * C + c];
                int8_t v10 = input->data[((iy + 1) * W_in + ix) * C + c];
                int8_t v11 = input->data[((iy + 1) * W_in + ix + 1) * C + c];

                // Max of 4
                int8_t m = v00;
                if (v01 > m) m = v01;
                if (v10 > m) m = v10;
                if (v11 > m) m = v11;

                output->data[(oy * W_out + ox) * C + c] = m;
            }
        }
    }
}

// ============================================================
// Global Average Pooling
// ============================================================

void global_avgpool_int8(
    const Tensor3D *input,
    Tensor1D *output,
    int8_t input_zp,
    float input_scale,
    float output_scale,
    int8_t output_zp)
{
    const int H = input->h;
    const int W = input->w;
    const int C = input->c;
    const int spatial = H * W;

    for (int c = 0; c < C; c++) {
        // Sum all spatial values for this channel
        int32_t sum = 0;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                sum += (int32_t)input->data[(y * W + x) * C + c];
            }
        }

        // Compute average in real domain
        // avg_real = (sum - spatial * input_zp) * input_scale / spatial
        float avg_real = ((float)sum - (float)spatial * (float)input_zp)
                         * input_scale / (float)spatial;

        // Quantize to output
        int32_t out_val = (int32_t)roundf(avg_real / output_scale + (float)output_zp);
        if (out_val < -128) out_val = -128;
        if (out_val > 127) out_val = 127;
        output->data[c] = (int8_t)out_val;
    }
}

// ============================================================
// Dense (Fully Connected)
// ============================================================

void dense_int8(
    const Tensor1D *input,
    const int8_t *weights,
    const int32_t *bias,
    Tensor1D *output,
    FixedPointMultiplier requant_m,
    int8_t input_zp,
    int8_t weight_zp,
    int8_t output_zp)
{
    const int in_len = input->len;
    const int out_len = output->len;

    for (int j = 0; j < out_len; j++) {
        int32_t acc = bias[j];

        const int8_t *w_row = &weights[j * in_len];
        for (int i = 0; i < in_len; i++) {
            int32_t input_val = (int32_t)input->data[i] - (int32_t)input_zp;
            int32_t weight_val = (int32_t)w_row[i] - (int32_t)weight_zp;
            acc += input_val * weight_val;
        }

        output->data[j] = requantize(acc, requant_m, output_zp);
    }
}

// ============================================================
// Argmax
// ============================================================

int argmax_int8(const int8_t *data, int len)
{
    int max_idx = 0;
    int8_t max_val = data[0];
    for (int i = 1; i < len; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// ============================================================
// Bitmap Sparsification
// ============================================================

uint32_t sparse_encode(const int8_t *data, uint32_t size, int8_t *out, int8_t zero_point)
{
    uint32_t bitmap_bytes = (size + 7) / 8;
    uint8_t *bitmap = (uint8_t *)out;
    memset(bitmap, 0, bitmap_bytes);

    int8_t *values = out + bitmap_bytes;
    uint32_t nz = 0;

    for (uint32_t i = 0; i < size; i++) {
        if (data[i] != zero_point) {
            bitmap[i / 8] |= (1 << (i % 8));
            values[nz++] = data[i];
        }
    }
    return bitmap_bytes + nz;
}

void sparse_decode(const int8_t *src, int8_t *dst, uint32_t original_size, int8_t zero_point)
{
    uint32_t bitmap_bytes = (original_size + 7) / 8;
    const uint8_t *bitmap = (const uint8_t *)src;
    const int8_t *values = src + bitmap_bytes;

    uint32_t vi = 0;
    for (uint32_t i = 0; i < original_size; i++) {
        if (bitmap[i / 8] & (1 << (i % 8))) {
            dst[i] = values[vi++];
        } else {
            dst[i] = zero_point;
        }
    }
}

// ============================================================
// Memory helpers
// ============================================================

int8_t *tensor_alloc(int size_bytes)
{
    return (int8_t *)malloc(size_bytes);
}

void tensor_free(int8_t *data)
{
    if (data) free(data);
}
