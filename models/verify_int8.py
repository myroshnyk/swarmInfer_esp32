"""
Verify INT8 inference matches ESP32 distributed pipeline.
Simulates exactly what the ESP32 workers + coordinator do.
"""
import numpy as np
from tensorflow import keras

cifar_classes = ["airplane","automobile","bird","cat","deer",
                 "dog","frog","horse","ship","truck"]

# Load model and data
model = keras.models.load_model("fatcnn_float32.keras")
(_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_test = x_test.astype('float32') / 255.0

test_img = x_test[0]  # [32,32,3] float32 in [0,1]
test_label = int(y_test[0][0])

# Extract weights
weights = {}
for layer in model.layers:
    w = layer.get_weights()
    if w:
        weights[layer.name] = w

print("=== Float32 prediction ===")
logits_f32 = model(test_img[np.newaxis, ...]).numpy()[0]
print(f"Prediction: {np.argmax(logits_f32)} ({cifar_classes[np.argmax(logits_f32)]})")
print(f"Logits: {logits_f32}")

# Print layer names to find correct keys
print("\n=== Layer names ===")
for layer in model.layers:
    w = layer.get_weights()
    if w:
        print(f"  {layer.name}: weights {[x.shape for x in w]}")

# ============================================================
# INT8 quantization helpers (match ESP32 tensor_ops.c)
# ============================================================

def quantize_tensor(x, scale, zp):
    """Quantize float to int8"""
    q = np.round(x / scale + zp).astype(np.int32)
    return np.clip(q, -128, 127).astype(np.int8)

def quantize_weights(w, scale, zp):
    return quantize_tensor(w, scale, zp)

def compute_requant_multiplier(input_scale, weight_scale, output_scale):
    """Match ESP32 fixed-point multiplier"""
    real_mult = (input_scale * weight_scale) / output_scale
    # Normalize to [0.5, 1.0) range
    shift = 0
    m = real_mult
    while m < 0.5:
        m *= 2
        shift += 1
    multiplier = int(round(m * (1 << 31)))
    shift += 31
    return multiplier, shift

def requantize(acc, multiplier, shift, output_zp):
    product = np.int64(acc) * np.int64(multiplier)
    shifted = (product + (1 << (shift - 1))) >> shift
    result = int(shifted) + output_zp
    return np.clip(result, -128, 127).astype(np.int8)

def conv2d_int8(input_data, kernel, bias, stride, padding,
                input_zp, weight_zp, output_zp, multiplier, shift):
    """INT8 conv2d matching ESP32 implementation exactly"""
    H, W, C_in = input_data.shape
    C_out, kH, kW, kC = kernel.shape

    if padding > 0:
        input_padded = np.full((H + 2*padding, W + 2*padding, C_in), input_zp, dtype=np.int8)
        input_padded[padding:padding+H, padding:padding+W, :] = input_data
    else:
        input_padded = input_data

    pH, pW = input_padded.shape[0], input_padded.shape[1]
    oH = (pH - kH) // stride + 1
    oW = (pW - kW) // stride + 1

    output = np.zeros((oH, oW, C_out), dtype=np.int8)

    for oc in range(C_out):
        for oh in range(oH):
            for ow in range(oW):
                acc = np.int32(bias[oc])
                for kh in range(kH):
                    for kw in range(kW):
                        for ic in range(kC):
                            ih = oh * stride + kh
                            iw = ow * stride + kw
                            iv = np.int32(input_padded[ih, iw, ic]) - np.int32(input_zp)
                            wv = np.int32(kernel[oc, kh, kw, ic]) - np.int32(weight_zp)
                            acc += iv * wv
                output[oh, ow, oc] = requantize(acc, multiplier, shift, output_zp)
    return output

def relu_int8(data, zp):
    return np.maximum(data, np.int8(zp))

def maxpool2x2_int8(data):
    H, W, C = data.shape
    out = np.zeros((H//2, W//2, C), dtype=np.int8)
    for h in range(H//2):
        for w in range(W//2):
            for c in range(C):
                out[h, w, c] = max(
                    data[2*h, 2*w, c], data[2*h, 2*w+1, c],
                    data[2*h+1, 2*w, c], data[2*h+1, 2*w+1, c])
    return out

def gap_int8(data, input_zp, input_scale, output_scale, output_zp):
    H, W, C = data.shape
    out = np.zeros(C, dtype=np.int8)
    for c in range(C):
        total = 0.0
        for h in range(H):
            for w in range(W):
                total += (float(data[h, w, c]) - input_zp) * input_scale
        avg = total / (H * W)
        q = round(avg / output_scale + output_zp)
        out[c] = np.clip(q, -128, 127).astype(np.int8)
    return out

def dense_int8(input_data, kernel, bias, input_zp, weight_zp, output_zp, multiplier, shift):
    """kernel shape: [in_features, out_features]"""
    in_f, out_f = kernel.shape
    output = np.zeros(out_f, dtype=np.int8)
    for o in range(out_f):
        acc = np.int32(bias[o])
        for i in range(in_f):
            iv = np.int32(input_data[i]) - np.int32(input_zp)
            wv = np.int32(kernel[i, o]) - np.int32(weight_zp)
            acc += iv * wv
        output[o] = requantize(acc, multiplier, shift, output_zp)
    return output

# ============================================================
# Quantization parameters (from weight headers)
# ============================================================
INPUT_SCALE = 1.0/255.0
INPUT_ZP = -128

CONV1_KSCALE = 0.00412539
CONV1_KZP = 24
CONV1_OSCALE = 0.01645606
CONV1_OZP = -128

CONV2_KSCALE = 0.00757448
CONV2_KZP = 13
CONV2_OSCALE = 0.06597094
CONV2_OZP = -128

CONV3_KSCALE = 0.00829005
CONV3_KZP = 48
CONV3_OSCALE = 0.11769509
CONV3_OZP = -128

DENSE1_KSCALE = 0.00780428
DENSE1_KZP = 14
DENSE1_OSCALE = 0.02797115
DENSE1_OZP = -128

DENSE2_KSCALE = 0.00719110
DENSE2_KZP = 17
DENSE2_OSCALE = 0.23556096
DENSE2_OZP = 19

# ============================================================
# Quantize input image (same as ESP32: uint8 → int8 with zp=-128)
# ============================================================
img_uint8 = (test_img * 255).astype(np.uint8)
img_int8 = img_uint8.astype(np.int16) - 128
img_int8 = img_int8.astype(np.int8)
print(f"\nInput[0,0]: {img_int8[0,0,:3]}")

# ============================================================
# Quantize all weights (same as fix_quantize.py)
# ============================================================
# Conv1: [kH, kW, C_in, C_out] → transpose to [C_out, kH, kW, C_in]
conv1_w_f32 = weights['conv1'][0]  # [5,5,3,64]
conv1_b_f32 = weights['conv1'][1]  # [64]
conv1_w_t = conv1_w_f32.transpose(3, 0, 1, 2)  # [64,5,5,3]
conv1_w_q = quantize_weights(conv1_w_t, CONV1_KSCALE, CONV1_KZP)
conv1_b_q = np.round(conv1_b_f32 / (INPUT_SCALE * CONV1_KSCALE)).astype(np.int32)

conv2_w_f32 = weights['conv2'][0]  # [3,3,64,128]
conv2_b_f32 = weights['conv2'][1]
conv2_w_t = conv2_w_f32.transpose(3, 0, 1, 2)  # [128,3,3,64]
conv2_w_q = quantize_weights(conv2_w_t, CONV2_KSCALE, CONV2_KZP)
conv2_b_q = np.round(conv2_b_f32 / (CONV1_OSCALE * CONV2_KSCALE)).astype(np.int32)

conv3_w_f32 = weights['conv3'][0]  # [3,3,128,256]
conv3_b_f32 = weights['conv3'][1]
conv3_w_t = conv3_w_f32.transpose(3, 0, 1, 2)  # [256,3,3,128]
conv3_w_q = quantize_weights(conv3_w_t, CONV3_KSCALE, CONV3_KZP)
conv3_b_q = np.round(conv3_b_f32 / (CONV2_OSCALE * CONV3_KSCALE)).astype(np.int32)

dense1_w_f32 = weights['dense1'][0]  # [256,128]
dense1_b_f32 = weights['dense1'][1]
dense1_w_q = quantize_weights(dense1_w_f32, DENSE1_KSCALE, DENSE1_KZP)
dense1_b_q = np.round(dense1_b_f32 / (CONV3_OSCALE * DENSE1_KSCALE)).astype(np.int32)

dense2_w_f32 = weights['dense2'][0]  # [128,10]
dense2_b_f32 = weights['dense2'][1]
dense2_w_q = quantize_weights(dense2_w_f32, DENSE2_KSCALE, DENSE2_KZP)
dense2_b_q = np.round(dense2_b_f32 / (DENSE1_OSCALE * DENSE2_KSCALE)).astype(np.int32)

# ============================================================
# Run INT8 inference (full model, no partitioning)
# ============================================================
print("\n=== Running INT8 inference (Python, no partitioning) ===")

m1, s1 = compute_requant_multiplier(INPUT_SCALE, CONV1_KSCALE, CONV1_OSCALE)
print("Conv1...", end=" ", flush=True)
conv1_out = conv2d_int8(img_int8, conv1_w_q, conv1_b_q, 1, 2, INPUT_ZP, CONV1_KZP, CONV1_OZP, m1, s1)
conv1_out = relu_int8(conv1_out, CONV1_OZP)
pool1_out = maxpool2x2_int8(conv1_out)
print(f"done. Shape: {pool1_out.shape}")
print(f"  L1 out[0,0,0:8]:  {pool1_out[0,0,0:8]}")
print(f"  L1 out[0,0,16:24]: {pool1_out[0,0,16:24]}")

# Debug: print conv2 weights for worker 0 (channels 0-31)
print(f"  conv2_w_q[0,0,0,0:8] (w0): {conv2_w_q[0, 0, 0, 0:8]}")
print(f"  conv2_b_q[0:4] (w0): {conv2_b_q[0:4]}")

m2, s2 = compute_requant_multiplier(CONV1_OSCALE, CONV2_KSCALE, CONV2_OSCALE)
print("Conv2...", end=" ", flush=True)
conv2_out = conv2d_int8(pool1_out, conv2_w_q, conv2_b_q, 1, 1, CONV1_OZP, CONV2_KZP, CONV2_OZP, m2, s2)
conv2_out = relu_int8(conv2_out, CONV2_OZP)
pool2_out = maxpool2x2_int8(conv2_out)
print(f"done. Shape: {pool2_out.shape}")
print(f"  L2 out[0,0,0:8]: {pool2_out[0,0,0:8]}")

m3, s3 = compute_requant_multiplier(CONV2_OSCALE, CONV3_KSCALE, CONV3_OSCALE)
print("Conv3...", end=" ", flush=True)
conv3_out = conv2d_int8(pool2_out, conv3_w_q, conv3_b_q, 1, 1, CONV2_OZP, CONV3_KZP, CONV3_OZP, m3, s3)
conv3_out = relu_int8(conv3_out, CONV3_OZP)
print(f"done. Shape: {conv3_out.shape}")

gap_out = gap_int8(conv3_out, CONV3_OZP, CONV3_OSCALE, CONV3_OSCALE, CONV3_OZP)
print(f"GAP done. Shape: {gap_out.shape}")
print(f"  GAP[0:8]:  {gap_out[0:8]}")
print(f"  GAP[64:72]: {gap_out[64:72]}")

md1, sd1 = compute_requant_multiplier(CONV3_OSCALE, DENSE1_KSCALE, DENSE1_OSCALE)
print(f"  dense1_b_q[0:8]: {dense1_b_q[0:8]}")
dense1_out = dense_int8(gap_out, dense1_w_q, dense1_b_q, CONV3_OZP, DENSE1_KZP, DENSE1_OZP, md1, sd1)
print(f"Dense1 pre-relu: {dense1_out[:8]}")
dense1_out = np.maximum(dense1_out, np.int8(DENSE1_OZP))
print(f"Dense1 post-relu: {dense1_out[:8]}")

md2, sd2 = compute_requant_multiplier(DENSE1_OSCALE, DENSE2_KSCALE, DENSE2_OSCALE)
dense2_out = dense_int8(dense1_out, dense2_w_q, dense2_b_q, DENSE1_OZP, DENSE2_KZP, DENSE2_OZP, md2, sd2)

print(f"\n=== INT8 Results ===")
predicted = np.argmax(dense2_out)
print(f"Prediction: {predicted} ({cifar_classes[predicted]})")
print(f"Ground truth: {test_label} ({cifar_classes[test_label]})")
print(f"Logits (INT8): {dense2_out}")
for i in range(10):
    print(f"  [{i}] {cifar_classes[i]:12s}: {dense2_out[i]}")
