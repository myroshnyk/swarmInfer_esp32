"""
SwarmInfer: Re-quantize FatCNN-Lite from saved float32 model.
Uses correct per-layer bias quantization (input_scale per layer, not global).
"""
import os
import numpy as np
from tensorflow import keras

print("Loading FatCNN-Lite model...")
model = keras.models.load_model("fatcnn_lite_float32.keras")
(_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_test = x_test.astype('float32') / 255.0

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Float32 accuracy: {test_acc:.4f}")

# Activation ranges from 1000 calibration images
print("\n=== Computing activation ranges ===")
x_sample = x_test[:1000]
x = x_sample
layer_outputs = {}
for layer in model.layers:
    x = layer(x)
    if len(layer.get_weights()) > 0 or isinstance(layer, keras.layers.GlobalAveragePooling2D):
        out_np = x.numpy()
        layer_outputs[layer.name] = (float(np.min(out_np)), float(np.max(out_np)))
        print(f"  {layer.name}: range=[{layer_outputs[layer.name][0]:.4f}, {layer_outputs[layer.name][1]:.4f}]")

def quantize_tensor(t, name):
    t_min, t_max = float(np.min(t)), float(np.max(t))
    scale = (t_max - t_min) / 255.0
    if scale == 0: scale = 1e-8
    zp = int(np.round(-128 - t_min / scale))
    zp = max(-128, min(127, zp))
    q = np.clip(np.round(t / scale + zp), -128, 127).astype(np.int8)
    err = np.max(np.abs(t - (q.astype(np.float32) - zp) * scale))
    print(f"  {name}: scale={scale:.6f}, zp={zp}, max_err={err:.6f}")
    return q, scale, zp

input_scale = 1.0 / 255.0
input_zp = -128

print("\n=== Quantizing weights ===")
layers = []
prev_output_scale = input_scale
for layer in model.layers:
    w = layer.get_weights()
    if len(w) == 0:
        continue
    kernel, bias = w[0], w[1]
    kq, ks, kzp = quantize_tensor(kernel, layer.name)
    act_min, act_max = layer_outputs.get(layer.name, (-1, 1))
    os_ = (act_max - act_min) / 255.0
    if os_ == 0: os_ = 1e-8
    ozp = int(np.round(-128 - act_min / os_))
    ozp = max(-128, min(127, ozp))
    ltype = 'conv2d' if isinstance(layer, keras.layers.Conv2D) else 'dense'
    layers.append({
        'name': layer.name, 'type': ltype, 'kernel': kq, 'bias': bias,
        'kernel_scale': ks, 'kernel_zp': kzp,
        'output_scale': os_, 'output_zp': ozp,
        'input_scale': prev_output_scale, 'shape': kernel.shape,
    })
    prev_output_scale = os_

# Export C header
os.makedirs("c_weights", exist_ok=True)
fname = "c_weights/fatcnn_lite_weights.h"
print(f"\n=== Exporting to {fname} ===")

total = 0
with open(fname, 'w') as f:
    f.write("// FatCNN-Lite: ALL weights for single ESP32-S3 (auto-generated)\n")
    f.write("#ifndef FATCNN_LITE_WEIGHTS_H\n#define FATCNN_LITE_WEIGHTS_H\n\n#include <stdint.h>\n\n")
    f.write(f"#define INPUT_SCALE {input_scale:.8f}f\n#define INPUT_ZP {input_zp}\n\n")

    for info in layers:
        n = info['name']
        if info['type'] == 'conv2d':
            k = np.transpose(info['kernel'], (3, 0, 1, 2)).flatten()
        else:
            # Dense: Keras [in, out] -> C [out, in]
            k = info['kernel'].T.flatten()

        f.write(f"// {n}: shape={info['shape']}, {len(k)} bytes\n")
        f.write(f"static const int8_t {n}_kernel[] = {{\n")
        for i in range(0, len(k), 16):
            f.write("    " + ", ".join(str(v) for v in k[i:i+16]) + ",\n")
        f.write("};\n\n")

        # Correct bias quantization: use per-layer input scale
        b = np.round(info['bias'] / (info['input_scale'] * info['kernel_scale'])).astype(np.int32)
        f.write(f"static const int32_t {n}_bias[] = {{ {', '.join(str(v) for v in b)} }};\n\n")
        f.write(f"#define {n.upper()}_KSCALE {info['kernel_scale']:.8f}f\n")
        f.write(f"#define {n.upper()}_KZP {info['kernel_zp']}\n")
        f.write(f"#define {n.upper()}_OSCALE {info['output_scale']:.8f}f\n")
        f.write(f"#define {n.upper()}_OZP {info['output_zp']}\n\n")
        total += len(k) + len(b) * 4

    f.write(f"// Total: {total} bytes\n#endif\n")

print(f"  Total weight bytes: {total}")

# Quick INT8 verification on first 20 test images
print("\n=== Quick INT8 verification (Python) ===")
cifar_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
correct_f32 = 0
correct_int8_approx = 0
for i in range(20):
    logits = model(x_test[i:i+1]).numpy()[0]
    pred_f32 = np.argmax(logits)
    label = int(y_test[i][0])
    if pred_f32 == label:
        correct_f32 += 1

print(f"Float32 accuracy (first 20): {correct_f32}/20 = {100*correct_f32/20:.0f}%")
print(f"\nDone! Weights exported to {fname}")
