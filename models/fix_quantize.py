"""
SwarmInfer: Quantize trained FatCNN and export C arrays.
Run after train_fatcnn.py has saved fatcnn_float32.keras
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ============================================================
# 1. LOAD MODEL AND DATA
# ============================================================
print("Loading model and data...")
model = keras.models.load_model("fatcnn_float32.keras")
(_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_test = x_test.astype('float32') / 255.0

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Float32 accuracy: {test_acc:.4f}")

# ============================================================
# 2. COMPUTE ACTIVATION RANGES
# ============================================================
print("\n=== Computing activation ranges ===")
x_sample = x_test[:1000]

# Run inference and capture intermediate outputs
layer_outputs = {}
x = x_sample
for layer in model.layers:
    x = layer(x)
    if len(layer.get_weights()) > 0 or isinstance(layer, keras.layers.GlobalAveragePooling2D):
        out_np = x.numpy() if hasattr(x, 'numpy') else np.array(x)
        layer_outputs[layer.name] = (float(np.min(out_np)), float(np.max(out_np)))
        print(f"  {layer.name}: range=[{layer_outputs[layer.name][0]:.4f}, {layer_outputs[layer.name][1]:.4f}]")

# ============================================================
# 3. QUANTIZE WEIGHTS
# ============================================================
def quantize_tensor(tensor_float, name="tensor"):
    t_min = float(np.min(tensor_float))
    t_max = float(np.max(tensor_float))
    scale = (t_max - t_min) / 255.0
    if scale == 0: scale = 1e-8
    zero_point = int(np.round(-128 - t_min / scale))
    zero_point = max(-128, min(127, zero_point))
    quantized = np.clip(np.round(tensor_float / scale + zero_point), -128, 127).astype(np.int8)
    dequantized = (quantized.astype(np.float32) - zero_point) * scale
    max_error = np.max(np.abs(tensor_float - dequantized))
    print(f"  {name}: shape={tensor_float.shape}, scale={scale:.6f}, zp={zero_point}, max_err={max_error:.6f}")
    return quantized, scale, zero_point

print("\n=== Quantizing weights ===")
input_scale = 1.0 / 255.0
input_zp = -128
print(f"  Input: scale={input_scale:.6f}, zp={input_zp}")

layers_info = []
for layer in model.layers:
    weights = layer.get_weights()
    if len(weights) == 0:
        continue

    name = layer.name
    print(f"\nLayer: {name}")
    kernel = weights[0]
    bias = weights[1]
    kernel_q, k_scale, k_zp = quantize_tensor(kernel, "kernel")

    # Output quantization from activation ranges
    if name in layer_outputs:
        act_min, act_max = layer_outputs[name]
    else:
        act_min, act_max = -1.0, 1.0
    out_scale = (act_max - act_min) / 255.0
    if out_scale == 0: out_scale = 1e-8
    out_zp = int(np.round(-128 - act_min / out_scale))
    out_zp = max(-128, min(127, out_zp))

    ltype = 'conv2d' if isinstance(layer, keras.layers.Conv2D) else 'dense'
    layers_info.append({
        'name': name, 'type': ltype,
        'kernel': kernel_q, 'kernel_scale': k_scale, 'kernel_zp': k_zp,
        'bias': bias,
        'output_scale': out_scale, 'output_zp': out_zp,
        'kernel_shape': kernel.shape,
    })

# ============================================================
# 4. PARTITION FOR N WORKERS
# ============================================================
N_WORKERS = 4
print(f"\n=== Partitioning for {N_WORKERS} workers ===")

# Compute per-layer input scales (output scale of previous layer)
# conv1 input = image (input_scale), conv2 input = conv1 output, etc.
layer_input_scales = {}
prev_scale = input_scale
for info in layers_info:
    layer_input_scales[info['name']] = prev_scale
    prev_scale = info['output_scale']

partitions = {w: [] for w in range(N_WORKERS)}

for info in layers_info:
    if info['type'] == 'conv2d':
        kernel = info['kernel']  # TF: [kH, kW, C_in, C_out]
        bias = info['bias']
        c_out = kernel.shape[3]
        chunk = c_out // N_WORKERS

        for w in range(N_WORKERS):
            s, e = w * chunk, (w + 1) * chunk
            w_kernel = np.transpose(kernel[:, :, :, s:e], (3, 0, 1, 2))  # → [C_out_partial, kH, kW, C_in]
            w_bias = bias[s:e]
            partitions[w].append({
                'name': info['name'], 'kernel': w_kernel, 'bias': w_bias,
                'channels': (s, e),
                'kernel_scale': info['kernel_scale'], 'kernel_zp': info['kernel_zp'],
                'output_scale': info['output_scale'], 'output_zp': info['output_zp'],
            })
        print(f"  {info['name']}: C_out={c_out}, {chunk}/worker, "
              f"kernel/worker={chunk * kernel.shape[0] * kernel.shape[1] * kernel.shape[2]} bytes")

# ============================================================
# 5. EXPORT C ARRAYS
# ============================================================
output_dir = "c_weights"
os.makedirs(output_dir, exist_ok=True)
print(f"\n=== Exporting C arrays to {output_dir}/ ===")

for w in range(N_WORKERS):
    fname = os.path.join(output_dir, f"worker_{w}_weights.h")
    total = 0
    with open(fname, 'w') as f:
        f.write(f"// SwarmInfer Worker {w} Weights (auto-generated)\n")
        f.write(f"#ifndef WORKER_{w}_WEIGHTS_H\n#define WORKER_{w}_WEIGHTS_H\n\n#include <stdint.h>\n\n")
        for layer in partitions[w]:
            n = layer['name']
            k = layer['kernel'].flatten()
            f.write(f"// {n}: channels [{layer['channels'][0]}..{layer['channels'][1]}), shape={layer['kernel'].shape}\n")
            f.write(f"static const int8_t {n}_w{w}_kernel[] = {{\n")
            for i in range(0, len(k), 16):
                f.write("    " + ", ".join(str(v) for v in k[i:i+16]) + ",\n")
            f.write("};\n\n")

            bias_input_scale = layer_input_scales[layer['name']]
            b = np.round(layer['bias'] / (bias_input_scale * layer['kernel_scale'])).astype(np.int32)
            f.write(f"static const int32_t {n}_w{w}_bias[] = {{ {', '.join(str(v) for v in b)} }};\n\n")
            f.write(f"#define {n.upper()}_W{w}_KSCALE {layer['kernel_scale']:.8f}f\n")
            f.write(f"#define {n.upper()}_W{w}_KZP {layer['kernel_zp']}\n")
            f.write(f"#define {n.upper()}_W{w}_OSCALE {layer['output_scale']:.8f}f\n")
            f.write(f"#define {n.upper()}_W{w}_OZP {layer['output_zp']}\n\n")
            total += len(k) + len(b) * 4
        f.write(f"// Total: {total} bytes\n#endif\n")
    print(f"  {fname} ({total} bytes)")

# Coordinator (dense layers)
fname = os.path.join(output_dir, "coordinator_weights.h")
total = 0
with open(fname, 'w') as f:
    f.write("// SwarmInfer Coordinator Weights (auto-generated)\n")
    f.write("#ifndef COORDINATOR_WEIGHTS_H\n#define COORDINATOR_WEIGHTS_H\n\n#include <stdint.h>\n\n")
    f.write(f"#define INPUT_SCALE {input_scale:.8f}f\n#define INPUT_ZP {input_zp}\n\n")
    # Write conv3 output scale/zp (needed by coordinator for dense1 input)
    for info in layers_info:
        if info['name'] == 'conv3':
            f.write(f"// conv3 output quantization (= dense1 input) — same for all workers\n")
            f.write(f"#define CONV3_OSCALE {info['output_scale']:.8f}f\n")
            f.write(f"#define CONV3_OZP {info['output_zp']}\n\n")
            break
    for info in layers_info:
        if info['type'] != 'dense': continue
        n = info['name']
        # Transpose dense kernel from Keras [in, out] to C [out, in]
        k = info['kernel'].T.flatten()
        f.write(f"// {n}: shape={info['kernel_shape']}\n")
        f.write(f"static const int8_t {n}_kernel[] = {{\n")
        for i in range(0, len(k), 16):
            f.write("    " + ", ".join(str(v) for v in k[i:i+16]) + ",\n")
        f.write("};\n\n")
        bias_input_scale = layer_input_scales[info['name']]
        b = np.round(info['bias'] / (bias_input_scale * info['kernel_scale'])).astype(np.int32)
        f.write(f"static const int32_t {n}_bias[] = {{ {', '.join(str(v) for v in b)} }};\n\n")
        f.write(f"#define {n.upper()}_KSCALE {info['kernel_scale']:.8f}f\n")
        f.write(f"#define {n.upper()}_KZP {info['kernel_zp']}\n")
        f.write(f"#define {n.upper()}_OSCALE {info['output_scale']:.8f}f\n")
        f.write(f"#define {n.upper()}_OZP {info['output_zp']}\n\n")
        total += len(k) + len(b) * 4
    f.write(f"// Total: {total} bytes\n#endif\n")
print(f"  {fname} ({total} bytes)")

# ============================================================
# 6. SUMMARY
# ============================================================
conv_bytes = sum(info['kernel'].nbytes for info in layers_info if info['type'] == 'conv2d')
dense_bytes = sum(info['kernel'].nbytes for info in layers_info if info['type'] == 'dense')
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  Float32 accuracy:     {test_acc:.4f}")
print(f"  Conv weights (INT8):  {conv_bytes:,} bytes")
print(f"  Dense weights (INT8): {dense_bytes:,} bytes")
print(f"  Total (INT8):         {conv_bytes + dense_bytes:,} bytes")
print(f"  Per worker (N=4):     ~{conv_bytes // 4:,} bytes")
print(f"  Files: {output_dir}/worker_{{0..3}}_weights.h + coordinator_weights.h")
print(f"{'='*60}")
