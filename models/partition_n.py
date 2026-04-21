"""
SwarmInfer: Partition FatCNN weights for N workers.
Usage: python partition_n.py <N>
Generates c_weights_n<N>/ directory with worker + coordinator headers.
Reuses quantization from fix_quantize.py (same scales/zp).
"""
import os
import sys
import numpy as np
from tensorflow import keras

if len(sys.argv) < 2:
    print("Usage: python partition_n.py <N_WORKERS>")
    sys.exit(1)

N_WORKERS = int(sys.argv[1])
print(f"Partitioning FatCNN for N={N_WORKERS} workers")

# ============================================================
# 1. Load model
# ============================================================
model = keras.models.load_model("fatcnn_float32.keras")

# ============================================================
# 2. Quantize (same logic as fix_quantize.py)
# ============================================================
(_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_test = x_test.astype('float32') / 255.0

input_scale = 1.0 / 255.0
input_zp = -128

def quantize_tensor(tensor_float, name="tensor"):
    t_min = float(np.min(tensor_float))
    t_max = float(np.max(tensor_float))
    scale = (t_max - t_min) / 255.0
    if scale == 0: scale = 1e-8
    zero_point = int(np.round(-128 - t_min / scale))
    zero_point = max(-128, min(127, zero_point))
    quantized = np.clip(np.round(tensor_float / scale + zero_point), -128, 127).astype(np.int8)
    return quantized, scale, zero_point

# Compute activation ranges
x_sample = x_test[:1000]
layer_outputs = {}
x = x_sample
for layer in model.layers:
    x = layer(x)
    if len(layer.get_weights()) > 0 or isinstance(layer, keras.layers.GlobalAveragePooling2D):
        out_np = x.numpy() if hasattr(x, 'numpy') else np.array(x)
        layer_outputs[layer.name] = (float(np.min(out_np)), float(np.max(out_np)))

layers_info = []
for layer in model.layers:
    weights = layer.get_weights()
    if len(weights) == 0:
        continue
    name = layer.name
    kernel = weights[0]
    bias = weights[1]
    kernel_q, k_scale, k_zp = quantize_tensor(kernel, "kernel")

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
        'kernel': kernel_q, 'kernel_float': kernel, 'bias': bias,
        'kernel_scale': k_scale, 'kernel_zp': k_zp,
        'output_scale': out_scale, 'output_zp': out_zp,
        'kernel_shape': kernel.shape,
    })

# Per-layer input scales
layer_input_scales = {}
prev_scale = input_scale
for info in layers_info:
    layer_input_scales[info['name']] = prev_scale
    prev_scale = info['output_scale']

# ============================================================
# 3. Partition conv layers
# ============================================================
for info in layers_info:
    if info['type'] == 'conv2d':
        c_out = info['kernel_shape'][3]
        if c_out % N_WORKERS != 0:
            print(f"ERROR: {info['name']} has {c_out} output channels, not divisible by {N_WORKERS}")
            sys.exit(1)

partitions = {w: [] for w in range(N_WORKERS)}
for info in layers_info:
    if info['type'] == 'conv2d':
        kernel = info['kernel']  # [kH, kW, C_in, C_out] INT8
        bias = info['bias']       # float32
        c_out = kernel.shape[3]
        chunk = c_out // N_WORKERS
        for w in range(N_WORKERS):
            s, e = w * chunk, (w + 1) * chunk
            w_kernel = np.transpose(kernel[:, :, :, s:e], (3, 0, 1, 2))
            w_bias = bias[s:e]
            partitions[w].append({
                'name': info['name'], 'kernel': w_kernel, 'bias': w_bias,
                'channels': (s, e),
                'kernel_scale': info['kernel_scale'], 'kernel_zp': info['kernel_zp'],
                'output_scale': info['output_scale'], 'output_zp': info['output_zp'],
            })
        print(f"  {info['name']}: C_out={c_out}, {chunk}/worker, "
              f"{chunk * kernel.shape[0] * kernel.shape[1] * kernel.shape[2]} bytes/worker")

# ============================================================
# 4. Export
# ============================================================
output_dir = f"c_weights_n{N_WORKERS}"
os.makedirs(output_dir, exist_ok=True)

# Worker headers
for w in range(N_WORKERS):
    fname = os.path.join(output_dir, f"worker_{w}_weights.h")
    total = 0
    with open(fname, 'w') as f:
        f.write(f"// SwarmInfer Worker {w} Weights (N={N_WORKERS}, auto-generated)\n")
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

# Coordinator header (dense layers + conv3 output params)
fname = os.path.join(output_dir, "coordinator_weights.h")
total = 0
with open(fname, 'w') as f:
    f.write(f"// SwarmInfer Coordinator Weights (N={N_WORKERS}, auto-generated)\n")
    f.write("#ifndef COORDINATOR_WEIGHTS_H\n#define COORDINATOR_WEIGHTS_H\n\n#include <stdint.h>\n\n")
    f.write(f"#define INPUT_SCALE {input_scale:.8f}f\n#define INPUT_ZP {input_zp}\n\n")
    # conv3 output scale/zp for dense1 input
    for info in layers_info:
        if info['name'] == 'conv3':
            f.write(f"// conv3 output quantization (= dense1 input)\n")
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
# 5. Generate swarm_protocol dimensions header
# ============================================================
conv_layers = [info for info in layers_info if info['type'] == 'conv2d']
# conv1: 5x5, pad=2, stride=1, maxpool → 32→16
# conv2: 3x3, pad=1, stride=1, maxpool → 16→8
# conv3: 3x3, pad=1, stride=1, GAP → 8→1

ch_per_worker = [info['kernel_shape'][3] // N_WORKERS for info in conv_layers]

fname = os.path.join(output_dir, "swarm_dims.h")
with open(fname, 'w') as f:
    f.write(f"// SwarmInfer dimension overrides for N={N_WORKERS} (auto-generated)\n")
    f.write(f"// Include AFTER swarm_protocol.h to override defaults\n")
    f.write(f"#ifndef SWARM_DIMS_N{N_WORKERS}_H\n#define SWARM_DIMS_N{N_WORKERS}_H\n\n")
    f.write(f"// Worker output channels for N={N_WORKERS}\n")
    f.write(f"#undef L1_WORKER_OC\n#define L1_WORKER_OC {ch_per_worker[0]}\n")
    f.write(f"#undef L2_WORKER_OC\n#define L2_WORKER_OC {ch_per_worker[1]}\n")
    f.write(f"#undef L3_WORKER_OC\n#define L3_WORKER_OC {ch_per_worker[2]}\n\n")
    # Redefine sizes that depend on WORKER_OC
    f.write(f"#undef L1_CONV_SIZE\n#define L1_CONV_SIZE (L1_CONV_H * L1_CONV_W * L1_WORKER_OC)\n")
    f.write(f"#undef L1_POOL_SIZE\n#define L1_POOL_SIZE (L1_POOL_H * L1_POOL_W * L1_WORKER_OC)\n")
    f.write(f"#undef L2_CONV_SIZE\n#define L2_CONV_SIZE (L2_CONV_H * L2_CONV_W * L2_WORKER_OC)\n")
    f.write(f"#undef L2_POOL_SIZE\n#define L2_POOL_SIZE (L2_POOL_H * L2_POOL_W * L2_WORKER_OC)\n")
    f.write(f"#undef L3_CONV_SIZE\n#define L3_CONV_SIZE (L3_CONV_H * L3_CONV_W * L3_WORKER_OC)\n")
    f.write(f"#undef L3_GAP_SIZE\n#define L3_GAP_SIZE L3_WORKER_OC\n\n")
    f.write(f"#endif\n")
print(f"  {fname}")

# Summary
per_worker_bytes = sum(
    info['kernel_shape'][0]*info['kernel_shape'][1]*info['kernel_shape'][2]*(info['kernel_shape'][3]//N_WORKERS)
    for info in conv_layers
)
print(f"\n=== Summary (N={N_WORKERS}) ===")
print(f"  Channels per worker: L1={ch_per_worker[0]}, L2={ch_per_worker[1]}, L3={ch_per_worker[2]}")
print(f"  Weights per worker:  ~{per_worker_bytes:,} bytes")
print(f"  Output dir: {output_dir}/")
