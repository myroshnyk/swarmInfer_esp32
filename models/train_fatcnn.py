"""
SwarmInfer: Train FatCNN on CIFAR-10, quantize to INT8, export weights as C arrays.

FatCNN Architecture:
  Layer 1: Conv2D 5x5, 3->64,  padding=2, stride=1 + MaxPool 2x2 + ReLU
  Layer 2: Conv2D 3x3, 64->128, padding=1, stride=1 + MaxPool 2x2 + ReLU
  Layer 3: Conv2D 3x3, 128->256, padding=1, stride=1 + GlobalAvgPool + ReLU
  Layer 4: Dense 256->128 + ReLU
  Layer 5: Dense 128->10

Total weights: ~407 KB (INT8) — does NOT fit in single ESP32-S3 (512 KB SRAM)
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Fixed seeds for reproducibility. The paper reports a single-seed result
# (this is documented as a limitation in sec. "Limitations"); these seeds
# reproduce the exact model used for the 2026-04 experiment suite.
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ============================================================
# 1. DEFINE MODEL
# ============================================================
def create_fatcnn():
    model = keras.Sequential([
        # Layer 1: Conv 5x5, 3->64 + MaxPool + ReLU
        keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu',
                           input_shape=(32, 32, 3), name='conv1'),
        keras.layers.MaxPooling2D((2, 2), name='pool1'),

        # Layer 2: Conv 3x3, 64->128 + MaxPool + ReLU
        keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                           name='conv2'),
        keras.layers.MaxPooling2D((2, 2), name='pool2'),

        # Layer 3: Conv 3x3, 128->256 + GlobalAvgPool + ReLU
        keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                           name='conv3'),
        keras.layers.GlobalAveragePooling2D(name='gap'),

        # Layer 4: Dense 256->128 + ReLU
        keras.layers.Dense(128, activation='relu', name='dense1'),

        # Layer 5: Dense 128->10
        keras.layers.Dense(10, name='dense2'),
    ])
    return model


# ============================================================
# 2. TRAIN
# ============================================================
def train_model(model, epochs=30):
    # Load CIFAR-10
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Train
    print("\n=== Training FatCNN on CIFAR-10 ===")
    model.summary()
    print()

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=128,
        validation_data=(x_test, y_test),
        verbose=1
    )

    # Evaluate on full 10k test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy (float32, full 10k): {test_acc:.4f}")

    # Evaluate on the first 1000 test images -- the same subset that the
    # firmware runs on-device, so this float32 number is directly comparable
    # to the on-device INT8 accuracy reported in the paper.
    subset_x, subset_y = x_test[:1000], y_test[:1000]
    subset_loss, subset_acc = model.evaluate(subset_x, subset_y, verbose=0)
    subset_preds = np.argmax(model(subset_x).numpy(), axis=1)
    subset_correct = int(np.sum(subset_preds == subset_y.flatten()))
    print(f"Test accuracy (float32, first 1000 images used in paper): "
          f"{subset_correct}/1000 = {subset_acc:.4f}")

    return model, history, (x_train, y_train, x_test, y_test)


# ============================================================
# 3. QUANTIZE TO INT8
# ============================================================
def quantize_tensor(tensor_float, name="tensor"):
    """Quantize a float32 tensor to INT8 with per-tensor scale and zero_point."""
    t_min = float(np.min(tensor_float))
    t_max = float(np.max(tensor_float))

    # Compute scale and zero_point for asymmetric quantization
    # Maps [t_min, t_max] -> [-128, 127]
    scale = (t_max - t_min) / 255.0
    if scale == 0:
        scale = 1e-8

    zero_point = int(np.round(-128 - t_min / scale))
    zero_point = max(-128, min(127, zero_point))

    # Quantize
    quantized = np.round(tensor_float / scale + zero_point).astype(np.int8)
    quantized = np.clip(quantized, -128, 127).astype(np.int8)

    # Verify dequantization error
    dequantized = (quantized.astype(np.float32) - zero_point) * scale
    max_error = np.max(np.abs(tensor_float - dequantized))

    print(f"  {name}: shape={tensor_float.shape}, range=[{t_min:.4f}, {t_max:.4f}], "
          f"scale={scale:.6f}, zp={zero_point}, max_error={max_error:.6f}")

    return quantized, scale, zero_point


def quantize_model(model, x_test, y_test):
    """Quantize all layers of the model to INT8."""
    print("\n=== Quantizing model to INT8 ===")

    layers_info = []

    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) == 0:
            continue  # Skip pooling, activation layers

        layer_name = layer.name
        print(f"\nLayer: {layer_name}")

        if isinstance(layer, keras.layers.Conv2D):
            # weights[0] = kernel [kH, kW, C_in, C_out] (TF format)
            # weights[1] = bias [C_out]
            kernel = weights[0]
            bias = weights[1]

            # Quantize kernel to INT8
            kernel_q, kernel_scale, kernel_zp = quantize_tensor(kernel, "kernel")

            # Bias stays INT32 (accumulated precision)
            bias_int32 = bias  # Will be converted after input quantization

            layers_info.append({
                'name': layer_name,
                'type': 'conv2d',
                'kernel': kernel_q,
                'kernel_scale': kernel_scale,
                'kernel_zp': kernel_zp,
                'bias': bias,
                'kernel_shape': kernel.shape,  # [kH, kW, C_in, C_out]
            })

        elif isinstance(layer, keras.layers.Dense):
            # weights[0] = kernel [in_features, out_features]
            # weights[1] = bias [out_features]
            kernel = weights[0]
            bias = weights[1]

            kernel_q, kernel_scale, kernel_zp = quantize_tensor(kernel, "kernel")

            layers_info.append({
                'name': layer_name,
                'type': 'dense',
                'kernel': kernel_q,
                'kernel_scale': kernel_scale,
                'kernel_zp': kernel_zp,
                'bias': bias,
                'kernel_shape': kernel.shape,
            })

    # Quantize input (CIFAR-10 images normalized to [0,1])
    # Input quantization: maps [0, 1] -> [-128, 127]
    input_scale = 1.0 / 255.0
    input_zp = -128
    print(f"\nInput: scale={input_scale:.6f}, zp={input_zp}")

    # For each layer, compute output quantization params
    # (simplified: use representative dataset to find activation ranges)
    print("\n=== Computing activation ranges ===")
    activations = compute_activation_ranges(model, x_test[:1000])

    for i, info in enumerate(layers_info):
        act_name = info['name']
        if act_name in activations:
            act_min, act_max = activations[act_name]
            act_scale = (act_max - act_min) / 255.0
            if act_scale == 0:
                act_scale = 1e-8
            act_zp = int(np.round(-128 - act_min / act_scale))
            act_zp = max(-128, min(127, act_zp))
            info['output_scale'] = act_scale
            info['output_zp'] = act_zp
            print(f"  {act_name}: act_range=[{act_min:.4f}, {act_max:.4f}], "
                  f"scale={act_scale:.6f}, zp={act_zp}")

    return layers_info, input_scale, input_zp


def compute_activation_ranges(model, x_sample):
    """Run a sample through the model and record min/max of each layer's output."""
    ranges = {}

    # Build sub-models for each layer
    for layer in model.layers:
        if len(layer.get_weights()) == 0 and not isinstance(layer, keras.layers.GlobalAveragePooling2D):
            continue

        sub_model = keras.Model(inputs=model.input, outputs=layer.output)
        output = sub_model.predict(x_sample, verbose=0)
        ranges[layer.name] = (float(np.min(output)), float(np.max(output)))

    return ranges


# ============================================================
# 4. PARTITION WEIGHTS FOR N WORKERS
# ============================================================
def partition_weights(layers_info, n_workers=4):
    """Split conv layer weights by output channels for N workers."""
    print(f"\n=== Partitioning weights for {n_workers} workers ===")

    partitions = {w: [] for w in range(n_workers)}

    for info in layers_info:
        if info['type'] == 'conv2d':
            # TF kernel shape: [kH, kW, C_in, C_out]
            kernel = info['kernel']
            bias = info['bias']
            c_out = kernel.shape[3]
            chunk = c_out // n_workers

            print(f"\n  {info['name']}: C_out={c_out}, {chunk} channels per worker")

            for w in range(n_workers):
                start = w * chunk
                end = (w + 1) * chunk
                # Slice output channels
                w_kernel = kernel[:, :, :, start:end]
                w_bias = bias[start:end]

                # Convert to channels-last layout for ESP32: [C_out_partial, kH, kW, C_in]
                w_kernel_esp = np.transpose(w_kernel, (3, 0, 1, 2))

                partitions[w].append({
                    'name': info['name'],
                    'kernel': w_kernel_esp,
                    'bias': w_bias,
                    'channels': (start, end),
                    'kernel_scale': info['kernel_scale'],
                    'kernel_zp': info['kernel_zp'],
                    'output_scale': info.get('output_scale', 1.0),
                    'output_zp': info.get('output_zp', 0),
                })

                print(f"    Worker {w}: channels [{start}..{end}), "
                      f"kernel={w_kernel_esp.shape}, bias={w_bias.shape}, "
                      f"bytes={w_kernel_esp.nbytes + w_bias.nbytes * 4}")

        elif info['type'] == 'dense':
            # Dense layers run on coordinator only
            print(f"\n  {info['name']}: Dense — coordinator only, "
                  f"shape={info['kernel_shape']}, bytes={info['kernel'].nbytes}")

    return partitions


# ============================================================
# 5. EXPORT TO C HEADER FILES
# ============================================================
def export_to_c_arrays(layers_info, partitions, input_scale, input_zp,
                       n_workers=4, output_dir="c_weights"):
    """Export quantized weights as C header files for each worker."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Exporting C arrays to {output_dir}/ ===")

    # Export worker weights
    for w in range(n_workers):
        filename = os.path.join(output_dir, f"worker_{w}_weights.h")
        with open(filename, 'w') as f:
            f.write(f"// SwarmInfer Worker {w} Weights (auto-generated)\n")
            f.write(f"// Do not edit manually\n\n")
            f.write(f"#ifndef WORKER_{w}_WEIGHTS_H\n")
            f.write(f"#define WORKER_{w}_WEIGHTS_H\n\n")
            f.write(f"#include <stdint.h>\n\n")

            total_bytes = 0
            for layer in partitions[w]:
                name = layer['name']
                kernel = layer['kernel']
                bias = layer['bias']

                # Write kernel (INT8)
                f.write(f"// {name}: channels [{layer['channels'][0]}..{layer['channels'][1]})\n")
                f.write(f"// Shape: {kernel.shape} = {kernel.nbytes} bytes\n")
                f.write(f"static const int8_t {name}_w{w}_kernel[] = {{\n")
                flat = kernel.flatten()
                for i in range(0, len(flat), 16):
                    chunk = flat[i:i+16]
                    f.write("    " + ", ".join(f"{v}" for v in chunk) + ",\n")
                f.write(f"}};\n\n")

                # Write bias (INT32 as float for now, will convert)
                bias_int32 = (bias / (input_scale * layer['kernel_scale'])).astype(np.int32)
                f.write(f"static const int32_t {name}_w{w}_bias[] = {{\n")
                f.write("    " + ", ".join(f"{v}" for v in bias_int32) + ",\n")
                f.write(f"}};\n\n")

                # Write quantization params
                f.write(f"#define {name.upper()}_W{w}_KERNEL_SCALE {layer['kernel_scale']:.8f}f\n")
                f.write(f"#define {name.upper()}_W{w}_KERNEL_ZP {layer['kernel_zp']}\n")
                f.write(f"#define {name.upper()}_W{w}_OUTPUT_SCALE {layer['output_scale']:.8f}f\n")
                f.write(f"#define {name.upper()}_W{w}_OUTPUT_ZP {layer['output_zp']}\n\n")

                total_bytes += kernel.nbytes + len(bias_int32) * 4

            f.write(f"// Total worker {w} weight bytes: {total_bytes}\n\n")
            f.write(f"#endif\n")

        print(f"  Worker {w}: {filename} ({total_bytes} bytes)")

    # Export coordinator weights (dense layers)
    filename = os.path.join(output_dir, "coordinator_weights.h")
    with open(filename, 'w') as f:
        f.write("// SwarmInfer Coordinator Weights (auto-generated)\n")
        f.write("// Dense layers only — run on coordinator\n\n")
        f.write("#ifndef COORDINATOR_WEIGHTS_H\n")
        f.write("#define COORDINATOR_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")

        # Input quantization
        f.write(f"#define INPUT_SCALE {input_scale:.8f}f\n")
        f.write(f"#define INPUT_ZP {input_zp}\n\n")

        total_bytes = 0
        for info in layers_info:
            if info['type'] != 'dense':
                continue

            name = info['name']
            kernel = info['kernel']
            bias = info['bias']

            # Kernel (INT8) — shape [in, out] → flatten
            f.write(f"// {name}: shape {kernel.shape} = {kernel.nbytes} bytes\n")
            f.write(f"static const int8_t {name}_kernel[] = {{\n")
            flat = kernel.flatten()
            for i in range(0, len(flat), 16):
                chunk = flat[i:i+16]
                f.write("    " + ", ".join(f"{v}" for v in chunk) + ",\n")
            f.write(f"}};\n\n")

            # Bias (INT32)
            bias_int32 = (bias / (input_scale * info['kernel_scale'])).astype(np.int32)
            f.write(f"static const int32_t {name}_bias[] = {{\n")
            f.write("    " + ", ".join(f"{v}" for v in bias_int32) + ",\n")
            f.write(f"}};\n\n")

            f.write(f"#define {name.upper()}_KERNEL_SCALE {info['kernel_scale']:.8f}f\n")
            f.write(f"#define {name.upper()}_KERNEL_ZP {info['kernel_zp']}\n")
            f.write(f"#define {name.upper()}_OUTPUT_SCALE {info.get('output_scale', 1.0):.8f}f\n")
            f.write(f"#define {name.upper()}_OUTPUT_ZP {info.get('output_zp', 0)}\n\n")

            total_bytes += kernel.nbytes + len(bias_int32) * 4

        f.write(f"// Total coordinator weight bytes: {total_bytes}\n\n")
        f.write("#endif\n")

    print(f"  Coordinator: {filename} ({total_bytes} bytes)")


# ============================================================
# 6. MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  SwarmInfer: FatCNN Training & Quantization Pipeline")
    print("=" * 60)

    # Create model
    model = create_fatcnn()

    # Train
    model, history, (x_train, y_train, x_test, y_test) = train_model(model, epochs=30)

    # Save float32 model
    model.save("fatcnn_float32.keras")
    print("\nFloat32 model saved to fatcnn_float32.keras")

    # Quantize
    layers_info, input_scale, input_zp = quantize_model(model, x_test, y_test)

    # Partition for 4 workers
    partitions = partition_weights(layers_info, n_workers=4)

    # Export C arrays
    export_to_c_arrays(layers_info, partitions, input_scale, input_zp,
                       n_workers=4, output_dir="c_weights")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    total_conv_weights = sum(info['kernel'].nbytes for info in layers_info if info['type'] == 'conv2d')
    total_dense_weights = sum(info['kernel'].nbytes for info in layers_info if info['type'] == 'dense')

    print(f"  Conv weights (INT8):    {total_conv_weights:,} bytes")
    print(f"  Dense weights (INT8):   {total_dense_weights:,} bytes")
    print(f"  Total weights (INT8):   {total_conv_weights + total_dense_weights:,} bytes")
    print(f"  Per worker (N=4):       {total_conv_weights // 4:,} bytes")
    print(f"  Coordinator (dense):    {total_dense_weights:,} bytes")
    print(f"\n  Output files in: c_weights/")
    print(f"    worker_0_weights.h .. worker_3_weights.h")
    print(f"    coordinator_weights.h")
    print(f"\n  Float32 accuracy:       {history.history['val_accuracy'][-1]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
