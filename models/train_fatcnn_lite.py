"""
SwarmInfer: FatCNN-Lite — reduced model that fits in single ESP32-S3.
Channels: 32-64-128 instead of 64-128-256.
Total weights ~102 KB (fits in 512 KB SRAM with OS).
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ============================================================
# 1. MODEL
# ============================================================
def create_fatcnn_lite():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (5,5), padding='same', activation='relu',
                           input_shape=(32,32,3), name='conv1'),
        keras.layers.MaxPooling2D((2,2), name='pool1'),
        keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', name='conv2'),
        keras.layers.MaxPooling2D((2,2), name='pool2'),
        keras.layers.Conv2D(128, (3,3), padding='same', activation='relu', name='conv3'),
        keras.layers.GlobalAveragePooling2D(name='gap'),
        keras.layers.Dense(64, activation='relu', name='dense1'),
        keras.layers.Dense(10, name='dense2'),
    ])
    return model

# ============================================================
# 2. TRAIN
# ============================================================
print("="*60)
print("  FatCNN-Lite Training")
print("="*60)

model = create_fatcnn_lite()
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model.summary()
model.fit(x_train, y_train, epochs=30, batch_size=128,
          validation_data=(x_test, y_test), verbose=1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy (float32): {test_acc:.4f}")
model.save("fatcnn_lite_float32.keras")

# ============================================================
# 3. QUANTIZE & COMPUTE ACTIVATION RANGES
# ============================================================
print("\n=== Quantizing ===")
input_scale = 1.0 / 255.0
input_zp = -128

# Activation ranges
x_sample = x_test[:1000]
x = x_sample
layer_outputs = {}
for layer in model.layers:
    x = layer(x)
    if len(layer.get_weights()) > 0 or isinstance(layer, keras.layers.GlobalAveragePooling2D):
        out_np = x.numpy() if hasattr(x, 'numpy') else np.array(x)
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

layers = []
for layer in model.layers:
    w = layer.get_weights()
    if len(w) == 0: continue
    kernel, bias = w[0], w[1]
    kq, ks, kzp = quantize_tensor(kernel, layer.name)
    act_min, act_max = layer_outputs.get(layer.name, (-1, 1))
    os_ = (act_max - act_min) / 255.0
    if os_ == 0: os_ = 1e-8
    ozp = int(np.round(-128 - act_min / os_))
    ozp = max(-128, min(127, ozp))
    ltype = 'conv2d' if isinstance(layer, keras.layers.Conv2D) else 'dense'
    layers.append({'name': layer.name, 'type': ltype, 'kernel': kq, 'bias': bias,
                   'kernel_scale': ks, 'kernel_zp': kzp,
                   'output_scale': os_, 'output_zp': ozp, 'shape': kernel.shape})

# ============================================================
# 4. EXPORT SINGLE C HEADER (all weights for one ESP32)
# ============================================================
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
            # TF: [kH, kW, C_in, C_out] → ESP: [C_out, kH, kW, C_in]
            k = np.transpose(info['kernel'], (3, 0, 1, 2)).flatten()
        else:
            k = info['kernel'].flatten()

        f.write(f"// {n}: shape={info['shape']}, {len(k)} bytes\n")
        f.write(f"static const int8_t {n}_kernel[] = {{\n")
        for i in range(0, len(k), 16):
            f.write("    " + ", ".join(str(v) for v in k[i:i+16]) + ",\n")
        f.write("};\n\n")

        b = (info['bias'] / (input_scale * info['kernel_scale'])).astype(np.int32)
        f.write(f"static const int32_t {n}_bias[] = {{ {', '.join(str(v) for v in b)} }};\n\n")
        f.write(f"#define {n.upper()}_KSCALE {info['kernel_scale']:.8f}f\n")
        f.write(f"#define {n.upper()}_KZP {info['kernel_zp']}\n")
        f.write(f"#define {n.upper()}_OSCALE {info['output_scale']:.8f}f\n")
        f.write(f"#define {n.upper()}_OZP {info['output_zp']}\n\n")
        total += len(k) + len(b) * 4

    f.write(f"// Total: {total} bytes\n#endif\n")

print(f"  Total weight bytes: {total}")

# ============================================================
# 5. EXPORT TEST IMAGE (first CIFAR-10 test image as C array)
# ============================================================
test_img = x_test[0]  # [32, 32, 3] float32 in [0, 1]
test_label = int(y_test[0])
# Quantize to INT8
img_q = np.clip(np.round(test_img / input_scale + input_zp), -128, 127).astype(np.int8)

fname_img = "c_weights/test_image.h"
with open(fname_img, 'w') as f:
    f.write("// CIFAR-10 test image #0 (auto-generated)\n")
    f.write("#ifndef TEST_IMAGE_H\n#define TEST_IMAGE_H\n\n#include <stdint.h>\n\n")
    f.write(f"#define TEST_IMAGE_LABEL {test_label}  // ground truth\n\n")
    flat = img_q.flatten()
    f.write(f"// Shape: [32, 32, 3] = {len(flat)} bytes\n")
    f.write("static const int8_t test_image[] = {\n")
    for i in range(0, len(flat), 16):
        f.write("    " + ", ".join(str(v) for v in flat[i:i+16]) + ",\n")
    f.write("};\n\n#endif\n")

# Get Python prediction for verification
logits = model(test_img[np.newaxis, ...]).numpy()[0]
pred_class = np.argmax(logits)
cifar_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
print(f"\n  Test image #0: label={test_label} ({cifar_classes[test_label]}), "
      f"predicted={pred_class} ({cifar_classes[pred_class]})")
print(f"  Logits: {logits}")
print(f"  Export: {fname_img}")

# ============================================================
# 6. SUMMARY
# ============================================================
conv_b = sum(info['kernel'].nbytes for info in layers if info['type'] == 'conv2d')
dense_b = sum(info['kernel'].nbytes for info in layers if info['type'] == 'dense')
print(f"\n{'='*60}")
print(f"  FATCNN-LITE SUMMARY")
print(f"{'='*60}")
print(f"  Accuracy:           {test_acc:.4f}")
print(f"  Conv weights:       {conv_b:,} bytes")
print(f"  Dense weights:      {dense_b:,} bytes")
print(f"  Total INT8:         {conv_b + dense_b:,} bytes")
print(f"  + bias + OS:        ~{conv_b + dense_b + 2000 + 60000:,} bytes")
print(f"  ESP32 SRAM:         524,288 bytes")
print(f"  Fits?               {'YES' if conv_b + dense_b + 70000 < 524288 else 'NO'}")
print(f"{'='*60}")
