"""
Export batch of CIFAR-10 test images for distributed accuracy test.
Generates test_images_batch.h with N images as INT8 arrays.

Default N is 1000, which matches the sample size used in the paper.
Pass a smaller N (e.g., 50 or 100) as argv[1] for faster bring-up.
"""
import sys
import numpy as np
from tensorflow import keras

N_IMAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

cifar_classes = ["airplane","automobile","bird","cat","deer",
                 "dog","frog","horse","ship","truck"]

print(f"Exporting {N_IMAGES} CIFAR-10 test images...")
(_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()

# INT8 quantization: same as ESP32 (uint8 - 128)
images_int8 = x_test[:N_IMAGES].astype(np.int16) - 128
labels = y_test[:N_IMAGES, 0].astype(np.int32)

outpath = "c_weights/test_images_batch.h"
with open(outpath, 'w') as f:
    f.write("// SwarmInfer: Batch test images (auto-generated)\n")
    f.write(f"// {N_IMAGES} CIFAR-10 test images, INT8 (uint8 - 128)\n")
    f.write("#ifndef TEST_IMAGES_BATCH_H\n#define TEST_IMAGES_BATCH_H\n\n")
    f.write("#include <stdint.h>\n\n")
    f.write(f"#define BATCH_SIZE {N_IMAGES}\n\n")

    # Labels array
    f.write(f"static const int batch_labels[BATCH_SIZE] = {{\n    ")
    f.write(", ".join(str(l) for l in labels))
    f.write("\n};\n\n")

    # All images in one flat array: [N_IMAGES][3072]
    f.write(f"static const int8_t batch_images[BATCH_SIZE][3072] = {{\n")
    for idx in range(N_IMAGES):
        img = images_int8[idx].flatten()
        f.write(f"  {{ // image {idx}: label={labels[idx]} ({cifar_classes[labels[idx]]})\n")
        for i in range(0, len(img), 24):
            f.write("    " + ", ".join(str(v) for v in img[i:i+24]) + ",\n")
        f.write(f"  }},\n")
    f.write("};\n\n")
    f.write("#endif\n")

print(f"Exported to {outpath}")
print(f"Labels distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

# Also compute Python INT8 predictions for comparison
print(f"\nComputing Python float32 predictions...")
model = keras.models.load_model("fatcnn_float32.keras")
x_batch = x_test[:N_IMAGES].astype('float32') / 255.0
logits = model(x_batch).numpy()
preds = np.argmax(logits, axis=1)
correct = np.sum(preds == labels)
print(f"Float32 accuracy on {N_IMAGES} images: {correct}/{N_IMAGES} = {100*correct/N_IMAGES:.1f}%")
for i in range(N_IMAGES):
    marker = "✓" if preds[i] == labels[i] else "✗"
    print(f"  [{i:2d}] true={labels[i]}({cifar_classes[labels[i]]:10s}) pred={preds[i]}({cifar_classes[preds[i]]:10s}) {marker}")
