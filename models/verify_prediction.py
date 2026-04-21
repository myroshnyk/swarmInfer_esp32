"""Quick verification: what does FatCNN predict for test image #0?"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

cifar_classes = ["airplane","automobile","bird","cat","deer",
                 "dog","frog","horse","ship","truck"]

model = keras.models.load_model("fatcnn_float32.keras")
(_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_test = x_test.astype('float32') / 255.0

test_img = x_test[0]
test_label = int(y_test[0][0])
logits = model(test_img[np.newaxis, ...]).numpy()[0]
pred_class = np.argmax(logits)

print(f"Ground truth: {test_label} ({cifar_classes[test_label]})")
print(f"FatCNN prediction: {pred_class} ({cifar_classes[pred_class]})")
print(f"Logits: {logits}")
print(f"Top-3: {np.argsort(logits)[::-1][:3]} = {[cifar_classes[i] for i in np.argsort(logits)[::-1][:3]]}")
