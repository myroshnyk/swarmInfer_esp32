"""
SwarmInfer / scaled MobileNet (R2-11 generalization experiment).

Trains a MobileNet-V1-style depthwise-separable CNN on CIFAR-10 upscaled to
96x96, as a higher-resolution, real-architecture demonstration that SwarmInfer's
output-channel / per-channel partitioning generalizes beyond the toy FatCNN.

Design goal: at least one pointwise (1x1) layer whose INT8 weights exceed a
single ESP32-S3's usable SRAM (block 8: 512->1024 = 512 KB), motivating
distribution. Activations live in PSRAM on-device.

Isolated: lives entirely under models/mobilenet/, touches nothing in the FatCNN
pipeline. BatchNorm is used for stable training and folded into the preceding
convolution before INT8 quantization (Phase 2).

Usage:
    conda activate swarm-ml
    python train_mbnet.py --smoke            # 3 epochs, quick pipeline check
    python train_mbnet.py --epochs 40        # full run
"""
import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

RES = 96
NUM_CLASSES = 10


def dwsep_block(x, out_ch, stride, name):
    """Depthwise 3x3 (+BN+ReLU) -> Pointwise 1x1 (+BN+ReLU)."""
    x = layers.DepthwiseConv2D(3, strides=stride, padding="same",
                               use_bias=False, name=f"{name}_dw")(x)
    x = layers.BatchNormalization(momentum=0.9, name=f"{name}_dwbn")(x)
    x = layers.ReLU(name=f"{name}_dwrelu")(x)
    x = layers.Conv2D(out_ch, 1, padding="same",
                      use_bias=False, name=f"{name}_pw")(x)
    x = layers.BatchNormalization(momentum=0.9, name=f"{name}_pwbn")(x)
    x = layers.ReLU(name=f"{name}_pwrelu")(x)
    return x


def build_model():
    """Scaled MobileNet for 96x96x3 -> 10 classes.

    Block table (output / pointwise INT8 weight size):
      conv0  3x3 s2  3->32     48x48x32
      b1     dwsep   32->64    48x48x64    2 KB
      b2     dwsep s2 64->128  24x24x128   8 KB
      b3     dwsep   128->128  24x24x128  16 KB
      b4     dwsep s2 128->256 12x12x256  32 KB
      b5     dwsep   256->256  12x12x256  64 KB
      b6     dwsep s2 256->512 6x6x512   128 KB
      b7     dwsep   512->512  6x6x512   256 KB
      b8     dwsep s2 512->1024 3x3x1024 512 KB  <- exceeds single-MCU SRAM
      GAP -> 1024 -> Dense 10
    """
    inp = layers.Input(shape=(RES, RES, 3), name="input")
    x = layers.Conv2D(32, 3, strides=2, padding="same",
                      use_bias=False, name="conv0")(inp)
    x = layers.BatchNormalization(momentum=0.9, name="conv0_bn")(x)
    x = layers.ReLU(name="conv0_relu")(x)

    x = dwsep_block(x, 64,   1, "b1")
    x = dwsep_block(x, 128,  2, "b2")
    x = dwsep_block(x, 128,  1, "b3")
    x = dwsep_block(x, 256,  2, "b4")
    x = dwsep_block(x, 256,  1, "b5")
    x = dwsep_block(x, 512,  2, "b6")
    x = dwsep_block(x, 512,  1, "b7")
    x = dwsep_block(x, 1024, 2, "b8")

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    out = layers.Dense(NUM_CLASSES, name="dense")(x)
    return models.Model(inp, out, name="scaled_mobilenet")


def make_ds(images, labels, training, batch):
    """tf.data pipeline: resize 32->96 on the fly (avoids a 5.5 GB preload).

    Training pipeline adds light augmentation (random horizontal flip and
    +-10% translation via zero-pad + random crop) to close the train/val gap.
    """
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(10000)

    def prep(img, lbl):
        img = tf.image.resize(tf.cast(img, tf.float32), [RES, RES],
                              method="bilinear") / 255.0
        if training:
            img = tf.image.random_flip_left_right(img)
            pad = int(RES * 0.1)                       # +-10% translation
            img = tf.image.resize_with_crop_or_pad(img, RES + 2 * pad,
                                                   RES + 2 * pad)
            img = tf.image.random_crop(img, [RES, RES, 3])
        return img, lbl

    return ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE) \
             .batch(batch).prefetch(tf.data.AUTOTUNE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--smoke", action="store_true",
                    help="3 epochs on a 5k subset to validate the pipeline")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "mbnet_float32.keras"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    (xtr, ytr), (xte, yte) = tf.keras.datasets.cifar10.load_data()
    ytr, yte = ytr.flatten(), yte.flatten()
    if args.smoke:
        xtr, ytr = xtr[:5000], ytr[:5000]
        xte, yte = xte[:1000], yte[:1000]
        args.epochs = 3

    # Hold out a validation split from train for best-checkpoint selection,
    # so the model put in the paper is NOT selected on the test set.
    val_n = 1000 if args.smoke else 5000
    xva, yva = xtr[-val_n:], ytr[-val_n:]
    xtr, ytr = xtr[:-val_n], ytr[:-val_n]

    train_ds = make_ds(xtr, ytr, True, args.batch)
    val_ds = make_ds(xva, yva, False, args.batch)    # checkpoint selection
    test_ds = make_ds(xte, yte, False, args.batch)   # untouched, final report

    model = build_model()
    model.summary()
    total = model.count_params()
    print(f"\nTotal params: {total:,}")

    model.compile(optimizer=optimizers.Adam(args.lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        args.out, monitor="val_accuracy", mode="max",
        save_best_only=True, verbose=1)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              verbose=2, callbacks=[ckpt])

    # best-on-val checkpoint is already on disk; report on the untouched test set
    best = tf.keras.models.load_model(args.out)
    loss, acc = best.evaluate(test_ds, verbose=0)
    print(f"\nBest-checkpoint test accuracy (10k): {acc*100:.2f}%")
    print(f"Saved best float model -> {args.out}")


if __name__ == "__main__":
    main()
