"""
SwarmInfer — R2-2 + R2-9: Training-variance seed sweep.

Trains BOTH FatCNN (64-128-256) and FatCNN-Lite (32-64-128) on CIFAR-10 for
3 random seeds each (0,1,2), 30 epochs, with hyperparameters IDENTICAL to the
canonical scripts train_fatcnn.py / train_fatcnn_lite.py:
    optimizer = Adam(lr=1e-3)
    loss      = SparseCategoricalCrossentropy(from_logits=True)
    batch     = 128
    epochs    = 30
    NO data augmentation (matching the canonical scripts and the paper pipeline).

For each trained model we record float32 test accuracy on:
    (a) the standard full 10,000-image CIFAR-10 test set,
    (b) the first-1,000-image subset (comparable to the on-device numbers).

Outputs results/seed_sweep.json with per-seed accuracies, per-architecture
mean +/- sample std, and the accuracy GAP (FatCNN - Lite) paired by seed:
mean +/- sd and a 95% t-confidence interval over the 3 paired gaps.

Checkpoints are written to models/seed_sweep_ckpts/ so the run is resumable;
the canonical models/c_weights/ are never touched.

Usage:
    python seed_sweep.py            # full run: 30 epochs, seeds 0,1,2
    python seed_sweep.py --smoke    # smoke: 3 epochs, validates pipeline + JSON
    SEED_SWEEP_SMOKE=1 python seed_sweep.py   # env-var equivalent
"""
import os
import sys
import json
import time
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(REPO, "results")
CKPT_DIR = os.path.join(HERE, "seed_sweep_ckpts")

SEEDS = [0, 1, 2]
BATCH = 128
LR = 1e-3

SMOKE = ("--smoke" in sys.argv) or (os.environ.get("SEED_SWEEP_SMOKE") == "1")
EPOCHS = 3 if SMOKE else 30


# ---------------------------------------------------------------------------
# Architectures (verbatim from canonical scripts)
# ---------------------------------------------------------------------------
def create_fatcnn():
    return keras.Sequential([
        keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu',
                            input_shape=(32, 32, 3), name='conv1'),
        keras.layers.MaxPooling2D((2, 2), name='pool1'),
        keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2'),
        keras.layers.MaxPooling2D((2, 2), name='pool2'),
        keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3'),
        keras.layers.GlobalAveragePooling2D(name='gap'),
        keras.layers.Dense(128, activation='relu', name='dense1'),
        keras.layers.Dense(10, name='dense2'),
    ])


def create_fatcnn_lite():
    return keras.Sequential([
        keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu',
                            input_shape=(32, 32, 3), name='conv1'),
        keras.layers.MaxPooling2D((2, 2), name='pool1'),
        keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2'),
        keras.layers.MaxPooling2D((2, 2), name='pool2'),
        keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3'),
        keras.layers.GlobalAveragePooling2D(name='gap'),
        keras.layers.Dense(64, activation='relu', name='dense1'),
        keras.layers.Dense(10, name='dense2'),
    ])


ARCHS = {"fatcnn": create_fatcnn, "fatcnn_lite": create_fatcnn_lite}


def set_all_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return x_train, y_train, x_test, y_test


def eval_acc(model, x, y):
    logits = model.predict(x, batch_size=256, verbose=0)
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == y.reshape(-1)))


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def mean_sd(vals):
    a = np.asarray(vals, dtype=np.float64)
    m = float(np.mean(a))
    sd = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    return m, sd


def t_ci_95(vals):
    """95% t-interval for the mean of a small sample."""
    a = np.asarray(vals, dtype=np.float64)
    n = len(a)
    m = float(np.mean(a))
    if n < 2:
        return m, m, m
    sd = float(np.std(a, ddof=1))
    se = sd / np.sqrt(n)
    # two-sided 95% t critical values for small df
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(n - 1, 1.96)
    half = tcrit * se
    return m, m - half, m + half


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("=" * 64)
    print(f"  SEED SWEEP  (SMOKE={SMOKE}, EPOCHS={EPOCHS}, SEEDS={SEEDS})")
    print("=" * 64)

    x_train, y_train, x_test, y_test = load_data()
    x_test_1k, y_test_1k = x_test[:1000], y_test[:1000]

    results = {
        "meta": {
            "smoke": SMOKE,
            "epochs": EPOCHS,
            "seeds": SEEDS,
            "batch": BATCH,
            "lr": LR,
            "augmentation": "none (matches canonical train_fatcnn*.py and the shipped INT8 weights)",
            "full_test_set": 10000,
            "subset": "first 1000 test images (comparable to on-device INT8)",
            "ci_method": "two-sided 95% Student-t interval over 3 paired-by-seed gaps",
        },
        "per_arch": {},
        "gap": {},
    }

    out_path = os.path.join(RESULTS_DIR,
                            "seed_sweep_smoke.json" if SMOKE else "seed_sweep.json")

    per_seed_acc10k = {a: {} for a in ARCHS}
    per_seed_acc1k = {a: {} for a in ARCHS}

    for arch_name, builder in ARCHS.items():
        results["per_arch"][arch_name] = {"per_seed": {}}
        for seed in SEEDS:
            tag = f"{arch_name}_seed{seed}{'_smoke' if SMOKE else ''}"
            ckpt = os.path.join(CKPT_DIR, f"{tag}.keras")

            if os.path.exists(ckpt):
                print(f"\n[{tag}] resuming from checkpoint")
                model = keras.models.load_model(ckpt)
            else:
                print(f"\n[{tag}] training {EPOCHS} epochs ...")
                set_all_seeds(seed)
                model = builder()
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=LR),
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'],
                )
                # No augmentation — matches the canonical train_fatcnn*.py scripts
                # (epochs=30, plain fit on normalized data) that produced the
                # shipped INT8 weights and the paper's on-device accuracies.
                t0 = time.time()
                hist = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH,
                                 validation_data=(x_test, y_test), verbose=2)
                dt = time.time() - t0
                model.save(ckpt)
                print(f"[{tag}] trained in {dt:.1f}s ({dt / EPOCHS:.1f}s/epoch)")

            acc10k = eval_acc(model, x_test, y_test)
            acc1k = eval_acc(model, x_test_1k, y_test_1k)
            per_seed_acc10k[arch_name][seed] = acc10k
            per_seed_acc1k[arch_name][seed] = acc1k
            results["per_arch"][arch_name]["per_seed"][str(seed)] = {
                "acc_10k": acc10k,
                "acc_1k": acc1k,
            }
            print(f"[{tag}] acc_10k={acc10k:.4f}  acc_1k={acc1k:.4f}")

            # write partial results after every model so progress is durable
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

        m10, sd10 = mean_sd(list(per_seed_acc10k[arch_name].values()))
        m1, sd1 = mean_sd(list(per_seed_acc1k[arch_name].values()))
        results["per_arch"][arch_name]["acc_10k_mean"] = m10
        results["per_arch"][arch_name]["acc_10k_sd"] = sd10
        results["per_arch"][arch_name]["acc_1k_mean"] = m1
        results["per_arch"][arch_name]["acc_1k_sd"] = sd1
        print(f"[{arch_name}] 10k: {m10:.4f} +/- {sd10:.4f}   "
              f"1k: {m1:.4f} +/- {sd1:.4f}")

    # --- Paired gap (FatCNN - Lite) per seed ---
    for subset, table in (("10k", per_seed_acc10k), ("1k", per_seed_acc1k)):
        gaps = [table["fatcnn"][s] - table["fatcnn_lite"][s] for s in SEEDS]
        gm, gsd = mean_sd(gaps)
        _, lo, hi = t_ci_95(gaps)
        results["gap"][subset] = {
            "per_seed_gap": {str(s): table["fatcnn"][s] - table["fatcnn_lite"][s]
                             for s in SEEDS},
            "gap_mean": gm,
            "gap_sd": gsd,
            "gap_ci95_low": lo,
            "gap_ci95_high": hi,
        }
        print(f"\nGAP ({subset}) FatCNN-Lite paired by seed: "
              f"{gm * 100:.2f} pp +/- {gsd * 100:.2f} pp   "
              f"95% CI [{lo * 100:.2f}, {hi * 100:.2f}] pp")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
