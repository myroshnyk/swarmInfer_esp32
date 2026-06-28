"""R2-6 (part A): distribution of post-ReLU activation sparsity across layers
and across images, for the FatCNN gather payloads (what bitmap sparsification
encodes). Reuses the validated INT8 calibration + engine.

Sparsity of a layer = fraction of its transmitted output equal to the ReLU
zero-point (i.e., zeros). The worker sends the POOLED output (Conv1/Conv2 after
2x2 maxpool) or the GAP vector (Conv3), so we measure sparsity on exactly those
tensors. Writes results/sparsity_dist.{json,md} and a figure
paper/figures/fig_sparsity.pdf.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

import int8_engine as eng
from quant_ablation import calibrate, INPUT_SCALE, INPUT_ZP, MODELS, HERE

REPO = os.path.join(HERE, "..")
N_IMAGES = 1000


def layer_outputs(params, img_int8):
    """Return the three gather-payload tensors (Conv1 pool, Conv2 pool, Conv3 GAP)
    and each layer's ReLU zero-point, exactly as the worker would send them."""
    def conv(name, x, stride, pad):
        p = params[name]
        m, s = eng.compute_requant_multiplier(p["input_scale"], p["pt_kscale"], p["oscale"])
        return eng.conv2d_int8(x, p["pt_kernel_q"], p["pt_bias_q"], stride, pad,
                               p["input_zp"], p["pt_kzp"], p["ozp"], m, s)

    o1 = eng.maxpool2x2_int8(eng.relu_int8(conv("conv1", img_int8, 1, 2), params["conv1"]["ozp"]))
    o2 = eng.maxpool2x2_int8(eng.relu_int8(conv("conv2", o1, 1, 1), params["conv2"]["ozp"]))
    c3 = eng.relu_int8(conv("conv3", o2, 1, 1), params["conv3"]["ozp"])
    p3 = params["conv3"]
    gap = eng.gap_int8(c3, p3["ozp"], p3["oscale"], p3["oscale"], p3["ozp"])
    return [(o1, params["conv1"]["ozp"]), (o2, params["conv2"]["ozp"]), (gap, p3["ozp"])]


def main():
    model = keras.models.load_model(os.path.join(HERE, MODELS["fatcnn"]))
    (_, _), (x_test, _) = keras.datasets.cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0
    params, _ = calibrate(model, x_test[:1000])

    names = ["Conv1 (pool)", "Conv2 (pool)", "Conv3 (GAP)"]
    per_image = [[], [], []]   # sparsity fraction per image, per layer
    for i in range(N_IMAGES):
        img = eng.quantize_tensor(x_test[i], INPUT_SCALE, INPUT_ZP)
        for li, (out, zp) in enumerate(layer_outputs(params, img)):
            zeros = int(np.sum(out == np.int8(zp)))
            per_image[li].append(zeros / out.size)

    stats = {}
    for li, name in enumerate(names):
        a = np.array(per_image[li])
        stats[name] = {
            "mean": round(float(a.mean()) * 100, 1),
            "std": round(float(a.std()) * 100, 1),
            "min": round(float(a.min()) * 100, 1),
            "max": round(float(a.max()) * 100, 1),
            "size_elems": int(layer_outputs(params, eng.quantize_tensor(x_test[0], INPUT_SCALE, INPUT_ZP))[li][0].size),
        }

    json.dump({"n_images": N_IMAGES, "per_layer": stats},
              open(os.path.join(REPO, "results", "sparsity_dist.json"), "w"), indent=2)

    # markdown
    md = ["# R2-6 (A): activation sparsity distribution (post-ReLU gather payloads)",
          f"\n{N_IMAGES} CIFAR-10 test images. Sparsity = fraction of the transmitted "
          "tensor equal to the ReLU zero-point.\n",
          "| Layer | Output elems | Mean sparsity | Std | Min | Max |",
          "|---|---|---|---|---|---|"]
    for name in names:
        s = stats[name]
        md.append(f"| {name} | {s['size_elems']} | {s['mean']}% | {s['std']}% | {s['min']}% | {s['max']}% |")
    open(os.path.join(REPO, "results", "sparsity_dist.md"), "w").write("\n".join(md) + "\n")

    # figure: box/violin of per-image sparsity per layer
    COL = 3.358
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.labelsize": 9, "figure.dpi": 600, "savefig.dpi": 600})
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    data = [100 * np.array(per_image[li]) for li in range(3)]
    bp = ax.boxplot(data, labels=names, showfliers=False, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], ["#4472C4", "#ED7D31", "#70AD47"]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_ylabel("Activation sparsity (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(REPO, "paper", "figures", "fig_sparsity.pdf"))
    print("\n".join(md))
    print("\nwrote results/sparsity_dist.{json,md} + paper/figures/fig_sparsity.pdf")


if __name__ == "__main__":
    main()
