"""
SwarmInfer — R2-5: Quantization ablation isolating quantization error.

For each architecture (FatCNN 64-128-256 and FatCNN-Lite 32-64-128) this
compares three inference modes on the CIFAR-10 test set, on both the
first-1,000-image subset (to match the on-device INT8 regime) and the full
10,000-image set:

    (a) float32                              (Keras, ground truth)
    (b) INT8 per-tensor                      (the current fix_quantize.py
                                              pipeline: one weight scale/zp per
                                              conv/dense layer)
    (c) INT8 per-channel weights             (per-output-channel symmetric
                                              weight scale; activations stay
                                              per-tensor)

The INT8 paths use int8_engine.py, which reproduces the exact integer
arithmetic of the ESP32 firmware (validated bit-exact against the scalar
reference in verify_int8.py). Activation ranges / output scales are calibrated
on the first 1,000 training-of-quantization images exactly as fix_quantize.py
does (x_test[:1000]).

It reports:
    - accuracy for (a),(b),(c) on 1k and 10k
    - deltas: float32 -> INT8 per-tensor, and per-tensor -> per-channel
    - per conv layer: observed real activation range (min/max) and weight range
    - per conv layer: INT32 accumulator SATURATION counts (how often the
      requantized int8 value clamps to +127 or -128), per-tensor vs per-channel

Outputs results/quant_ablation.json and results/quant_ablation.md.

Usage:
    python quant_ablation.py            # full: 1k + 10k, both archs
    python quant_ablation.py --smoke    # smoke: 40 images, validates pipeline
"""
import os
import sys
import json

import numpy as np
from tensorflow import keras

import int8_engine as eng

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(REPO, "results")

SMOKE = ("--smoke" in sys.argv) or (os.environ.get("QUANT_ABLATION_SMOKE") == "1")

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

INPUT_SCALE = 1.0 / 255.0
INPUT_ZP = -128

MODELS = {
    "fatcnn": "fatcnn_float32.keras",
    "fatcnn_lite": "fatcnn_lite_float32.keras",
}
CONV_NAMES = ["conv1", "conv2", "conv3"]


# ---------------------------------------------------------------------------
# Calibration: activation ranges + per-tensor / per-channel weight params
# ---------------------------------------------------------------------------
def calibrate(model, x_calib):
    """Return per-layer quant params, mirroring fix_quantize.py calibration."""
    # Activation ranges via sequential forward pass (matches fix_quantize.py)
    layer_ranges = {}
    x = x_calib
    for layer in model.layers:
        x = layer(x)
        if len(layer.get_weights()) > 0 or isinstance(layer, keras.layers.GlobalAveragePooling2D):
            arr = x.numpy() if hasattr(x, "numpy") else np.array(x)
            layer_ranges[layer.name] = (float(np.min(arr)), float(np.max(arr)))

    weights = {l.name: l.get_weights() for l in model.layers if l.get_weights()}

    params = {}
    prev_oscale = INPUT_SCALE
    prev_ozp = INPUT_ZP
    for layer in model.layers:
        w = layer.get_weights()
        if not w:
            continue
        name = layer.name
        kernel_tf, bias = w[0], w[1]
        is_conv = isinstance(layer, keras.layers.Conv2D)

        # ESP weight layout
        if is_conv:
            kernel_esp = kernel_tf.transpose(3, 0, 1, 2)        # [C_out,kH,kW,C_in]
        else:
            kernel_esp = kernel_tf.T                            # [out,in]

        # output (activation) quant
        a_min, a_max = layer_ranges.get(name, (-1.0, 1.0))
        oscale = (a_max - a_min) / 255.0
        if oscale == 0:
            oscale = 1e-8
        ozp = int(np.round(-128 - a_min / oscale))
        ozp = max(-128, min(127, ozp))

        # per-tensor weight quant
        kscale, kzp, w_min, w_max = eng.quantize_tensor_params(kernel_esp)
        kernel_q_pt = eng.quantize_tensor(kernel_esp, kscale, kzp)

        # per-channel weight quant (symmetric per out-channel)
        pc_scale, pc_zp = eng.quantize_per_channel_params(kernel_esp, axis=0)
        kernel_q_pc = eng.quantize_per_channel(kernel_esp, pc_scale, pc_zp)

        # int32 bias = bias / (input_scale * weight_scale)
        bias_pt = np.round(bias / (prev_oscale * kscale)).astype(np.int64)
        bias_pc = np.round(bias / (prev_oscale * pc_scale)).astype(np.int64)

        params[name] = {
            "is_conv": is_conv,
            "input_scale": prev_oscale,
            "input_zp": prev_ozp,
            "kernel_esp_f32": kernel_esp,
            "bias_f32": bias,
            "act_min": a_min, "act_max": a_max,
            "w_min": w_min, "w_max": w_max,
            "oscale": oscale, "ozp": ozp,
            # per-tensor
            "pt_kscale": kscale, "pt_kzp": kzp,
            "pt_kernel_q": kernel_q_pt, "pt_bias_q": bias_pt,
            # per-channel
            "pc_scale": pc_scale, "pc_zp": pc_zp,
            "pc_kernel_q": kernel_q_pc, "pc_bias_q": bias_pc,
        }
        prev_oscale, prev_ozp = oscale, ozp

    return params, layer_ranges


# ---------------------------------------------------------------------------
# INT8 forward pass (per-tensor or per-channel), with instrumentation
# ---------------------------------------------------------------------------
def int8_forward(img_int8, params, mode, instr):
    """mode in {'per_tensor','per_channel'}. instr: dict of per-layer dicts."""
    pt = (mode == "per_tensor")

    def conv_args(name):
        p = params[name]
        if pt:
            kq, kzp, bq, wscale = p["pt_kernel_q"], p["pt_kzp"], p["pt_bias_q"], p["pt_kscale"]
        else:
            kq, kzp, bq, wscale = p["pc_kernel_q"], p["pc_zp"], p["pc_bias_q"], p["pc_scale"]
        mult, shift = eng.compute_requant_multiplier(p["input_scale"], wscale, p["oscale"])
        return p, kq, kzp, bq, mult, shift

    # conv1: stride1 pad2 (5x5); conv2/conv3: stride1 pad1 (3x3)
    p1, k1, z1, b1, m1, s1 = conv_args("conv1")
    out = eng.conv2d_int8(img_int8, k1, b1, 1, 2, p1["input_zp"], z1,
                          p1["ozp"], m1, s1, instr=instr.setdefault("conv1", {}))
    out = eng.relu_int8(out, p1["ozp"])
    out = eng.maxpool2x2_int8(out)

    p2, k2, z2, b2, m2, s2 = conv_args("conv2")
    out = eng.conv2d_int8(out, k2, b2, 1, 1, p2["input_zp"], z2,
                          p2["ozp"], m2, s2, instr=instr.setdefault("conv2", {}))
    out = eng.relu_int8(out, p2["ozp"])
    out = eng.maxpool2x2_int8(out)

    p3, k3, z3, b3, m3, s3 = conv_args("conv3")
    out = eng.conv2d_int8(out, k3, b3, 1, 1, p3["input_zp"], z3,
                          p3["ozp"], m3, s3, instr=instr.setdefault("conv3", {}))
    out = eng.relu_int8(out, p3["ozp"])

    gap = eng.gap_int8(out, p3["ozp"], p3["oscale"], p3["oscale"], p3["ozp"])

    # dense1 (+relu), dense2
    pd1 = params["dense1"]
    if pt:
        d1 = eng.dense_int8(gap, pd1["pt_kernel_q"].T, pd1["pt_bias_q"],
                            p3["ozp"], pd1["pt_kzp"], pd1["ozp"],
                            *eng.compute_requant_multiplier(pd1["input_scale"], pd1["pt_kscale"], pd1["oscale"]))
    else:
        m, s = eng.compute_requant_multiplier(pd1["input_scale"], pd1["pc_scale"], pd1["oscale"])
        d1 = eng.dense_int8(gap, pd1["pc_kernel_q"].T, pd1["pc_bias_q"],
                            p3["ozp"], pd1["pc_zp"], pd1["ozp"], m, s)
    d1 = np.maximum(d1, np.int8(pd1["ozp"]))

    pd2 = params["dense2"]
    if pt:
        d2 = eng.dense_int8(d1, pd2["pt_kernel_q"].T, pd2["pt_bias_q"],
                            pd1["ozp"], pd2["pt_kzp"], pd2["ozp"],
                            *eng.compute_requant_multiplier(pd2["input_scale"], pd2["pt_kscale"], pd2["oscale"]))
    else:
        m, s = eng.compute_requant_multiplier(pd2["input_scale"], pd2["pc_scale"], pd2["oscale"])
        d2 = eng.dense_int8(d1, pd2["pc_kernel_q"].T, pd2["pc_bias_q"],
                            pd1["ozp"], pd2["pc_zp"], pd2["ozp"], m, s)
    return int(np.argmax(d2))


def validate_bit_exact(params):
    """Confirm vectorized conv1 == scalar reference on one random patch."""
    rng = np.random.default_rng(0)
    img = rng.integers(-128, 128, size=(8, 8, 3)).astype(np.int8)
    p = params["conv1"]
    k, kzp, bq, kscale = p["pt_kernel_q"], p["pt_kzp"], p["pt_bias_q"], p["pt_kscale"]
    m, s = eng.compute_requant_multiplier(p["input_scale"], kscale, p["oscale"])
    fast = eng.conv2d_int8(img, k, bq, 1, 2, p["input_zp"], kzp, p["ozp"], m, s)
    ref = eng.conv2d_int8_ref(img, k, bq, 1, 2, p["input_zp"], kzp, p["ozp"], m, s)
    ok = bool(np.array_equal(fast, ref))
    print(f"  bit-exact vectorized-vs-reference (conv1): {ok}")
    return ok


# ---------------------------------------------------------------------------
def run_arch(arch_name, x_test, y_test, n_eval):
    print(f"\n{'=' * 64}\n  {arch_name}  (evaluating {n_eval} images)\n{'=' * 64}")
    model = keras.models.load_model(os.path.join(HERE, MODELS[arch_name]))

    x_calib = x_test[:1000]
    params, layer_ranges = calibrate(model, x_calib)
    bit_exact = validate_bit_exact(params)

    # Pre-quantize int8 input images
    y = y_test.reshape(-1)

    res = {"bit_exact": bit_exact, "subsets": {}, "per_layer": {}}

    # accuracy across modes; we run 10k once and slice 1k from the same preds
    subsets = {"1k": 1000, "10k": n_eval}
    # float32 predictions (vectorized by Keras)
    logits = model.predict(x_test[:n_eval], batch_size=256, verbose=0)
    pred_f32 = np.argmax(logits, axis=1)

    # INT8 predictions per image (both modes), with per-layer instrumentation
    instr_pt = {}
    instr_pc = {}
    pred_pt = np.zeros(n_eval, dtype=np.int32)
    pred_pc = np.zeros(n_eval, dtype=np.int32)
    for i in range(n_eval):
        img_int8 = (x_test[i] * 255).astype(np.int16)
        img_int8 = np.clip(img_int8 - 128, -128, 127).astype(np.int8)
        pred_pt[i] = int8_forward(img_int8, params, "per_tensor", instr_pt)
        pred_pc[i] = int8_forward(img_int8, params, "per_channel", instr_pc)
        if (i + 1) % 200 == 0:
            print(f"    {arch_name}: {i + 1}/{n_eval} images int8-evaluated")

    for sub, nsub in subsets.items():
        nsub = min(nsub, n_eval)
        yy = y[:nsub]
        a_f32 = float(np.mean(pred_f32[:nsub] == yy))
        a_pt = float(np.mean(pred_pt[:nsub] == yy))
        a_pc = float(np.mean(pred_pc[:nsub] == yy))
        res["subsets"][sub] = {
            "n": nsub,
            "acc_float32": a_f32,
            "acc_int8_per_tensor": a_pt,
            "acc_int8_per_channel": a_pc,
            "delta_f32_to_pt": a_pt - a_f32,
            "delta_pt_to_pc": a_pc - a_pt,
        }
        print(f"  [{sub}] f32={a_f32:.4f}  int8/tensor={a_pt:.4f}  "
              f"int8/chan={a_pc:.4f}  d(f32->pt)={a_pt - a_f32:+.4f}  "
              f"d(pt->pc)={a_pc - a_pt:+.4f}")

    # per-layer ranges + saturation (over the full n_eval images)
    for name in CONV_NAMES:
        p = params[name]
        ip = instr_pt.get(name, {})
        ic = instr_pc.get(name, {})

        def sat(d):
            n = d.get("n_outputs", 0)
            hi = d.get("sat_hi", 0)
            lo = d.get("sat_lo", 0)
            return {
                "sat_hi_127": hi,
                "sat_lo_neg128": lo,
                "n_outputs": n,
                # True accumulator/requant saturation = clamp to the UPPER bound (+127).
                "sat_hi_frac": hi / n if n else 0.0,
                # Clamp to the LOWER bound (-128) is the post-ReLU zero point, i.e.
                # activation sparsity — NOT harmful saturation. Reported separately.
                "relu_zero_frac": lo / n if n else 0.0,
                "sat_frac": (hi + lo) / n if n else 0.0,  # legacy combined (do not use as "saturation")
            }

        res["per_layer"][name] = {
            "activation_real_range": [p["act_min"], p["act_max"]],
            "weight_real_range": [p["w_min"], p["w_max"]],
            "output_scale": p["oscale"],
            "output_zp": p["ozp"],
            "pt_weight_scale": p["pt_kscale"],
            "pt_weight_zp": p["pt_kzp"],
            "pc_weight_scale_min": float(np.min(p["pc_scale"])),
            "pc_weight_scale_max": float(np.max(p["pc_scale"])),
            "saturation_per_tensor": sat(ip),
            "saturation_per_channel": sat(ic),
        }
        pl = res["per_layer"][name]
        print(f"  {name}: act=[{p['act_min']:.3f},{p['act_max']:.3f}] "
              f"w=[{p['w_min']:.3f},{p['w_max']:.3f}]  "
              f"sat+127(pt)={pl['saturation_per_tensor']['sat_hi_frac'] * 100:.3f}%  "
              f"sat+127(pc)={pl['saturation_per_channel']['sat_hi_frac'] * 100:.3f}%  "
              f"(reluZero={pl['saturation_per_tensor']['relu_zero_frac'] * 100:.1f}%)")

    return res


def write_markdown(results, path):
    lines = []
    lines.append("# Quantization Ablation (R2-5)\n")
    lines.append(f"Smoke run: **{results['meta']['smoke']}**. "
                 f"Images evaluated per arch: **{results['meta']['n_eval']}**. "
                 "Activations per-tensor in all INT8 modes; weights per-tensor "
                 "vs per-output-channel.\n")
    lines.append("## Accuracy\n")
    lines.append("| Arch | Subset | float32 | INT8 per-tensor | INT8 per-channel | Δ f32→pt | Δ pt→pc |")
    lines.append("|------|--------|---------|-----------------|------------------|----------|---------|")
    for arch, r in results["arch"].items():
        for sub, s in r["subsets"].items():
            lines.append(f"| {arch} | {sub} | {s['acc_float32'] * 100:.1f}% | "
                         f"{s['acc_int8_per_tensor'] * 100:.1f}% | "
                         f"{s['acc_int8_per_channel'] * 100:.1f}% | "
                         f"{s['delta_f32_to_pt'] * 100:+.1f} pp | "
                         f"{s['delta_pt_to_pc'] * 100:+.1f} pp |")
    lines.append("\n## Per-layer ranges & accumulator saturation\n")
    lines.append("**Accumulator saturation** = fraction of requantized INT8 outputs that clamp to the "
                 "UPPER bound +127 (true requant/accumulator overflow). Clamps to the lower bound "
                 "-128 are the post-ReLU zero point (activation sparsity), not harmful saturation, "
                 "and are listed separately as *ReLU-zero %*.\n")
    lines.append("| Arch | Layer | Act range | Weight range | Sat→+127 (pt) | Sat→+127 (pc) | ReLU-zero % |")
    lines.append("|------|-------|-----------|--------------|---------------|---------------|-------------|")
    for arch, r in results["arch"].items():
        for name, pl in r["per_layer"].items():
            ar = pl["activation_real_range"]
            wr = pl["weight_real_range"]
            sp = pl["saturation_per_tensor"]["sat_hi_frac"]
            sc = pl["saturation_per_channel"]["sat_hi_frac"]
            rz = pl["saturation_per_tensor"]["relu_zero_frac"]
            lines.append(f"| {arch} | {name} | [{ar[0]:.2f}, {ar[1]:.2f}] | "
                         f"[{wr[0]:.3f}, {wr[1]:.3f}] | {sp * 100:.3f}% | {sc * 100:.3f}% | {rz * 100:.1f}% |")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0

    n_eval = 40 if SMOKE else 10000

    results = {
        "meta": {
            "smoke": SMOKE,
            "n_eval": n_eval,
            "calibration": "activation ranges from x_test[:1000] (matches fix_quantize.py)",
            "modes": ["float32", "int8_per_tensor", "int8_per_channel"],
            "per_channel": "symmetric per-output-channel weight scale; activations per-tensor",
            "note": "INT8 paths bit-exact with ESP32 firmware integer math",
        },
        "arch": {},
    }

    for arch_name in MODELS:
        results["arch"][arch_name] = run_arch(arch_name, x_test, y_test, n_eval)

    suffix = "_smoke" if SMOKE else ""
    json_path = os.path.join(RESULTS_DIR, f"quant_ablation{suffix}.json")
    md_path = os.path.join(RESULTS_DIR, f"quant_ablation{suffix}.md")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    write_markdown(results, md_path)
    print(f"\nWrote {json_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
