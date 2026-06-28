"""SwarmInfer / N-scaling + non-divisible-channels experiment (R2-7, R2-8).

ISOLATED experiment: extends FatCNN distribution to worker counts whose channel
counts are NOT divisible by N (e.g., N=3, N=5), using UNEVEN output-channel
partitioning (the first C_out mod N workers each receive one extra channel).
Does NOT touch the released even-split pipeline (partition_n.py, swarm_worker,
swarm_coordinator). Emits headers for the isolated swarm_*_nscale firmware.

Strategy for non-divisible C_out: uneven shards. Worker w gets
  size_w = C_out // N + (1 if w < C_out % N else 0)
contiguous output channels. Partitioning stays mathematically exact, so the
concatenated result is bit-identical to the single-node INT8 reference; the only
cost is load imbalance (the +1-channel workers do slightly more work).

Reuses the validated FatCNN INT8 calibration + numpy engine (quant_ablation.py,
int8_engine.py) so the partition is checked bit-exact on the host before any flash.

Usage:
    conda activate swarm-ml
    python partition_nscale.py --workers 5 --eval 200
    python partition_nscale.py --workers 3 --eval 200
"""
import argparse
import os

import numpy as np
from tensorflow import keras

import int8_engine as eng
from quant_ablation import calibrate, INPUT_SCALE, INPUT_ZP, MODELS, HERE


def split_channels(c_out, n):
    """Uneven contiguous split: first (c_out % n) workers get one extra channel."""
    base, rem = divmod(c_out, n)
    sizes = [base + (1 if w < rem else 0) for w in range(n)]
    offs = [sum(sizes[:w]) for w in range(n)]
    return sizes, offs


def conv_args(params, name):
    p = params[name]
    kq, kzp, bq, wscale = p["pt_kernel_q"], p["pt_kzp"], p["pt_bias_q"], p["pt_kscale"]
    mult, shift = eng.compute_requant_multiplier(p["input_scale"], wscale, p["oscale"])
    return p, kq, kzp, bq, mult, shift


def conv_full(params, name, x, stride, pad):
    p, kq, kzp, bq, m, s = conv_args(params, name)
    return eng.conv2d_int8(x, kq, bq, stride, pad, p["input_zp"], kzp, p["ozp"], m, s)


def conv_sharded(params, name, x, stride, pad, n):
    """Compute the conv as N uneven output-channel shards, then concatenate.

    Each shard uses the SAME per-tensor scales/zp; only the kernel/bias rows
    differ. Concatenation along the channel axis must reproduce conv_full exactly.
    """
    p, kq, kzp, bq, m, s = conv_args(params, name)
    c_out = kq.shape[0]
    sizes, offs = split_channels(c_out, n)
    shards = []
    for w in range(n):
        a, b = offs[w], offs[w] + sizes[w]
        if sizes[w] == 0:
            continue
        shard = eng.conv2d_int8(x, kq[a:b], bq[a:b], stride, pad,
                                p["input_zp"], kzp, p["ozp"], m, s)
        shards.append(shard)
    return np.concatenate(shards, axis=2)  # [H,W,C], channel axis


def forward(params, img_int8, n=None, check=None):
    """Full FatCNN INT8 forward. If n is set, each conv is also computed as N
    uneven shards and asserted bit-identical to the full conv (check=list to log)."""
    def step(name, x, stride, pad):
        full = conv_full(params, name, x, stride, pad)
        if n is not None:
            sh = conv_sharded(params, name, x, stride, pad, n)
            ok = np.array_equal(full, sh)
            if check is not None:
                check.append((name, ok))
            if not ok:
                raise AssertionError(f"{name}: sharded != full (N={n})")
        return full

    p1 = params["conv1"]
    out = step("conv1", img_int8, 1, 2)
    out = eng.relu_int8(out, p1["ozp"]); out = eng.maxpool2x2_int8(out)
    p2 = params["conv2"]
    out = step("conv2", out, 1, 1)
    out = eng.relu_int8(out, p2["ozp"]); out = eng.maxpool2x2_int8(out)
    p3 = params["conv3"]
    out = step("conv3", out, 1, 1)
    out = eng.relu_int8(out, p3["ozp"])
    gap = eng.gap_int8(out, p3["ozp"], p3["oscale"], p3["oscale"], p3["ozp"])
    pd1 = params["dense1"]
    d1 = eng.dense_int8(gap, pd1["pt_kernel_q"].T, pd1["pt_bias_q"], p3["ozp"],
                        pd1["pt_kzp"], pd1["ozp"],
                        *eng.compute_requant_multiplier(pd1["input_scale"], pd1["pt_kscale"], pd1["oscale"]))
    d1 = np.maximum(d1, np.int8(pd1["ozp"]))
    pd2 = params["dense2"]
    d2 = eng.dense_int8(d1, pd2["pt_kernel_q"].T, pd2["pt_bias_q"], pd1["ozp"],
                        pd2["pt_kzp"], pd2["ozp"],
                        *eng.compute_requant_multiplier(pd2["input_scale"], pd2["pt_kscale"], pd2["oscale"]))
    return int(np.argmax(d2))


def emit_headers(params, n, out_dir):
    """Emit isolated-firmware headers: per-worker weights (+ MY_L*_OC), coordinator
    (dense + per-worker OC arrays), swarm_nscale_dims.h (spatial dims)."""
    os.makedirs(out_dir, exist_ok=True)
    conv_names = ["conv1", "conv2", "conv3"]
    sizes_by_layer = {}
    offs_by_layer = {}
    for name in conv_names:
        c_out = params[name]["pt_kernel_q"].shape[0]
        sizes_by_layer[name], offs_by_layer[name] = split_channels(c_out, n)

    def fmt_i8(arr):
        out = []
        for i in range(0, len(arr), 16):
            out.append("    " + ", ".join(str(int(v)) for v in arr[i:i+16]) + ",")
        return "\n".join(out)

    # per-worker weight headers
    for w in range(n):
        path = os.path.join(out_dir, f"worker_{w}_weights.h")
        with open(path, "w") as f:
            f.write(f"// SwarmInfer-nscale Worker {w} weights (N={n}, uneven). Auto-generated.\n")
            f.write(f"#ifndef WORKER_{w}_WEIGHTS_H\n#define WORKER_{w}_WEIGHTS_H\n#include <stdint.h>\n\n")
            for li, name in enumerate(conv_names, 1):
                p = params[name]
                a = offs_by_layer[name][w]; sz = sizes_by_layer[name][w]
                b = a + sz
                kq = p["pt_kernel_q"][a:b].flatten()       # [C_out,kH,kW,Cin] sliced
                bias = p["pt_bias_q"][a:b].astype(np.int32)
                f.write(f"// {name}: channels [{a}..{b}) = {sz} of {p['pt_kernel_q'].shape[0]}\n")
                f.write(f"#define MY_L{li}_OC {sz}\n")
                f.write(f"static const int8_t {name}_w{w}_kernel[] = {{\n{fmt_i8(kq)}\n}};\n")
                f.write(f"static const int32_t {name}_w{w}_bias[] = {{ {', '.join(str(int(v)) for v in bias)} }};\n")
                f.write(f"#define {name.upper()}_W{w}_KSCALE {p['pt_kscale']:.8f}f\n")
                f.write(f"#define {name.upper()}_W{w}_KZP {p['pt_kzp']}\n")
                f.write(f"#define {name.upper()}_W{w}_OSCALE {p['oscale']:.8f}f\n")
                f.write(f"#define {name.upper()}_W{w}_OZP {p['ozp']}\n\n")
            f.write("#endif\n")

    # coordinator header (dense weights + per-worker OC arrays + conv3 out params)
    path = os.path.join(out_dir, "coordinator_weights.h")
    with open(path, "w") as f:
        f.write(f"// SwarmInfer-nscale Coordinator weights (N={n}, uneven). Auto-generated.\n")
        f.write("#ifndef COORDINATOR_WEIGHTS_H\n#define COORDINATOR_WEIGHTS_H\n#include <stdint.h>\n\n")
        f.write(f"#define INPUT_SCALE {INPUT_SCALE:.8f}f\n#define INPUT_ZP {INPUT_ZP}\n\n")
        for li, name in enumerate(conv_names, 1):
            f.write(f"static const int L{li}_OC[{n}] = {{ {', '.join(str(s) for s in sizes_by_layer[name])} }};\n")
            f.write(f"static const int L{li}_OFF[{n}] = {{ {', '.join(str(o) for o in offs_by_layer[name])} }};\n")
        p3 = params["conv3"]
        f.write(f"\n#define CONV3_OSCALE {p3['oscale']:.8f}f\n#define CONV3_OZP {p3['ozp']}\n\n")
        for name in ["dense1", "dense2"]:
            p = params[name]
            # calibrate already stores dense kernel as [out,in] (kernel_esp = kernel_tf.T),
            # which is exactly the firmware dense_int8 layout (w[j*in_len + i]). Emit as-is.
            k = p["pt_kernel_q"].flatten()
            bias = p["pt_bias_q"].astype(np.int32)
            f.write(f"static const int8_t {name}_kernel[] = {{\n{fmt_i8(k)}\n}};\n")
            f.write(f"static const int32_t {name}_bias[] = {{ {', '.join(str(int(v)) for v in bias)} }};\n")
            f.write(f"#define {name.upper()}_KSCALE {p['pt_kscale']:.8f}f\n")
            f.write(f"#define {name.upper()}_KZP {p['pt_kzp']}\n")
            f.write(f"#define {name.upper()}_OSCALE {p['oscale']:.8f}f\n")
            f.write(f"#define {name.upper()}_OZP {p['ozp']}\n\n")
        f.write("#endif\n")
    return sizes_by_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--eval", type=int, default=200)
    ap.add_argument("--calib", type=int, default=1000)
    ap.add_argument("--emit", action="store_true", help="write firmware headers")
    args = ap.parse_args()
    n = args.workers

    model = keras.models.load_model(os.path.join(HERE, MODELS["fatcnn"]))
    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0
    y_test = y_test.flatten()

    params, _ = calibrate(model, x_test[:args.calib])

    # uneven split summary
    print(f"=== Uneven partition for N={n} ===")
    for name in ["conv1", "conv2", "conv3"]:
        c_out = params[name]["pt_kernel_q"].shape[0]
        sizes, offs = split_channels(c_out, n)
        print(f"  {name}: C_out={c_out} -> shards {sizes} (max {max(sizes)}, "
              f"imbalance {max(sizes)/(c_out/n):.3f}x)")

    # host bit-exact: full vs uneven-sharded, per conv layer + prediction
    n_ok = 0
    layer_ok = {"conv1": True, "conv2": True, "conv3": True}
    for i in range(args.eval):
        img = eng.quantize_tensor(x_test[i], INPUT_SCALE, INPUT_ZP)
        check = []
        pred = forward(params, img, n=n, check=check)
        for name, ok in check:
            layer_ok[name] = layer_ok[name] and ok
        if pred == int(y_test[i]):
            n_ok += 1
    print(f"\nHost bit-exact (full vs uneven-sharded), {args.eval} imgs:")
    for name in ["conv1", "conv2", "conv3"]:
        print(f"  {name}: {'BIT-EXACT' if layer_ok[name] else 'MISMATCH'}")
    print(f"INT8 accuracy on {args.eval} imgs: {100*n_ok/args.eval:.1f}%")

    if args.emit:
        out_dir = os.path.join(HERE, f"c_weights_nscale_n{n}")
        emit_headers(params, n, out_dir)
        print(f"\nEmitted firmware headers -> {out_dir}/")


if __name__ == "__main__":
    main()
