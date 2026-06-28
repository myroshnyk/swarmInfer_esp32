"""
Phase 2: INT8 quantization for the scaled MobileNet (R2-11).

Pipeline:
  1. Load the trained float keras model.
  2. Fold each BatchNorm into its preceding convolution (conv0, depthwise,
     pointwise) -> a BN-free conv with folded weights+bias.
  3. Calibrate per-tensor activation scales on a few hundred CIFAR-96 images.
  4. Quantize weights (per-tensor, or per-channel for depthwise via --pc-dw).
  5. Run a numpy INT8 reference (reuses ../int8_engine for pointwise=conv2d and
     dense; adds depthwise_int8) and report INT8 test accuracy.

The integer arithmetic matches common/tensor_ops.c (fixed-point requant), so the
numpy reference is the bit-exact target the C kernels (Phase 3) must reproduce.

Isolated: imports the shared engine read-only; writes nothing outside this dir.

Usage:
    conda activate swarm-ml
    python quantize_mbnet.py --calib 300 --eval 1000          # per-tensor
    python quantize_mbnet.py --calib 300 --eval 1000 --pc-dw  # per-channel DW
"""
import argparse
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import int8_engine as eng  # noqa: E402

RES = 96
EPS = 1e-3  # keras BN default epsilon


# ---------------------------------------------------------------------------
# BN folding
# ---------------------------------------------------------------------------
def fold_bn_into_conv(conv_w, bn_gamma, bn_beta, bn_mean, bn_var, depthwise):
    """Fold BN into a (bias-free) conv. Returns ESP-layout weights + bias.

    conv_w keras layout:
      standard:  [kH, kW, Cin, Cout]
      depthwise: [kH, kW, Cin, 1]
    Returns:
      w_esp: [Cout, kH, kW, Cin]  (depthwise: Cout==Cin, kC==1)
      b:     [Cout]   (folded bias)
    The BN per-channel factor s = gamma / sqrt(var + eps); folded_b = beta - mean*s.
    """
    s = bn_gamma / np.sqrt(bn_var + EPS)          # [C]
    if depthwise:
        # keras [kH,kW,C,1] -> scale along C (axis 2)
        w = conv_w[:, :, :, 0]                    # [kH,kW,C]
        w = w * s.reshape(1, 1, -1)               # scale per channel
        w_esp = np.transpose(w, (2, 0, 1))[:, :, :, None]  # [C,kH,kW,1]
    else:
        # keras [kH,kW,Cin,Cout] -> scale along Cout (axis 3)
        w = conv_w * s.reshape(1, 1, 1, -1)
        w_esp = np.transpose(w, (3, 0, 1, 2))     # [Cout,kH,kW,Cin]
    b = bn_beta - bn_mean * s                     # [Cout]
    return w_esp.astype(np.float64), b.astype(np.float64)


# ---------------------------------------------------------------------------
# Depthwise INT8 (numpy reference; matches the requant convention of the engine)
# ---------------------------------------------------------------------------
def depthwise_int8(input_data, kernel, bias, stride, padding,
                   input_zp, weight_zp, output_zp, multiplier, shift):
    """input_data int8 [H,W,C]; kernel int8 [C,kH,kW,1]; output channel c uses
    only input channel c. weight_zp/multiplier/shift scalar or [C]."""
    H, W, C = input_data.shape
    _, kH, kW, _ = kernel.shape
    if padding > 0:
        ip = np.full((H + 2 * padding, W + 2 * padding, C), input_zp, dtype=np.int8)
        ip[padding:padding + H, padding:padding + W, :] = input_data
    else:
        ip = input_data
    pH, pW = ip.shape[0], ip.shape[1]
    oH = (pH - kH) // stride + 1
    oW = (pW - kW) // stride + 1
    acc = np.zeros((oH, oW, C), dtype=np.int64)
    wz = np.asarray(weight_zp)
    for kh in range(kH):
        for kw in range(kW):
            patch = ip[kh:kh + stride * oH:stride, kw:kw + stride * oW:stride, :]
            patch = patch.astype(np.int64) - np.int64(input_zp)
            k = kernel[:, kh, kw, 0].astype(np.int64)  # [C]
            if wz.ndim == 0:
                k = k - np.int64(weight_zp)
            else:
                k = k - wz.astype(np.int64)
            acc += patch * k.reshape(1, 1, C)
    acc = acc + np.asarray(bias, dtype=np.int64).reshape(1, 1, C)
    mult = np.asarray(multiplier)
    sh = np.asarray(shift)
    if mult.ndim == 0:
        out = eng.requantize_vec(acc, multiplier, shift, output_zp)
    else:
        out = eng.requantize_vec(acc, mult.reshape(1, 1, C), sh.reshape(1, 1, C), output_zp)
    return out.reshape(oH, oW, C)


# ---------------------------------------------------------------------------
# Build the folded float graph + calibrate + quantize
# ---------------------------------------------------------------------------
BLOCKS = [  # (name, out_ch, stride)
    ("b1", 64, 1), ("b2", 128, 2), ("b3", 128, 1), ("b4", 256, 2),
    ("b5", 256, 1), ("b6", 512, 2), ("b7", 512, 1), ("b8", 1024, 2),
]


def get_bn(model, name):
    l = model.get_layer(name)
    g, b, m, v = l.get_weights()
    return g, b, m, v


def build_folded(model):
    """Return an ordered list of folded float layers describing the network."""
    layers = []
    # conv0: standard 3x3 s2, padding same -> pad 1
    w = model.get_layer("conv0").get_weights()[0]
    g, b, m, v = get_bn(model, "conv0_bn")
    we, be = fold_bn_into_conv(w, g, b, m, v, depthwise=False)
    layers.append(dict(type="conv", name="conv0", w=we, b=be, stride=2, pad=1))
    for name, out_ch, stride in BLOCKS:
        wdw = model.get_layer(f"{name}_dw").get_weights()[0]
        g, b, m, v = get_bn(model, f"{name}_dwbn")
        we, be = fold_bn_into_conv(wdw, g, b, m, v, depthwise=True)
        layers.append(dict(type="dw", name=f"{name}_dw", w=we, b=be, stride=stride, pad=1))
        wpw = model.get_layer(f"{name}_pw").get_weights()[0]
        g, b, m, v = get_bn(model, f"{name}_pwbn")
        we, be = fold_bn_into_conv(wpw, g, b, m, v, depthwise=False)
        layers.append(dict(type="pw", name=f"{name}_pw", w=we, b=be, stride=1, pad=0))
    # dense
    dw, db = model.get_layer("dense").get_weights()  # [1024,10],[10]
    layers.append(dict(type="dense", name="dense", w=dw.astype(np.float64),
                       b=db.astype(np.float64)))
    return layers


# ---------------------------------------------------------------------------
# Float forward over the folded graph (numpy) — validates BN folding and is the
# structure the INT8 path mirrors.
# ---------------------------------------------------------------------------
def same_pad(n, k, s):
    """TF 'SAME' padding for one spatial dim: returns (before, after).
    out = ceil(n/s); total = max((out-1)*s + k - n, 0); before = total//2."""
    out = -(-n // s)  # ceil
    total = max((out - 1) * s + k - n, 0)
    return total // 2, total - total // 2


def _pad_same(x, kH, kW, stride, val=0.0):
    H, W, C = x.shape
    pt, pb = same_pad(H, kH, stride)
    pl, pr = same_pad(W, kW, stride)
    if pt == pb == pl == pr == 0:
        return x
    o = np.full((H + pt + pb, W + pl + pr, C), val, dtype=x.dtype)
    o[pt:pt + H, pl:pl + W, :] = x
    return o


def conv_float(x, w_esp, b, stride):
    """x [H,W,Cin]; w_esp [Cout,kH,kW,Cin]. TF 'SAME' padding."""
    Cout, kH, kW, Cin = w_esp.shape
    xp = _pad_same(x, kH, kW, stride)
    pH, pW, _ = xp.shape
    oH, oW = (pH - kH) // stride + 1, (pW - kW) // stride + 1
    cols = np.empty((oH * oW, kH * kW * Cin))
    idx = 0
    for kh in range(kH):
        for kw in range(kW):
            patch = xp[kh:kh + stride * oH:stride, kw:kw + stride * oW:stride, :]
            cols[:, idx:idx + Cin] = patch.reshape(oH * oW, Cin)
            idx += Cin
    wf = w_esp.reshape(Cout, -1)
    out = cols @ wf.T + b.reshape(1, Cout)
    return out.reshape(oH, oW, Cout)


def dw_float(x, w_esp, b, stride):
    """x [H,W,C]; w_esp [C,kH,kW,1] depthwise. TF 'SAME' padding."""
    C, kH, kW, _ = w_esp.shape
    xp = _pad_same(x, kH, kW, stride)
    pH, pW, _ = xp.shape
    oH, oW = (pH - kH) // stride + 1, (pW - kW) // stride + 1
    out = np.zeros((oH, oW, C))
    for kh in range(kH):
        for kw in range(kW):
            patch = xp[kh:kh + stride * oH:stride, kw:kw + stride * oW:stride, :]
            out += patch * w_esp[:, kh, kw, 0].reshape(1, 1, C)
    return out + b.reshape(1, 1, C)


def forward_float(folded, x):
    """x [H,W,3] float -> logits[10]. Captures post-ReLU activations for calib."""
    acts = {"input": x}
    h = x
    for L in folded:
        if L["type"] == "conv":
            h = np.maximum(conv_float(h, L["w"], L["b"], L["stride"]), 0.0)
        elif L["type"] == "dw":
            h = np.maximum(dw_float(h, L["w"], L["b"], L["stride"]), 0.0)
        elif L["type"] == "pw":
            h = np.maximum(conv_float(h, L["w"], L["b"], L["stride"]), 0.0)
        elif L["type"] == "dense":
            g = h.reshape(-1, h.shape[-1]).mean(axis=0)  # GAP
            acts["gap"] = g
            return g @ L["w"] + L["b"]
        acts[L["name"]] = h
    return None


# ---------------------------------------------------------------------------
# Calibration + INT8 quantization + INT8 forward
# ---------------------------------------------------------------------------
def act_chain(folded):
    """Yield (layer, input_act_key, output_act_key) in execution order."""
    prev = "input"
    for L in folded:
        if L["type"] == "dense":
            yield L, "gap", "logits"
        else:
            yield L, prev, L["name"]
            prev = L["name"]


def calibrate(folded, imgs):
    """Per-tensor activation scale/zp from observed min/max over calib images."""
    ranges = {}
    for x in imgs:
        acts = {"input": x}
        h = x
        for L in folded:
            if L["type"] == "dw":
                h = np.maximum(dw_float(h, L["w"], L["b"], L["stride"]), 0.0)
            elif L["type"] in ("conv", "pw"):
                h = np.maximum(conv_float(h, L["w"], L["b"], L["stride"]), 0.0)
            elif L["type"] == "dense":
                g = h.reshape(-1, h.shape[-1]).mean(axis=0)
                acts["gap"] = g
                break
            acts[L["name"]] = h
        for k, v in acts.items():
            lo, hi = float(v.min()), float(v.max())
            if k in ranges:
                ranges[k] = (min(ranges[k][0], lo), max(ranges[k][1], hi))
            else:
                ranges[k] = (lo, hi)
    aq = {}
    for k, (lo, hi) in ranges.items():
        scale = (hi - lo) / 255.0 or 1e-8
        zp = int(np.clip(round(-128 - lo / scale), -128, 127))
        aq[k] = (scale, zp)
    return aq


def quantize_model(folded, aq, pc_dw):
    """Quantize weights+bias and precompute requant per layer."""
    q = []
    for L, ik, ok in act_chain(folded):
        in_s, in_zp = aq[ik]
        out_s, out_zp = aq[ok] if ok in aq else (1.0, 0)
        if L["type"] == "dense":
            w = L["w"]  # [in,out] float
            w_s, w_zp, *_ = eng.quantize_tensor_params(w)
            qw = eng.quantize_tensor(w, w_s, w_zp)
            mult, sh = eng.compute_requant_multiplier(in_s, w_s, out_s)
            qb = np.round(L["b"] / (in_s * w_s)).astype(np.int64)
            q.append(dict(type="dense", name="dense", w=qw, b=qb, in_zp=in_zp,
                          w_zp=w_zp, out_zp=out_zp, mult=mult, sh=sh,
                          in_s=in_s, w_s=w_s, out_s=out_s))
            continue
        w = L["w"]  # [Cout,kH,kW,Cin] or [C,kH,kW,1]
        if L["type"] == "dw" and pc_dw:
            w_s, w_zp = eng.quantize_per_channel_params(w, axis=0)
            qw = eng.quantize_per_channel(w, w_s, w_zp)
            mult, sh = eng.compute_requant_multiplier(in_s, w_s, out_s)
            qb = np.round(L["b"] / (in_s * w_s)).astype(np.int64)
        else:
            w_s, w_zp, *_ = eng.quantize_tensor_params(w)
            qw = eng.quantize_tensor(w, w_s, w_zp)
            mult, sh = eng.compute_requant_multiplier(in_s, w_s, out_s)
            qb = np.round(L["b"] / (in_s * w_s)).astype(np.int64)
        q.append(dict(type=L["type"], name=L["name"], w=qw, b=qb, stride=L["stride"],
                      in_zp=in_zp, w_zp=w_zp, out_zp=out_zp, mult=mult, sh=sh,
                      in_s=in_s, w_s=w_s, out_s=out_s))
    return q


def _pad_asym(h, kH, kW, stride, in_zp):
    """Apply TF-SAME asymmetric padding with int8 zero-point fill (then the
    engine runs with padding=0)."""
    pt, pb = same_pad(h.shape[0], kH, stride)
    pl, pr = same_pad(h.shape[1], kW, stride)
    if (pt, pb, pl, pr) == (0, 0, 0, 0):
        return h
    H, W, C = h.shape
    hp = np.full((H + pt + pb, W + pl + pr, C), in_zp, np.int8)
    hp[pt:pt + H, pl:pl + W, :] = h
    return hp


def forward_int8(q, x_q, aq, collect=None):
    """x_q int8 [H,W,3]. q is the quantized layer list; aq has activation params.
    Returns predicted class index. If `collect` is a list, append per-layer
    (name, int64 output checksum) for host-validation."""
    h = x_q
    last_zp, last_s = None, None
    for L in q:
        if L["type"] in ("conv", "pw"):
            h2 = _pad_asym(h, L["w"].shape[1], L["w"].shape[2], L["stride"], L["in_zp"])
            h = eng.conv2d_int8(h2, L["w"], L["b"], L["stride"], 0,
                                L["in_zp"], L["w_zp"], L["out_zp"], L["mult"], L["sh"])
            h = eng.relu_int8(h, L["out_zp"])
            last_zp, last_s = L["out_zp"], L["out_s"]
        elif L["type"] == "dw":
            h2 = _pad_asym(h, L["w"].shape[1], L["w"].shape[2], L["stride"], L["in_zp"])
            h = depthwise_int8(h2, L["w"], L["b"], L["stride"], 0,
                               L["in_zp"], L["w_zp"], L["out_zp"], L["mult"], L["sh"])
            h = eng.relu_int8(h, L["out_zp"])
            last_zp, last_s = L["out_zp"], L["out_s"]
        elif L["type"] == "dense":
            gap_s, gap_zp = aq["gap"]
            g = eng.gap_int8(h, last_zp, last_s, gap_s, gap_zp)   # int8 [C]
            if collect is not None:
                collect.append(("gap", int(np.sum(g.astype(np.int64)))))
            out = eng.dense_int8(g, L["w"], L["b"], L["in_zp"], L["w_zp"],
                                 L["out_zp"], L["mult"], L["sh"])
            if collect is not None:
                collect.append(("dense", int(np.sum(out.astype(np.int64)))))
            return int(np.argmax(out.astype(np.int32)))
        if collect is not None:
            collect.append((L["name"], int(np.sum(h.astype(np.int64)))))
    raise RuntimeError("no dense layer")


TYPE_ENUM = {"conv": "MB_CONV", "dw": "MB_DW", "pw": "MB_PW", "dense": "MB_DENSE"}


def _arr_i8(f, name, a):
    flat = a.astype(np.int8).flatten()
    f.write(f"static const int8_t {name}[{flat.size}] = {{\n")
    for i in range(0, flat.size, 20):
        f.write("  " + ",".join(str(int(v)) for v in flat[i:i + 20]) + ",\n")
    f.write("};\n")


def _arr_i32(f, name, a):
    flat = a.astype(np.int64).flatten()
    f.write(f"static const int32_t {name}[{flat.size}] = {{ "
            + ",".join(str(int(v)) for v in flat) + " };\n")


def export_c(q, aq, outdir, test_img_q, test_label, exp_pred, checksums):
    """Write mbnet_weights.h (weights + layer table) and mbnet_testvec.h."""
    wpath = os.path.join(outdir, "mbnet_weights.h")
    with open(wpath, "w") as f:
        f.write("// scaled MobileNet INT8 weights (auto-generated by quantize_mbnet.py)\n")
        f.write("#ifndef MBNET_WEIGHTS_H\n#define MBNET_WEIGHTS_H\n#include <stdint.h>\n")
        f.write('#include "mbnet_ops.h"\n\n')
        for i, L in enumerate(q):
            # C dense_int8 expects [out][in]; Keras/numpy dense weight is [in,out].
            w = L["w"].T if L["type"] == "dense" else L["w"]
            _arr_i8(f, f"L{i:02d}_w", w)
            _arr_i32(f, f"L{i:02d}_b", L["b"])
        f.write(f"\n#define MB_NUM_LAYERS {len(q)}\n")
        in_s, in_zp = aq["input"]
        f.write(f"#define MB_INPUT_SCALE {in_s:.8f}f\n#define MB_INPUT_ZP {in_zp}\n")
        # GAP params: input scale = last conv (pw) out scale; output = gap scale/zp.
        gap_s, gap_zp = aq["gap"]
        last_conv_out_s = q[-2]["out_s"]  # b8_pw
        f.write(f"#define MB_GAP_IN_SCALE {last_conv_out_s:.8f}f\n")
        f.write(f"#define MB_GAP_IN_ZP {q[-2]['out_zp']}\n")
        f.write(f"#define MB_GAP_SCALE {gap_s:.8f}f\n#define MB_GAP_ZP {gap_zp}\n\n")
        f.write("static const MbLayer mb_layers[MB_NUM_LAYERS] = {\n")
        for i, L in enumerate(q):
            if L["type"] == "dense":
                cin, cout = L["w"].shape
                kH = kW = 1; stride = 1
            else:
                cout, kH, kW, cin = L["w"].shape
                if L["type"] == "dw":
                    cout, cin = L["w"].shape[0], L["w"].shape[0]  # depthwise: C channels
                stride = L["stride"]
            f.write(f"  {{ {TYPE_ENUM[L['type']]}, {kH},{kW},{stride}, {cin},{cout}, "
                    f"L{i:02d}_w, L{i:02d}_b, {L['in_zp']},{L['w_zp']},{L['out_zp']}, "
                    f"{L['mult']},{L['sh']} }},\n")
        f.write("};\n#endif\n")
    print(f"  wrote {wpath}")

    tpath = os.path.join(outdir, "mbnet_testvec.h")
    H, W, C = test_img_q.shape
    with open(tpath, "w") as f:
        f.write("// host-validation test vector (auto-generated)\n")
        f.write("#ifndef MBNET_TESTVEC_H\n#define MBNET_TESTVEC_H\n#include <stdint.h>\n\n")
        f.write(f"#define MB_IN_H {H}\n#define MB_IN_W {W}\n#define MB_IN_C {C}\n")
        f.write(f"#define MB_TEST_LABEL {int(test_label)}\n#define MB_EXPECTED_PRED {int(exp_pred)}\n\n")
        _arr_i8(f, "mb_test_input", test_img_q)
        f.write(f"\n#define MB_NUM_CHECKSUMS {len(checksums)}\n")
        f.write("static const int64_t mb_checksums[MB_NUM_CHECKSUMS] = { "
                + ",".join(str(c) for _, c in checksums) + " };\n")
        f.write("static const char *mb_checksum_names[MB_NUM_CHECKSUMS] = { "
                + ",".join('"%s"' % n for n, _ in checksums) + " };\n")
        f.write("#endif\n")
    print(f"  wrote {tpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__),
                                                    "mbnet_float32.keras"))
    ap.add_argument("--eval", type=int, default=200)
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--pc-dw", action="store_true")
    ap.add_argument("--export", action="store_true",
                    help="write mbnet_weights.h + mbnet_testvec.h for host/firmware")
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model)
    (_, _), (xte, yte) = tf.keras.datasets.cifar10.load_data()
    yte = yte.flatten()

    def prep(imgs):
        t = tf.image.resize(tf.cast(imgs, tf.float32), [RES, RES], method="bilinear") / 255.0
        return t.numpy().astype(np.float64)

    folded = build_folded(model)
    print("Folded graph:", len(folded), "layers")

    # ---- VALIDATE BN FOLDING: folded-float logits must match keras logits ----
    n = min(args.eval, 200)
    xb = prep(xte[:n])
    keras_logits = model.predict(xb, verbose=0)
    fold_logits = np.stack([forward_float(folded, xb[i]) for i in range(n)])
    max_abs = float(np.max(np.abs(keras_logits - fold_logits)))
    keras_acc = float(np.mean(np.argmax(keras_logits, 1) == yte[:n])) * 100
    fold_acc = float(np.mean(np.argmax(fold_logits, 1) == yte[:n])) * 100
    agree = float(np.mean(np.argmax(keras_logits, 1) == np.argmax(fold_logits, 1))) * 100
    print(f"BN-folding check (n={n}): keras_acc={keras_acc:.1f}%  "
          f"folded_acc={fold_acc:.1f}%  pred_agree={agree:.1f}%  "
          f"max|logit_diff|={max_abs:.4f}")
    if max_abs < 1e-2 and agree > 99.0:
        print("  OK: BN folding is correct (folded float == keras).")
    else:
        print("  WARN: folding mismatch — investigate before quantizing.")

    # ---- INT8 quantization + accuracy ----
    print(f"\nCalibrating on {args.calib} images...")
    aq = calibrate(folded, prep(xte[:args.calib]))
    q = quantize_model(folded, aq, args.pc_dw)
    in_s, in_zp = aq["input"]
    ne = args.eval
    xe = prep(xte[:ne])
    correct = 0
    for i in range(ne):
        xq = eng.quantize_tensor(xe[i], in_s, in_zp)
        correct += (forward_int8(q, xq, aq) == yte[i])
    int8_acc = correct / ne * 100
    kl = model.predict(xe, verbose=0)
    keras_e = float(np.mean(np.argmax(kl, 1) == yte[:ne])) * 100
    print(f"INT8 eval (n={ne}, pc_dw={args.pc_dw}): keras={keras_e:.1f}%  "
          f"int8={int8_acc:.1f}%  drop={keras_e - int8_acc:+.1f}pp")

    if args.export:
        print("\nExporting C weights + test vector...")
        img0 = prep(xte[:1])[0]
        xq0 = eng.quantize_tensor(img0, in_s, in_zp)
        chk = []
        pred0 = forward_int8(q, xq0, aq, collect=chk)
        export_c(q, aq, os.path.dirname(__file__), xq0, yte[0], pred0, chk)
        print(f"  test image 0: label={int(yte[0])} expected_pred={pred0} "
              f"({len(chk)} layer checksums)")


if __name__ == "__main__":
    main()
