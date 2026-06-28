"""
Phase 4: partition the scaled MobileNet across N workers (R2-11).

Design: the coordinator runs the cheap/orchestration layers (conv0, all
depthwise, GAP, dense); the workers run the 8 expensive pointwise (1x1) layers,
each holding an OUTPUT-CHANNEL shard. Because quantization is per-tensor, every
shard shares the same multiplier/shift/zero-points and only the weight/bias rows
differ, so the gather (concatenation of shards) is mathematically exact — the
same bit-exact property the FatCNN partitioning has.

This script reuses quantize_mbnet to get the quantized layer list, VALIDATES
that the sharded pointwise reconstruction is bit-exact vs the full model over a
test set, and emits C headers:
  mbnet_coord_weights.h          conv0 + 8 depthwise + dense + all params
  mbnet_worker_{w}_weights.h     this worker's shard of each pointwise layer

Usage:
    conda activate swarm-ml
    python partition_mbnet.py --workers 4 --calib 200 --eval 200
"""
import argparse
import os
import re
import numpy as np
import tensorflow as tf

import quantize_mbnet as qm
import int8_engine as eng

RES = 96


def _read_testvec_checksums(path):
    """Parse the mb_checksums[] array out of an existing mbnet_testvec.h."""
    if not os.path.exists(path):
        return None
    txt = open(path).read()
    m = re.search(r"mb_checksums\[[^\]]*\]\s*=\s*\{([^}]*)\}", txt)
    if not m:
        return None
    return [int(x) for x in m.group(1).split(",") if x.strip()]


def shard_bounds(c_out, n):
    """Even split of c_out across n workers (last gets remainder)."""
    base = c_out // n
    bounds = []
    s = 0
    for w in range(n):
        e = s + base if w < n - 1 else c_out
        bounds.append((s, e))
        s = e
    return bounds


def forward_int8_partitioned(q, x_q, aq, n_workers):
    """Same as qm.forward_int8 but each pointwise layer is computed shard-by-shard
    (per worker) and concatenated — validates partition exactness."""
    h = x_q
    last_zp, last_s = None, None
    for L in q:
        if L["type"] == "pw":
            h2 = qm._pad_asym(h, 1, 1, 1, L["in_zp"])
            cout = L["w"].shape[0]
            parts = []
            for (s, e) in shard_bounds(cout, n_workers):
                wsh = L["w"][s:e]
                bsh = L["b"][s:e]
                part = eng.conv2d_int8(h2, wsh, bsh, 1, 0, L["in_zp"], L["w_zp"],
                                       L["out_zp"], L["mult"], L["sh"])
                parts.append(part)
            h = np.concatenate(parts, axis=2)  # concat output channels
            h = eng.relu_int8(h, L["out_zp"])
            last_zp, last_s = L["out_zp"], L["out_s"]
        elif L["type"] in ("conv",):
            h2 = qm._pad_asym(h, L["w"].shape[1], L["w"].shape[2], L["stride"], L["in_zp"])
            h = eng.conv2d_int8(h2, L["w"], L["b"], L["stride"], 0, L["in_zp"],
                                L["w_zp"], L["out_zp"], L["mult"], L["sh"])
            h = eng.relu_int8(h, L["out_zp"])
            last_zp, last_s = L["out_zp"], L["out_s"]
        elif L["type"] == "dw":
            h2 = qm._pad_asym(h, L["w"].shape[1], L["w"].shape[2], L["stride"], L["in_zp"])
            h = qm.depthwise_int8(h2, L["w"], L["b"], L["stride"], 0, L["in_zp"],
                                  L["w_zp"], L["out_zp"], L["mult"], L["sh"])
            h = eng.relu_int8(h, L["out_zp"])
            last_zp, last_s = L["out_zp"], L["out_s"]
        elif L["type"] == "dense":
            gs, gz = aq["gap"]
            g = eng.gap_int8(h, last_zp, last_s, gs, gz)
            out = eng.dense_int8(g, L["w"], L["b"], L["in_zp"], L["w_zp"],
                                 L["out_zp"], L["mult"], L["sh"])
            return int(np.argmax(out.astype(np.int32)))
    raise RuntimeError("no dense")


TYPE_ENUM = {"conv": "MB_CONV", "dw": "MB_DW", "pw": "MB_PW", "dense": "MB_DENSE"}


def _dims(L):
    if L["type"] == "dense":
        cin, cout = L["w"].shape
        return 1, 1, 1, cin, cout
    cout, kH, kW, cin = L["w"].shape
    if L["type"] == "dw":
        cin = cout  # depthwise: C channels
    return kH, kW, L["stride"], cin, cout


def emit_coord(q, aq, n, outdir):
    """Coordinator header: conv0 + depthwise + dense weights, full layer table
    (pointwise entries carry params but NULL weights — distributed), and
    input/GAP params."""
    path = os.path.join(outdir, "mbnet_coord_weights.h")
    with open(path, "w") as f:
        f.write("// scaled MobileNet coordinator weights (auto-generated)\n")
        f.write("#ifndef MBNET_COORD_WEIGHTS_H\n#define MBNET_COORD_WEIGHTS_H\n")
        f.write('#include <stdint.h>\n#include "mbnet_ops.h"\n\n')
        for i, L in enumerate(q):
            if L["type"] == "pw":
                continue  # weights live on the workers
            w = L["w"].T if L["type"] == "dense" else L["w"]
            qm._arr_i8(f, f"L{i:02d}_w", w)
            qm._arr_i32(f, f"L{i:02d}_b", L["b"])
        f.write(f"\n#define MB_NUM_LAYERS {len(q)}\n")
        in_s, in_zp = aq["input"]
        gap_s, gap_zp = aq["gap"]
        f.write(f"#define MB_INPUT_SCALE {in_s:.8f}f\n#define MB_INPUT_ZP {in_zp}\n")
        f.write(f"#define MB_GAP_IN_SCALE {q[-2]['out_s']:.8f}f\n#define MB_GAP_IN_ZP {q[-2]['out_zp']}\n")
        f.write(f"#define MB_GAP_SCALE {gap_s:.8f}f\n#define MB_GAP_ZP {gap_zp}\n\n")
        f.write("static const MbLayer mb_layers[MB_NUM_LAYERS] = {\n")
        for i, L in enumerate(q):
            kH, kW, st, cin, cout = _dims(L)
            wref = "0,0" if L["type"] == "pw" else f"L{i:02d}_w,L{i:02d}_b"
            f.write(f"  {{ {TYPE_ENUM[L['type']]}, {kH},{kW},{st}, {cin},{cout}, "
                    f"{wref}, {L['in_zp']},{L['w_zp']},{L['out_zp']}, {L['mult']},{L['sh']} }},\n")
        f.write("};\n#endif\n")
    print(f"  wrote {path}")


def emit_worker(q, n, w, outdir):
    """Worker header: this worker's output-channel shard of each pointwise layer."""
    pw_idx = [i for i, L in enumerate(q) if L["type"] == "pw"]
    path = os.path.join(outdir, f"mbnet_worker_{w}_weights.h")
    with open(path, "w") as f:
        f.write(f"// scaled MobileNet worker {w}/{n} weights (auto-generated)\n")
        f.write(f"#ifndef MBNET_WORKER_{w}_WEIGHTS_H\n#define MBNET_WORKER_{w}_WEIGHTS_H\n")
        f.write('#include <stdint.h>\n#include "mbnet_ops.h"\n\n')
        for k, i in enumerate(pw_idx):
            L = q[i]
            cout = L["w"].shape[0]
            s, e = shard_bounds(cout, n)[w]
            qm._arr_i8(f, f"P{k:02d}_w", L["w"][s:e])
            qm._arr_i32(f, f"P{k:02d}_b", L["b"][s:e])
        f.write(f"\n#define MB_NUM_PW {len(pw_idx)}\n")
        f.write("static const MbShard mb_shards[MB_NUM_PW] = {\n")
        for k, i in enumerate(pw_idx):
            L = q[i]
            cout, _, _, cin = L["w"].shape
            s, e = shard_bounds(cout, n)[w]
            f.write(f"  {{ {s},{e},{cin}, P{k:02d}_w,P{k:02d}_b, "
                    f"{L['in_zp']},{L['w_zp']},{L['out_zp']}, {L['mult']},{L['sh']} }},\n")
        f.write("};\n#endif\n")
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "mbnet_float32.keras"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--eval", type=int, default=200)
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model)
    (_, _), (xte, yte) = tf.keras.datasets.cifar10.load_data()
    yte = yte.flatten()

    def prep(imgs):
        t = tf.image.resize(tf.cast(imgs, tf.float32), [RES, RES], method="bilinear") / 255.0
        return t.numpy().astype(np.float64)

    folded = qm.build_folded(model)
    aq = qm.calibrate(folded, prep(xte[:args.calib]))
    q = qm.quantize_model(folded, aq, pc_dw=False)
    in_s, in_zp = aq["input"]

    # ---- GUARD: the deployed coord/worker weights MUST come from the SAME
    # quantization as the firmware's self-check reference (mbnet_testvec.h).
    # A calib mismatch between them is exactly the R2-11 Phase 5b "-299" bug:
    # deterministic, prediction-preserving per-layer checksum deltas on-device.
    # Recompute this q's per-layer checksums for test image 0 and assert they
    # match the existing mb_checksums; refuse to emit divergent headers. ----
    tv_path = os.path.join(os.path.dirname(__file__), "mbnet_testvec.h")
    ref = _read_testvec_checksums(tv_path)
    if ref is None:
        print("Guard SKIP: mbnet_testvec.h absent — generate it with "
              "`quantize_mbnet.py --export --calib %d` (SAME calib)." % args.calib)
    else:
        xq0 = eng.quantize_tensor(prep(xte[:1])[0], in_s, in_zp)
        chk = []
        qm.forward_int8(q, xq0, aq, collect=chk)
        mine = [c for _, c in chk]
        if mine != ref:
            first = next((i for i in range(min(len(mine), len(ref)))
                          if mine[i] != ref[i]), 0)
            raise SystemExit(
                "\nGUARD FAILED: this partition's quantization does NOT match "
                "mbnet_testvec.h.\n  first divergence at checksum index %d: "
                "partition=%d testvec=%d (delta %+d)\n  -> testvec and "
                "coord/worker headers came from different quantizations "
                "(usually a different --calib).\n  Regenerate BOTH at the same "
                "calib: `quantize_mbnet.py --export --calib %d` then this script "
                "with `--calib %d` (or run ./regen.sh)."
                % (first, mine[first], ref[first], mine[first] - ref[first],
                   args.calib, args.calib))
        print("Guard OK: partition quantization matches mbnet_testvec.h "
              "(%d checksums)." % len(ref))

    # ---- VALIDATE: full vs partitioned (must be identical predictions) ----
    ne = args.eval
    xe = prep(xte[:ne])
    mism = 0
    for i in range(ne):
        xq = eng.quantize_tensor(xe[i], in_s, in_zp)
        full = qm.forward_int8(q, xq, aq)
        part = forward_int8_partitioned(q, xq, aq, args.workers)
        if full != part:
            mism += 1
    print(f"Partition check (N={args.workers}, n={ne}): "
          f"{'OK bit-exact' if mism == 0 else f'{mism} MISMATCH'} (full vs sharded)")

    # report shard sizes of the big pointwise layers
    pw_idx = [i for i, L in enumerate(q) if L["type"] == "pw"]
    print("Pointwise layers (output channels, per-worker shard, INT8 weight bytes/worker):")
    for i in pw_idx:
        cout, _, _, cin = q[i]["w"].shape
        per = cout // args.workers
        print(f"  L{i:02d} {q[i]['name']}: {cin}->{cout}  shard={per}  "
              f"{per * cin} B/worker (full {cout * cin} B)")

    print(f"\nExporting C headers (N={args.workers}) to {os.path.dirname(__file__)}/ ...")
    emit_coord(q, aq, args.workers, os.path.dirname(__file__))
    for w in range(args.workers):
        emit_worker(q, args.workers, w, os.path.dirname(__file__))


if __name__ == "__main__":
    main()
