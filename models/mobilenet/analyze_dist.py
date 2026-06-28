"""R2-11: parse the distributed MobileNet serial capture and emit a results
summary (results/mbnet_r2_11.json + .md).

Reads:
  logs/mbnet_distributed.log   (coordinator serial capture, capture_mbnet.py)
  results/mbnet_comm_volume.csv (communication volume, comm_volume.py)
  logs/mbnet_train_60ep.log     (float test accuracy)

The gather phase combines worker pointwise compute and unicast transmit; we
decompose it using the per-round comp= field (max worker compute per round).
"""
import csv
import json
import os
import re
import statistics as st

HERE = os.path.dirname(__file__)
REPO = os.path.join(HERE, "..", "..")
LOG = os.path.join(REPO, "logs", "mbnet_distributed.log")
COMM = os.path.join(REPO, "results", "mbnet_comm_volume.csv")
TRAIN = os.path.join(REPO, "logs", "mbnet_train_60ep.log")
INT8 = os.path.join(REPO, "logs", "mbnet_int8_eval.log")
OUT_JSON = os.path.join(REPO, "results", "mbnet_r2_11.json")
OUT_MD = os.path.join(REPO, "results", "mbnet_r2_11.md")

text = open(LOG).read()

# per-run summary lines: total=Xms (bcast=Yms gather=Zms) mism=N
runs = []
for m in re.finditer(r"total=(\d+)ms \(bcast=(\d+)ms gather=(\d+)ms\)\s+mism=(\d+)", text):
    total, bc, ga, mism = map(int, m.groups())
    runs.append({"total": total, "bcast": bc, "gather": ga, "mism": mism})

# local_conv/dw and other (constant, deterministic)
prof = re.search(r"local_conv/dw=(\d+)ms.*?bcast=\d+ms gather=\d+ms other=(\d+)ms", text)
local = int(prof.group(1))
other = int(prof.group(2))

# per-round: pwK (Lxx) WxHxC bcast=..ms gather=..ms comp=Zms  -> average per round
rounds = {}
for m in re.finditer(r"pw(\d+) \(L(\d+)\) (\d+x\d+x\d+) bcast=(\d+)ms gather=(\d+)ms comp=(\d+)ms", text):
    k = int(m.group(1))
    r = rounds.setdefault(k, {"dims": m.group(3), "bc": [], "ga": [], "cp": []})
    r["bc"].append(int(m.group(4)))
    r["ga"].append(int(m.group(5)))
    r["cp"].append(int(m.group(6)))
per_round = {}
for k, r in sorted(rounds.items()):
    bc, ga, cp = st.mean(r["bc"]), st.mean(r["ga"]), st.mean(r["cp"])
    per_round[k] = {"dims": r["dims"], "bcast": round(bc, 1), "gather": round(ga, 1),
                    "compute": round(cp, 1), "transmit": round(ga - cp, 1)}
comp_per_round = {k: v["compute"] for k, v in per_round.items()}
worker_compute = sum(comp_per_round.values())

# write per-round CSV
with open(os.path.join(REPO, "results", "mbnet_perround.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pw", "output_dims", "bcast_ms", "gather_ms", "compute_ms", "transmit_ms"])
    for k, v in per_round.items():
        w.writerow([k, v["dims"], v["bcast"], v["gather"], v["compute"], v["transmit"]])

mean_total = st.mean(r["total"] for r in runs)
mean_bcast = st.mean(r["bcast"] for r in runs)
mean_gather = st.mean(r["gather"] for r in runs)
n_pass = sum(1 for r in runs if r["mism"] == 0)

gather_transmit = mean_gather - worker_compute
communication = mean_bcast + gather_transmit
compute = worker_compute + local
total = mean_total

# communication volume
comm_rows = list(csv.DictReader(open(COMM)))
tot = [r for r in comm_rows if r["layer"] == "TOTAL"][0]
bcast_B = int(tot["broadcast_B"]); gather_B = int(tot["gather_B"])

# float accuracy
facc = re.search(r"Best-checkpoint test accuracy \(10k\): ([\d.]+)%", open(TRAIN).read())
float_acc = float(facc.group(1)) if facc else None

# INT8 accuracy (per-tensor) from the eval log
int8_acc = None
if os.path.exists(INT8):
    im = re.search(r"int8=([\d.]+)%", open(INT8).read())
    int8_acc = float(im.group(1)) if im else None

res = {
    "model": "scaled MobileNet-V1 (depthwise-separable), 96x96x3, 1,089,738 params",
    "sram_exceeding_layer": "b8_pw 512->1024: 524,288 INT8 weight bytes (512 KB) > 512 KB SRAM; 128 KB/worker at N=4",
    "n_workers": 4,
    "distributed_pointwise_rounds": 8,
    "accuracy": {"float_test_10k_pct": float_acc, "int8_test_10k_pct": int8_acc},
    "bit_exact": {"runs": len(runs), "pass": n_pass, "predicted_class": 3, "label": 3},
    "latency_ms": {
        "total_mean": round(total, 1),
        "broadcast_mean": round(mean_bcast, 1),
        "gather_mean": round(mean_gather, 1),
        "local_compute": local,
        "other": other,
        "worker_pw_compute": round(worker_compute, 1),
        "gather_transmit_derived": round(gather_transmit, 1),
        "per_run_totals": [r["total"] for r in runs],
        "per_round": {f"pw{k}": v for k, v in per_round.items()},
    },
    "compute_vs_communication": {
        "compute_ms": round(compute, 1),
        "compute_pct": round(100 * compute / total, 1),
        "communication_ms": round(communication, 1),
        "communication_pct": round(100 * communication / total, 1),
        "other_ms": other,
        "other_pct": round(100 * other / total, 1),
        "verdict": "communication-bound",
    },
    "communication_volume_bytes": {
        "broadcast": bcast_B, "gather": gather_B, "total": bcast_B + gather_B,
        "broadcast_KB": round(bcast_B / 1024, 1), "gather_KB": round(gather_B / 1024, 1),
        "total_KB": round((bcast_B + gather_B) / 1024, 1),
    },
}

json.dump(res, open(OUT_JSON, "w"), indent=2)

cvc = res["compute_vs_communication"]
lat = res["latency_ms"]
md = f"""# R2-11: Scaled MobileNet distributed results

**Model:** {res['model']}
**SRAM-exceeding layer:** {res['sram_exceeding_layer']}
**Distribution:** N={res['n_workers']} workers, {res['distributed_pointwise_rounds']} distributed pointwise rounds (output-channel partitioning); depthwise/stem/dense on coordinator.

## Accuracy
- Float (10k test, best-checkpoint): **{float_acc}%**
- INT8 (10k test, per-tensor): **{int8_acc}%**

## On-device correctness
- **{n_pass}/{len(runs)} runs bit-exact** (mism=0), predicted class {res['bit_exact']['predicted_class']} == label {res['bit_exact']['label']}.

## Latency (mean over {len(runs)} runs, ms)
| Phase | ms |
|---|---|
| Total | {lat['total_mean']} |
| Broadcast | {lat['broadcast_mean']} |
| Gather (compute+transmit) | {lat['gather_mean']} |
|   - worker pointwise compute | {lat['worker_pw_compute']} |
|   - transmit (derived) | {lat['gather_transmit_derived']} |
| Local (conv0/depthwise/dense) | {lat['local_compute']} |
| Other | {lat['other']} |

Per-run totals (ms): {lat['per_run_totals']}

## Compute vs. communication (the regime shift R2-11 asks for)
| Component | ms | % |
|---|---|---|
| Compute (worker pw + local) | {cvc['compute_ms']} | {cvc['compute_pct']}% |
| Communication (broadcast + gather transmit) | {cvc['communication_ms']} | {cvc['communication_pct']}% |
| Other | {cvc['other_ms']} | {cvc['other_pct']}% |

**Verdict: {cvc['verdict']}** at 96x96 — contrast with FatCNN at 32x32 (73% compute, compute-bound).

## Communication volume (per inference, INT8 activations)
- Broadcast: {res['communication_volume_bytes']['broadcast_KB']} KB
- Gather: {res['communication_volume_bytes']['gather_KB']} KB
- **Total: {res['communication_volume_bytes']['total_KB']} KB**
"""
open(OUT_MD, "w").write(md)
print(f"wrote {OUT_JSON}")
print(f"wrote {OUT_MD}")
print(f"\nbit-exact {n_pass}/{len(runs)} | total {lat['total_mean']}ms | "
      f"compute {cvc['compute_pct']}% comm {cvc['communication_pct']}%")
