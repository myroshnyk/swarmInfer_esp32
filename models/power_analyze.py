"""
SwarmInfer — R1-6 Power & Energy from MEASURED INA219 data.

Parses serial captures from the POWER_MEASURE firmware builds and derives the
single-node and N=4 cluster power/energy reported in the manuscript (Table 12,
Section "Power and Energy"). This REPLACES the datasheet estimate in
pub/scripts/power_estimation.py — every number here comes from on-device INA219
measurement, not a datasheet.

The firmware (common/power_meter.c, 0.1 ohm shunt on the 5 V rail, SDA=8/SCL=9,
~1 kHz sampling) emits three line types this script understands:

  PWRDUMP,<tag>,<count>,<mean_mw>,<mean_ma>,<min_mw>,<max_mw>   # authoritative per-board aggregate
  PWRSAVED,<tag>,<count>,<mean_mw>,<mean_ma>,<min_mw>,<max_mw>  # same, restored from NVS on next boot
  PWRW,<img>,<w0_mw>,<w0_ma>,<w1_mw>,<w1_ma>,...                # per-worker means relayed to coordinator
  PWR,<count>,<inst_mw>,<inst_ma>,<bus_mV>                      # raw ~1 kHz samples (fallback only)

Tags: SINGLE (single_inference), COORD (swarm_coordinator). Worker boards report
their lifetime mean to the coordinator inside RESULT_DONE, surfaced as PWRW.

Cluster power  = COORD mean + sum(worker means).
Energy/inf (J) = power (W) * latency (s), with latency taken from the
log-verified runs (single 1.897 s, N=4 2.115 s) unless overridden.

Usage:
    # one coordinator log (all workers instrumented + relayed) + one single-node log:
    python power_analyze.py --single logs/power_runs/single.log \
                            --coord  logs/power_runs/coord_n4.log
    # or board-by-board, one INA219 moved between boards:
    python power_analyze.py --single single.log --coord coord.log \
                            --worker w0.log --worker w1.log --worker w2.log --worker w3.log

Outputs results/power.json and prints the Table 12 cells. A board log is accepted
only if it carries a real, non-zero aggregate (mean_mw > 0); a zero-current
capture (broken/disconnected shunt) is rejected with a clear error so a dead
sensor can never silently become a published number.
"""
import argparse
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# Log-verified latencies (ms) — see logs/reference_paper_runs (analyze_logs.py).
LAT_SINGLE_MS = 1897.0
LAT_N4_MS = 2115.0

MIN_PLAUSIBLE_MW = 50  # an active ESP32-S3 draws >> 50 mW; below this the shunt was not reading


def _parse_aggregate(path, want_tag):
    """Return (mean_mw, mean_ma, source) for the given tag from PWRDUMP/PWRSAVED,
    else fall back to the mean of raw PWR samples. Raises on missing/zero data."""
    dump = None          # (mean_mw, mean_ma) from PWRDUMP/PWRSAVED for want_tag
    raw_mw = []
    with open(path, errors="replace") as fh:
        for line in fh:
            if "PWRDUMP," in line or "PWRSAVED," in line:
                body = line.split("PWRDUMP," if "PWRDUMP," in line else "PWRSAVED,", 1)[1]
                f = body.strip().split(",")
                # <tag>,<count>,<mean_mw>,<mean_ma>,<min_mw>,<max_mw>
                if len(f) >= 4 and f[0] == want_tag and f[1].isdigit() and int(f[1]) > 0:
                    dump = (int(f[2]), int(f[3]))
            elif "PWR," in line and "PWRW" not in line and "PWRDUMP" not in line:
                m = re.search(r"PWR,(\d+),(-?\d+),(-?\d+),(\d+)", line)
                if m:
                    raw_mw.append(abs(int(m.group(2))))
    if dump is not None:
        return dump[0], dump[1], f"PWRDUMP/{want_tag}"
    if raw_mw:
        return int(statistics.mean(raw_mw)), 0, "raw PWR mean"
    raise ValueError(f"{path}: no PWRDUMP/PWRSAVED,{want_tag} and no PWR samples")


def _check_alive(path, mean_mw, tag):
    if mean_mw < MIN_PLAUSIBLE_MW:
        sys.exit(f"ERROR: {path} ({tag}) mean power {mean_mw} mW < {MIN_PLAUSIBLE_MW} mW.\n"
                 f"The INA219 current channel was not reading (dead/disconnected shunt).\n"
                 f"Re-wire the shunt in series on the 5 V rail and re-capture — refusing to\n"
                 f"publish a zero-current measurement.")


def _parse_pwrw(path):
    """Per-worker power (mW) from PWRW lines in a coordinator log. Each PWRW value
    is the worker's *cumulative lifetime mean* up to that image, so the converged
    estimate is the LAST non-zero value per worker (not the average over images)."""
    cols = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            if "PWRW," not in line:
                continue
            f = line.split("PWRW,", 1)[1].strip().split(",")
            vals = f[1:]  # drop img index
            for w in range(len(vals) // 2):
                mw = vals[2 * w]
                if mw.lstrip("-").isdigit() and abs(int(mw)) > 0:
                    cols[w] = abs(int(mw))   # keep overwriting -> last value wins
    return [cols[w] for w in sorted(cols)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", help="single_inference POWER_MEASURE log (tag SINGLE)")
    ap.add_argument("--coord", help="swarm_coordinator POWER_MEASURE log (tag COORD; may carry PWRW)")
    ap.add_argument("--worker", action="append", default=[],
                    help="per-worker POWER_MEASURE log; repeat. Optional if --coord has PWRW.")
    ap.add_argument("--pwrw", help="coordinator-run log (workers on battery) whose PWRW lines carry "
                    "the per-worker means; use when coord's own power is in a separate --coord log.")
    ap.add_argument("--lat-single-ms", type=float, default=LAT_SINGLE_MS)
    ap.add_argument("--lat-n4-ms", type=float, default=LAT_N4_MS)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "power.json"))
    args = ap.parse_args()

    result = {"meta": {"shunt_ohm": 0.1, "source": "INA219 high-side, measured",
                       "lat_single_ms": args.lat_single_ms, "lat_n4_ms": args.lat_n4_ms}}

    # ---- single node ----
    if args.single:
        mw, ma, src = _parse_aggregate(args.single, "SINGLE")
        _check_alive(args.single, mw, "SINGLE")
        p_w = mw / 1000.0
        e_j = p_w * args.lat_single_ms / 1000.0
        result["single_node"] = {"power_w": round(p_w, 3), "energy_j": round(e_j, 3),
                                 "mean_mw": mw, "mean_ma": ma, "src": src}
        print(f"Single-node: {p_w:.3f} W  ->  {e_j:.3f} J/inf  ({src})")

    # ---- cluster (coordinator + workers) ----
    if args.coord:
        c_mw, c_ma, c_src = _parse_aggregate(args.coord, "COORD")
        _check_alive(args.coord, c_mw, "COORD")
        worker_mw = []
        for wlog in args.worker:
            wmw, wma, wsrc = _parse_aggregate(wlog, "WORKER")
            _check_alive(wlog, wmw, "WORKER")
            worker_mw.append(wmw)
        if not worker_mw and args.pwrw:
            worker_mw = _parse_pwrw(args.pwrw)   # workers relayed over ESP-NOW (separate run)
        if not worker_mw:
            worker_mw = _parse_pwrw(args.coord)  # workers relayed in the same coord log
        if not worker_mw:
            sys.exit("ERROR: no worker power found — give --worker logs or a --coord log with PWRW lines.")
        for i, w in enumerate(worker_mw):
            _check_alive(args.coord, w, f"worker{i} (PWRW)")
        cluster_mw = c_mw + sum(worker_mw)
        p_w = cluster_mw / 1000.0
        e_j = p_w * args.lat_n4_ms / 1000.0
        result["cluster_n4"] = {
            "power_w": round(p_w, 3), "energy_j": round(e_j, 3),
            "coord_mw": c_mw, "worker_mw": worker_mw,
            "worker_mean_mw": int(statistics.mean(worker_mw)), "n_workers": len(worker_mw),
        }
        print(f"Coordinator: {c_mw/1000:.3f} W;  workers (mW): {worker_mw} "
              f"(mean {statistics.mean(worker_mw)/1000:.3f} W)")
        print(f"Cluster N={len(worker_mw)}: {p_w:.3f} W  ->  {e_j:.3f} J/inf")

    if "single_node" in result and "cluster_n4" in result:
        s, c = result["single_node"], result["cluster_n4"]
        result["energy_ratio"] = round(c["energy_j"] / s["energy_j"], 1)
        print(f"\nEnergy ratio (cluster / single): {result['energy_ratio']}x")
        print("\n--- Table 12 cells ---")
        print(f"Single-node          & {s['power_w']:.2f} & {args.lat_single_ms:.0f} & {s['energy_j']:.2f} \\\\")
        print(f"Distributed $N{{=}}4$  & {c['power_w']:.2f} & {args.lat_n4_ms:.0f} & {c['energy_j']:.2f} \\\\")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
