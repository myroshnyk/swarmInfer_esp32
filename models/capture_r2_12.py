"""R2-12: RF-robustness distance sweep capture (released FatCNN N=4).

Captures the coordinator serial while the spatially-distributed cluster runs at a
given board-group separation. Records, per distance config: timeout events,
per-image retries (a layer timeout makes the coordinator re-broadcast and retry
the image), end-to-end latency distribution, and prediction bit-exactness vs the
benchtop N=4 reference (exact partitioning -> predictions should stay correct;
degraded RF should manifest as latency/retries, not errors).

Run under ESP-IDF python (pyserial):
    source ~/.espressif/v6.0/esp-idf/export.sh
    python capture_r2_12.py --dist d0_1m --count 100
"""
import argparse
import gzip
import os
import re
import statistics as st
import time

import serial

REPO = os.path.join(os.path.dirname(__file__), "..")
REF = os.path.join(REPO, "pub/logs/reference_paper_runs/fatcnn_n4.log.gz")
CSV_RE = re.compile(r"CSV,fatcnn_n\d+,(\d+),(\d+),(\d+),(\d+),(\d+)")
TIMEOUT_RE = re.compile(r"TIMEOUT|retry", re.IGNORECASE)


def load_ref():
    preds = {}
    with gzip.open(REF, "rt") as f:
        for line in f:
            m = CSV_RE.search(line)
            if m:
                preds[int(m.group(1))] = int(m.group(3))
    return preds


def reset(ser):
    ser.dtr = False; ser.rts = True; time.sleep(0.15); ser.rts = False
    time.sleep(0.05); ser.reset_input_buffer()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", required=True, help="config label, e.g. d0_1m / d1_5m / d2_10m")
    ap.add_argument("--port", default="/dev/cu.usbmodem212301")
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    log_path = os.path.join(REPO, "logs", f"rf_sweep_{args.dist}.log")
    ser = serial.Serial(args.port, 115200, timeout=1)
    reset(ser)

    rows = []           # (img, label, pred, match, total_us)
    timeout_events = 0
    t0 = time.time()
    with open(log_path, "w") as f:
        while len(rows) < args.count and time.time() - t0 < args.timeout:
            line = ser.readline().decode("utf-8", "replace").rstrip()
            if not line:
                continue
            f.write(line + "\n"); f.flush()
            if TIMEOUT_RE.search(line) and "CSV" not in line:
                timeout_events += 1
            m = CSV_RE.search(line)
            if m:
                img, label, pred, match, total = (int(m.group(i)) for i in range(1, 6))
                rows.append((img, label, pred, match, total))
                if len(rows) % 25 == 0:
                    print(f"  {args.dist}: {len(rows)}/{args.count} (timeouts so far: {timeout_events})")
    ser.close()

    lat = [r[4] / 1000.0 for r in rows]
    lat_s = sorted(lat)
    p95 = lat_s[int(0.95 * (len(lat_s) - 1))]
    acc = 100.0 * sum(r[3] for r in rows) / len(rows)
    ref = load_ref()
    mism = [(img, pred, ref[img]) for img, _, pred, _, _ in rows if img in ref and pred != ref[img]]
    checked = sum(1 for img, *_ in rows if img in ref)

    print(f"\n=== R2-12 {args.dist} ===")
    print(f"images: {len(rows)}   timeout/retry events: {timeout_events}")
    print(f"latency ms: mean {st.mean(lat):.0f}  sd {st.pstdev(lat):.0f}  "
          f"min {min(lat):.0f}  p95 {p95:.0f}  max {max(lat):.0f}")
    print(f"accuracy: {acc:.1f}%   pred vs reference: {checked-len(mism)}/{checked} "
          f"{'BIT-EXACT' if not mism else 'MISMATCH '+str(mism[:5])}")
    print(f"raw -> {log_path}")


if __name__ == "__main__":
    main()
