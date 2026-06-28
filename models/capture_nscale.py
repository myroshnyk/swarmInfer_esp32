"""Capture the nscale FatCNN coordinator (R2-7/R2-8) over serial.

The coordinator runs a continuous batch; we reset it, collect the first --count
CSV lines, save the raw log to logs/, and validate predictions bit-exact against
the released N=4 reference (predictions must be identical — partitioning is exact).

Run under the ESP-IDF python environment (pyserial):
    source ~/.espressif/v6.0/esp-idf/export.sh
    python capture_nscale.py --n 3 --count 150
"""
import argparse
import gzip
import os
import re
import time

import serial

REPO = os.path.join(os.path.dirname(__file__), "..")
REF = {
    2: os.path.join(REPO, "pub/logs/reference_paper_runs/fatcnn_n2.log.gz"),
    4: os.path.join(REPO, "pub/logs/reference_paper_runs/fatcnn_n4.log.gz"),
}
CSV_RE = re.compile(r"CSV,fatcnn_n\d+,(\d+),(\d+),(\d+),(\d+),(\d+)")


def load_ref(path):
    preds = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            m = CSV_RE.search(line)
            if m:
                img, _, pred = int(m.group(1)), int(m.group(2)), int(m.group(3))
                preds[img] = pred
    return preds


def reset(ser):
    ser.dtr = False; ser.rts = True; time.sleep(0.15); ser.rts = False
    time.sleep(0.05); ser.reset_input_buffer()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="number of workers (config)")
    ap.add_argument("--port", default="/dev/cu.usbmodem212301")
    ap.add_argument("--count", type=int, default=150)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    log_path = os.path.join(REPO, "logs", f"fatcnn_nscale_n{args.n}.log")
    ser = serial.Serial(args.port, 115200, timeout=1)
    reset(ser)

    rows = []          # (img, label, pred, match, total_us)
    t0 = time.time()
    with open(log_path, "w") as f:
        while len(rows) < args.count and time.time() - t0 < args.timeout:
            line = ser.readline().decode("utf-8", "replace").rstrip()
            if not line:
                continue
            f.write(line + "\n")
            m = CSV_RE.search(line)
            if m:
                img, label, pred, match, total = (int(m.group(i)) for i in range(1, 6))
                rows.append((img, label, pred, match, total))
                if len(rows) % 25 == 0:
                    print(f"  {len(rows)}/{args.count} captured...")
    ser.close()
    print(f"Captured {len(rows)} inferences -> {log_path}")

    # accuracy + latency
    acc = 100.0 * sum(r[3] for r in rows) / len(rows)
    lat = [r[4] / 1000.0 for r in rows]
    print(f"Accuracy: {acc:.1f}%   mean latency: {sum(lat)/len(lat):.0f} ms")

    # bit-exact vs reference (N=4): predictions must match per image
    ref = load_ref(REF[4])
    mism = [(img, pred, ref[img]) for img, _, pred, _, _ in rows if img in ref and pred != ref[img]]
    checked = sum(1 for img, *_ in rows if img in ref)
    print(f"Prediction match vs N=4 reference: {checked - len(mism)}/{checked} "
          f"({'BIT-EXACT' if not mism else 'MISMATCH: ' + str(mism[:5])})")


if __name__ == "__main__":
    main()
