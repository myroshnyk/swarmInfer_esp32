"""Capture distributed MobileNet (R2-11) coordinator runs over serial.

The coordinator runs ONE inference per boot, then idles. We reset it via the
RTS/DTR pulse and read its serial output until the bit-exact verdict line, then
repeat for N runs. Output is appended to logs/mbnet_distributed.log.

Run under the ESP-IDF python environment (has pyserial):
    source ~/.espressif/v6.0/esp-idf/export.sh
    python capture_mbnet.py --port /dev/cu.usbmodem212301 --runs 5
"""
import argparse
import os
import time

import serial

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
LOG = os.path.join(REPO, "logs", "mbnet_distributed.log")


def reset(ser):
    ser.dtr = False
    ser.rts = True
    time.sleep(0.15)
    ser.rts = False
    time.sleep(0.05)
    ser.reset_input_buffer()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/cu.usbmodem212301")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds per run")
    args = ap.parse_args()

    ser = serial.Serial(args.port, 115200, timeout=1)
    with open(LOG, "a") as f:
        f.write(f"\n===== mbnet distributed capture: {args.runs} runs, "
                f"port {args.port} =====\n")
        passes = 0
        for r in range(args.runs):
            reset(ser)
            header = f"----- RUN {r+1}/{args.runs} -----"
            print(header)
            f.write(header + "\n")
            t0 = time.time()
            done = False
            while time.time() - t0 < args.timeout:
                line = ser.readline().decode("utf-8", "replace").rstrip()
                if not line:
                    continue
                print(line)
                f.write(line + "\n")
                if "DISTRIBUTED BIT-EXACT" in line or "FAIL" in line:
                    if "PASS" in line:
                        passes += 1
                    done = True
                    break
            f.flush()
            if not done:
                print(f"  (run {r+1} TIMED OUT after {args.timeout}s)")
                f.write(f"  (run {r+1} TIMED OUT)\n")
        summary = f"===== SUMMARY: {passes}/{args.runs} bit-exact PASS ====="
        print(summary)
        f.write(summary + "\n")
    ser.close()


if __name__ == "__main__":
    main()
