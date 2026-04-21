#!/usr/bin/env python3
"""
Capture serial output from ESP32 until batch inference completes.
Watches for "BATCH ACCURACY RESULTS" marker, then reads summary and exits.

Usage: python capture_serial.py <port> <logfile> [timeout_sec]
  port: serial port (e.g., /dev/cu.usbmodem212301)
  logfile: output log file path
  timeout_sec: max wait time in seconds (default: 600)

Requires: pyserial (included in ESP-IDF venv)
"""

import serial
import sys
import time


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <port> <logfile> [timeout_sec]")
        sys.exit(1)

    port = sys.argv[1]
    logfile = sys.argv[2]
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 600

    print(f"Capturing {port} → {logfile} (timeout={timeout}s)")

    ser = serial.Serial(port, 115200, timeout=1)
    # Reset the board by toggling DTR
    ser.dtr = False
    time.sleep(0.1)
    ser.dtr = True

    start = time.time()
    found_marker = False

    with open(logfile, 'w') as f:
        while time.time() - start < timeout:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode('utf-8', errors='replace')
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()

            if 'BATCH ACCURACY RESULTS' in line:
                found_marker = True
                # Read remaining summary lines
                for _ in range(20):
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode('utf-8', errors='replace')
                    sys.stdout.write(line)
                    f.write(line)
                    if '====' in line and 'BATCH' not in line:
                        break
                break

    ser.close()

    if found_marker:
        print(f"\nBatch complete. Log saved to {logfile}")
    else:
        print(f"\nTimeout ({timeout}s). Partial log saved to {logfile}")
        sys.exit(1)


if __name__ == '__main__':
    main()
