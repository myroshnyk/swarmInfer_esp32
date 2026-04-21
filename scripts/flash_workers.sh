#!/bin/bash
# Flash swarm_worker firmware to all worker boards.
# Usage: ./flash_workers.sh [N_WORKERS] [WEIGHT_DIR]
#   N_WORKERS: 2 or 4 (default: 4)
#   WEIGHT_DIR: path to weight headers (default: auto from N_WORKERS)
#
# Prerequisites:
#   - ESP-IDF v6.0 sourced (source $IDF_PATH/export.sh)
#   - idf.py set-target esp32s3 run once in firmware/swarm_worker/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKER_DIR="$PROJECT_DIR/firmware/swarm_worker"

N_WORKERS=${1:-4}

# Worker board ports (update these for your setup)
WORKER_PORTS=(
    "/dev/cu.usbmodem212401"   # Worker 0
    "/dev/cu.usbmodem212201"   # Worker 1
    "/dev/cu.usbmodem2121401"  # Worker 2
    "/dev/cu.usbmodem2121301"  # Worker 3
)

echo "=== SwarmInfer: Flash $N_WORKERS Workers ==="
echo "Project: $PROJECT_DIR"
echo ""

cd "$WORKER_DIR"

for (( i=0; i<N_WORKERS; i++ )); do
    PORT="${WORKER_PORTS[$i]}"
    echo "--- Worker $i → $PORT ---"
    SWARM_WORKER_ID=$i idf.py build
    idf.py -p "$PORT" flash
    echo "Worker $i flashed OK"
    echo ""
done

echo "=== All $N_WORKERS workers flashed ==="
