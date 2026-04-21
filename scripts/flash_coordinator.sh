#!/bin/bash
# Flash swarm_coordinator firmware.
# Usage: ./flash_coordinator.sh [N_WORKERS]
#   N_WORKERS: 2 or 4 (default: 4)
#
# Prerequisites:
#   - ESP-IDF v6.0 sourced (source $IDF_PATH/export.sh)
#   - idf.py set-target esp32s3 run once in firmware/swarm_coordinator/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COORD_DIR="$PROJECT_DIR/firmware/swarm_coordinator"

N_WORKERS=${1:-4}
COORD_PORT="/dev/cu.usbmodem212301"  # Update for your setup

echo "=== SwarmInfer: Flash Coordinator (N=$N_WORKERS) ==="
echo "Project: $PROJECT_DIR"
echo "Port: $COORD_PORT"
echo ""

cd "$COORD_DIR"
SWARM_N_WORKERS=$N_WORKERS idf.py build
idf.py -p "$COORD_PORT" flash

echo "=== Coordinator flashed (N=$N_WORKERS) ==="
echo "Run monitor: idf.py -p $COORD_PORT monitor"
