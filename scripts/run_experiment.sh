#!/bin/bash
# SwarmInfer: Full experiment suite for IEEE Access paper.
#
# Runs all configurations, captures logs, generates LaTeX tables.
#
# Configs:
#   1. FatCNN-Lite single-node (baseline)
#   2. FatCNN distributed N=2
#   3. FatCNN distributed N=4
#
# Usage: ./run_experiment.sh [step]
#   step: "all" (default), "single", "n2", "n4", "analyze"
#
# Prerequisites:
#   - ESP-IDF v6.0 sourced (source $IDF_PATH/export.sh)
#   - conda activate swarm-ml (for partition_n.py and analyze_logs.py)
#   - All boards connected via USB
#
# NOTE: This script operates on the root project (../../),
# NOT the pub/ snapshot. Weights and firmware build from root dirs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="$ROOT_DIR/models"
LOG_DIR="$ROOT_DIR/logs"
RESULTS_DIR="$ROOT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Python with pyserial (ESP-IDF venv) for serial capture.
# Override via IDF_PYTHON env var if your venv lives elsewhere.
IDF_PYTHON="${IDF_PYTHON:-$(command -v python)}"

# Ports (update for your setup)
COORD_PORT="/dev/cu.usbmodem212301"
WORKER_PORTS=(
    "/dev/cu.usbmodem212401"   # Worker 0
    "/dev/cu.usbmodem212201"   # Worker 1
    "/dev/cu.usbmodem2121401"  # Worker 2
    "/dev/cu.usbmodem2121301"  # Worker 3
)

# Timeouts (seconds) — 1000 images × estimated per-image latency + margin
SINGLE_TIMEOUT=3600
N2_TIMEOUT=5400
N4_TIMEOUT=3600

mkdir -p "$LOG_DIR"

STEP=${1:-all}

echo "============================================"
echo "  SwarmInfer Experiment Suite"
echo "  Timestamp: $TIMESTAMP"
echo "  Root: $ROOT_DIR"
echo "  Step: $STEP"
echo "============================================"
echo ""

# ── Helper: kill any process on a port ──
kill_port() {
    local port=$1
    local pids
    pids=$(lsof -t "$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  Killing processes on $port"
        kill $pids 2>/dev/null || true
        sleep 1
    fi
}

# ── Helper: build firmware with error checking ──
build_firmware() {
    local project_dir=$1
    shift
    # Pass remaining args as env vars for idf.py
    echo "  Building in $project_dir ..."
    cd "$project_dir"
    if ! env "$@" idf.py build; then
        echo "ERROR: Build failed in $project_dir"
        echo "Check the build log above for details."
        exit 1
    fi
    echo "  Build OK."
}

# ── Helper: flash firmware with error checking ──
flash_firmware() {
    local port=$1
    kill_port "$port"
    echo "  Flashing to $port ..."
    if ! idf.py -p "$port" flash; then
        echo "ERROR: Flash failed on $port"
        exit 1
    fi
    echo "  Flash OK."
}

# ================================================================
# Step 0: Generate batch test images (100)
# ================================================================
setup_batch_images() {
    echo "--- Generating batch test images ---"
    cd "$MODELS_DIR"
    conda run -n swarm-ml --no-capture-output python export_batch.py
    # Copy to all c_weights_nX directories (skip symlink target)
    local link_target
    link_target=$(readlink c_weights 2>/dev/null || true)
    for d in c_weights_n*/; do
        [ -d "$d" ] && [ "$d" != "${link_target}/" ] && cp c_weights/test_images_batch.h "$d/"
    done
    echo "--- Batch images ready ---"
    echo ""
}

# ================================================================
# Config 1: Single-node baseline (FatCNN-Lite)
# ================================================================
run_single() {
    echo ""
    echo "================================================================"
    echo "  CONFIG 1/3: FatCNN-Lite Single-Node Baseline"
    echo "================================================================"
    echo ""

    LOGFILE="$LOG_DIR/lite_n1_${TIMESTAMP}.log"

    # Ensure c_weights points to n4 (has fatcnn_lite_weights.h + test_images_batch.h)
    cd "$MODELS_DIR"
    rm -f c_weights
    ln -s c_weights_n4 c_weights

    # Clean build to ensure fresh binary
    cd "$ROOT_DIR/single_inference"
    idf.py fullclean 2>/dev/null || true

    # Build
    build_firmware "$ROOT_DIR/single_inference"

    # Flash
    flash_firmware "$COORD_PORT"

    # Wait for boot
    sleep 3

    # Capture
    echo "--- Capturing output → $LOGFILE ---"
    $IDF_PYTHON "$SCRIPT_DIR/capture_serial.py" "$COORD_PORT" "$LOGFILE" "$SINGLE_TIMEOUT"

    # Verify log contains expected data
    if ! grep -q "CSV,lite_n1" "$LOGFILE"; then
        echo "ERROR: Log does not contain lite_n1 CSV data. Wrong firmware?"
        exit 1
    fi

    echo ""
    echo "=== Single-node done ==="
}

# ================================================================
# Config 2/3: Distributed FatCNN (parametric N)
# ================================================================
run_distributed() {
    local N=$1
    local TIMEOUT=$2

    echo ""
    echo "================================================================"
    echo "  CONFIG: FatCNN Distributed N=$N"
    echo "================================================================"
    echo ""

    LOGFILE="$LOG_DIR/fatcnn_n${N}_${TIMESTAMP}.log"

    # Setup weights
    echo "--- Setup weights N=$N ---"
    cd "$MODELS_DIR"
    conda run -n swarm-ml --no-capture-output python partition_n.py "$N"
    rm -f c_weights
    ln -s "c_weights_n${N}" c_weights

    # Flash workers (fullclean needed when WORKER_ID changes)
    echo "--- Flashing $N workers ---"
    cd "$ROOT_DIR/swarm_worker"
    for (( i=0; i<N; i++ )); do
        PORT="${WORKER_PORTS[$i]}"
        echo "  Worker $i → $PORT"
        idf.py fullclean 2>/dev/null || true
        build_firmware "$ROOT_DIR/swarm_worker" "SWARM_WORKER_ID=$i"
        flash_firmware "$PORT"
    done

    # Flash coordinator (fullclean to pick up new NUM_WORKERS)
    echo "--- Flashing coordinator N=$N ---"
    cd "$ROOT_DIR/swarm_coordinator"
    idf.py fullclean 2>/dev/null || true
    build_firmware "$ROOT_DIR/swarm_coordinator" "SWARM_N_WORKERS=$N"
    flash_firmware "$COORD_PORT"

    # Wait for workers to boot and send WORKER_READY
    echo "--- Waiting 5s for worker boot ---"
    sleep 5

    # Capture
    echo "--- Capturing output → $LOGFILE ---"
    $IDF_PYTHON "$SCRIPT_DIR/capture_serial.py" "$COORD_PORT" "$LOGFILE" "$TIMEOUT"

    # Verify log contains expected data
    if ! grep -q "CSV,fatcnn_n${N}" "$LOGFILE"; then
        echo "ERROR: Log does not contain fatcnn_n${N} CSV data. Wrong firmware?"
        exit 1
    fi

    echo ""
    echo "=== N=$N done ==="
}

# ================================================================
# Analysis: parse logs, generate LaTeX tables
# ================================================================
run_analyze() {
    echo ""
    echo "================================================================"
    echo "  Analyzing Results"
    echo "================================================================"
    echo ""

    # Find latest log files
    LATEST_SINGLE=$(ls -t "$LOG_DIR"/lite_n1_*.log 2>/dev/null | head -1)
    LATEST_N2=$(ls -t "$LOG_DIR"/fatcnn_n2_*.log 2>/dev/null | head -1)
    LATEST_N4=$(ls -t "$LOG_DIR"/fatcnn_n4_*.log 2>/dev/null | head -1)

    LOGS=""
    [ -n "$LATEST_SINGLE" ] && LOGS="$LOGS $LATEST_SINGLE" && echo "  Single: $LATEST_SINGLE"
    [ -n "$LATEST_N2" ]     && LOGS="$LOGS $LATEST_N2"     && echo "  N=2:    $LATEST_N2"
    [ -n "$LATEST_N4" ]     && LOGS="$LOGS $LATEST_N4"     && echo "  N=4:    $LATEST_N4"

    if [ -z "$LOGS" ]; then
        echo "ERROR: No log files found in $LOG_DIR/"
        exit 1
    fi

    cd "$ROOT_DIR"
    conda run -n swarm-ml --no-capture-output python "$SCRIPT_DIR/analyze_logs.py" $LOGS

    echo ""
    echo "=== Analysis complete ==="
    echo "  Tables: $RESULTS_DIR/"
    echo "  Stats:  $RESULTS_DIR/stats.json"
}

# ================================================================
# Run selected step(s)
# ================================================================
case "$STEP" in
    all)
        setup_batch_images
        run_single
        run_distributed 2 "$N2_TIMEOUT"
        run_distributed 4 "$N4_TIMEOUT"
        run_analyze
        ;;
    single)
        setup_batch_images
        run_single
        ;;
    n2)
        setup_batch_images
        run_distributed 2 "$N2_TIMEOUT"
        ;;
    n4)
        setup_batch_images
        run_distributed 4 "$N4_TIMEOUT"
        ;;
    analyze)
        run_analyze
        ;;
    *)
        echo "Unknown step: $STEP"
        echo "Valid: all, single, n2, n4, analyze"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  Experiment suite finished!"
echo "  Logs:   $LOG_DIR/"
echo "  Tables: $RESULTS_DIR/"
echo "============================================"
