#!/bin/bash
# Generate quantized weights and partition for N workers.
# Usage: ./setup_weights.sh [N_WORKERS]
#   N_WORKERS: 2 or 4 (default: 4)
#
# Prerequisites:
#   - conda activate swarm-ml (or Python 3.11+ with tensorflow, numpy)
#   - models/fatcnn_float32.keras must exist (run train_fatcnn.py first)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$PROJECT_DIR/models"

N_WORKERS=${1:-4}

echo "=== SwarmInfer: Generate Weights (N=$N_WORKERS) ==="
echo ""

cd "$MODELS_DIR"

# Step 1: Quantize and export base weights (if not already done)
if [ ! -d "c_weights_n4" ] && [ ! -d "c_weights_n2" ]; then
    echo "--- Step 1: Quantize FatCNN (fix_quantize.py) ---"
    python fix_quantize.py
    # fix_quantize.py outputs to c_weights/, rename to c_weights_n4
    mv c_weights c_weights_n4
fi

# Step 2: Partition for N workers
echo "--- Step 2: Partition for N=$N_WORKERS (partition_n.py) ---"
python partition_n.py "$N_WORKERS"

# Step 3: Export batch test images (always regenerate)
echo "--- Step 3: Export batch test images ---"
python export_batch.py
# Copy to all c_weights_nX directories
for d in c_weights_n*/; do
    cp c_weights/test_images_batch.h "$d"
done

# Step 4: Symlink c_weights → c_weights_nN
echo "--- Step 4: Symlink c_weights → c_weights_n${N_WORKERS} ---"
rm -f c_weights
ln -s "c_weights_n${N_WORKERS}" c_weights

echo ""
echo "=== Weights ready: models/c_weights → c_weights_n${N_WORKERS} ==="
ls -la c_weights
