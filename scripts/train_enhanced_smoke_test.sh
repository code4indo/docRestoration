#!/bin/bash

# Smoke Test Script for train_enhanced.py
# Extended validation test (2 epochs, 200 steps) for the ENHANCED generator.

echo "======================================================"
echo "🧪 GAN-HTR Smoke Test - ENHANCED GENERATOR (Extended)"
echo "======================================================"
echo ""
echo "Purpose: Extended validation of the enhanced generator"
echo "  • Epochs: 2"
echo "  • Steps: 200"
echo "  • Generator: Enhanced (ResNet + Attention)"
echo ""
echo "======================================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "📍 Project root: $PROJECT_ROOT"

# Set workspace environment variable
if [ -z "$WORKSPACE" ]; then
    WORKSPACE="$PROJECT_ROOT"
    echo "🏠 Local execution detected"
else
    echo "☁️  Cloud execution detected"
}

# Create logbook directory
LOGBOOK_DIR="$WORKSPACE/logbook"
mkdir -p "$LOGBOOK_DIR"

LOG_FILE="$LOGBOOK_DIR/smoke_test_enhanced_extended_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Training output will be saved to: $LOG_FILE"
echo ""

# Determine Python executable
if command -v poetry &> /dev/null && [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
    PYTHON_CMD="poetry run python"
    echo "🐍 Using Poetry Python environment"
else
    PYTHON_CMD="python3"
    echo "🐍 Using system Python: python3"
fi

echo "🐍 Python command: $PYTHON_CMD"
echo ""

# Set paths
DATASET_PATH="$PROJECT_ROOT/dual_modal_gan/data/dataset_gan.tfrecord"
CHARSET_PATH="$PROJECT_ROOT/real_data_preparation/real_data_charlist.txt"
RECOGNIZER_WEIGHTS="$PROJECT_ROOT/models/best_htr_recognizer/best_model.weights.h5"
CHECKPOINT_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/checkpoints_enhanced_smoke"
SAMPLE_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/samples_enhanced_smoke"
TRAIN_SCRIPT="$PROJECT_ROOT/dual_modal_gan/scripts/train_enhanced.py"

# Verify required files exist
echo "🔍 Checking required files..."
for file in "$DATASET_PATH" "$CHARSET_PATH" "$RECOGNIZER_WEIGHTS" "$TRAIN_SCRIPT"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: Required file not found: $file"
        exit 1
    else
        echo "✅ Found: $(basename "$file")"
    fi
done
echo ""

# Create output directories
mkdir -p "$CHECKPOINT_DIR" "$SAMPLE_DIR"

echo "🚀 Starting smoke test training for ENHANCED generator..."
echo ""

# Run training with all paths relative to project root
cd "$PROJECT_ROOT" || {
    echo "❌ Error: Cannot change to project directory: $PROJECT_ROOT"
    exit 1
}

# Set GPU environment variables
export CUDA_VISIBLE_DEVICES=0

$PYTHON_CMD dual_modal_gan/scripts/train_enhanced.py \
  --generator_version enhanced \
  --tfrecord_path "$DATASET_PATH" \
  --charset_path "$CHARSET_PATH" \
  --recognizer_weights "$RECOGNIZER_WEIGHTS" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --sample_dir "$SAMPLE_DIR" \
  --gpu_id 0 \
  --epochs 2 \
  --steps_per_epoch 200 \
  --batch_size 2 \
  --eval_interval 1 \
  --save_interval 1 \
  --no_restore \
  --seed 42 2>&1 | tee "$LOG_FILE"

TRAIN_EXIT_CODE=$?

echo ""
echo "========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Enhanced smoke test completed successfully!"
else
    echo "❌ Enhanced smoke test failed with exit code: $TRAIN_EXIT_CODE"
fi
echo "========================================="
echo ""
