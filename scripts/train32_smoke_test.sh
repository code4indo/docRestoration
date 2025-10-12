#!/bin/bash

# Smoke Test Script for train32.py (Pure FP32)
# Quick validation test (2 epochs) before full training
# Works both locally and on cloud providers (RunPod, etc.)

echo "=========================================="
echo "🧪 GAN-HTR Smoke Test - Pure FP32"
echo "=========================================="
echo ""
echo "Purpose: Quick validation (2 epochs)"
echo "  • Fast iteration for testing changes"
echo "  • Validate convergence pattern"
echo "  • Check for errors before full run"
echo "  • Early stopping enabled (patience=3)"
echo ""
echo "========================================="
echo ""

# Get script directory (works for both local and cloud execution)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "📍 Project root: $PROJECT_ROOT"
echo "📍 Script directory: $SCRIPT_DIR"
echo ""

# Set workspace environment variable for cloud compatibility
if [ -z "$WORKSPACE" ]; then
    # Local execution - use project root as workspace
    WORKSPACE="$PROJECT_ROOT"
    echo "🏠 Local execution detected"
else
    # Cloud execution - use provided workspace
    echo "☁️  Cloud execution detected"
fi

# Create logbook directory if it doesn't exist
LOGBOOK_DIR="$WORKSPACE/logbook"
mkdir -p "$LOGBOOK_DIR"

# Output redirected to logfile untuk monitoring
LOG_FILE="$LOGBOOK_DIR/smoke_test_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Training output will be saved to: $LOG_FILE"
echo ""

# Determine Python executable
if command -v poetry &> /dev/null && [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
    PYTHON_CMD="poetry run python"
    echo "🐍 Using Poetry Python environment"
elif [ ! -z "$PYTHON_BIN" ]; then
    PYTHON_CMD="$PYTHON_BIN"
    echo "🐍 Using custom Python: $PYTHON_BIN"
else
    PYTHON_CMD="python3"
    echo "🐍 Using system Python: python3"
fi

echo "🐍 Python command: $PYTHON_CMD"
echo ""

# Set paths (relative to project root)
DATASET_PATH="$PROJECT_ROOT/dual_modal_gan/data/dataset_gan.tfrecord"
CHARSET_PATH="$PROJECT_ROOT/real_data_preparation/real_data_charlist.txt"
RECOGNIZER_WEIGHTS="$PROJECT_ROOT/models/best_htr_recognizer/best_model.weights.h5"
CHECKPOINT_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/checkpoints_fp32_smoke_test"
SAMPLE_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/samples_fp32_smoke_test"
TRAIN_SCRIPT="$PROJECT_ROOT/dual_modal_gan/scripts/train32.py"

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

echo "📊 Starting MLflow UI in the background..."
poetry run mlflow ui &
echo "   MLflow UI running at http://localhost:5000"
echo ""

echo "🚀 Starting smoke test training..."
echo ""

# Run training with all paths relative to project root
cd "$PROJECT_ROOT" || {
    echo "❌ Error: Cannot change to project directory: $PROJECT_ROOT"
    exit 1
}

# Set GPU environment variables for training
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export ALLOW_GPU=1

# Additional environment variables for TensorFlow stability
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export TF_XLA_FLAGS=--tf_xla_auto_jit=0

$PYTHON_CMD dual_modal_gan/scripts/train32.py \
  --tfrecord_path "$DATASET_PATH" \
  --charset_path "$CHARSET_PATH" \
  --recognizer_weights "$RECOGNIZER_WEIGHTS" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --sample_dir "$SAMPLE_DIR" \
  --gpu_id 0 \
  --epochs 8 \
  --warmup_epochs 2 \
  --annealing_epochs 4 \
  --steps_per_epoch 50 \
  --batch_size 2 \
  --lr_g 0.0004 \
  --lr_d 0.0004 \
  --pixel_loss_weight 10.0 \
  --ctc_loss_weight 10.0 \
  --adv_loss_weight 5.0 \
  --gradient_clip_norm 1.0 \
  --ctc_loss_clip_max 100.0 \
  --eval_interval 5 \
  --save_interval 5 \
  --discriminator_mode ground_truth \
  --no_restore \
  --early_stopping \
  --patience 10 \
  --min_delta 0.1 \
  --restore_best_weights \
  --seed 42 2>&1 | tee "$LOG_FILE"

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Smoke test completed successfully!"
else
    echo "❌ Smoke test failed with exit code: $TRAIN_EXIT_CODE"
fi
echo "=========================================="
echo ""
echo "Check results:"
echo "  📊 Metrics: $CHECKPOINT_DIR/metrics/training_metrics_fp32.json"
echo "  🖼️  Samples: $SAMPLE_DIR/"
echo "  📝 Log file: $LOG_FILE"
echo ""

# Check if key output files were created
if [ -f "$CHECKPOINT_DIR/metrics/training_metrics_fp32.json" ]; then
    echo "✅ Training metrics file created"
else
    echo "⚠️  Warning: Training metrics file not found"
fi

if [ -d "$SAMPLE_DIR" ] && [ "$(ls -A "$SAMPLE_DIR" 2>/dev/null)" ]; then
    echo "✅ Sample images created"
else
    echo "⚠️  Warning: No sample images found"
fi

echo ""
echo "🏁 Smoke test finished!"
echo ""
