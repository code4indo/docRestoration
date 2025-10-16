#!/bin/bash

# HPO Launcher Script for Loss Weight Optimization
# Runs Bayesian Optimization with Optuna in background
# Monitors progress via log file

echo "=========================================="
echo "🔬 HPO - Loss Weight Optimization"
echo "=========================================="
echo ""
echo "Method: Bayesian Optimization (Optuna TPE)"
echo "Target: Find optimal loss weights in ~30-40 min"
echo "GPUs: 2 (parallel execution)"
echo "Trials: 30"
echo ""
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "📍 Project root: $PROJECT_ROOT"
echo ""

# Create logbook directory if it doesn't exist
LOGBOOK_DIR="$PROJECT_ROOT/logbook"
mkdir -p "$LOGBOOK_DIR"

# Output log file
LOG_FILE="$LOGBOOK_DIR/hpo_loss_weights_$(date +%Y%m%d_%H%M%S).log"
echo "📝 HPO output will be saved to: $LOG_FILE"
echo ""

# Verify required files exist
echo "🔍 Checking required files..."
DATASET_PATH="$PROJECT_ROOT/dual_modal_gan/data/dataset_gan.tfrecord"
CHARSET_PATH="$PROJECT_ROOT/real_data_preparation/real_data_charlist.txt"
RECOGNIZER_WEIGHTS="$PROJECT_ROOT/models/best_htr_recognizer/best_model.weights.h5"
HPO_SCRIPT="$PROJECT_ROOT/scripts/hpo_loss_weights.py"

for file in "$DATASET_PATH" "$CHARSET_PATH" "$RECOGNIZER_WEIGHTS" "$HPO_SCRIPT"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: Required file not found: $file"
        exit 1
    else
        echo "✅ Found: $(basename "$file")"
    fi
done
echo ""

# Check GPU availability
echo "🎮 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    N_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    echo ""
    echo "✅ Found $N_GPUS GPU(s)"
else
    echo "⚠️  nvidia-smi not found, assuming 2 GPUs"
    N_GPUS=2
fi
echo ""

# Ask for confirmation
read -p "🚀 Start HPO optimization? This will take ~30-40 minutes. [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ HPO cancelled by user"
    exit 0
fi

# Change to project root
cd "$PROJECT_ROOT" || {
    echo "❌ Error: Cannot change to project directory: $PROJECT_ROOT"
    exit 1
}

# Set GPU environment variables
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export ALLOW_GPU=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export TF_XLA_FLAGS=--tf_xla_auto_jit=0

echo "🚀 Starting HPO in background..."
echo ""

# Run HPO in background
nohup poetry run python "$HPO_SCRIPT" \
  --project_root "$PROJECT_ROOT" \
  --tfrecord_path "$DATASET_PATH" \
  --charset_path "$CHARSET_PATH" \
  --recognizer_weights "$RECOGNIZER_WEIGHTS" \
  --n_trials 30 \
  --n_gpus "$N_GPUS" \
  --epochs 2 \
  --steps_per_epoch 30 \
  --batch_size 2 \
  --warmup_epochs 1 \
  --annealing_epochs 1 \
  --eval_interval 2 \
  --save_interval 2 \
  --seed 42 \
  > "$LOG_FILE" 2>&1 &

HPO_PID=$!

# Save PID for later reference
echo $HPO_PID > "$PROJECT_ROOT/hpo_pid.txt"

echo "✅ HPO started in background (PID: $HPO_PID)"
echo ""
echo "📊 Monitor progress:"
echo "   tail -f $LOG_FILE"
echo ""
echo "📈 View real-time dashboard (in another terminal):"
echo "   poetry run optuna-dashboard sqlite:///hpo_study.db"
echo ""
echo "🛑 Stop HPO:"
echo "   kill $HPO_PID"
echo ""
echo "⏱️  Estimated completion: $(date -d '+40 minutes' '+%H:%M:%S')"
echo ""
echo "=========================================="
echo ""

# Ask if user wants to tail the log
read -p "📖 Monitor log file now? [Y/n] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "Monitoring HPO progress (Ctrl+C to stop monitoring, HPO will continue)..."
    echo ""
    tail -f "$LOG_FILE"
fi

echo ""
echo "🏁 HPO monitoring stopped (process still running in background)"
echo "   Check status: ps -p $HPO_PID"
echo ""
