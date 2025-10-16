#!/bin/bash

# Test Script Setup Validation
# Validates that all paths and dependencies are correctly configured

echo "=========================================="
echo "🔧 GAN-HTR Script Setup Validation"
echo "=========================================="
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

echo ""
echo "🔍 Checking required files..."

# Check required files
DATASET_PATH="$PROJECT_ROOT/dual_modal_gan/data/dataset_gan.tfrecord"
CHARSET_PATH="$PROJECT_ROOT/real_data_preparation/real_data_charlist.txt"
RECOGNIZER_WEIGHTS="$PROJECT_ROOT/models/best_htr_recognizer/best_model.weights.h5"
TRAIN_SCRIPT="$PROJECT_ROOT/dual_modal_gan/scripts/train32.py"
SMOKE_TEST_SCRIPT="$PROJECT_ROOT/scripts/train32_smoke_test.sh"

ALL_FILES_FOUND=true
for file in "$DATASET_PATH" "$CHARSET_PATH" "$RECOGNIZER_WEIGHTS" "$TRAIN_SCRIPT" "$SMOKE_TEST_SCRIPT"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: Required file not found: $file"
        ALL_FILES_FOUND=false
    else
        echo "✅ Found: $(basename "$file")"
    fi
done

echo ""
echo "🐍 Checking Python environment..."

# Check Python environment
if command -v poetry &> /dev/null && [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
    PYTHON_CMD="poetry run python"
    echo "✅ Poetry Python environment found"

    # Test if key packages are available
    if poetry run python -c "import tensorflow, numpy, cv2, matplotlib" 2>/dev/null; then
        echo "✅ Core Python packages available"
    else
        echo "❌ Some Python packages missing"
        ALL_FILES_FOUND=false
    fi

elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "✅ System Python found"

    # Test if key packages are available
    if python3 -c "import tensorflow, numpy, cv2, matplotlib" 2>/dev/null; then
        echo "✅ Core Python packages available"
    else
        echo "❌ Some Python packages missing"
        ALL_FILES_FOUND=false
    fi
else
    echo "❌ No Python environment found"
    ALL_FILES_FOUND=false
fi

echo ""
echo "🎯 Checking GPU availability..."

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "✅ NVIDIA GPUs detected: $GPU_COUNT"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  NVIDIA SMI not found - GPU may not be available"
fi

echo ""
echo "🚀 Script execution test..."

# Test script execution (dry run)
echo "Testing smoke test script setup..."
if [ -x "$SMOKE_TEST_SCRIPT" ]; then
    echo "✅ Smoke test script is executable"

    # Test script dry run (just check if it can start)
    cd "$PROJECT_ROOT" && timeout 5s bash "$SMOKE_TEST_SCRIPT" 2>/dev/null | head -10
    if [ $? -eq 124 ]; then
        echo "✅ Script starts correctly (timeout after 5s as expected)"
    else
        echo "⚠️  Script may have issues - check manually"
    fi
else
    echo "❌ Smoke test script not executable"
    echo "   Run: chmod +x scripts/train32_smoke_test.sh"
    ALL_FILES_FOUND=false
fi

echo ""
echo "=========================================="
if [ "$ALL_FILES_FOUND" = true ]; then
    echo "✅ All checks passed! Script is ready to run."
    echo ""
    echo "To run smoke test:"
    echo "  cd $PROJECT_ROOT"
    echo "  ./scripts/train32_smoke_test.sh"
    echo ""
    echo "For cloud execution:"
    echo "  export WORKSPACE=/workspace/docRestoration"
    echo "  ./scripts/train32_smoke_test.sh"
else
    echo "❌ Some checks failed. Please fix the issues above."
    echo ""
    echo "Common fixes:"
    echo "  • Install dependencies: poetry install"
    echo "  • Make script executable: chmod +x scripts/train32_smoke_test.sh"
    echo "  • Check file paths and permissions"
fi
echo "=========================================="
echo ""