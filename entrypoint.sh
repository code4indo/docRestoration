#!/bin/bash
# Entry point script for GAN-HTR container
# Handles auto-download from HuggingFace and environment setup

set -e

echo "🚀 Starting GAN-HTR container..."

# Function to check if directory exists and is not empty
check_data_exists() {
    local dir="$1"
    if [ -d "$dir" ] && [ "$(ls -A $dir 2>/dev/null)" ]; then
        return 0
    else
        return 1
    fi
}

# Function to download data from HuggingFace
download_data() {
    echo "📥 Downloading data from HuggingFace..."
    cd /workspace

    # Download datasets
    if ! check_data_exists "/workspace/dual_modal_gan/data"; then
        echo "  📊 Downloading GAN dataset..."
        python3 /workspace/download_from_huggingface.py
    else
        echo "  ✅ GAN dataset already exists, skipping download"
    fi

    # Download models
    if ! check_data_exists "/workspace/models/best_htr_recognizer"; then
        echo "  🤖 Downloading HTR models..."
        mkdir -p /workspace/models/best_htr_recognizer
        # Models will be downloaded by the download script
    else
        echo "  ✅ HTR models already exist, skipping download"
    fi

    echo "✅ Data download completed"
}

# Function to setup environment
setup_environment() {
    echo "🔧 Setting up environment..."

    # Set proper permissions
    chmod +x /workspace/scripts/*.sh 2>/dev/null || true

    # Create necessary directories
    mkdir -p /workspace/dual_modal_gan/{outputs,logs}
    mkdir -p /workspace/logbook
    mkdir -p /workspace/mlruns

    # Initialize MLflow if needed
    if [ ! -f "/workspace/mlruns/.mlflow_configured" ]; then
        touch /workspace/mlruns/.mlflow_configured
        echo "✅ MLflow initialized"
    fi

    echo "✅ Environment setup completed"
}

# Function to validate installation
validate_installation() {
    echo "🔍 Validating installation..."

    # Check Python
    python3 --version
    echo "✅ Python version: $(python3 --version)"

    # Check TensorFlow
    python3 -c "import tensorflow as tf; print('✅ TensorFlow version:', tf.__version__)"

    # Check GPU (if available)
    python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'✅ GPU detected: {len(gpus)} device(s)')
    for gpu in gpus:
        print(f'  - {gpu.name}')
else:
    print('⚠️ No GPU detected, using CPU mode')
"

    # Check critical files
    critical_files=(
        "/workspace/dual_modal_gan/scripts/train.py"
        "/workspace/network/layers.py"
        "/workspace/real_data_preparation/real_data_charlist.txt"
    )

    for file in "${critical_files[@]}"; do
        if [ -f "$file" ]; then
            echo "✅ Found: $file"
        else
            echo "⚠️ Missing: $file"
        fi
    done

    echo "✅ Installation validation completed"
}

# Main execution logic
main() {
    echo "🎯 GAN-HTR Container Entry Point"
    echo "=================================="

    # Determine mode
    MODE=${MODE:-"production"}
    TEST_MODE=${TEST_MODE:-"false"}

    echo "📋 Configuration:"
    echo "  - Mode: $MODE"
    echo "  - Test Mode: $TEST_MODE"
    echo "  - GPU: $CUDA_VISIBLE_DEVICES"
    echo "  - Working Directory: $(pwd)"

    # Setup environment
    setup_environment

    # Download data if in production mode or data doesn't exist
    if [ "$MODE" = "production" ] || [ "$TEST_MODE" = "true" ]; then
        download_data
    fi

    # Validate installation
    validate_installation

    # Execute the provided command or default training
    if [ $# -gt 0 ]; then
        echo "🏃 Executing command: $@"
        exec "$@"
    else
        echo "🏃 Default: Starting training..."
        exec python3 /workspace/dual_modal_gan/scripts/train.py
    fi
}

# Run main function with all arguments
main "$@"