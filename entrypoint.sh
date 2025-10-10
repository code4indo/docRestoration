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

    # Check critical files instead of directories
    local dataset_file="/workspace/dual_modal_gan/data/dataset_gan.tfrecord"
    local model_file="/workspace/models/best_htr_recognizer/best_model.weights.h5"
    local charlist_file="/workspace/real_data_preparation/real_data_charlist.txt"

    # Check if all critical files exist
    if [ -f "$dataset_file" ] && [ -f "$model_file" ] && [ -f "$charlist_file" ]; then
        echo "  ✅ All data files already exist, skipping download"
        echo "    - Dataset: $(du -h $dataset_file | cut -f1)"
        echo "    - Model: $(du -h $model_file | cut -f1)"
        echo "    - Charlist: $(wc -l < $charlist_file) characters"
    else
        echo "  📊 Downloading data from HuggingFace repository..."
        echo "    Repository: ${HF_USERNAME:-jatnikonm}/${HF_REPO_NAME:-HTR_VOC}"
        
        # Run download script
        python3 /workspace/docRestoration/scripts/download_from_huggingface.py
        
        if [ $? -eq 0 ]; then
            echo "  ✅ Download completed successfully"
        else
            echo "  ❌ Download failed, check logs above"
            exit 1
        fi
    fi

    echo "✅ Data validation completed"
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

    # Check critical files (code + data)
    critical_code_files=(
        "/workspace/docRestoration/dual_modal_gan/scripts/train.py"
    )
    
    critical_data_files=(
        "/workspace/dual_modal_gan/data/dataset_gan.tfrecord"
        "/workspace/models/best_htr_recognizer/best_model.weights.h5"
        "/workspace/real_data_preparation/real_data_charlist.txt"
    )

    echo "📂 Checking critical code files..."
    for file in "${critical_code_files[@]}"; do
        if [ -f "$file" ]; then
            echo "✅ Found: $file"
        else
            echo "❌ Missing: $file (This file should be in the image/volume)"
            exit 1
        fi
    done
    
    echo "📊 Checking critical data files..."
    for file in "${critical_data_files[@]}"; do
        if [ -f "$file" ]; then
            size=$(du -h "$file" | cut -f1)
            echo "✅ Found: $file ($size)"
        else
            echo "⚠️ Missing: $file (Will be downloaded)"
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

    # Execute the provided command or training script
    if [ $# -gt 0 ]; then
        echo "🏃 Executing command: $@"
        exec "$@"
    else
        # Determine training script from environment variable
        TRAINING_SCRIPT=${TRAINING_SCRIPT:-"dual_modal_gan/scripts/train.py"}
        
        echo "🏃 Starting training with: $TRAINING_SCRIPT"
        
        # Check if script is shell script or python
        if [[ "$TRAINING_SCRIPT" == *.sh ]]; then
            # Shell script - make executable and run
            chmod +x "/workspace/docRestoration/$TRAINING_SCRIPT"
            exec bash "/workspace/docRestoration/$TRAINING_SCRIPT"
        elif [[ "$TRAINING_SCRIPT" == *.py ]]; then
            # Python script
            exec python3 "/workspace/docRestoration/$TRAINING_SCRIPT"
        else
            echo "❌ Unknown script type: $TRAINING_SCRIPT"
            echo "   Supported: .py or .sh files"
            exit 1
        fi
    fi
}

# Run main function with all arguments
main "$@"