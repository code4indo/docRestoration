#!/bin/bash
################################################################################
# Full-Size Document Restoration Inference Launcher
# 
# This script runs inference on DIBCO2016 dataset using the trained 
# production_v3 model with GPU acceleration and comprehensive logging.
#
# Features:
# - GPU 1 by default (falls back to CPU if unavailable)
# - Automatic output directory creation with timestamp
# - Background execution with process monitoring
# - Comprehensive logging
# - Metrics calculation and visualization
#
# Usage:
#   ./scripts/run_inference_dibco2016.sh [gpu_id]
#
# Examples:
#   ./scripts/run_inference_dibco2016.sh        # Use GPU 1 (default)
#   ./scripts/run_inference_dibco2016.sh 0      # Use GPU 0
#   ./scripts/run_inference_dibco2016.sh -1     # Use CPU only
#
# Author: AI Assistant
# Date: 2025-10-22
################################################################################

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Project paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${PROJECT_ROOT}/dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model"
CHECKPOINT_NAME="ckpt-88"
INPUT_DIR="${PROJECT_ROOT}/dibco_datasets/DIPCO2016_dataset"
GT_DIR="${PROJECT_ROOT}/dibco_datasets/DIPCO2016_Dataset_GT"
OUTPUT_BASE="${PROJECT_ROOT}/results/inference_production_v3"

# GPU configuration
GPU_ID=${1:-1}  # Default to GPU 1, can be overridden by argument

# Script path
INFERENCE_SCRIPT="${PROJECT_ROOT}/dual_modal_gan/scripts/inference_production_v3.py"

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ SUCCESS: $1"
}

check_file_exists() {
    if [ ! -f "$1" ]; then
        log_error "File not found: $1"
        return 1
    fi
    return 0
}

check_dir_exists() {
    if [ ! -d "$1" ]; then
        log_error "Directory not found: $1"
        return 1
    fi
    return 0
}

# ============================================================================
# Validation
# ============================================================================

log_info "Starting Full-Size Document Restoration Inference"
log_info "=================================================="
log_info ""
log_info "Configuration:"
log_info "  Project root: ${PROJECT_ROOT}"
log_info "  Checkpoint: ${CHECKPOINT_DIR}/${CHECKPOINT_NAME}"
log_info "  Input directory: ${INPUT_DIR}"
log_info "  GT directory: ${GT_DIR}"
log_info "  GPU ID: ${GPU_ID}"
log_info ""

# Check if script exists
if ! check_file_exists "${INFERENCE_SCRIPT}"; then
    log_error "Inference script not found"
    exit 1
fi

# Check if checkpoint exists
if ! check_file_exists "${CHECKPOINT_DIR}/${CHECKPOINT_NAME}.index"; then
    log_error "Checkpoint not found: ${CHECKPOINT_DIR}/${CHECKPOINT_NAME}"
    log_info "Available checkpoints in ${CHECKPOINT_DIR}:"
    ls -lh "${CHECKPOINT_DIR}/" || true
    exit 1
fi

# Check if input directory exists
if ! check_dir_exists "${INPUT_DIR}"; then
    log_error "Input directory not found"
    exit 1
fi

# Count input images
NUM_IMAGES=$(ls "${INPUT_DIR}"/*.bmp 2>/dev/null | wc -l)
if [ "${NUM_IMAGES}" -eq 0 ]; then
    log_error "No BMP images found in ${INPUT_DIR}"
    exit 1
fi
log_info "Found ${NUM_IMAGES} images to process"

# Check if GT directory exists (optional but recommended)
if [ ! -d "${GT_DIR}" ]; then
    log_info "⚠️  Warning: GT directory not found, metrics will not be calculated"
    GT_DIR=""
fi

# ============================================================================
# Environment Setup
# ============================================================================

log_info "Setting up environment..."

# Activate virtual environment if it exists
if [ -d "${PROJECT_ROOT}/.venv" ]; then
    log_info "Activating virtual environment..."
    source "${PROJECT_ROOT}/.venv/bin/activate"
elif [ -d "${PROJECT_ROOT}/venv" ]; then
    log_info "Activating virtual environment..."
    source "${PROJECT_ROOT}/venv/bin/activate"
else
    log_info "⚠️  No virtual environment found, using system Python"
fi

# Check Python and TensorFlow
log_info "Checking dependencies..."
python --version || { log_error "Python not found"; exit 1; }

if ! python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')" 2>/dev/null; then
    log_error "TensorFlow not installed"
    log_info "Install with: poetry install"
    exit 1
fi

# Check GPU availability
if [ "${GPU_ID}" -ge 0 ]; then
    log_info "Checking GPU availability..."
    GPU_COUNT=$(python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))" 2>/dev/null || echo "0")
    
    if [ "${GPU_COUNT}" -gt 0 ]; then
        log_success "Found ${GPU_COUNT} GPU(s)"
        python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); [print(f'  GPU {i}: {gpu.name}') for i, gpu in enumerate(gpus)]" 2>/dev/null || true
    else
        log_info "⚠️  No GPU found, will use CPU (slower)"
        GPU_ID=-1
    fi
fi

# ============================================================================
# Output Directory Setup
# ============================================================================

# Create output directory with timestamp
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="${OUTPUT_BASE}/dibco2016_${TIMESTAMP}"

log_info "Creating output directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# ============================================================================
# Run Inference
# ============================================================================

log_info ""
log_info "================================================================"
log_info "STARTING INFERENCE"
log_info "================================================================"
log_info ""

# Build command
CMD="python ${INFERENCE_SCRIPT} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --checkpoint_name ${CHECKPOINT_NAME} \
    --input_dir ${INPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --gpu_id ${GPU_ID} \
    --image_ext .bmp"

# Add GT directory if available
if [ -n "${GT_DIR}" ]; then
    CMD="${CMD} --gt_dir ${GT_DIR}"
fi

log_info "Executing command:"
log_info "${CMD}"
log_info ""

# Run inference
START_TIME=$(date +%s)

if eval "${CMD}"; then
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))
    
    log_success "Inference completed successfully"
    log_info "Elapsed time: ${MINUTES}m ${SECONDS}s"
else
    log_error "Inference failed"
    exit 1
fi

# ============================================================================
# Results Summary
# ============================================================================

log_info ""
log_info "================================================================"
log_info "RESULTS SUMMARY"
log_info "================================================================"
log_info ""

# Count output files
NUM_RESTORED=$(ls "${OUTPUT_DIR}"/*_restored.png 2>/dev/null | wc -l)
NUM_COMPARISONS=$(ls "${OUTPUT_DIR}"/*_comparison.png 2>/dev/null | wc -l)

log_info "Output directory: ${OUTPUT_DIR}"
log_info "Processed images: ${NUM_RESTORED}"
log_info "Comparison images: ${NUM_COMPARISONS}"

# Display metrics if available
if [ -f "${OUTPUT_DIR}/metrics.csv" ]; then
    log_info ""
    log_info "Metrics Summary:"
    log_info "----------------"
    
    # Extract and display average metrics (last line of CSV)
    tail -n 1 "${OUTPUT_DIR}/metrics.csv" | awk -F',' '{
        printf "  PSNR:      %s dB\n", $2
        printf "  SSIM:      %s\n", $3
        printf "  F-Measure: %s\n", $4
        printf "  NRM:       %s\n", $5
        printf "  MPM:       %s\n", $6
    }'
    
    log_info ""
    log_info "Full metrics saved to: ${OUTPUT_DIR}/metrics.csv"
fi

# Display log location
if ls "${OUTPUT_DIR}"/inference_*.log 1> /dev/null 2>&1; then
    LOG_FILE=$(ls "${OUTPUT_DIR}"/inference_*.log | head -n 1)
    log_info "Detailed log: ${LOG_FILE}"
fi

# Display summary JSON
if [ -f "${OUTPUT_DIR}/summary.json" ]; then
    log_info "Summary JSON: ${OUTPUT_DIR}/summary.json"
fi

log_info ""
log_info "================================================================"
log_success "ALL TASKS COMPLETED"
log_info "================================================================"
log_info ""
log_info "To view results:"
log_info "  cd ${OUTPUT_DIR}"
log_info "  ls -lh"
log_info ""
log_info "To view metrics:"
log_info "  cat ${OUTPUT_DIR}/metrics.csv"
log_info ""
log_info "To view comparison images:"
log_info "  xdg-open ${OUTPUT_DIR}/*_comparison.png  # Linux"
log_info "  open ${OUTPUT_DIR}/*_comparison.png      # macOS"
log_info ""

exit 0
