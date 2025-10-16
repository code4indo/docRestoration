#!/bin/bash

# Ablation Study Script: Baseline vs Iterative Refinement
# This script runs two training experiments to compare:
# 1. Baseline: Single generator pass without iterative refinement
# 2. Iterative: Two-pass generator with attention-guided refinement

set -e  # Exit on any error

# Configuration
GPU_ID="1"
EPOCHS=10
BATCH_SIZE=4
STEPS_PER_EPOCH=100
RECOGNIZER_WEIGHTS="models/best_htr_recognizer/best_model.weights.h5"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section headers
print_header() {
    echo -e "\n${BLUE}===========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===========================================${NC}"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to clean previous experiment data
clean_experiment_data() {
    local exp_name=$1
    local project_root=$2
    
    local CHECKPOINT_DIR="$project_root/dual_modal_gan/outputs/checkpoints_ablation/$exp_name"
    local SAMPLE_DIR="$project_root/dual_modal_gan/outputs/samples_ablation/$exp_name"
    local LOG_FILE="$project_root/logbook/ablation_exp_$exp_name.log"
    
    print_warning "Cleaning previous data for $exp_name..."
    if [ -d "$CHECKPOINT_DIR" ]; then
        rm -rf "$CHECKPOINT_DIR"
        print_warning "  Removed $CHECKPOINT_DIR"
    fi
    if [ -d "$SAMPLE_DIR" ]; then
        rm -rf "$SAMPLE_DIR"
        print_warning "  Removed $SAMPLE_DIR"
    fi
    if [ -f "$LOG_FILE" ]; then
        rm "$LOG_FILE"
        print_warning "  Removed $LOG_FILE"
    fi
    print_success "Cleaned data for $exp_name"
}

# Check if required files exist
check_dependencies() {
    print_header "Checking Dependencies"
    
    if [ ! -f "$RECOGNIZER_WEIGHTS" ]; then
        print_error "Recognizer weights not found: $RECOGNIZER_WEIGHTS"
        exit 1
    fi
    print_success "Recognizer weights found"
    
    if [ ! -f "dual_modal_gan/data/dataset_gan.tfrecord" ]; then
        print_error "Dataset not found: dual_modal_gan/data/dataset_gan.tfrecord"
        exit 1
    fi
    print_success "Dataset found"
    
    if [ ! -f "real_data_preparation/real_data_charlist.txt" ]; then
        print_error "Charset not found: real_data_preparation/real_data_charlist.txt"
        exit 1
    fi
    print_success "Charset found"
}

# Function to run baseline experiment
run_baseline() {
    # Setup paths and python command
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    PYTHON_CMD="poetry run python"

    # Define unique output directories
    EXP_NAME="baseline"
    CHECKPOINT_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/checkpoints_ablation/$EXP_NAME"
    SAMPLE_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/samples_ablation/$EXP_NAME"
    LOG_FILE="$PROJECT_ROOT/logbook/ablation_exp_$EXP_NAME.log"

    # Clean previous data for this experiment
    clean_experiment_data "$EXP_NAME" "$PROJECT_ROOT"

    echo "📍 Project root: $PROJECT_ROOT"
    echo "🐍 Python command: $PYTHON_CMD"
    echo ""
    echo "Starting baseline training with the following configuration:"
    echo "  - Mode: Single generator pass (no iterative refinement)"
    echo "  - GPU: $GPU_ID"
    echo "  - Epochs: $EPOCHS"
    echo "  - Batch Size: $BATCH_SIZE"
    echo "  - Steps per Epoch: $STEPS_PER_EPOCH"
    echo ""
    
    # Define unique output directories
    EXP_NAME="baseline"
    CHECKPOINT_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/checkpoints_ablation/$EXP_NAME"
    SAMPLE_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/samples_ablation/$EXP_NAME"
    LOG_FILE="$PROJECT_ROOT/logbook/ablation_exp_$EXP_NAME.log"
    
    mkdir -p "$CHECKPOINT_DIR" "$SAMPLE_DIR"
    echo "📝 Log file: $LOG_FILE"
    
    # Change to project root to run and set PYTHONPATH
    cd "$PROJECT_ROOT" || exit 1
    export PYTHONPATH="$PROJECT_ROOT/dual_modal_gan/src:$PROJECT_ROOT:$PYTHONPATH"
    
    $PYTHON_CMD dual_modal_gan/scripts/train32.py \
        --no_iterative_refinement \
        --gpu_id $GPU_ID \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --steps_per_epoch $STEPS_PER_EPOCH \
        --recognizer_weights $RECOGNIZER_WEIGHTS \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --sample_dir "$SAMPLE_DIR" \
        --early_stopping \
        --patience 15 \
        --min_delta 0.01 \
        --restore_best_weights \
        --save_interval 5 \
        --eval_interval 1 \
        --contrastive_loss_weight 0.0 \
        --no_restore \
        --seed 42 2>&1 | tee "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        print_success "Baseline training completed successfully"
        return 0
    else
        print_error "Baseline training failed"
        return 1
    fi
}

# Function to run iterative refinement experiment
run_iterative() {
    print_header "Running ITERATIVE Refinement Experiment (Two Pass)"
    
    # Setup paths and python command
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    PYTHON_CMD="poetry run python"

    # Define unique output directories
    EXP_NAME="iterative"
    CHECKPOINT_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/checkpoints_ablation/$EXP_NAME"
    SAMPLE_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/samples_ablation/$EXP_NAME"
    LOG_FILE="$PROJECT_ROOT/logbook/ablation_exp_$EXP_NAME.log"

    # Clean previous data for this experiment
    clean_experiment_data "$EXP_NAME" "$PROJECT_ROOT"

    echo "📍 Project root: $PROJECT_ROOT"
    echo "🐍 Python command: $PYTHON_CMD"
    echo ""
    echo "Starting iterative refinement training with the following configuration:"
    echo "  - Mode: Two-pass generator with attention-guided refinement"
    echo "  - GPU: $GPU_ID"
    echo "  - Epochs: $EPOCHS"
    echo "  - Batch Size: $BATCH_SIZE"
    echo "  - Steps per Epoch: $STEPS_PER_EPOCH"
    echo ""
    
    # Define unique output directories
    EXP_NAME="iterative"
    CHECKPOINT_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/checkpoints_ablation/$EXP_NAME"
    SAMPLE_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/samples_ablation/$EXP_NAME"
    LOG_FILE="$PROJECT_ROOT/logbook/ablation_exp_$EXP_NAME.log"
    
    mkdir -p "$CHECKPOINT_DIR" "$SAMPLE_DIR"
    echo "📝 Log file: $LOG_FILE"
    
    # Change to project root to run and set PYTHONPATH
    cd "$PROJECT_ROOT" || exit 1
    export PYTHONPATH="$PROJECT_ROOT/dual_modal_gan/src:$PROJECT_ROOT:$PYTHONPATH"
    
    # Run iterative training
    $PYTHON_CMD dual_modal_gan/scripts/train32.py \
        --gpu_id $GPU_ID \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --steps_per_epoch $STEPS_PER_EPOCH \
        --recognizer_weights $RECOGNIZER_WEIGHTS \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --sample_dir "$SAMPLE_DIR" \
        --early_stopping \
        --patience 15 \
        --min_delta 0.01 \
        --restore_best_weights \
        --save_interval 5 \
        --eval_interval 1 \
        --contrastive_loss_weight 0.0 \
        --no_restore \
        --seed 42 2>&1 | tee "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        print_success "Iterative refinement training completed successfully"
        return 0
    else
        print_error "Iterative refinement training failed"
        return 1
    fi
}

# Function to generate comparison report
generate_report() {
    print_header "Generating Ablation Study Report"
    
    REPORT_FILE="ablation_study_report_$(date +%Y%m%d_%H%M%S).md"
    
    echo "# Ablation Study Report: Baseline vs Iterative Refinement" > "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "**Date:** $(date)" >> "$REPORT_FILE"
    echo "**GPU:** $GPU_ID" >> "$REPORT_FILE"
    echo "**Epochs:** $EPOCHS" >> "$REPORT_FILE"
    echo "**Batch Size:** $BATCH_SIZE" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    echo "## Experiment Configurations" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "### 1. Baseline (Single Pass)" >> "$REPORT_FILE"
    echo "- **Generator Mode:** Single pass with dummy attention map" >> "$REPORT_FILE"
    echo "- **Iterative Refinement:** Disabled" >> "$REPORT_FILE"
    echo "- **Output Directory:** 
ual_modal_gan/outputs/checkpoints_ablation/baseline" >> "$REPORT_FILE"
    echo "- **Sample Directory:** 
ual_modal_gan/outputs/samples_ablation/baseline" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    echo "### 2. Iterative Refinement (Two Pass)" >> "$REPORT_FILE"
    echo "- **Generator Mode:** Two-pass with attention-guided refinement" >> "$REPORT_FILE"
    echo "- **Pass 1:** Generate initial image with dummy attention" >> "$REPORT_FILE"
    echo "- **Pass 2:** Generate refined image using attention from Pass 1" >> "$REPORT_FILE"
    echo "- **Output Directory:** 
ual_modal_gan/outputs/checkpoints_ablation/iterative" >> "$REPORT_FILE"
    echo "- **Output Directory:** 
ual_modal_gan/outputs/samples_ablation/iterative" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    echo "## Results Summary" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # Extract results from baseline
    if [ -f "dual_modal_gan/outputs/checkpoints_ablation/baseline/metrics/training_metrics_fp32.json" ]; then
        echo "### Baseline Results" >> "$REPORT_FILE"
        echo "
```json
" >> "$REPORT_FILE"
        cat dual_modal_gan/outputs/checkpoints_ablation/baseline/metrics/training_metrics_fp32.json >> "$REPORT_FILE"
        echo "
```
" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
    
    # Extract results from iterative
    if [ -f "dual_modal_gan/outputs/checkpoints_ablation/iterative/metrics/training_metrics_fp32.json" ]; then
        echo "### Iterative Refinement Results" >> "$REPORT_FILE"
        echo "
```json
" >> "$REPORT_FILE"
        cat dual_modal_gan/outputs/checkpoints_ablation/iterative/metrics/training_metrics_fp32.json >> "$REPORT_FILE"
        echo "
```
" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
    
    echo "## Key Findings" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "- **Training Time:** [To be filled after experiments complete]" >> "$REPORT_FILE"
    echo "- **Best PSNR:** [To be filled after experiments complete]" >> "$REPORT_FILE"
    echo "- **Best CER:** [To be filled after experiments complete]" >> "$REPORT_FILE"
    echo "- **Early Stopping:** [To be filled after experiments complete]" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    echo "## Sample Images" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "Baseline samples are saved in: 
ual_modal_gan/outputs/samples_ablation/baseline/" >> "$REPORT_FILE"
    echo "Iterative refinement samples are saved in: 
ual_modal_gan/outputs/samples_ablation/iterative/" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    echo "## MLflow Tracking" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "Both experiments use MLflow for tracking. View results at:" >> "$REPORT_FILE"
    echo "
```bash
" >> "$REPORT_FILE"
    echo "poetry run mlflow ui" >> "$REPORT_FILE"
    echo "# Then visit http://localhost:5000" >> "$REPORT_FILE"
    echo "
```
" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    print_success "Report generated: $REPORT_FILE"
}

# Main execution
main() {
    print_header "Ablation Study: Baseline vs Iterative Refinement"
    echo "This study compares two generator architectures:"
    echo "1. Baseline: Single pass without iterative refinement"
    echo "2. Iterative: Two-pass with attention-guided refinement"
    echo ""
    
    # Check dependencies
    check_dependencies
    
    
    # Run experiments
    baseline_success=false
    iterative_success=false
    
    # Run baseline experiment
    if run_baseline; then
        baseline_success=true
        print_success "Baseline experiment completed"
    else
        print_error "Baseline experiment failed"
    fi
    
    # Run iterative experiment
    if run_iterative; then
        iterative_success=true
        print_success "Iterative refinement experiment completed"
    else
        print_error "Iterative refinement training failed"
    fi
    
    # Generate report
    if [ "$baseline_success" = true ] || [ "$iterative_success" = true ]; then
        generate_report
        
        print_header "Ablation Study Summary"
        if [ "$baseline_success" = true ]; then
            print_success "Baseline: ✓ Completed"
        else
            print_error "Baseline: ✗ Failed"
        fi
        
        if [ "$iterative_success" = true ]; then
            print_success "Iterative: ✓ Completed"
        else
            print_error "Iterative: ✗ Failed"
        fi
        
        echo ""
        echo "Next steps:"
        echo "1. Review the generated report"
        echo "2. Compare training metrics"
        echo "3. Analyze sample images"
        echo "4. Draw conclusions about iterative refinement effectiveness"
    else
        print_error "Both experiments failed. Please check the logs."
        exit 1
    fi
}

# Run main function
main "$@"
