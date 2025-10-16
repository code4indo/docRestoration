#!/bin/bash

# =============================================================================
# Bayesian Optimization Script for GAN-HTR Loss Weights
# =============================================================================
# This script runs Optuna-based hyperparameter optimization to find the optimal
# loss weights (pixel_loss_weight, rec_feat_loss_weight, adv_loss_weight)
# for the GAN-HTR model.
#
# Usage: ./run_bayesian_optimization.sh [n_trials]
# =============================================================================

set -e  # Exit on any error

# --- Configuration ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
HPO_DIR="${SCRIPTS_DIR}/hpo"
OBJECTIVE_SCRIPT="${HPO_DIR}/objective.py"
MONITOR_SCRIPT="${HPO_DIR}/monitor_hpo.py"

# --- Default Parameters ---
N_TRIALS=${1:-50}  # Default 50 trials if not specified
EPOCHS_PER_TRIAL=3    # Short epochs for fast iteration
BATCH_SIZE=4           # Safe batch size to avoid OOM
GPU_ID="1"             # GPU to use

# --- Color Codes for Output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Helper Functions ---
print_header() {
    echo -e "${BLUE}=====================================================================${NC}"
    echo -e "${BLUE}  GAN-HTR Bayesian Optimization - Loss Weights Search${NC}"
    echo -e "${BLUE}=====================================================================${NC}"
    echo ""
}

print_config() {
    echo -e "${GREEN}Configuration:${NC}"
    echo -e "  Project Root: ${PROJECT_ROOT}"
    echo -e "  Objective Script: ${OBJECTIVE_SCRIPT}"
    echo -e "  Monitor Script: ${MONITOR_SCRIPT}"
    echo -e "  Number of Trials: ${N_TRIALS}"
    echo -e "  Epochs per Trial: ${EPOCHS_PER_TRIAL}"
    echo -e "  Batch Size: ${BATCH_SIZE}"
    echo -e "  GPU ID: ${GPU_ID}"
    echo ""
}

print_status() {
    echo -e "${YELLOW}Status: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check if Python is available
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check if poetry is available
    if ! command -v poetry &> /dev/null; then
        print_error "Poetry is not installed or not in PATH"
        exit 1
    fi
    
    # Check if objective script exists
    if [[ ! -f "${OBJECTIVE_SCRIPT}" ]]; then
        print_error "Objective script not found: ${OBJECTIVE_SCRIPT}"
        exit 1
    fi
    
    # Check if monitor script exists
    if [[ ! -f "${MONITOR_SCRIPT}" ]]; then
        print_error "Monitor script not found: ${MONITOR_SCRIPT}"
        exit 1
    fi
    
    print_success "All dependencies checked"
    echo ""
}

setup_environment() {
    print_status "Setting up environment..."
    
    # Change to project root
    cd "${PROJECT_ROOT}"
    
    # Activate virtual environment
    source ".venv/bin/activate"
    
    # Set GPU environment variable
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    
    print_success "Environment setup complete"
    echo ""
}

run_optimization() {
    print_status "Starting Bayesian optimization..."
    echo -e "${YELLOW}This will run ${N_TRIALS} trials in the background.${NC}"
    echo -e "${YELLOW}Use the monitor script to track progress.${NC}"
    echo ""
    
    # Create output directories
    mkdir -p "${PROJECT_ROOT}/dual_modal_gan/outputs/hpo_checkpoints"
    mkdir -p "${PROJECT_ROOT}/dual_modal_gan/outputs/hpo_samples"
    
    # Run optimization in background
    nohup poetry run python "${OBJECTIVE_SCRIPT}" --n_trials "${N_TRIALS}" \
        > "${PROJECT_ROOT}/hpo_optimization.log" 2>&1 &
    
    OPTUNA_PID=$!
    echo "Optuna process started with PID: ${OPTUNA_PID}"
    echo "Log file: ${PROJECT_ROOT}/hpo_optimization.log"
    echo ""
    
    # Save PID for later reference
    echo "${OPTUNA_PID}" > "${PROJECT_ROOT}/optuna_pid.txt"
    
    print_success "Bayesian optimization started in background"
    echo ""
}

show_monitoring_commands() {
    echo -e "${GREEN}Monitoring Commands:${NC}"
    echo ""
    echo -e "1. Check process status:"
    echo -e "   ${YELLOW}ps aux | grep objective.py${NC}"
    echo ""
    echo -e "2. Monitor HPO progress:"
    echo -e "   ${YELLOW}poetry run python ${MONITOR_SCRIPT}${NC}"
    echo ""
    echo -e "3. View optimization logs:"
    echo -e "   ${YELLOW}tail -f ${PROJECT_ROOT}/hpo_optimization.log${NC}"
    echo ""
    echo -e "4. Stop optimization:"
    echo -e "   ${YELLOW}kill \$(cat ${PROJECT_ROOT}/optuna_pid.txt)${NC}"
    echo ""
    echo -e "5. View MLflow results:"
    echo -e "   ${YELLOW}poetry run mlflow ui${NC}"
    echo -e "   Then open: http://localhost:5000"
    echo ""
}

show_expected_output() {
    echo -e "${GREEN}Expected Output:${NC}"
    echo ""
    echo -e "The optimization will search for the best combination of:"
    echo -e "  • pixel_loss_weight: [1.0, 200.0]"
    echo -e "  • rec_feat_loss_weight: [1.0, 150.0]"
    echo -e "  • adv_loss_weight: [0.1, 10.0]"
    echo ""
    echo -e "Objective function to maximize:"
    echo -e "  ${YELLOW}score = PSNR + SSIM - (100.0 * CER)${NC}"
    echo ""
    echo -e "Best weights will be saved to Optuna study database:"
    echo -e "  ${YELLOW}${HPO_DIR}/optuna_study.db${NC}"
    echo ""
}

# --- Main Execution ---
main() {
    print_header
    print_config
    check_dependencies
    setup_environment
    run_optimization
    show_monitoring_commands
    show_expected_output
    
    echo -e "${GREEN}Bayesian optimization setup complete!${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "1. Monitor progress using: ${YELLOW}poetry run python ${MONITOR_SCRIPT}${NC}"
    echo -e "2. Wait for all ${N_TRIALS} trials to complete"
    echo -e "3. Analyze results using Optuna visualization tools"
    echo -e "4. Run full training with best weights found"
    echo ""
}

# --- Run Script ---
main "$@"