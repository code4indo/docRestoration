#!/bin/bash
# Launcher script for Integrated Document Restoration + HTR System
# Usage: ./launch_integrated_system.sh [OPTIONS]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "  Integrated Document Restoration + HTR System"
echo "========================================================================"

# Default configuration
USE_GPU=true
PORT=7860
LOG_FILE="integrated_system.log"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            USE_GPU=false
            shift
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help)
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cpu          Use CPU instead of GPU"
            echo "  --gpu N        Use specific GPU (default: 0)"
            echo "  --port N       Use specific port (default: 7860)"
            echo "  --help         Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --gpu 0 --port 7860"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check prerequisites
echo -e "${YELLOW}[1/4] Checking prerequisites...${NC}"

if ! command -v poetry &> /dev/null; then
    echo -e "${RED}ERROR: Poetry not found${NC}"
    exit 1
fi

if [ "$USE_GPU" = true ]; then
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}WARNING: nvidia-smi not found, falling back to CPU${NC}"
        USE_GPU=false
    else
        echo -e "${GREEN}✓ GPU available${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    fi
fi

# Check model checkpoint
echo -e "${YELLOW}[2/4] Checking model checkpoint...${NC}"
CHECKPOINT_DIR="dual_modal_gan/checkpoints/production_full_coverage_vgg_v1"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo -e "${RED}ERROR: Checkpoint not found: $CHECKPOINT_DIR${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Model checkpoint found${NC}"

# Check Loghi HTR service (optional)
echo -e "${YELLOW}[3/4] Checking Loghi HTR service...${NC}"
HTR_ADDRESS="${LOGHI_ADDRESS:-http://localhost:5001}"
if curl -s --connect-timeout 2 "$HTR_ADDRESS" &> /dev/null; then
    echo -e "${GREEN}✓ Loghi HTR service is running at $HTR_ADDRESS${NC}"
else
    echo -e "${YELLOW}⚠ Loghi HTR service not detected at $HTR_ADDRESS${NC}"
    echo -e "${YELLOW}  HTR functionality will be unavailable (restoration will still work)${NC}"
fi

# Launch application
echo -e "${YELLOW}[4/4] Starting application...${NC}"

# Build command - use enhanced version
CMD="poetry run python dual_modal_gan/scripts/gradio_integrated_htr_enhanced.py"

# Add GPU configuration if enabled
if [ "$USE_GPU" = true ]; then
    GPU_ID="${GPU_ID:-0}"
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    echo -e "${GREEN}Using GPU: $GPU_ID${NC}"
else
    export CUDA_VISIBLE_DEVICES=""
    echo -e "${YELLOW}Using CPU${NC}"
fi

# Set Gradio configuration
export GRADIO_SERVER_PORT=$PORT

echo ""
echo -e "${GREEN}========================================================================"
echo "  Application Starting"
echo -e "========================================================================${NC}"
echo ""
echo "  URL: http://localhost:$PORT"
echo "  GPU: $([ "$USE_GPU" = true ] && echo "Enabled (GPU $GPU_ID)" || echo "Disabled (CPU)")"
echo "  HTR: $(curl -s --connect-timeout 2 "$HTR_ADDRESS" &> /dev/null && echo "Available" || echo "Unavailable")"
echo "  Log: $LOG_FILE"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Run in foreground (for easy Ctrl+C)
$CMD 2>&1 | tee "$LOG_FILE"
