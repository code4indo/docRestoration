#!/bin/bash
################################################################################
# Resize Line Images to Model Format (1024×128)
# 
# Quick launcher script for resizing handwritten line images to model format
#
# Usage:
#   ./scripts/run_resize_lines.sh <input_dir> <output_dir> [mode]
#
# Examples:
#   ./scripts/run_resize_lines.sh DokumenRusak/lines_original DokumenRusak/lines_resized
#   ./scripts/run_resize_lines.sh DokumenRusak/lines_original DokumenRusak/lines_resized preserve_aspect
#   ./scripts/run_resize_lines.sh DokumenRusak/lines_original DokumenRusak/lines_resized with_border
#
################################################################################

set -e  # Exit on error

# Default values
MODE="${3:-preserve_aspect}"
BORDER_SIZE=10

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 2 ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo ""
    echo "Usage: $0 <input_dir> <output_dir> [mode]"
    echo ""
    echo "Modes:"
    echo "  preserve_aspect  - Maintain aspect ratio with padding (default)"
    echo "  stretch          - Stretch to fit (may distort)"
    echo "  with_border      - Maintain aspect ratio with guaranteed border"
    echo ""
    echo "Examples:"
    echo "  $0 DokumenRusak/lines_original DokumenRusak/lines_resized"
    echo "  $0 DokumenRusak/lines_original DokumenRusak/lines_resized preserve_aspect"
    echo "  $0 DokumenRusak/lines_original DokumenRusak/lines_resized with_border"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}Error: Input directory not found: $INPUT_DIR${NC}"
    exit 1
fi

# Print configuration
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Resize Line Images to Model Format${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Input:  ${YELLOW}$INPUT_DIR${NC}"
echo -e "Output: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "Mode:   ${YELLOW}$MODE${NC}"
echo -e "Target: ${YELLOW}1024×128 pixels (W×H)${NC}"
echo ""

# Activate virtual environment if exists
if [ -f ".venv/bin/activate" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

# Run resize script
echo -e "${GREEN}Processing images...${NC}"
echo ""

python scripts/resize_lines_to_model_format.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --mode "$MODE" \
    --border_size "$BORDER_SIZE" \
    --create_comparison \
    --num_samples 5

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Resize completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "Resized images saved to: ${YELLOW}$OUTPUT_DIR${NC}"
    echo -e "Comparison grid: ${YELLOW}$OUTPUT_DIR/comparison_grid.png${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  1. Review comparison grid to verify quality"
    echo "  2. Run inference on resized images:"
    echo -e "     ${YELLOW}python dual_modal_gan/scripts/inference_production_v3.py \\${NC}"
    echo -e "       ${YELLOW}--input_dir $OUTPUT_DIR \\${NC}"
    echo -e "       ${YELLOW}--output_dir results/inference_production_v3/lines_restored \\${NC}"
    echo -e "       ${YELLOW}--gpu_id 1 --image_ext .png${NC}"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Resize failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
