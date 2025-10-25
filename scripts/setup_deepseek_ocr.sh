#!/bin/bash
###############################################################################
# DeepSeek-OCR Line Detection Setup & Test Script
###############################################################################
#
# This script sets up a separate Python environment for DeepSeek-OCR to avoid
# conflicts between PyTorch and TensorFlow.
#
# Usage:
#   ./setup_deepseek_ocr.sh
#
###############################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEEPSEEK_ENV="$PROJECT_ROOT/.venv_deepseek"

echo "=============================================================================="
echo "DeepSeek-OCR Environment Setup"
echo "=============================================================================="
echo ""

# Step 1: Create separate virtual environment for DeepSeek-OCR
if [ ! -d "$DEEPSEEK_ENV" ]; then
    echo "📦 Creating separate virtual environment for DeepSeek-OCR..."
    python3 -m venv "$DEEPSEEK_ENV"
    echo "✓ Virtual environment created: $DEEPSEEK_ENV"
else
    echo "✓ Virtual environment already exists: $DEEPSEEK_ENV"
fi

# Step 2: Activate environment
echo ""
echo "🔧 Activating DeepSeek-OCR environment..."
source "$DEEPSEEK_ENV/bin/activate"

# Step 3: Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Step 4: Install dependencies
echo ""
echo "📦 Installing DeepSeek-OCR dependencies..."
echo "  This may take a few minutes..."

# Install PyTorch (CPU-only to avoid large download)
echo "  - Installing PyTorch (CPU version)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1

# Install transformers and other deps
echo "  - Installing transformers..."
pip install transformers > /dev/null 2>&1

echo "  - Installing supporting libraries..."
pip install einops addict easydict pillow opencv-python > /dev/null 2>&1

# Optional: Flash Attention (commented out, requires compilation)
# echo "  - Installing flash-attn (optional, may take time)..."
# pip install flash-attn --no-build-isolation

echo ""
echo "✓ All dependencies installed successfully!"

# Step 5: Test installation
echo ""
echo "=============================================================================="
echo "Testing DeepSeek-OCR Installation"
echo "=============================================================================="
echo ""

python3 << 'PYTEST'
import sys
try:
    import torch
    import transformers
    import einops
    import addict
    import easydict
    import cv2
    from PIL import Image
    
    print("✓ PyTorch version:", torch.__version__)
    print("✓ Transformers version:", transformers.__version__)
    print("✓ All dependencies loaded successfully!")
    print("")
    print("Device availability:")
    print("  - CUDA available:", torch.cuda.is_available())
    print("  - CPU available: True")
    
    sys.exit(0)
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
PYTEST

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================================================="
    echo "✓ SETUP COMPLETE!"
    echo "=============================================================================="
    echo ""
    echo "To use DeepSeek-OCR:"
    echo ""
    echo "  1. Activate the environment:"
    echo "     source $DEEPSEEK_ENV/bin/activate"
    echo ""
    echo "  2. Run line detection test:"
    echo "     python dual_modal_gan/scripts/line_detection_deepseek.py \\"
    echo "       --input dibco_datasets/DIPCO2016_dataset/1.bmp \\"
    echo "       --output_dir results/line_detection_deepseek \\"
    echo "       --device cpu"
    echo ""
    echo "  3. Deactivate when done:"
    echo "     deactivate"
    echo ""
else
    echo ""
    echo "❌ Setup failed. Please check the error messages above."
    exit 1
fi
