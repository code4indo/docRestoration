#!/bin/bash
# Required data files for GAN-HTR training - Download from HuggingFace

echo "=== Checking Required Data Files ==="
echo ""

# Check if required files exist
MISSING_FILES=0

echo "1. Checking dataset..."
if [ -f "dual_modal_gan/data/dataset_gan.tfrecord" ]; then
    SIZE=$(du -h dual_modal_gan/data/dataset_gan.tfrecord | cut -f1)
    echo "   ✅ Dataset found: $SIZE"
else
    echo "   ❌ Dataset NOT found: dual_modal_gan/data/dataset_gan.tfrecord"
    echo "   → Will download from HuggingFace (~5GB)"
    MISSING_FILES=$((MISSING_FILES + 1))
fi

echo "2. Checking model weights..."
if [ -f "models/best_htr_recognizer/best_model.weights.h5" ]; then
    SIZE=$(du -h models/best_htr_recognizer/best_model.weights.h5 | cut -f1)
    echo "   ✅ Model weights found: $SIZE"
else
    echo "   ❌ Model weights NOT found: models/best_htr_recognizer/best_model.weights.h5"
    echo "   → Will download from HuggingFace (~500MB)"
    MISSING_FILES=$((MISSING_FILES + 1))
fi

echo "3. Checking character list..."
if [ -f "real_data_preparation/real_data_charlist.txt" ]; then
    SIZE=$(du -h real_data_preparation/real_data_charlist.txt | cut -f1)
    echo "   ✅ Character list found: $SIZE"
else
    echo "   ❌ Character list NOT found: real_data_preparation/real_data_charlist.txt"
    echo "   → Will download from HuggingFace (~1KB)"
    MISSING_FILES=$((MISSING_FILES + 1))
fi

echo ""
if [ $MISSING_FILES -eq 0 ]; then
    echo "🎉 All required data files present!"
    echo ""
    echo "You can run training directly:"
    echo "  ./scripts/train32_smoke_test.sh"
else
    echo "⚠️  $MISSING_FILES required files missing."
    echo ""
    echo "Download data from HuggingFace:"
    echo "  python scripts/download_from_huggingface.py"
    echo ""
    echo "This will download ~5.2GB of data (one-time only)."
fi

echo ""
echo "=== Check Complete ==="
