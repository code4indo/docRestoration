#!/bin/bash

# Quick access to ALL 155 fine-tuning samples

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         ALL 155 FINE-TUNING SAMPLES VIEWER                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

SAMPLES_DIR="visualization/finetuning/all_samples"
HTML_FILE="$SAMPLES_DIR/all_samples.html"

if [ ! -f "$HTML_FILE" ]; then
    echo "❌ Samples not extracted yet!"
    echo ""
    echo "📦 Run extraction first:"
    echo "   poetry run python scripts/extract_all_finetuning_samples.py"
    echo ""
    exit 1
fi

echo "✅ All 155 samples extracted!"
echo ""

# Count files
TRAIN_COUNT=$(ls $SAMPLES_DIR/train/pairs/*.png 2>/dev/null | wc -l)
VAL_COUNT=$(ls $SAMPLES_DIR/val/pairs/*.png 2>/dev/null | wc -l)
TEST_COUNT=$(ls $SAMPLES_DIR/test/pairs/*.png 2>/dev/null | wc -l)
TOTAL=$((TRAIN_COUNT + VAL_COUNT + TEST_COUNT))

echo "📊 Dataset Summary:"
echo "   Train:      $TRAIN_COUNT samples"
echo "   Validation: $VAL_COUNT samples"
echo "   Test:       $TEST_COUNT samples"
echo "   ─────────────────────────"
echo "   TOTAL:      $TOTAL samples"
echo ""

echo "📁 File Structure:"
echo "   ├─ train/   (108 samples)"
echo "   │  ├─ clean/       - Ground truth images"
echo "   │  ├─ degraded/    - Input degraded images"
echo "   │  └─ pairs/       - Side-by-side comparisons"
echo "   ├─ val/     (23 samples)"
echo "   └─ test/    (24 samples)"
echo ""

echo "🌐 Opening interactive browser view..."
echo ""

# Open in browser
if command -v xdg-open &> /dev/null; then
    xdg-open "$HTML_FILE" 2>/dev/null &
    echo "   ✅ Opened: $HTML_FILE"
elif command -v firefox &> /dev/null; then
    firefox "$HTML_FILE" 2>/dev/null &
    echo "   ✅ Opened in Firefox"
elif command -v google-chrome &> /dev/null; then
    google-chrome "$HTML_FILE" 2>/dev/null &
    echo "   ✅ Opened in Chrome"
else
    echo "   ℹ️  No browser detected. Open manually:"
    echo "      file://$(pwd)/$HTML_FILE"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "🎯 QUICK ACTIONS"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "View grid visualizations:"
echo "  eog $SAMPLES_DIR/train_grid.png    # 50 train samples"
echo "  eog $SAMPLES_DIR/val_grid.png      # 23 val samples"
echo "  eog $SAMPLES_DIR/test_grid.png     # 24 test samples"
echo ""
echo "Browse individual folders:"
echo "  nautilus $SAMPLES_DIR/train/pairs  # All train pairs"
echo "  nautilus $SAMPLES_DIR/val/pairs    # All val pairs"
echo "  nautilus $SAMPLES_DIR/test/pairs   # All test pairs"
echo ""
echo "Count samples:"
echo "  ls $SAMPLES_DIR/train/pairs | wc -l"
echo "  ls $SAMPLES_DIR/val/pairs | wc -l"
echo "  ls $SAMPLES_DIR/test/pairs | wc -l"
echo ""
