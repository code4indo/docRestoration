#!/bin/bash

# Quick access to fine-tuning dataset visualization

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         FINE-TUNING DATASET VISUALIZATION                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

VIS_DIR="visualization/finetuning"
HTML_FILE="$VIS_DIR/index.html"

echo "📁 Visualization files:"
echo ""
ls -lh "$VIS_DIR" | grep -E '\.(png|html)$' | awk '{print "   " $9 " (" $5 ")"}'
echo ""

if [ -f "$HTML_FILE" ]; then
    echo "🌐 Opening visualization in browser..."
    echo ""
    
    # Try to open in browser (works on most Linux systems with GUI)
    if command -v xdg-open &> /dev/null; then
        xdg-open "$HTML_FILE" 2>/dev/null &
        echo "   ✅ Opened: $HTML_FILE"
    elif command -v firefox &> /dev/null; then
        firefox "$HTML_FILE" 2>/dev/null &
        echo "   ✅ Opened in Firefox: $HTML_FILE"
    elif command -v google-chrome &> /dev/null; then
        google-chrome "$HTML_FILE" 2>/dev/null &
        echo "   ✅ Opened in Chrome: $HTML_FILE"
    else
        echo "   ℹ️  No browser detected. Please open manually:"
        echo "      file://$(pwd)/$HTML_FILE"
    fi
    
    echo ""
    echo "📊 Or view individual images:"
    echo "   eog $VIS_DIR/pair_comparison_*.png"
    echo "   eog $VIS_DIR/finetuning_preview_*.png"
else
    echo "❌ Visualization not found. Generate it first:"
    echo "   poetry run python scripts/visualize_finetuning_strips.py"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
