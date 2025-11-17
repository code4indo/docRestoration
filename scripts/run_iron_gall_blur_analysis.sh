#!/bin/bash
# Skrip untuk menjalankan analisis slight blur pada region iron gall corrosion

set -e

# Aktivasi virtual environment
source .venv/bin/activate

echo "======================================"
echo "Iron Gall Blur Analysis"
echo "======================================"
echo ""

# Setup direktori
DEGRADED_DIR="DokumenRusak/forPaper"
RESTORED_DIR="DokumenRusak/forPaper_results"
OUTPUT_DIR="analysis_results/iron_gall_blur_$(date +%Y%m%d_%H%M%S)"

echo "Degraded images: $DEGRADED_DIR"
echo "Restored images: $RESTORED_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Jalankan analisis
poetry run python scripts/analyze_iron_gall_blur.py \
    --degraded-dir "$DEGRADED_DIR" \
    --restored-dir "$RESTORED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --blur-threshold-slight 100.0 \
    --blur-threshold-moderate 50.0

echo ""
echo "======================================"
echo "Analysis Complete!"
echo "======================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Check the following:"
echo "  1. $OUTPUT_DIR/analysis_report.json - Summary report"
echo "  2. $OUTPUT_DIR/analysis_*.png - Visualizations"
echo "  3. $OUTPUT_DIR/crops/ - Extracted crops for paper"
echo ""
