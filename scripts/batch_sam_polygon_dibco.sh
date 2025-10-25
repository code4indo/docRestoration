#!/bin/bash
# Batch SAM Polygon Segmentation for DIBCO Dataset
# Processes all DIBCO images and generates visualizations

set -e

PROJECT_ROOT="/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration"
INPUT_DIR="$PROJECT_ROOT/dibco_datasets/DIPCO2016_dataset"
OUTPUT_BASE="$PROJECT_ROOT/results/sam_polygon_dibco_batch"
SAM_CHECKPOINT="$PROJECT_ROOT/models/sam/sam_vit_b_01ec64.pth"

echo "============================================================"
echo "SAM POLYGON SEGMENTATION - DIBCO BATCH PROCESSING"
echo "============================================================"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_BASE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Count BMP files
num_files=$(find "$INPUT_DIR" -name "*.bmp" | wc -l)
echo "Found $num_files BMP images"
echo ""

# Process each image
current=0
total_polygons=0
total_lines=0

for img_path in "$INPUT_DIR"/*.bmp; do
    current=$((current + 1))
    img_name=$(basename "$img_path" .bmp)
    output_dir="$OUTPUT_BASE/$img_name"
    
    echo "[$current/$num_files] Processing: $img_name.bmp"
    
    # Run SAM polygon segmentation
    cd "$PROJECT_ROOT"
    poetry run python dual_modal_gan/scripts/test_sam_polygon.py \
        --input_image "$img_path" \
        --sam_checkpoint "$SAM_CHECKPOINT" \
        --output_dir "$output_dir" \
        2>&1 | tee "$output_dir/process.log" | \
        grep -E "(Detected|Segmented|vertices|coverage)" | head -15
    
    # Extract stats
    segmented=$(grep "Segmented:" "$output_dir/process.log" | tail -1 | awk '{print $3}')
    detected=$(grep "Detected.*text lines" "$output_dir/process.log" | tail -1 | awk '{print $2}')
    
    if [ -n "$segmented" ] && [ -n "$detected" ]; then
        total_polygons=$((total_polygons + segmented))
        total_lines=$((total_lines + detected))
        echo "  → $segmented polygons / $detected lines"
    fi
    
    echo ""
done

echo "============================================================"
echo "BATCH PROCESSING COMPLETED"
echo "============================================================"
echo "Total images: $num_files"
echo "Total lines detected: $total_lines"
echo "Total polygons extracted: $total_polygons"
if [ $total_lines -gt 0 ]; then
    polygon_rate=$(awk "BEGIN {printf \"%.1f\", ($total_polygons/$total_lines)*100}")
    echo "Polygon extraction rate: $polygon_rate%"
fi
echo ""
echo "Results saved to: $OUTPUT_BASE/"
echo ""

# Create summary HTML
echo "Generating summary visualization..."
cat > "$OUTPUT_BASE/index.html" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>SAM Polygon Segmentation - DIBCO Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .image-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .image-card img { width: 100%; border: 1px solid #ddd; }
        .image-card h3 { margin-top: 0; color: #0066cc; }
        .stats { background: #f0f0f0; padding: 10px; border-radius: 4px; margin: 10px 0; }
        .polygon { color: green; font-weight: bold; }
        .rectangle { color: #999; }
    </style>
</head>
<body>
    <h1>🔬 SAM Polygon Segmentation - DIBCO Dataset Results</h1>
    <p><strong>Dataset:</strong> DIBCO 2016 | <strong>Model:</strong> SAM ViT-B | <strong>Date:</strong> October 23, 2025</p>
    
    <div class="image-grid">
EOF

# Add each image to HTML
for img_dir in "$OUTPUT_BASE"/*/; do
    if [ -d "$img_dir" ] && [ "$img_dir" != "$OUTPUT_BASE/" ]; then
        img_name=$(basename "$img_dir")
        
        # Check if visualization exists
        if [ -f "$img_dir/polygon_vs_rectangle.png" ]; then
            # Extract stats from log
            segmented=$(grep "Segmented:" "$img_dir/process.log" 2>/dev/null | tail -1 | awk '{print $3}' || echo "0")
            total=$(grep "Total lines:" "$img_dir/process.log" 2>/dev/null | tail -1 | awk '{print $4}' || echo "0")
            coverage=$(grep "Avg mask coverage:" "$img_dir/process.log" 2>/dev/null | tail -1 | awk '{print $5}' || echo "0.0%")
            
            cat >> "$OUTPUT_BASE/index.html" << CARD
        <div class="image-card">
            <h3>Image: $img_name</h3>
            <img src="$img_name/polygon_vs_rectangle.png" alt="$img_name">
            <div class="stats">
                <strong>Lines detected:</strong> $total<br>
                <strong class="polygon">Polygons extracted:</strong> $segmented<br>
                <strong>Avg coverage:</strong> $coverage
            </div>
        </div>
CARD
        fi
    fi
done

cat >> "$OUTPUT_BASE/index.html" << 'EOF'
    </div>
    
    <hr>
    <p><em>Green polygons = SAM segmentation following text contour | Red rectangles = Projection profile detection</em></p>
</body>
</html>
EOF

echo "✓ Summary HTML generated: $OUTPUT_BASE/index.html"
echo ""
echo "To view results, open: file://$OUTPUT_BASE/index.html"
