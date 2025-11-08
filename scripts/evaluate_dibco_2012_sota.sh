#!/bin/bash
# DIBCO 2012 EVALUATION - PROVE SOTA SUPERIORITY
# Evaluate best checkpoint on held-out DIBCO 2012 test set

set -e

echo "🎯 DIBCO 2012 TEST SET EVALUATION"
echo "=================================="
echo ""

# Configuration
CHECKPOINT_DIR="dual_modal_gan/checkpoints/dibco_pure_sota_comparison_v1/best_model"
DIBCO_2012_INPUT="dibco_datasets/2012/imgs"
DIBCO_2012_GT="dibco_datasets/2012/gt_imgs"
OUTPUT_DIR="results/dibco_2012_pure_sota_evaluation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Error: Best model checkpoint not found at $CHECKPOINT_DIR"
    echo "   Training may not be complete yet."
    exit 1
fi

# Find checkpoint file
CKPT_FILE=$(ls -t "$CHECKPOINT_DIR"/ckpt-*.index 2>/dev/null | head -1 | sed 's/.index$//')
if [ -z "$CKPT_FILE" ]; then
    echo "❌ Error: No checkpoint file found in $CHECKPOINT_DIR"
    exit 1
fi

CKPT_NAME=$(basename "$CKPT_FILE")
echo "✅ Found checkpoint: $CKPT_NAME"
echo ""

echo "📊 EVALUATION SETUP:"
echo "   Model:         Pure DIBCO fine-tuned (Enhanced V2)"
echo "   Checkpoint:    $CKPT_NAME"
echo "   Test dataset:  DIBCO 2012 (14 images, held-out)"
echo "   Output dir:    $OUTPUT_DIR"
echo ""

echo "🏆 SOTA BASELINES TO COMPARE:"
echo "   DocEnTR:   22.29 dB PSNR, 95.31% F-Measure"
echo "   DE-GAN:    22.00 dB PSNR, 95.18% F-Measure"
echo "   Souibgui:  ~22-23 dB PSNR"
echo ""

echo "📈 OUR ZERO-SHOT BASELINE:"
echo "   ckpt-88:   16.46 dB PSNR, 98.40% F-Measure"
echo "   Expected improvement: +6 dB (to 20-23 dB range)"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR/restored"
mkdir -p "$OUTPUT_DIR/metrics"

echo "🚀 Step 1: Running inference on DIBCO 2012 test set..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run inference
poetry run python dual_modal_gan/scripts/inference_portrait_overlap_experiment.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --checkpoint_name "$CKPT_NAME" \
    --input "$DIBCO_2012_INPUT" \
    --output_dir "$OUTPUT_DIR/restored" \
    --gpu_id 0 \
    --image_ext .png \
    --output_format png

echo ""
echo "✅ Inference completed!"
echo ""

echo "📊 Step 2: Calculating metrics (PSNR, SSIM, F-Measure)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create Python script for metric calculation
cat > "$OUTPUT_DIR/calculate_metrics.py" << 'PYTHON_SCRIPT'
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import json

def calculate_f_measure(restored, gt, threshold=127):
    """Calculate F-Measure for binarization quality"""
    # Binarize images
    _, restored_bin = cv2.threshold(restored, threshold, 255, cv2.THRESH_BINARY)
    _, gt_bin = cv2.threshold(gt, threshold, 255, cv2.THRESH_BINARY)
    
    # Calculate TP, FP, FN
    tp = np.sum((restored_bin == 255) & (gt_bin == 255))
    fp = np.sum((restored_bin == 255) & (gt_bin == 0))
    fn = np.sum((restored_bin == 0) & (gt_bin == 255))
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F-Measure
    f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f_measure, precision, recall

def evaluate_dibco_2012(restored_dir, gt_dir, output_file):
    restored_dir = Path(restored_dir)
    gt_dir = Path(gt_dir)
    
    results = []
    
    # Process each image
    for gt_path in sorted(gt_dir.glob("*.png")):
        img_name = gt_path.stem
        restored_path = restored_dir / f"{img_name}.png"
        
        if not restored_path.exists():
            print(f"⚠️  Skipping {img_name}: restored image not found")
            continue
        
        # Load images
        gt_img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        restored_img = cv2.imread(str(restored_path), cv2.IMREAD_GRAYSCALE)
        
        # Resize if needed
        if gt_img.shape != restored_img.shape:
            restored_img = cv2.resize(restored_img, (gt_img.shape[1], gt_img.shape[0]))
        
        # Calculate metrics
        psnr_val = psnr(gt_img, restored_img, data_range=255)
        ssim_val = ssim(gt_img, restored_img, data_range=255)
        f_measure, precision, recall = calculate_f_measure(restored_img, gt_img)
        
        result = {
            "image": img_name,
            "psnr": float(psnr_val),
            "ssim": float(ssim_val),
            "f_measure": float(f_measure),
            "precision": float(precision),
            "recall": float(recall)
        }
        results.append(result)
        
        print(f"✅ {img_name}: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}, F-Measure={f_measure:.4f}")
    
    # Calculate averages
    avg_psnr = np.mean([r["psnr"] for r in results])
    avg_ssim = np.mean([r["ssim"] for r in results])
    avg_f_measure = np.mean([r["f_measure"] for r in results])
    std_psnr = np.std([r["psnr"] for r in results])
    std_ssim = np.std([r["ssim"] for r in results])
    std_f_measure = np.std([r["f_measure"] for r in results])
    
    summary = {
        "total_images": len(results),
        "average": {
            "psnr": float(avg_psnr),
            "ssim": float(avg_ssim),
            "f_measure": float(avg_f_measure)
        },
        "std": {
            "psnr": float(std_psnr),
            "ssim": float(std_ssim),
            "f_measure": float(std_f_measure)
        },
        "per_image": results
    }
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total images: {len(results)}")
    print(f"   Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"   Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"   Average F-Measure: {avg_f_measure:.4f} ± {std_f_measure:.4f}")
    
    return summary

if __name__ == "__main__":
    import sys
    restored_dir = sys.argv[1]
    gt_dir = sys.argv[2]
    output_file = sys.argv[3]
    
    evaluate_dibco_2012(restored_dir, gt_dir, output_file)
PYTHON_SCRIPT

# Run metric calculation
poetry run python "$OUTPUT_DIR/calculate_metrics.py" \
    "$OUTPUT_DIR/restored" \
    "$DIBCO_2012_GT" \
    "$OUTPUT_DIR/metrics/results_${TIMESTAMP}.json"

echo ""
echo "✅ Metrics calculated!"
echo ""

# Display results comparison
echo "🏆 RESULTS COMPARISON WITH SOTA:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Parse results
RESULTS_FILE="$OUTPUT_DIR/metrics/results_${TIMESTAMP}.json"
if [ -f "$RESULTS_FILE" ]; then
    OUR_PSNR=$(python3 -c "import json; print(f\"{json.load(open('$RESULTS_FILE'))['average']['psnr']:.2f}\")")
    OUR_SSIM=$(python3 -c "import json; print(f\"{json.load(open('$RESULTS_FILE'))['average']['ssim']:.4f}\")")
    OUR_FM=$(python3 -c "import json; print(f\"{json.load(open('$RESULTS_FILE'))['average']['f_measure']:.4f}\")")
    
    echo "| Method            | PSNR (dB) | SSIM   | F-Measure | Status      |"
    echo "|-------------------|-----------|--------|-----------|-------------|"
    echo "| DocEnTR (SOTA)    | 22.29     | -      | 0.9531    | Baseline    |"
    echo "| DE-GAN (SOTA)     | 22.00     | -      | 0.9518    | Baseline    |"
    echo "| Ours (zero-shot)  | 16.46     | 0.9046 | 0.9840    | Before FT   |"
    echo "| Ours (pure DIBCO) | $OUR_PSNR     | $OUR_SSIM | $OUR_FM    | 🔥 THIS RUN |"
    echo ""
    
    # Determine status
    if (( $(echo "$OUR_PSNR > 22.50" | bc -l) )); then
        echo "🏆🎉 CONGRATULATIONS! NEW SOTA CHAMPION!"
        echo "   Your Enhanced V2 architecture BEAT all baselines!"
        echo "   Improvement over DocEnTR: +$(echo "$OUR_PSNR - 22.29" | bc -l) dB"
    elif (( $(echo "$OUR_PSNR > 22.29" | bc -l) )); then
        echo "🎉 SUCCESS! Beat DocEnTR (SOTA #1)!"
        echo "   Improvement: +$(echo "$OUR_PSNR - 22.29" | bc -l) dB"
    elif (( $(echo "$OUR_PSNR > 22.00" | bc -l) )); then
        echo "✅ SUCCESS! Beat DE-GAN (SOTA #2)!"
        echo "   Improvement: +$(echo "$OUR_PSNR - 22.00" | bc -l) dB"
    elif (( $(echo "$OUR_PSNR > 20.00" | bc -l) )); then
        echo "✅ GOOD! Competitive with SOTA (within 2 dB)"
        echo "   Gap to SOTA: -$(echo "22.29 - $OUR_PSNR" | bc -l) dB"
    else
        echo "⚠️  Below SOTA target, but significant improvement over zero-shot"
        echo "   Improvement from baseline: +$(echo "$OUR_PSNR - 16.46" | bc -l) dB"
    fi
    
    echo ""
    echo "📁 Results saved to:"
    echo "   Metrics: $RESULTS_FILE"
    echo "   Images:  $OUTPUT_DIR/restored/"
    echo ""
fi

echo "✅ Evaluation complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Review visual quality: check $OUTPUT_DIR/restored/"
echo "   2. Analyze per-image metrics in $RESULTS_FILE"
echo "   3. Prepare publication materials"
echo "   4. Document architecture advantages"
echo ""

exit 0
