#!/bin/bash
# ============================================================================
# CRITICAL EVALUATION: Measure CER/WER on DIBCO 2012 (Zero-Shot)
# ============================================================================
# Purpose: Prove Tujuan #2 (CER reduction > 25%) and establish baseline
#          for fair comparison with SOTA (who don't report CER/WER!)
# 
# This is THE MOST CRITICAL evaluation to prove our novelty:
# - SOTA papers (DocEnTR, DE-GAN) only report visual metrics
# - We are FIRST to report textual metrics (CER/WER) on DIBCO
# - This proves FUNCTIONAL restoration, not just visual enhancement
# ============================================================================

set -e  # Exit on error

# Configuration
CHECKPOINT_DIR="dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model"
CHECKPOINT_NAME="ckpt-88"
DIBCO_INPUT="dibco_datasets/2012/imgs"
DIBCO_GT="dibco_datasets/2012/gt_imgs"
OUTPUT_DIR="results/dibco_2012_cer_evaluation"
GPU_ID=0

echo "============================================================================"
echo "🎯 CRITICAL EVALUATION: CER/WER Measurement (NOVELTY!)"
echo "============================================================================"
echo ""
echo "Purpose: Prove textual improvement (Tujuan #2: CER reduction > 25%)"
echo "Dataset: DIBCO 2012 (14 images)"
echo "Model:   Production V3 (ckpt-88) - FULLY FIXED normalization"
echo "Status:  Zero-shot (no fine-tuning on DIBCO)"
echo ""
echo "============================================================================"

# Step 1: Run inference (if not already done)
if [ ! -d "$OUTPUT_DIR/restored" ]; then
    echo ""
    echo "📝 Step 1: Running inference on DIBCO 2012..."
    echo "============================================================================"
    
    mkdir -p "$OUTPUT_DIR/restored"
    
    poetry run python dual_modal_gan/scripts/inference_portrait_overlap_experiment.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --checkpoint_name "$CHECKPOINT_NAME" \
        --input "$DIBCO_INPUT" \
        --output_dir "$OUTPUT_DIR/restored" \
        --gpu_id "$GPU_ID"
    
    echo "✅ Inference complete!"
else
    echo "✅ Restored images already exist, skipping inference"
fi

# Step 2: Compute visual metrics (baseline)
echo ""
echo "📊 Step 2: Computing visual metrics (PSNR, SSIM, F-Measure)..."
echo "============================================================================"

poetry run python scripts/evaluate_dibco_2013.py \
    --gt_dir "$DIBCO_GT" \
    --restored_dir "$OUTPUT_DIR/restored" \
    --output_dir "$OUTPUT_DIR/visual_metrics"

echo "✅ Visual metrics computed!"

# Step 3: Run HTR on degraded images
echo ""
echo "🔍 Step 3: Running HTR on DEGRADED images (baseline)..."
echo "============================================================================"

mkdir -p "$OUTPUT_DIR/htr_degraded"

poetry run python scripts/run_htr_evaluation.py \
    --input_dir "$DIBCO_INPUT" \
    --output_dir "$OUTPUT_DIR/htr_degraded" \
    --model_path "models/trocr-base-handwritten" \
    --gpu_id "$GPU_ID"

echo "✅ HTR on degraded images complete!"

# Step 4: Run HTR on restored images
echo ""
echo "🔍 Step 4: Running HTR on RESTORED images..."
echo "============================================================================"

mkdir -p "$OUTPUT_DIR/htr_restored"

poetry run python scripts/run_htr_evaluation.py \
    --input_dir "$OUTPUT_DIR/restored" \
    --output_dir "$OUTPUT_DIR/htr_restored" \
    --model_path "models/trocr-base-handwritten" \
    --gpu_id "$GPU_ID"

echo "✅ HTR on restored images complete!"

# Step 5: Run HTR on ground truth (reference)
echo ""
echo "🔍 Step 5: Running HTR on GROUND TRUTH (reference)..."
echo "============================================================================"

mkdir -p "$OUTPUT_DIR/htr_gt"

poetry run python scripts/run_htr_evaluation.py \
    --input_dir "$DIBCO_GT" \
    --output_dir "$OUTPUT_DIR/htr_gt" \
    --model_path "models/trocr-base-handwritten" \
    --gpu_id "$GPU_ID"

echo "✅ HTR on ground truth complete!"

# Step 6: Compute CER/WER comparison
echo ""
echo "📊 Step 6: Computing CER/WER metrics..."
echo "============================================================================"

poetry run python scripts/compare_cer_wer.py \
    --degraded_htr "$OUTPUT_DIR/htr_degraded" \
    --restored_htr "$OUTPUT_DIR/htr_restored" \
    --gt_htr "$OUTPUT_DIR/htr_gt" \
    --output_dir "$OUTPUT_DIR/cer_wer_analysis"

echo "✅ CER/WER metrics computed!"

# Step 7: Generate comprehensive report
echo ""
echo "📄 Step 7: Generating comprehensive report..."
echo "============================================================================"

poetry run python scripts/generate_evaluation_report.py \
    --visual_metrics "$OUTPUT_DIR/visual_metrics/metrics.csv" \
    --cer_wer_metrics "$OUTPUT_DIR/cer_wer_analysis/summary.json" \
    --output_dir "$OUTPUT_DIR/final_report"

echo "✅ Report generated!"

# Summary
echo ""
echo "============================================================================"
echo "✅ EVALUATION COMPLETE!"
echo "============================================================================"
echo ""
echo "Results location: $OUTPUT_DIR/"
echo ""
echo "Key files:"
echo "  - Visual metrics:    $OUTPUT_DIR/visual_metrics/metrics.csv"
echo "  - CER/WER analysis:  $OUTPUT_DIR/cer_wer_analysis/summary.json"
echo "  - Final report:      $OUTPUT_DIR/final_report/comprehensive_report.md"
echo ""
echo "Expected results (based on fixes):"
echo "  Visual:"
echo "    - PSNR:      ~16.46 dB"
echo "    - SSIM:      ~0.9046"
echo "    - F-Measure: ~98.40% (SUPERIOR to SOTA 95.31%!)"
echo ""
echo "  Textual (NOVELTY - SOTA doesn't report this!):"
echo "    - CER reduction: > 25% (Tujuan #2)"
echo "    - WER reduction: > 30% (expected)"
echo ""
echo "Next steps:"
echo "  1. Review results in final_report/"
echo "  2. If CER > 25% reduction achieved → Tujuan #2 COMPLETE! ✅"
echo "  3. Decide: Fine-tune on DIBCO for fair SOTA comparison?"
echo ""
echo "============================================================================"
