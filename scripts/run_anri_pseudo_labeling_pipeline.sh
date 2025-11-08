#!/bin/bash
#
# ANRI Pseudo-Labeling Fine-tuning Pipeline
# Complete workflow for domain adaptation on real ANRI documents
#
# Author: Pseudo-Labeling Pipeline
# Date: 2025-10-28
#

set -e  # Exit on error

echo "================================================================================"
echo "ANRI PSEUDO-LABELING FINE-TUNING PIPELINE"
echo "================================================================================"
echo ""
echo "This pipeline will:"
echo "  1. Extract patches from 33 ANRI full pages"
echo "  2. Generate pseudo ground-truth with ckpt-99"
echo "  3. Filter patches by quality"
echo "  4. Create TFRecord dataset"
echo "  5. Fine-tune model (visual-only mode)"
echo "  6. Evaluate on held-out pages"
echo ""

# Configuration
GPU_ID=${GPU_ID:-0}
CHECKPOINT_DIR="dual_modal_gan/checkpoints/thin_stroke_preservation_v1_academic/best_model"
CHECKPOINT_NAME="ckpt-99"

echo "Configuration:"
echo "  GPU ID: $GPU_ID"
echo "  Base checkpoint: $CHECKPOINT_DIR/$CHECKPOINT_NAME"
echo ""

read -p "Press Enter to start, or Ctrl+C to cancel..."
echo ""

# Step 1: Extract patches
echo "================================================================================"
echo "STEP 1: EXTRACT PATCHES FROM FULL PAGES"
echo "================================================================================"
echo ""

poetry run python scripts/extract_patches_from_full_pages.py \
    --input_dir DokumenRusak/full_pages_ANRI \
    --output_dir DokumenRusak/anri_patches \
    --patch_size 256 \
    --overlap 64 \
    --train_ratio 0.85 \
    --seed 42

echo ""
echo "✅ Step 1 complete"
echo ""

# Step 2: Generate pseudo-labels
echo "================================================================================"
echo "STEP 2: GENERATE PSEUDO GROUND-TRUTH"
echo "================================================================================"
echo ""

poetry run python scripts/generate_pseudo_labels_patches.py \
    --input_dir DokumenRusak/anri_patches \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --checkpoint_name "$CHECKPOINT_NAME" \
    --gpu_id $GPU_ID

echo ""
echo "✅ Step 2 complete"
echo ""

# Step 3: Filter patches
echo "================================================================================"
echo "STEP 3: QUALITY FILTERING"
echo "================================================================================"
echo ""

poetry run python scripts/filter_pseudo_patches.py \
    --input_dir DokumenRusak/anri_patches \
    --output_dir DokumenRusak/anri_patches_filtered \
    --ssim_min 0.6 \
    --ssim_max 0.95 \
    --text_preservation_min 0.7 \
    --restored_std_min 20.0 \
    --top_percentile 0.7 \
    --inspection_samples 20

echo ""
echo "✅ Step 3 complete"
echo ""
echo "IMPORTANT: Check visual inspection samples in:"
echo "  DokumenRusak/anri_patches_filtered/visual_inspection/"
echo ""
read -p "Review samples and press Enter to continue, or Ctrl+C to stop..."
echo ""

# Step 4: Create TFRecord
echo "================================================================================"
echo "STEP 4: CREATE TFRECORD DATASET"
echo "================================================================================"
echo ""

poetry run python scripts/create_pseudo_tfrecord.py \
    --input_dir DokumenRusak/anri_patches_filtered \
    --output_dir DokumenRusak/anri_tfrecords \
    --verify

echo ""
echo "✅ Step 4 complete"
echo ""

# Step 5: Fine-tuning
echo "================================================================================"
echo "STEP 5: FINE-TUNING (VISUAL-ONLY MODE)"
echo "================================================================================"
echo ""

echo "Launching fine-tuning in background..."
echo "Monitor with: tail -f logs/finetune_anri_pseudo_v1.log"
echo ""

nohup ./scripts/universal_train_from_json.sh \
    configs/finetune_anri_pseudo_visual_only.json \
    > logs/finetune_anri_pseudo_v1.log 2>&1 &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"
echo ""
echo "Training started in background."
echo "This will take approximately 8-12 hours."
echo ""
echo "To monitor:"
echo "  tail -f logs/finetune_anri_pseudo_v1.log"
echo ""
echo "To stop:"
echo "  kill $TRAIN_PID"
echo ""

read -p "Wait for training to complete, then press Enter to continue with evaluation..."
echo ""

# Step 6: Evaluation
echo "================================================================================"
echo "STEP 6: EVALUATE ON HELD-OUT PAGES"
echo "================================================================================"
echo ""

poetry run python scripts/evaluate_finetuned_on_anri.py \
    --val_pages_dir DokumenRusak/full_pages_ANRI \
    --extraction_metadata DokumenRusak/anri_patches/extraction_metadata.json \
    --baseline_checkpoint_dir "$CHECKPOINT_DIR" \
    --baseline_checkpoint_name "$CHECKPOINT_NAME" \
    --finetuned_checkpoint_dir dual_modal_gan/checkpoints/finetune_anri_pseudo_v1/best_model \
    --finetuned_checkpoint_name ckpt-best \
    --output_dir results/anri_finetuning_evaluation \
    --gpu_id $GPU_ID

echo ""
echo "✅ Step 6 complete"
echo ""

# Summary
echo "================================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  Patches: DokumenRusak/anri_patches_filtered/"
echo "  TFRecords: DokumenRusak/anri_tfrecords/"
echo "  Fine-tuned model: dual_modal_gan/checkpoints/finetune_anri_pseudo_v1/"
echo "  Evaluation: results/anri_finetuning_evaluation/"
echo ""
echo "Check visual comparisons:"
echo "  results/anri_finetuning_evaluation/comparisons/"
echo ""
echo "Review metrics:"
echo "  results/anri_finetuning_evaluation/evaluation_results.json"
echo ""
