#!/bin/bash
# Ensemble Inference Script untuk DIBCO 2012
# Runs multiple checkpoints separately then fuses results

set -e

# Directories
TEST_DIR="dibco_datasets/2012/imgs"
GT_DIR="dibco_datasets/2012/gt_imgs"
OUTPUT_BASE="dual_modal_gan/outputs/ensemble_dibco2012"

# Checkpoints configuration
declare -a CHECKPOINTS=(
    "dual_modal_gan/checkpoints/dibco_transfer_from_production_v3_fixed/best_model/ckpt-214:0.30:dibco_transfer_best"
    "dual_modal_gan/checkpoints/dibco_transfer_learning_v2_enhanced/best_model/ckpt-119:0.25:dibco_v2_enhanced"
    "dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model/ckpt-88:0.25:production_v3_synthetic"
    "dual_modal_gan/checkpoints/dibco_pure_sota_comparison_v1/best_model/ckpt-115:0.20:dibco_pure_sota"
)

echo "=== ENSEMBLE INFERENCE FOR DIBCO 2012 ===" 
echo ""
echo "Test images: $TEST_DIR"
echo "Ground truth: $GT_DIR"
echo "Output base: $OUTPUT_BASE"
echo ""

# Step 1: Run inference for each checkpoint
echo "Step 1: Running inference for each checkpoint..."
echo ""

for entry in "${CHECKPOINTS[@]}"; do
    IFS=':' read -r ckpt_path weight name <<< "$entry"
    
    output_dir="$OUTPUT_BASE/${name}"
    mkdir -p "$output_dir"
    
    echo "→ Processing: $name (weight=$weight)"
    echo "  Checkpoint: $ckpt_path"
    echo "  Output: $output_dir"
    
    # Run inference using existing production script
    poetry run python dual_modal_gan/scripts/inference_production_v3.py \
        --checkpoint_path "$ckpt_path" \
        --input_dir "$TEST_DIR" \
        --output_dir "$output_dir" \
        --generator_version "enhanced" \
        --gpu_id "0" \
        > "$output_dir/inference.log" 2>&1
    
    echo "  ✅ Done"
    echo ""
done

# Step 2: Fuse results
echo "Step 2: Fusing results from all checkpoints..."
echo ""

poetry run python << 'PYTHON_SCRIPT'
import os
import numpy as np
import cv2
from pathlib import Path
import json

# Configuration
output_base = Path("dual_modal_gan/outputs/ensemble_dibco2012")
gt_dir = Path("dibco_datasets/2012/gt_imgs")

checkpoints_config = [
    ("dibco_transfer_best", 0.30),
    ("dibco_v2_enhanced", 0.25),
    ("production_v3_synthetic", 0.25),
    ("dibco_pure_sota", 0.20)
]

# Normalize weights
weights = np.array([w for _, w in checkpoints_config])
weights = weights / weights.sum()

print(f"Normalized weights: {weights}")
print()

# Get list of images from first checkpoint
first_checkpoint_dir = output_base / checkpoints_config[0][0]
image_files = sorted(list(first_checkpoint_dir.glob("*.png")))

if len(image_files) == 0:
    print("ERROR: No output images found!")
    exit(1)

print(f"Found {len(image_files)} images to fuse")
print()

# Create fusion output directory
fusion_dir = output_base / "ensemble_fused"
fusion_dir.mkdir(exist_ok=True)

# Fusion
results = []

for img_file in image_files:
    img_name = img_file.stem
    
    # Load all checkpoint outputs for this image
    outputs = []
    for checkpoint_name, _ in checkpoints_config:
        ckpt_dir = output_base / checkpoint_name
        ckpt_img_path = ckpt_dir / img_file.name
        
        if not ckpt_img_path.exists():
            print(f"WARNING: Missing {ckpt_img_path}, skipping this checkpoint")
            continue
        
        img = cv2.imread(str(ckpt_img_path), cv2.IMREAD_GRAYSCALE)
        outputs.append(img.astype(np.float32))
    
    if len(outputs) == 0:
        print(f"ERROR: No outputs for {img_name}")
        continue
    
    # Weighted fusion
    fused = np.zeros_like(outputs[0])
    for i, output in enumerate(outputs):
        if i < len(weights):
            fused += weights[i] * output
    
    # Clip and convert
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    
    # Save fused image
    output_path = fusion_dir / img_file.name
    cv2.imwrite(str(output_path), fused)
    
    # Calculate metrics if GT available
    gt_name = img_name.replace("_restored", "")
    gt_path = gt_dir / f"{gt_name}_gt.png"
    if not gt_path.exists():
        gt_path = gt_dir / f"{gt_name}.png"
    
    if gt_path.exists():
        gt_img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        
        # PSNR
        mse = np.mean((fused.astype(float) - gt_img.astype(float)) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        results.append({
            'image': img_name,
            'psnr': float(psnr)
        })
        
        print(f"{img_name}: PSNR = {psnr:.2f} dB")

# Summary
if results:
    avg_psnr = np.mean([r['psnr'] for r in results])
    
    print()
    print("=== ENSEMBLE RESULTS ===")
    print(f"Number of images: {len(results)}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print()
    print("Comparison with SOTA:")
    print(f"  DE-GAN (22.00 dB):   {avg_psnr - 22.00:+.2f} dB")
    print(f"  DocEnTR (22.29 dB):  {avg_psnr - 22.29:+.2f} dB")
    print(f"  Target (22.50 dB):   {avg_psnr - 22.50:+.2f} dB")
    
    # Save results
    results_path = fusion_dir / 'ensemble_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'checkpoints': [{'name': name, 'weight': float(w)} for name, w in checkpoints_config],
            'weights_normalized': weights.tolist(),
            'average_psnr': float(avg_psnr),
            'num_images': len(results),
            'results': results
        }, f, indent=2)
    
    print()
    print(f"Results saved to: {results_path}")
    print(f"Fused images saved to: {fusion_dir}")

PYTHON_SCRIPT

echo ""
echo "=== ENSEMBLE INFERENCE COMPLETE ==="
