#!/usr/bin/env python3
"""
Test Single Checkpoint with EXACT Training Preprocessing
=========================================================
NO contrast stretching, NO extra normalization.
HANYA simple /255.0 then *2-1 untuk match training exactly.
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from dual_modal_gan.src.models.generator_enhanced import unet_enhanced

TARGET_LINE_WIDTH = 128
TARGET_LINE_HEIGHT = 1024

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def preprocess_image_training_mode(img):
    """
    EXACT preprocessing seperti training pipeline.
    NO contrast stretching, NO fancy normalization.
    
    Training pipeline:
    1. TFRecord data already in [0, 1]
    2. Normalize to [-1, 1] for tanh generator: x * 2 - 1
    
    Inference pipeline (to match):
    1. Load image [0, 255] uint8
    2. Normalize to [0, 1]: x / 255.0
    3. Normalize to [-1, 1]: x * 2 - 1
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: [0, 255] uint8 → [0, 1] float32
    img_normalized = img.astype(np.float32) / 255.0
    
    # Step 2: [0, 1] → [-1, 1] (for tanh generator)
    img_tanh = img_normalized * 2.0 - 1.0
    
    return img_tanh

def tile_inference_training_mode(generator, img_tanh, tile_width=TARGET_LINE_WIDTH, tile_height=TARGET_LINE_HEIGHT):
    """
    Process image using tiling with TRAINING preprocessing.
    Input: img_tanh in [-1, 1] range (same as training)
    Output: restored in [0, 1] range (denormalized from [-1, 1])
    """
    h, w = img_tanh.shape
    
    num_tiles_x = int(np.ceil(w / tile_width))
    num_tiles_y = int(np.ceil(h / tile_height))
    
    output = np.zeros_like(img_tanh)
    
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            x_start = i * tile_width
            x_end = min((i + 1) * tile_width, w)
            y_start = j * tile_height
            y_end = min((j + 1) * tile_height, h)
            
            tile = img_tanh[y_start:y_end, x_start:x_end]
            
            # Resize tile to model input size
            tile_resized = cv2.resize(tile, (tile_width, tile_height), 
                                     interpolation=cv2.INTER_AREA)
            
            # Prepare input (already in [-1, 1])
            tile_input = np.expand_dims(np.expand_dims(tile_resized, axis=-1), axis=0)
            
            # Run inference (generator outputs [-1, 1])
            tile_output = generator(tile_input, training=False)
            tile_output = tile_output[0, :, :, 0].numpy()
            
            # Resize back
            tile_output = cv2.resize(tile_output, (x_end - x_start, y_end - y_start),
                                    interpolation=cv2.INTER_CUBIC)
            
            output[y_start:y_end, x_start:x_end] = tile_output
    
    # Denormalize from [-1, 1] to [0, 1]
    output_normalized = (output + 1.0) / 2.0
    
    return output_normalized

def main():
    checkpoint_path = "dual_modal_gan/checkpoints/dibco_transfer_from_production_v3_fixed/best_model/ckpt-214"
    input_dir = "dibco_datasets/2012/imgs"
    gt_dir = "dibco_datasets/2012/gt_imgs"
    output_dir = "dual_modal_gan/outputs/single_ckpt214_training_preproc"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("SINGLE CHECKPOINT TEST - EXACT TRAINING PREPROCESSING")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Preprocessing: /255.0 → *2-1 (NO contrast stretching)")
    print(f"Test set: {input_dir}")
    print("")
    
    # Load generator
    print("Loading generator...")
    generator = unet_enhanced(input_size=(TARGET_LINE_HEIGHT, TARGET_LINE_WIDTH, 1))
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(checkpoint_path).expect_partial()
    print("✅ Generator loaded\n")
    
    results = []
    
    # Process each image
    for img_name in sorted(os.listdir(input_dir)):
        if not img_name.endswith('.png'):
            continue
        
        print(f"Processing: {img_name}")
        
        # Load input
        input_path = os.path.join(input_dir, img_name)
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        print(f"  Raw image: [{img.min()}, {img.max()}], mean={img.mean():.1f}")
        
        # Preprocess (EXACT training mode)
        img_tanh = preprocess_image_training_mode(img)
        print(f"  After preproc: [{img_tanh.min():.3f}, {img_tanh.max():.3f}], mean={img_tanh.mean():.3f}")
        
        # Run inference
        restored_normalized = tile_inference_training_mode(generator, img_tanh)
        
        # Convert to [0, 255] for saving
        restored = np.clip(restored_normalized * 255.0, 0, 255).astype(np.uint8)
        
        # Save output
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, restored)
        
        # Calculate PSNR
        gt_path = os.path.join(gt_dir, img_name)
        if os.path.exists(gt_path):
            gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            
            if gt_image.shape != restored.shape:
                gt_image = cv2.resize(gt_image, (restored.shape[1], restored.shape[0]),
                                     interpolation=cv2.INTER_AREA)
            
            psnr = calculate_psnr(restored, gt_image)
            results.append({'image': img_name, 'psnr': psnr})
            print(f"  → PSNR: {psnr:.2f} dB\n")
        else:
            print(f"  → GT not found\n")
    
    # Calculate statistics
    if results:
        psnrs = [r['psnr'] for r in results]
        avg_psnr = np.mean(psnrs)
        std_psnr = np.std(psnrs)
        
        print("=" * 80)
        print("📊 FINAL RESULTS")
        print("=" * 80)
        print(f"Average PSNR:  {avg_psnr:.2f} ± {std_psnr:.2f} dB")
        print(f"Min PSNR:      {np.min(psnrs):.2f} dB")
        print(f"Max PSNR:      {np.max(psnrs):.2f} dB")
        print(f"Images:        {len(results)}/14")
        
        print("\n" + "=" * 80)
        print("🎯 COMPARISON")
        print("=" * 80)
        print(f"Previous (with contrast stretch):  14.00 dB  ❌")
        print(f"Current (training preproc):        {avg_psnr:.2f} dB")
        print(f"Improvement:                       {avg_psnr - 14.00:+.2f} dB")
        print("")
        print(f"Validation PSNR (reported):        21.53 dB")
        print(f"Test PSNR (this run):              {avg_psnr:.2f} dB")
        print(f"Gap (overfitting):                 {avg_psnr - 21.53:+.2f} dB")
        print("")
        print(f"DE-GAN (2020):       22.00 dB → Gap: {avg_psnr - 22.00:+.2f} dB")
        print(f"DocEnTR (2022):      22.29 dB → Gap: {avg_psnr - 22.29:+.2f} dB")
        print("=" * 80)
        
        # Save summary
        summary = {
            "preprocessing": "training_mode_exact (no contrast stretching)",
            "checkpoint": checkpoint_path,
            "num_images": len(results),
            "average_psnr": float(avg_psnr),
            "std_psnr": float(std_psnr),
            "min_psnr": float(np.min(psnrs)),
            "max_psnr": float(np.max(psnrs)),
            "results": results
        }
        
        summary_file = os.path.join(output_dir, "training_preproc_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Summary saved: {summary_file}")

if __name__ == "__main__":
    main()
