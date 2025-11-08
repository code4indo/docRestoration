#!/usr/bin/env python3
"""
Test Single Checkpoint on DIBCO 2012
=====================================
Verify if single best checkpoint really achieves 21.53 dB on test set.
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import json

# Add project root to path
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

def preprocess_image(img):
    """Preprocess image for model input"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img

def tile_inference(generator, img, tile_width=TARGET_LINE_WIDTH, tile_height=TARGET_LINE_HEIGHT):
    """Process image using tiling strategy"""
    h, w = img.shape
    
    # Calculate number of tiles
    num_tiles_x = int(np.ceil(w / tile_width))
    num_tiles_y = int(np.ceil(h / tile_height))
    
    # Create output canvas
    output = np.zeros_like(img)
    
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            # Extract tile
            x_start = i * tile_width
            x_end = min((i + 1) * tile_width, w)
            y_start = j * tile_height
            y_end = min((j + 1) * tile_height, h)
            
            tile = img[y_start:y_end, x_start:x_end]
            
            # Resize tile to model input size
            tile_resized = cv2.resize(tile, (tile_width, tile_height), 
                                     interpolation=cv2.INTER_AREA)
            
            # Prepare input
            tile_input = np.expand_dims(np.expand_dims(tile_resized, axis=-1), axis=0)
            
            # Run inference
            tile_output = generator(tile_input, training=False)
            tile_output = tile_output[0, :, :, 0].numpy()
            
            # Resize back
            tile_output = cv2.resize(tile_output, (x_end - x_start, y_end - y_start),
                                    interpolation=cv2.INTER_CUBIC)
            
            # Place in output
            output[y_start:y_end, x_start:x_end] = tile_output
    
    return output

def main():
    checkpoint_path = "dual_modal_gan/checkpoints/dibco_transfer_from_production_v3_fixed/best_model/ckpt-214"
    input_dir = "dibco_datasets/2012/imgs"
    gt_dir = "dibco_datasets/2012/gt_imgs"
    output_dir = "dual_modal_gan/outputs/single_ckpt214_dibco_2012"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("SINGLE CHECKPOINT TEST - DIBCO 2012")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
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
        
        # Preprocess
        img_preprocessed = preprocess_image(img)
        
        # Run inference
        restored = tile_inference(generator, img_preprocessed)
        
        # Denormalize to [0, 255]
        restored = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
        
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
        print(f"Single Model:        {avg_psnr:.2f} dB")
        print(f"Ensemble (4 ckpts):  18.80 dB")
        print(f"Difference:          {avg_psnr - 18.80:+.2f} dB")
        print("")
        print(f"DE-GAN (2020):       22.00 dB → Gap: {avg_psnr - 22.00:+.2f} dB")
        print(f"DocEnTR (2022):      22.29 dB → Gap: {avg_psnr - 22.29:+.2f} dB")
        print("=" * 80)
        
        # Save summary
        summary = {
            "checkpoint": checkpoint_path,
            "num_images": len(results),
            "average_psnr": float(avg_psnr),
            "std_psnr": float(std_psnr),
            "results": results
        }
        
        summary_file = os.path.join(output_dir, "single_model_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Summary saved: {summary_file}")

if __name__ == "__main__":
    main()
