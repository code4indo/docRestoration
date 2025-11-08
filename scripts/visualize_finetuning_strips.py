#!/usr/bin/env python3
"""
Visualize Fine-tuning Dataset Strips

This script:
1. Shows original full-page images (GT vs Degraded)
2. Extracts and displays sample strips
3. Shows how sequential split works
4. Generates preview of augmentation
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def extract_strips_preview(image_path, strip_height=128, max_strips=5):
    """Extract first few strips for preview"""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    h, w = image.shape
    strips = []
    
    y = 0
    count = 0
    while y + strip_height <= h and count < max_strips:
        strip = image[y:y+strip_height, :]
        strips.append(strip)
        y += strip_height
        count += 1
    
    return strips, (h, w)

def resize_strip_for_model(strip, target_width=1024):
    """Resize strip to model input size"""
    h, w = strip.shape
    resized = cv2.resize(strip, (target_width, 128), interpolation=cv2.INTER_AREA)
    return resized

def augment_strip_preview(image):
    """Show augmentation examples"""
    augmented = []
    labels = []
    
    # Original
    augmented.append(image)
    labels.append("Original")
    
    # Horizontal flip
    augmented.append(cv2.flip(image, 1))
    labels.append("H-Flip")
    
    # Brightness +10%
    aug = np.clip(image * 1.1, 0, 255).astype(np.uint8)
    augmented.append(aug)
    labels.append("Bright +10%")
    
    # Brightness -10%
    aug = np.clip(image * 0.9, 0, 255).astype(np.uint8)
    augmented.append(aug)
    labels.append("Bright -10%")
    
    # Contrast
    alpha = 1.1
    aug = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    augmented.append(aug)
    labels.append("Contrast +10%")
    
    return augmented, labels

def main(args):
    """Main visualization function"""
    gt_dir = Path(args.gt_dir)
    deg_dir = Path(args.deg_dir)
    
    # Find image pairs
    gt_images = sorted(list(gt_dir.glob("*.png")) + list(gt_dir.glob("*.jpg")))
    deg_images = sorted(list(deg_dir.glob("*.png")) + list(deg_dir.glob("*.jpg")))
    
    if len(gt_images) == 0 or len(deg_images) == 0:
        print(f"❌ No images found!")
        print(f"   GT dir: {gt_dir}")
        print(f"   DEG dir: {deg_dir}")
        return
    
    # Match pairs
    pairs = []
    for gt_path in gt_images:
        stem = gt_path.stem
        for deg_path in deg_images:
            if deg_path.stem == stem:
                pairs.append((gt_path, deg_path))
                break
    
    if len(pairs) == 0:
        print(f"❌ No matching pairs found!")
        return
    
    print(f"✅ Found {len(pairs)} image pair(s)")
    
    for idx, (gt_path, deg_path) in enumerate(pairs):
        print(f"\n{'='*80}")
        print(f"PAIR {idx+1}: {gt_path.name}")
        print(f"{'='*80}")
        
        # Load full images
        gt_full = cv2.imread(str(gt_path))
        deg_full = cv2.imread(str(deg_path))
        
        if gt_full is None or deg_full is None:
            print(f"❌ Failed to load images")
            continue
        
        gt_gray = cv2.cvtColor(gt_full, cv2.COLOR_BGR2GRAY)
        deg_gray = cv2.cvtColor(deg_full, cv2.COLOR_BGR2GRAY)
        
        h, w = gt_gray.shape
        print(f"\n📏 Full Image Dimensions: {h} x {w} px")
        
        # Calculate strip statistics
        num_full_strips = h // 128
        remaining = h % 128
        total_strips = num_full_strips + (1 if remaining > 64 else 0)
        
        print(f"\n📊 Strip Statistics:")
        print(f"   Full strips (128px):     {num_full_strips}")
        print(f"   Remaining pixels:        {remaining}px")
        print(f"   Total strips (w/ last):  {total_strips}")
        print(f"   With augmentation (5x):  {total_strips * 5}")
        
        # Sequential split
        total_with_aug = total_strips * 5
        train_count = int(total_with_aug * 0.7)
        val_count = int(total_with_aug * 0.15)
        test_count = total_with_aug - train_count - val_count
        
        print(f"\n📐 Sequential Split:")
        print(f"   Train (70%):  {train_count:3d} samples (strips 0-{int(total_strips*0.7)})")
        print(f"   Val (15%):    {val_count:3d} samples (strips {int(total_strips*0.7)}-{int(total_strips*0.85)})")
        print(f"   Test (15%):   {test_count:3d} samples (strips {int(total_strips*0.85)}-{total_strips})")
        
        # Extract sample strips
        print(f"\n🔍 Extracting sample strips for visualization...")
        gt_strips, _ = extract_strips_preview(gt_path, max_strips=5)
        deg_strips, _ = extract_strips_preview(deg_path, max_strips=5)
        
        # Create visualization
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Full images comparison
        ax1 = plt.subplot(4, 1, 1)
        # Downsample for display
        display_width = 2000
        scale = display_width / w
        display_height = int(h * scale)
        gt_display = cv2.resize(gt_gray, (display_width, display_height))
        deg_display = cv2.resize(deg_gray, (display_width, display_height))
        comparison = np.hstack([gt_display, np.ones((display_height, 20), dtype=np.uint8)*255, deg_display])
        ax1.imshow(comparison, cmap='gray')
        ax1.set_title(f'Full Page Comparison: GT (Left) vs Degraded (Right)\nOriginal: {h}x{w}px → {total_strips} strips @ 128px height', 
                     fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. Sample strips (GT vs DEG, side by side)
        ax2 = plt.subplot(4, 1, 2)
        strip_vis = []
        for i, (gt_strip, deg_strip) in enumerate(zip(gt_strips[:3], deg_strips[:3])):
            # Resize to model input size
            gt_resized = resize_strip_for_model(gt_strip, target_width=1024)
            deg_resized = resize_strip_for_model(deg_strip, target_width=1024)
            
            # Stack horizontally with separator
            pair = np.hstack([gt_resized, np.ones((128, 10), dtype=np.uint8)*255, deg_resized])
            strip_vis.append(pair)
            if i < len(gt_strips) - 1:
                strip_vis.append(np.ones((10, pair.shape[1]), dtype=np.uint8)*255)
        
        strip_comparison = np.vstack(strip_vis)
        ax2.imshow(strip_comparison, cmap='gray')
        ax2.set_title(f'Sample Strips (Resized to 1024x128 - Model Input Size)\nGT (Left) vs Degraded (Right) - First 3 strips', 
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # 3. Augmentation preview (on first degraded strip)
        ax3 = plt.subplot(4, 1, 3)
        if len(deg_strips) > 0:
            first_strip_resized = resize_strip_for_model(deg_strips[0], target_width=1024)
            augmented, labels = augment_strip_preview(first_strip_resized)
            
            aug_vis = []
            for aug_img, label in zip(augmented, labels):
                # Add label on image
                aug_with_label = aug_img.copy()
                cv2.putText(aug_with_label, label, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                aug_vis.append(aug_with_label)
                aug_vis.append(np.ones((128, 10), dtype=np.uint8)*255)
            
            aug_comparison = np.hstack(aug_vis[:-1])  # Remove last separator
            ax3.imshow(aug_comparison, cmap='gray')
            ax3.set_title(f'Augmentation Examples (5x multiplier)\nFirst Degraded Strip with All Augmentation Variants', 
                         fontsize=12, fontweight='bold')
            ax3.axis('off')
        
        # 4. Sequential split visualization
        ax4 = plt.subplot(4, 1, 4)
        # Create colored visualization of split strategy
        split_vis = np.zeros((200, 1024, 3), dtype=np.uint8)
        
        # Train region (green)
        train_end = int(1024 * 0.7)
        split_vis[:, :train_end, :] = [50, 200, 50]  # Green
        cv2.putText(split_vis, f'TRAIN (70%): {train_count} samples', (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Val region (blue)
        val_end = int(1024 * 0.85)
        split_vis[:, train_end:val_end, :] = [200, 100, 50]  # Blue
        cv2.putText(split_vis, f'VAL (15%): {val_count}', (train_end + 20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Test region (orange)
        split_vis[:, val_end:, :] = [50, 100, 200]  # Orange
        cv2.putText(split_vis, f'TEST (15%): {test_count}', (val_end + 20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        ax4.imshow(split_vis)
        ax4.set_title(f'Sequential Split Strategy (NO Data Leakage)\nTop strips → Train, Middle → Val, Bottom → Test', 
                     fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"finetuning_preview_{gt_path.stem}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Visualization saved: {output_path}")
        
        if args.show:
            plt.show()
        else:
            plt.close()
        
        print(f"\n{'='*80}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize fine-tuning dataset')
    parser.add_argument('--gt_dir', type=str, default='DokumenRusak/manual_restoration/gt',
                        help='Directory with ground truth images')
    parser.add_argument('--deg_dir', type=str, default='DokumenRusak/manual_restoration/deg',
                        help='Directory with degraded images')
    parser.add_argument('--output_dir', type=str, default='visualization/finetuning',
                        help='Output directory for visualizations')
    parser.add_argument('--show', action='store_true',
                        help='Show visualization in window (default: save only)')
    
    args = parser.parse_args()
    main(args)
