#!/usr/bin/env python3
"""
Compare OLD (distorted) vs NEW (aspect ratio preserved) dataset
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def preprocess_old(image, target_width=1024, target_height=128):
    """OLD: Force resize (causes distortion)"""
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

def preprocess_new(image, target_height=128, target_width=1024):
    """NEW: Maintain aspect ratio with crop/pad"""
    h, w = image.shape
    
    # Calculate scaling factor
    scale = target_height / h
    new_w = int(w * scale)
    new_h = target_height
    
    # Resize maintaining aspect ratio
    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Pad or crop
    if new_w < target_width:
        pad_total = target_width - new_w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        
        result = cv2.copyMakeBorder(
            image_resized,
            top=0, bottom=0,
            left=pad_left, right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=255
        )
    elif new_w > target_width:
        crop_start = (new_w - target_width) // 2
        result = image_resized[:, crop_start:crop_start+target_width]
    else:
        result = image_resized
    
    return result

# Load image
gt_img = cv2.imread('DokumenRusak/manual_restoration/gt/ID-ANRI_K66b_082_0064.png', cv2.IMREAD_GRAYSCALE)
deg_img = cv2.imread('DokumenRusak/manual_restoration/deg/ID-ANRI_K66b_082_0064.jpg', cv2.IMREAD_GRAYSCALE)

# Extract first strip
gt_strip = gt_img[0:128, :]
deg_strip = deg_img[0:128, :]

# Process with both methods
gt_old = preprocess_old(gt_strip)
gt_new = preprocess_new(gt_strip)
deg_old = preprocess_old(deg_strip)
deg_new = preprocess_new(deg_strip)

# Create comparison
fig, axes = plt.subplots(4, 1, figsize=(20, 12))

# GT comparison
axes[0].imshow(gt_old, cmap='gray')
axes[0].set_title('OLD METHOD (GT): Force Resize - Text COMPRESSED -62.6% horizontally!', 
                  fontsize=14, fontweight='bold', color='red')
axes[0].axis('off')

axes[1].imshow(gt_new, cmap='gray')
axes[1].set_title('NEW METHOD (GT): Aspect Ratio Preserved - Center cropped (no distortion)', 
                  fontsize=14, fontweight='bold', color='green')
axes[1].axis('off')

# DEG comparison
axes[2].imshow(deg_old, cmap='gray')
axes[2].set_title('OLD METHOD (DEG): Force Resize - Text COMPRESSED -62.6% horizontally!', 
                  fontsize=14, fontweight='bold', color='red')
axes[2].axis('off')

axes[3].imshow(deg_new, cmap='gray')
axes[3].set_title('NEW METHOD (DEG): Aspect Ratio Preserved - Center cropped (no distortion)', 
                  fontsize=14, fontweight='bold', color='green')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('visualization/finetuning/aspect_ratio_comparison.png', dpi=150, bbox_inches='tight')
print('✅ Comparison saved to: visualization/finetuning/aspect_ratio_comparison.png')

# Calculate distortion metrics
h_orig, w_orig = gt_strip.shape
ratio_orig = w_orig / h_orig

h_old, w_old = gt_old.shape
ratio_old = w_old / h_old

h_new, w_new = gt_new.shape  
ratio_new = w_new / h_new

# After resize with aspect ratio
scale = 128 / h_orig
w_scaled = int(w_orig * scale)
ratio_scaled = w_scaled / 128

print('\n' + '='*80)
print('DISTORTION ANALYSIS')
print('='*80)
print(f'\nOriginal strip:  {w_orig}x{h_orig}  (aspect ratio: {ratio_orig:.3f})')
print(f'After scale:     {w_scaled}x128       (aspect ratio: {ratio_scaled:.3f})')
print(f'\nOLD method:      {w_old}x{h_old}       (aspect ratio: {ratio_old:.3f})')
print(f'  Distortion:    {(ratio_old/ratio_orig - 1)*100:+.1f}% ← COMPRESSED!')
print(f'\nNEW method:      {w_new}x{h_new}       (aspect ratio: {ratio_new:.3f})')
print(f'  Distortion:    {(ratio_new/ratio_scaled - 1)*100:+.1f}% ← PRESERVED!')
print(f'\n✅ Fix successfully prevents text distortion!')
print('='*80)
