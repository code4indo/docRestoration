#!/usr/bin/env python3
"""
Quick Image Pair Viewer

Shows GT vs Degraded side-by-side comparison
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def view_pair(gt_path, deg_path):
    """Display GT vs Degraded comparison"""
    
    # Load images
    gt_img = cv2.imread(str(gt_path))
    deg_img = cv2.imread(str(deg_path))
    
    if gt_img is None or deg_img is None:
        print(f"❌ Failed to load images")
        return
    
    # Convert to RGB for matplotlib
    gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    deg_rgb = cv2.cvtColor(deg_img, cv2.COLOR_BGR2RGB)
    
    h, w, c = gt_rgb.shape
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    
    # Ground Truth
    ax1.imshow(gt_rgb)
    ax1.set_title(f'Ground Truth (Clean)\n{gt_path.name}\nSize: {h}x{w}px', 
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Degraded
    ax2.imshow(deg_rgb)
    ax2.set_title(f'Degraded (Input)\n{deg_path.name}\nSize: {h}x{w}px', 
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('visualization/finetuning')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pair_comparison_{gt_path.stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    plt.close()

if __name__ == '__main__':
    gt_dir = Path('DokumenRusak/manual_restoration/gt')
    deg_dir = Path('DokumenRusak/manual_restoration/deg')
    
    # Find pairs
    gt_images = sorted(list(gt_dir.glob("*.png")) + list(gt_dir.glob("*.jpg")))
    
    for gt_path in gt_images:
        stem = gt_path.stem
        
        # Find matching degraded image
        deg_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = deg_dir / f"{stem}{ext}"
            if candidate.exists():
                deg_path = candidate
                break
        
        if deg_path:
            print(f"\n📷 Processing: {stem}")
            print(f"   GT:  {gt_path.name}")
            print(f"   DEG: {deg_path.name}")
            view_pair(gt_path, deg_path)
        else:
            print(f"⚠️  No matching degraded image for: {gt_path.name}")
