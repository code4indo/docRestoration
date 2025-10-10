#!/usr/bin/env python3
"""
Visual Comparison: Alpha Blending vs Background Fade
Untuk validasi apakah ghosting sudah hilang
"""

import cv2
import numpy as np
from pathlib import Path
import sys

def extract_crop(img_path: Path, x: int, y: int, w: int, h: int):
    """Extract crop dari dokumen untuk detail inspection"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    crop = img[y:y+h, x:x+w]
    return crop

def create_comparison_grid(doc_name: str, crop_coords: tuple):
    """Create side-by-side comparison untuk 1 dokumen"""
    x, y, w, h = crop_coords
    
    base_path = Path("/home/lambda_one/tesis/GAN-HTR-ORI/outputs")
    alpha_path = base_path / "TestAlphaBlending" / doc_name / f"{doc_name}_restored.png"
    bgfade_path = base_path / "TestBackgroundFade" / doc_name / f"{doc_name}_restored.png"
    
    # Extract crops
    alpha_crop = extract_crop(alpha_path, x, y, w, h)
    bgfade_crop = extract_crop(bgfade_path, x, y, w, h)
    
    if alpha_crop is None or bgfade_crop is None:
        print(f"❌ Failed to load images for {doc_name}")
        return None
    
    # Add labels
    label_alpha = np.ones((30, w, 3), dtype=np.uint8) * 255
    label_bgfade = np.ones((30, w, 3), dtype=np.uint8) * 255
    cv2.putText(label_alpha, "Alpha Blending (Ghosting)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(label_bgfade, "Background Fade (Fixed)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Stack: label + crop
    alpha_panel = np.vstack([label_alpha, alpha_crop])
    bgfade_panel = np.vstack([label_bgfade, bgfade_crop])
    
    # Horizontal comparison
    comparison = np.hstack([alpha_panel, bgfade_panel])
    
    return comparison

def main():
    """Generate comparison images untuk 3 test documents"""
    
    # Koordinat crop yang berisi overlap region (area rawan ghosting)
    # Format: (x, y, width, height)
    crops = {
        "ID-ANRI_K66b_005_0526_ori": (800, 1500, 1200, 600),  # Mid document overlap
        "ID-ANRI_K66a_2482_0562": (1000, 2000, 1200, 600),    # Complex overlap (182 lines)
        "ID-ANRI_K66a_2482_0564": (900, 1800, 1200, 600),     # Mid overlap (147 lines)
    }
    
    output_dir = Path("/home/lambda_one/tesis/GAN-HTR-ORI/outputs/GhostingComparison")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("GHOSTING FIX VISUAL VALIDATION")
    print("=" * 80)
    print("Comparing: Alpha Blending vs Background Fade")
    print("Focus: Overlap regions (area rawan ghosting)")
    print()
    
    for doc_name, crop_coords in crops.items():
        print(f"📄 {doc_name}")
        print(f"   Crop region: x={crop_coords[0]}, y={crop_coords[1]}, "
              f"w={crop_coords[2]}, h={crop_coords[3]}")
        
        comparison = create_comparison_grid(doc_name, crop_coords)
        if comparison is not None:
            output_path = output_dir / f"{doc_name}_ghosting_comparison.png"
            cv2.imwrite(str(output_path), comparison)
            print(f"   ✅ Saved: {output_path}")
            print(f"   Size: {comparison.shape[1]}x{comparison.shape[0]} pixels")
        else:
            print(f"   ❌ Failed to create comparison")
        print()
    
    print("=" * 80)
    print("✅ Comparison complete")
    print(f"📁 Output: {output_dir}")
    print()
    print("VALIDATION CHECKLIST:")
    print("  [?] Alpha Blending: Apakah terlihat bayangan/tulisan ganda?")
    print("  [?] Background Fade: Apakah teks terlihat bersih tanpa bayangan?")
    print("  [?] Edge smoothness: Apakah transisi ke background smooth?")
    print("=" * 80)

if __name__ == "__main__":
    main()
