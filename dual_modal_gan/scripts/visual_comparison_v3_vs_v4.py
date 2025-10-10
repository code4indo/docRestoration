#!/usr/bin/env python3
"""
Visual Comparison: v3 (2-directional) vs v4 (4-directional fade)
Untuk validasi eliminasi garis horizontal
"""

import cv2
import numpy as np
from pathlib import Path

def extract_crop(img_path: Path, x: int, y: int, w: int, h: int):
    """Extract crop dari dokumen untuk detail inspection"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    crop = img[y:y+h, x:x+w]
    return crop

def create_triple_comparison(doc_name: str, crop_coords: tuple):
    """Create side-by-side comparison: Alpha Blend | v3 | v4"""
    x, y, w, h = crop_coords
    
    base_path = Path("/home/lambda_one/tesis/GAN-HTR-ORI/outputs")
    alpha_path = base_path / "TestAlphaBlending" / doc_name / f"{doc_name}_restored.png"
    v3_path = base_path / "TestBackgroundFade" / doc_name / f"{doc_name}_restored.png"
    v4_path = base_path / "TestBackgroundFade_v4" / doc_name / f"{doc_name}_restored.png"
    
    # Extract crops
    alpha_crop = extract_crop(alpha_path, x, y, w, h)
    v3_crop = extract_crop(v3_path, x, y, w, h)
    v4_crop = extract_crop(v4_path, x, y, w, h)
    
    if alpha_crop is None or v3_crop is None or v4_crop is None:
        print(f"❌ Failed to load images for {doc_name}")
        return None
    
    # Add labels
    label_h = 40
    label_alpha = np.ones((label_h, w, 3), dtype=np.uint8) * 255
    label_v3 = np.ones((label_h, w, 3), dtype=np.uint8) * 255
    label_v4 = np.ones((label_h, w, 3), dtype=np.uint8) * 255
    
    cv2.putText(label_alpha, "Alpha Blend (Ghosting)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    cv2.putText(label_v3, "v3: 2-Dir Fade (Horiz Lines)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 0), 2)
    cv2.putText(label_v4, "v4: 4-Dir Fade (FIXED)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    
    # Stack: label + crop
    alpha_panel = np.vstack([label_alpha, alpha_crop])
    v3_panel = np.vstack([label_v3, v3_crop])
    v4_panel = np.vstack([label_v4, v4_crop])
    
    # Horizontal comparison
    comparison = np.hstack([alpha_panel, v3_panel, v4_panel])
    
    return comparison

def main():
    """Generate comparison images untuk 3 test documents"""
    
    # Koordinat crop yang berisi overlap regions dan area rawan garis horizontal
    crops = {
        "ID-ANRI_K66b_005_0526_ori": (800, 1500, 800, 500),   # Mid document overlap
        "ID-ANRI_K66a_2482_0562": (1000, 2000, 800, 500),     # Complex (182 lines)
        "ID-ANRI_K66a_2482_0564": (900, 1800, 800, 500),      # Mid overlap (147 lines)
    }
    
    output_dir = Path("/home/lambda_one/tesis/GAN-HTR-ORI/outputs/ComparisonV3_V4")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("VISUAL COMPARISON: v3 vs v4 (4-DIRECTIONAL FADE)")
    print("=" * 80)
    print("Versions:")
    print("  - Alpha Blend: Weighted average (ghosting issue)")
    print("  - v3: 2-directional fade (top/bottom only → horizontal lines remain)")
    print("  - v4: 4-directional fade (top/bottom/left/right → no horizontal lines)")
    print()
    
    for doc_name, crop_coords in crops.items():
        print(f"📄 {doc_name}")
        print(f"   Crop region: x={crop_coords[0]}, y={crop_coords[1]}, "
              f"w={crop_coords[2]}, h={crop_coords[3]}")
        
        comparison = create_triple_comparison(doc_name, crop_coords)
        if comparison is not None:
            output_path = output_dir / f"{doc_name}_v3_vs_v4_comparison.png"
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
    print("  [?] Alpha Blend: Ghosting/double text visible?")
    print("  [?] v3 (2-dir): Horizontal lines cutting text?")
    print("  [?] v4 (4-dir): Clean text, smooth edges, NO horizontal lines?")
    print("=" * 80)

if __name__ == "__main__":
    main()
