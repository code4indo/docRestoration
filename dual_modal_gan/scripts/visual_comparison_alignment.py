#!/usr/bin/env python3
"""
Visual Comparison: v5 (Misaligned) vs v6 (Precise Alignment)
Focus: Line alignment precision
"""

import cv2
import numpy as np
from pathlib import Path

def extract_crop(img_path: Path, x: int, y: int, w: int, h: int):
    """Extract crop dari dokumen untuk alignment inspection"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    crop = img[y:y+h, x:x+w]
    return crop

def draw_alignment_grid(img, grid_spacing=50):
    """Draw grid overlay untuk visualisasi alignment"""
    h, w = img.shape[:2]
    overlay = img.copy()
    
    # Draw vertical lines
    for x in range(0, w, grid_spacing):
        cv2.line(overlay, (x, 0), (x, h), (0, 255, 0), 1)
    
    # Draw horizontal lines
    for y in range(0, h, grid_spacing):
        cv2.line(overlay, (0, y), (w, y), (0, 255, 0), 1)
    
    # Blend overlay
    result = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    return result

def create_alignment_comparison(doc_name: str, crop_coords: tuple):
    """Create side-by-side comparison dengan grid overlay"""
    x, y, w, h = crop_coords
    
    base_path = Path("/home/lambda_one/tesis/GAN-HTR-ORI/outputs")
    v5_path = base_path / "TestContentAware_v5" / doc_name / f"{doc_name}_restored.png"
    v6_path = base_path / "TestPreciseAlignment_v6" / doc_name / f"{doc_name}_restored.png"
    
    # Extract crops
    v5_crop = extract_crop(v5_path, x, y, w, h)
    v6_crop = extract_crop(v6_path, x, y, w, h)
    
    if v5_crop is None or v6_crop is None:
        print(f"❌ Failed to load images for {doc_name}")
        return None
    
    # Add alignment grid
    v5_grid = draw_alignment_grid(v5_crop, grid_spacing=40)
    v6_grid = draw_alignment_grid(v6_crop, grid_spacing=40)
    
    # Add labels
    label_h = 50
    label_v5 = np.ones((label_h, w, 3), dtype=np.uint8) * 255
    label_v6 = np.ones((label_h, w, 3), dtype=np.uint8) * 255
    
    cv2.putText(label_v5, "v5: Misaligned (Padding Bug)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    cv2.putText(label_v6, "v6: Precise Alignment (FIXED)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
    
    # Stack: label + crop with grid
    v5_panel = np.vstack([label_v5, v5_grid])
    v6_panel = np.vstack([label_v6, v6_grid])
    
    # Vertical comparison
    comparison = np.vstack([v5_panel, v6_panel])
    
    return comparison

def main():
    """Generate comparison images focusing on line alignment"""
    
    # Koordinat crop yang berisi multiple lines untuk inspect alignment
    crops = {
        "ID-ANRI_K66b_005_0526_ori": [
            (400, 1000, 1000, 600, "multi_line_alignment"),
            (800, 2500, 1000, 600, "line_junction_test"),
        ],
        "ID-ANRI_K66a_2482_0562": [
            (600, 1500, 1000, 600, "dense_text_alignment"),
            (1200, 3000, 1000, 600, "baseline_precision"),
        ],
        "ID-ANRI_K66a_2482_0564": [
            (500, 1200, 1000, 600, "horizontal_alignment"),
            (1000, 2800, 1000, 600, "vertical_spacing"),
        ],
    }
    
    output_dir = Path("/home/lambda_one/tesis/GAN-HTR-ORI/outputs/ComparisonAlignment")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("ALIGNMENT PRECISION COMPARISON: v5 vs v6")
    print("=" * 80)
    print("Focus: Line reconstruction precision (puzzle alignment)")
    print()
    print("Versions:")
    print("  - v5: Content-aware fade (with padding bug)")
    print("       Issue: Bbox includes padding → misaligned placement")
    print("  - v6: Precise alignment fix")
    print("       Solution: Separate original bbox (placement) vs padded bbox (dimensions)")
    print()
    print("Grid overlay: Green lines every 40px untuk visual alignment check")
    print()
    
    for doc_name, crop_list in crops.items():
        print(f"📄 {doc_name}")
        
        for x, y, w, h, label in crop_list:
            print(f"   {label}: x={x}, y={y}, w={w}, h={h}")
            
            comparison = create_alignment_comparison(doc_name, (x, y, w, h))
            if comparison is not None:
                output_path = output_dir / f"{doc_name}_{label}_alignment_comparison.png"
                cv2.imwrite(str(output_path), comparison)
                print(f"   ✅ Saved: {output_path.name}")
            else:
                print(f"   ❌ Failed")
        print()
    
    print("=" * 80)
    print("✅ Alignment comparison complete")
    print(f"📁 Output: {output_dir}")
    print()
    print("VALIDATION CHECKLIST:")
    print("  [?] v5: Apakah lines terlihat 'zigzag' atau puzzle tidak pas?")
    print("  [?] v6: Apakah lines sejajar sempurna horizontal?")
    print("  [?] v6: Apakah spacing vertical consistent antar lines?")
    print("  [?] v6: Apakah tidak ada gap/overlap yang tidak seharusnya?")
    print()
    print("INSPECT: Look for grid alignment patterns (should be smooth in v6)")
    print("=" * 80)

if __name__ == "__main__":
    main()
