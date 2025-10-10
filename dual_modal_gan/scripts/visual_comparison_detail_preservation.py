#!/usr/bin/env python3
"""
Visual Comparison: v4 (Uniform fade) vs v5 (Content-aware fade)
Focus: Fine stroke detail preservation
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

def create_detail_comparison(doc_name: str, crop_coords: tuple):
    """Create side-by-side comparison untuk detail inspection"""
    x, y, w, h = crop_coords
    
    base_path = Path("/home/lambda_one/tesis/GAN-HTR-ORI/outputs")
    v4_path = base_path / "TestBackgroundFade_v4" / doc_name / f"{doc_name}_restored.png"
    v5_path = base_path / "TestContentAware_v5" / doc_name / f"{doc_name}_restored.png"
    
    # Extract crops
    v4_crop = extract_crop(v4_path, x, y, w, h)
    v5_crop = extract_crop(v5_path, x, y, w, h)
    
    if v4_crop is None or v5_crop is None:
        print(f"❌ Failed to load images for {doc_name}")
        return None
    
    # Upscale 2x untuk better detail visibility
    v4_crop = cv2.resize(v4_crop, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    v5_crop = cv2.resize(v5_crop, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    
    # Add labels
    label_h = 50
    label_v4 = np.ones((label_h, w*2, 3), dtype=np.uint8) * 255
    label_v5 = np.ones((label_h, w*2, 3), dtype=np.uint8) * 255
    
    cv2.putText(label_v4, "v4: Uniform Fade (Detail Loss)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 100, 0), 2)
    cv2.putText(label_v5, "v5: Content-Aware (Detail Preserve)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
    
    # Stack: label + crop
    v4_panel = np.vstack([label_v4, v4_crop])
    v5_panel = np.vstack([label_v5, v5_crop])
    
    # Vertical comparison untuk detail side-by-side
    comparison = np.vstack([v4_panel, v5_panel])
    
    return comparison

def main():
    """Generate comparison images focusing on fine stroke details"""
    
    # Koordinat crop yang berisi fine strokes di edge regions
    # Focus: Area dengan goresan halus yang menghubungkan huruf
    crops = {
        "ID-ANRI_K66b_005_0526_ori": [
            (500, 800, 600, 150, "line_edge_strokes"),
            (1200, 1500, 600, 150, "character_connections"),
        ],
        "ID-ANRI_K66a_2482_0562": [
            (800, 1200, 600, 150, "fine_details_top"),
            (1400, 2400, 600, 150, "edge_preservation"),
        ],
        "ID-ANRI_K66a_2482_0564": [
            (700, 1000, 600, 150, "stroke_connection"),
            (1300, 2200, 600, 150, "edge_integrity"),
        ],
    }
    
    output_dir = Path("/home/lambda_one/tesis/GAN-HTR-ORI/outputs/ComparisonDetailPreservation")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("DETAIL PRESERVATION COMPARISON: v4 vs v5")
    print("=" * 80)
    print("Focus: Fine stroke details at line edges")
    print()
    print("Versions:")
    print("  - v4: Uniform 4-directional fade (15-40px)")
    print("       Issue: Aggressive fade erases fine strokes")
    print("  - v5: Content-aware fade (10-20px, ink-sensitive)")
    print("       Solution: Detect ink → preserve (90-100% alpha)")
    print("                Background → fade normally (0-100% alpha)")
    print()
    
    for doc_name, crop_list in crops.items():
        print(f"📄 {doc_name}")
        
        for x, y, w, h, label in crop_list:
            print(f"   {label}: x={x}, y={y}, w={w}, h={h}")
            
            comparison = create_detail_comparison(doc_name, (x, y, w, h))
            if comparison is not None:
                output_path = output_dir / f"{doc_name}_{label}_detail_comparison.png"
                cv2.imwrite(str(output_path), comparison)
                print(f"   ✅ Saved: {output_path.name}")
            else:
                print(f"   ❌ Failed")
        print()
    
    print("=" * 80)
    print("✅ Detail comparison complete")
    print(f"📁 Output: {output_dir}")
    print()
    print("VALIDATION CHECKLIST:")
    print("  [?] v4: Apakah goresan halus di edge terlihat pudar/hilang?")
    print("  [?] v5: Apakah goresan halus tetap terlihat jelas?")
    print("  [?] v5: Apakah background tetap smooth (tidak ada garis)?")
    print("  [?] v5: Apakah integritas dokumen terjaga (fine strokes preserved)?")
    print()
    print("ZOOM IN untuk melihat detail halus (images sudah 2× upscaled)")
    print("=" * 80)

if __name__ == "__main__":
    main()
