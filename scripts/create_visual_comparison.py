"""
Create side-by-side visual comparison to see the actual differences.
"""

import cv2
import numpy as np
from pathlib import Path

def create_comparison(img1_path, img2_path, degraded_path, output_path):
    """Create side-by-side comparison."""
    # Load images
    degraded = cv2.imread(str(degraded_path), cv2.IMREAD_GRAYSCALE)
    restored1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
    restored2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)
    
    if degraded is None or restored1 is None or restored2 is None:
        return False
    
    # Create labels
    h, w = degraded.shape
    label_height = 30
    
    def add_label(img, text, color=255):
        canvas = np.ones((h + label_height, w), dtype=np.uint8) * color
        canvas[label_height:, :] = img
        cv2.putText(canvas, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
        return canvas
    
    degraded_labeled = add_label(degraded, "DEGRADED INPUT")
    restored1_labeled = add_label(restored1, "NO POST-PROCESS")
    restored2_labeled = add_label(restored2, "WITH POST-PROCESS")
    
    # Stack horizontally
    comparison = np.hstack([degraded_labeled, restored1_labeled, restored2_labeled])
    
    cv2.imwrite(str(output_path), comparison)
    return True

def main():
    # Paths
    degraded_dir = Path("DokumenRusak/data_latih")
    results_dir1 = Path("results/restored_data_latih/restored")
    results_dir2 = Path("RusakRingan/data_latih")
    output_dir = Path("results/visual_comparison_post_vs_no_post")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating visual comparisons...")
    print(f"Output directory: {output_dir}\n")
    
    # Find degraded images
    degraded_files = list(degraded_dir.glob("*.png"))
    
    for deg_path in degraded_files:
        base_name = deg_path.stem
        
        # Find corresponding restored files
        restored1_path = results_dir1 / f"{base_name}_restored.png"
        restored2_path = results_dir2 / f"{base_name}_restored.png"
        
        if not restored1_path.exists() or not restored2_path.exists():
            print(f"⚠️  Skipping {base_name}: restored files not found")
            continue
        
        output_path = output_dir / f"{base_name}_comparison.png"
        
        success = create_comparison(restored1_path, restored2_path, deg_path, output_path)
        
        if success:
            print(f"✓ Created: {output_path.name}")
        else:
            print(f"✗ Failed: {base_name}")
    
    print(f"\n✅ Comparison images saved to: {output_dir}")
    print("\nSilakan buka file comparison untuk melihat perbedaan visual!")

if __name__ == "__main__":
    main()
