#!/usr/bin/env python3
"""
Recreate ANRI qualitative figures with text-focused cropping.
Crop area tengah yang kaya teks, hindari area kosong.
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

# Konfigurasi
INPUT_DIR = Path("DokumenRusak/forPaper")
RESULTS_DIR = Path("DokumenRusak/forPaper_results")
OUTPUT_DIR = Path("Paper/data_dukung/evaluasi_anri")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 6 dokumen yang dipilih (sesuai urutan panel a-f)
SELECTED_DOCS = [
    ("ID-ANRI_K66b_065_119", "a"),      # High contrast 63.54
    ("ID-ANRI_K66b_070_179", "b"),      # High contrast 65.80
    ("ID-ANRI_K66b_005_0526_ori", "c"), # Highest contrast 73.61
    ("ID-ANRI_K66b_059_067", "d"),      # Low contrast 14.96
    ("ID-ANRI_K66a_2525_0024", "e"),    # Low contrast 29.74
    ("ID-ANRI_K66b_064_278", "f")       # Medium contrast 52.00
]

# Crop settings untuk fokus pada area teks
# Format: (crop_start_y_ratio, crop_end_y_ratio, crop_start_x_ratio, crop_end_x_ratio)
CROP_REGIONS = {
    "ID-ANRI_K66b_065_119": (0.25, 0.65, 0.1, 0.7),      # Crop tengah-atas
    "ID-ANRI_K66b_070_179": (0.30, 0.70, 0.15, 0.75),    # Crop tengah
    "ID-ANRI_K66b_005_0526_ori": (0.20, 0.60, 0.1, 0.7), # Crop atas-tengah
    "ID-ANRI_K66b_059_067": (0.35, 0.75, 0.15, 0.75),    # Crop tengah-bawah
    "ID-ANRI_K66a_2525_0024": (0.25, 0.65, 0.1, 0.7),    # Crop tengah
    "ID-ANRI_K66b_064_278": (0.30, 0.70, 0.15, 0.75)     # Crop tengah
}

def smart_crop_text_area(img, crop_region):
    """Crop area fokus teks berdasarkan ratio"""
    width, height = img.size
    y_start_ratio, y_end_ratio, x_start_ratio, x_end_ratio = crop_region
    
    x_start = int(width * x_start_ratio)
    x_end = int(width * x_end_ratio)
    y_start = int(height * y_start_ratio)
    y_end = int(height * y_end_ratio)
    
    return img.crop((x_start, y_start, x_end, y_end))

def create_side_by_side_panel(img_name, panel_label, target_width=1290, target_height=980):
    """Create degraded | restored comparison panel with text focus"""
    
    # Load images
    degraded_path = INPUT_DIR / f"{img_name}.jpg"
    restored_path = RESULTS_DIR / f"{img_name}_restored.tiff"
    
    if not degraded_path.exists() or not restored_path.exists():
        print(f"⚠️  Missing files for {img_name}")
        return None
    
    degraded = Image.open(degraded_path).convert('RGB')
    restored = Image.open(restored_path).convert('RGB')
    
    # Apply smart crop untuk fokus area teks
    crop_region = CROP_REGIONS.get(img_name, (0.25, 0.75, 0.1, 0.9))  # Default
    degraded_crop = smart_crop_text_area(degraded, crop_region)
    restored_crop = smart_crop_text_area(restored, crop_region)
    
    # Resize kedua gambar ke ukuran yang sama
    crop_width = target_width // 2 - 20  # Split untuk 2 gambar + gap
    
    # Hitung aspect ratio crop
    crop_aspect = degraded_crop.size[0] / degraded_crop.size[1]
    crop_height = int(crop_width / crop_aspect)
    
    # Limit tinggi maksimal
    if crop_height > target_height - 100:
        crop_height = target_height - 100
        crop_width = int(crop_height * crop_aspect)
    
    degraded_resized = degraded_crop.resize((crop_width, crop_height), Image.Resampling.LANCZOS)
    restored_resized = restored_crop.resize((crop_width, crop_height), Image.Resampling.LANCZOS)
    
    # Create panel dengan white background
    panel_width = crop_width * 2 + 60  # 2 images + gap + margins
    panel_height = crop_height + 120   # Image + top/bottom margins + labels
    panel = Image.new('RGB', (panel_width, panel_height), 'white')
    draw = ImageDraw.Draw(panel)
    
    # Try load font
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
    
    # Draw panel label (a), (b), etc.
    draw.text((20, 20), f"({panel_label})", fill='black', font=font_title)
    
    # Paste images side by side
    y_offset = 70
    x_degraded = 20
    x_restored = 20 + crop_width + 20
    
    panel.paste(degraded_resized, (x_degraded, y_offset))
    panel.paste(restored_resized, (x_restored, y_offset))
    
    # Draw labels di bawah gambar
    label_y = y_offset + crop_height + 10
    
    # Center align labels
    deg_bbox = draw.textbbox((0, 0), "Terdegradasi", font=font_label)
    deg_text_width = deg_bbox[2] - deg_bbox[0]
    deg_x = x_degraded + (crop_width - deg_text_width) // 2
    
    res_bbox = draw.textbbox((0, 0), "Hasil Restorasi", font=font_label)
    res_text_width = res_bbox[2] - res_bbox[0]
    res_x = x_restored + (crop_width - res_text_width) // 2
    
    draw.text((deg_x, label_y), "Terdegradasi", fill='black', font=font_label)
    draw.text((res_x, label_y), "Hasil Restorasi", fill='black', font=font_label)
    
    # Draw vertical separator line
    separator_x = 20 + crop_width + 10
    draw.line([(separator_x, y_offset), (separator_x, y_offset + crop_height)], 
              fill='gray', width=2)
    
    return panel

def create_combined_figure():
    """Create 2x3 grid of all panels"""
    
    panels = []
    for img_name, label in SELECTED_DOCS:
        print(f"Creating panel ({label}): {img_name}...")
        panel = create_side_by_side_panel(img_name, label)
        if panel:
            # Save individual panel
            panel_path = OUTPUT_DIR / f"panel_{label}_{img_name}.png"
            panel.save(panel_path, dpi=(300, 300))
            print(f"  ✓ Saved: {panel_path}")
            panels.append(panel)
        else:
            print(f"  ✗ Failed to create panel")
    
    if len(panels) != 6:
        print(f"⚠️  Only {len(panels)}/6 panels created")
        return None
    
    # Create 2x3 grid
    panel_width, panel_height = panels[0].size
    
    grid_width = panel_width * 3 + 80   # 3 columns + margins
    grid_height = panel_height * 2 + 100 # 2 rows + margins
    
    combined = Image.new('RGB', (grid_width, grid_height), 'white')
    
    # Paste panels dalam grid 2x3
    positions = [
        (40, 40),                                    # (a) top-left
        (40 + panel_width + 20, 40),                 # (b) top-center
        (40 + 2*(panel_width + 20), 40),             # (c) top-right
        (40, 40 + panel_height + 20),                # (d) bottom-left
        (40 + panel_width + 20, 40 + panel_height + 20),   # (e) bottom-center
        (40 + 2*(panel_width + 20), 40 + panel_height + 20) # (f) bottom-right
    ]
    
    for panel, (x, y) in zip(panels, positions):
        combined.paste(panel, (x, y))
    
    return combined

def main():
    print("=" * 70)
    print("RECREATING ANRI QUALITATIVE FIGURES - TEXT FOCUSED CROPPING")
    print("=" * 70)
    
    # Create individual panels
    print("\n1. Creating individual panels...")
    
    # Create combined figure
    print("\n2. Creating combined 2×3 grid...")
    combined = create_combined_figure()
    
    if combined:
        # Save as PNG
        png_path = OUTPUT_DIR / "fig_anri_qualitative_results.png"
        combined.save(png_path, dpi=(300, 300))
        print(f"\n✓ PNG saved: {png_path}")
        print(f"  Size: {combined.size[0]}×{combined.size[1]}px")
        
        # Save as PDF
        pdf_path = OUTPUT_DIR / "fig_anri_qualitative_results.pdf"
        combined.save(pdf_path, dpi=(300, 300), resolution=300)
        print(f"✓ PDF saved: {pdf_path}")
        
        # Check file size
        import os
        pdf_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        print(f"  PDF size: {pdf_size_mb:.2f} MB")
        
        print("\n" + "=" * 70)
        print("✅ ANRI figures recreated successfully with text-focused cropping!")
        print("=" * 70)
    else:
        print("\n❌ Failed to create combined figure")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
