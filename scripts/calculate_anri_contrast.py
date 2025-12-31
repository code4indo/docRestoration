#!/usr/bin/env python3
"""
Script to calculate contrast metrics for ANRI documents.
Generates objective data to support the categorization in Table V.7.

Contrast is measured using:
1. Standard deviation of pixel intensities (primary metric)
2. Michelson contrast: (Imax - Imin) / (Imax + Imin)
3. RMS contrast

Categories based on std deviation:
- High (>=60): Excellent restoration potential
- Medium (30-60): Good restoration with some residual noise
- Low (<30): Limited improvement due to extreme degradation
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

def calculate_contrast_metrics(image_path: str) -> dict:
    """Calculate various contrast metrics for an image."""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    pixels = np.array(img, dtype=np.float64)
    
    # Standard deviation (primary metric)
    std_contrast = np.std(pixels)
    
    # Michelson contrast
    i_max, i_min = pixels.max(), pixels.min()
    if (i_max + i_min) > 0:
        michelson = (i_max - i_min) / (i_max + i_min)
    else:
        michelson = 0.0
    
    # RMS contrast (normalized)
    mean_intensity = np.mean(pixels)
    if mean_intensity > 0:
        rms_contrast = np.sqrt(np.mean((pixels - mean_intensity) ** 2)) / mean_intensity
    else:
        rms_contrast = 0.0
    
    return {
        'std_contrast': round(std_contrast, 2),
        'michelson_contrast': round(michelson * 100, 2),  # Percentage
        'rms_contrast': round(rms_contrast * 100, 2),     # Percentage
        'mean_intensity': round(mean_intensity, 2),
        'width': img.width,
        'height': img.height
    }

def categorize_contrast(std_value: float) -> str:
    """Categorize contrast level based on standard deviation."""
    if std_value >= 60:
        return 'Tinggi'
    elif std_value >= 30:
        return 'Sedang'
    else:
        return 'Rendah'

def get_effectiveness_label(category: str) -> str:
    """Get effectiveness label based on contrast category."""
    labels = {
        'Tinggi': 'Baik Sekali',
        'Sedang': 'Baik',
        'Rendah': 'Terbatas'
    }
    return labels.get(category, 'Unknown')

def main():
    # Directory containing ANRI documents
    input_dir = Path('/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/DokumenRusak/forPaper')
    output_dir = Path('/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/Paper/data_dukung')
    
    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        sys.exit(1)
    
    # Get all image files
    image_files = sorted([f for f in input_dir.glob('*.jpg')] + 
                         [f for f in input_dir.glob('*.png')] +
                         [f for f in input_dir.glob('*.tif')] +
                         [f for f in input_dir.glob('*.tiff')])
    
    print(f"Found {len(image_files)} documents in {input_dir}")
    print("=" * 80)
    
    results = []
    category_counts = {'Tinggi': 0, 'Sedang': 0, 'Rendah': 0}
    
    for img_path in image_files:
        metrics = calculate_contrast_metrics(str(img_path))
        category = categorize_contrast(metrics['std_contrast'])
        effectiveness = get_effectiveness_label(category)
        category_counts[category] += 1
        
        result = {
            'filename': img_path.name,
            'std_contrast': metrics['std_contrast'],
            'michelson_contrast': metrics['michelson_contrast'],
            'rms_contrast': metrics['rms_contrast'],
            'mean_intensity': metrics['mean_intensity'],
            'category': category,
            'effectiveness': effectiveness,
            'dimensions': f"{metrics['width']}x{metrics['height']}"
        }
        results.append(result)
        
        print(f"{img_path.name:40} | Std: {metrics['std_contrast']:6.2f} | "
              f"Category: {category:8} | Effectiveness: {effectiveness}")
    
    print("=" * 80)
    print("\n📊 SUMMARY:")
    print(f"  Total documents: {len(results)}")
    for cat, count in category_counts.items():
        pct = (count / len(results)) * 100 if results else 0
        print(f"  {cat}: {count} ({pct:.1f}%)")
    
    # Save results to CSV
    csv_path = output_dir / 'anri_contrast_analysis.csv'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✅ Results saved to: {csv_path}")
    
    # Save summary JSON
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_documents': len(results),
        'category_distribution': category_counts,
        'category_thresholds': {
            'high': '>=60 (std deviation)',
            'medium': '30-60 (std deviation)',
            'low': '<30 (std deviation)'
        },
        'documents': results
    }
    
    json_path = output_dir / 'anri_contrast_analysis.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Summary saved to: {json_path}")
    
    # Print LaTeX-ready table
    print("\n" + "=" * 80)
    print("📋 LaTeX-ready table data:")
    print("=" * 80)
    print(f"{'Kategori Kontras':<20} | {'n':<3} | {'% Total':<8} | {'Efektivitas':<15}")
    print("-" * 60)
    for cat in ['Tinggi', 'Sedang', 'Rendah']:
        count = category_counts[cat]
        pct = (count / len(results)) * 100 if results else 0
        eff = get_effectiveness_label(cat)
        print(f"{cat:<20} | {count:<3} | {pct:>6.1f}%  | {eff:<15}")

if __name__ == '__main__':
    main()
