#!/usr/bin/env python3
"""
Analisis Gap Putih dan Probabilitas Kesempurnaan Hasil
"""

import cv2
import numpy as np
from pathlib import Path
import json

def analyze_white_gaps(image_path):
    """Analisis gap putih dalam reconstructed document"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Detect white regions (pixel > 250 = very white)
    white_mask = gray > 250
    white_pixels = np.sum(white_mask)
    total_pixels = h * w
    white_ratio = white_pixels / total_pixels
    
    # Find white gaps (connected components)
    # Invert: white becomes foreground
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    
    # Find contours of white regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Analyze gap sizes
    gap_sizes = []
    large_gaps = []  # Gaps > 100x100 pixels (suspicious)
    
    for contour in contours:
        x, y, w_box, h_box = cv2.boundingRect(contour)
        area = w_box * h_box
        gap_sizes.append(area)
        
        # Large gap = potential problem
        if area > 10000:  # 100x100 pixels
            large_gaps.append({
                'x': int(x),
                'y': int(y),
                'width': int(w_box),
                'height': int(h_box),
                'area': int(area)
            })
    
    # Statistics
    if gap_sizes:
        avg_gap = np.mean(gap_sizes)
        max_gap = np.max(gap_sizes)
        num_large_gaps = len(large_gaps)
    else:
        avg_gap = 0
        max_gap = 0
        num_large_gaps = 0
    
    return {
        'image_size': (w, h),
        'white_ratio': float(white_ratio),
        'num_gaps': len(gap_sizes),
        'avg_gap_size': float(avg_gap),
        'max_gap_size': float(max_gap),
        'num_large_gaps': num_large_gaps,
        'large_gaps': large_gaps[:10]  # Top 10 largest
    }

def calculate_perfection_probability(analysis):
    """Hitung probabilitas kesempurnaan berdasarkan metrics"""
    
    # Faktor-faktor yang mempengaruhi kesempurnaan
    factors = {}
    
    # 1. White ratio (ideal: 0.4-0.6 untuk dokumen dengan teks)
    white_ratio = analysis['white_ratio']
    if 0.4 <= white_ratio <= 0.6:
        factors['white_ratio_score'] = 100.0
    elif 0.3 <= white_ratio < 0.4 or 0.6 < white_ratio <= 0.7:
        factors['white_ratio_score'] = 80.0
    else:
        factors['white_ratio_score'] = 60.0
    
    # 2. Large gaps (ideal: 0 large gaps)
    num_large = analysis['num_large_gaps']
    if num_large == 0:
        factors['gap_score'] = 100.0
    elif num_large <= 5:
        factors['gap_score'] = 70.0
    elif num_large <= 10:
        factors['gap_score'] = 50.0
    else:
        factors['gap_score'] = 30.0
    
    # 3. Average gap size (ideal: small gaps only)
    avg_gap = analysis['avg_gap_size']
    if avg_gap < 1000:  # < 32x32 pixels average
        factors['avg_gap_score'] = 100.0
    elif avg_gap < 5000:  # < 71x71 pixels
        factors['avg_gap_score'] = 80.0
    else:
        factors['avg_gap_score'] = 60.0
    
    # 4. Max gap size (ideal: no huge gaps)
    max_gap = analysis['max_gap_size']
    if max_gap < 5000:  # < 71x71 pixels
        factors['max_gap_score'] = 100.0
    elif max_gap < 20000:  # < 141x141 pixels
        factors['max_gap_score'] = 75.0
    elif max_gap < 50000:  # < 224x224 pixels
        factors['max_gap_score'] = 50.0
    else:
        factors['max_gap_score'] = 25.0
    
    # Weighted average
    weights = {
        'white_ratio_score': 0.2,
        'gap_score': 0.4,  # Most important
        'avg_gap_score': 0.2,
        'max_gap_score': 0.2
    }
    
    probability = sum(factors[k] * weights[k] for k in factors.keys())
    
    return probability, factors

def main():
    """Analisis semua hasil reconstruction"""
    
    # Check for command line override
    import builtins
    if hasattr(builtins, 'OUTPUT_DIR_OVERRIDE'):
        base_path = Path(builtins.OUTPUT_DIR_OVERRIDE)
    else:
        base_path = Path("/home/lambda_one/tesis/GAN-HTR-ORI/outputs/TestPreciseAlignment_v6")
    
    results = []
    
    print("=" * 80)
    print("ANALISIS GAP PUTIH & PROBABILITAS KESEMPURNAAN")
    print("=" * 80)
    print()
    
    for doc_dir in sorted(base_path.glob("ID-ANRI_*")):
        doc_name = doc_dir.name
        restored_path = doc_dir / f"{doc_name}_restored.png"
        
        if not restored_path.exists():
            continue
        
        print(f"📄 {doc_name}")
        
        # Analisis
        analysis = analyze_white_gaps(restored_path)
        if analysis is None:
            print(f"   ❌ Failed to analyze")
            continue
        
        # Hitung probabilitas
        probability, factors = calculate_perfection_probability(analysis)
        
        print(f"   Image size: {analysis['image_size'][0]}×{analysis['image_size'][1]}")
        print(f"   White ratio: {analysis['white_ratio']:.2%}")
        print(f"   Total gaps: {analysis['num_gaps']}")
        print(f"   Large gaps (>10k px): {analysis['num_large_gaps']}")
        print(f"   Avg gap size: {analysis['avg_gap_size']:.0f} px²")
        print(f"   Max gap size: {analysis['max_gap_size']:.0f} px²")
        print()
        print(f"   📊 PROBABILITY BREAKDOWN:")
        print(f"      White ratio: {factors['white_ratio_score']:.1f}%")
        print(f"      Gap presence: {factors['gap_score']:.1f}%")
        print(f"      Avg gap size: {factors['avg_gap_score']:.1f}%")
        print(f"      Max gap size: {factors['max_gap_score']:.1f}%")
        print(f"   🎯 PERFECTION PROBABILITY: {probability:.1f}%")
        
        # Diagnosis
        if probability >= 90:
            status = "✅ EXCELLENT"
        elif probability >= 75:
            status = "✓ GOOD (minor gaps)"
        elif probability >= 60:
            status = "⚠️  ACCEPTABLE (visible gaps)"
        else:
            status = "❌ POOR (major gaps)"
        
        print(f"   Status: {status}")
        
        # Top 3 largest gaps
        if analysis['large_gaps']:
            print(f"   📍 Top 3 largest gaps:")
            for i, gap in enumerate(analysis['large_gaps'][:3], 1):
                print(f"      {i}. ({gap['x']}, {gap['y']}) size: {gap['width']}×{gap['height']} = {gap['area']:,} px²")
        
        print()
        
        results.append({
            'document': doc_name,
            'probability': probability,
            'factors': factors,
            'analysis': analysis
        })
    
    # Overall statistics
    if results:
        avg_prob = np.mean([r['probability'] for r in results])
        min_prob = min([r['probability'] for r in results])
        max_prob = max([r['probability'] for r in results])
        
        print("=" * 80)
        print("OVERALL STATISTICS")
        print("=" * 80)
        print(f"Documents analyzed: {len(results)}")
        print(f"Average probability: {avg_prob:.1f}%")
        print(f"Range: {min_prob:.1f}% - {max_prob:.1f}%")
        print()
        
        # Kategorisasi
        excellent = sum(1 for r in results if r['probability'] >= 90)
        good = sum(1 for r in results if 75 <= r['probability'] < 90)
        acceptable = sum(1 for r in results if 60 <= r['probability'] < 75)
        poor = sum(1 for r in results if r['probability'] < 60)
        
        print("QUALITY DISTRIBUTION:")
        print(f"  ✅ Excellent (≥90%): {excellent}/{len(results)}")
        print(f"  ✓  Good (75-89%): {good}/{len(results)}")
        print(f"  ⚠️  Acceptable (60-74%): {acceptable}/{len(results)}")
        print(f"  ❌ Poor (<60%): {poor}/{len(results)}")
        print()
        
        # Root cause analysis
        print("ROOT CAUSE ANALYSIS:")
        print()
        print("Gap putih disebabkan oleh:")
        print("1. ❌ FADE AREA - Content-aware fade menghilangkan konten di edge")
        print("   → Area yang di-fade menjadi putih (background)")
        print("   → Fix: Reduce fade width atau disable fade di non-overlap area")
        print()
        print("2. ❌ BASELINE GAPS - Laypa tidak detect area antar baseline")
        print("   → Area yang tidak ter-cover oleh baseline = white gap")
        print("   → Fix: Expand baseline bbox atau use original image as base")
        print()
        print("3. ❌ PADDING REMOVED - Area padding yang di-crop")
        print("   → Padding berguna tapi di-remove saat crop ke original")
        print("   → Already fixed in v6 (precise alignment)")
        print()
        
        print("RECOMMENDED SOLUTION:")
        print("v7 - Composite Rendering:")
        print("  1. Start dengan ORIGINAL IMAGE sebagai base (bukan white canvas)")
        print("  2. Place restored lines dengan alpha blending")
        print("  3. Fade hanya untuk anti-aliasing edge, bukan full edge")
        print("  4. Preserve background information dari original")
        print()
        print("Expected improvement: +20-30% probability")
        print("Target probability: 90%+ (excellent)")
        print("=" * 80)
        
        # Save results
        output_path = Path("/home/lambda_one/tesis/GAN-HTR-ORI/outputs/gap_analysis_v6.json")
        with open(output_path, 'w') as f:
            json.dump({
                'summary': {
                    'average_probability': avg_prob,
                    'min_probability': min_prob,
                    'max_probability': max_prob,
                    'quality_distribution': {
                        'excellent': excellent,
                        'good': good,
                        'acceptable': acceptable,
                        'poor': poor
                    }
                },
                'documents': results
            }, f, indent=2)
        
        print(f"📄 Detailed results saved: {output_path}")

if __name__ == "__main__":
    import sys
    
    # Support command line argument untuk output directory
    if len(sys.argv) > 1:
        output_dir_arg = sys.argv[1]
        # Patch output_dir global (hacky tapi cepat)
        import builtins
        builtins.OUTPUT_DIR_OVERRIDE = output_dir_arg
    
    main()
