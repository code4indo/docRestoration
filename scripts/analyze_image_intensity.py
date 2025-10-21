#!/usr/bin/env python3
"""
Script untuk menganalisis intensitas pixel dari gambar hasil training
Tujuan: Mendapatkan data objektif tentang seberapa hitam tulisan yang dihasilkan
"""

import cv2
import numpy as np
import os
from pathlib import Path
import json

def analyze_comparison_image(comparison_path, verbose=True):
    """Analisis gambar comparison yang berisi 3 bagian (degraded, GT, restored)"""
    img = cv2.imread(comparison_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"❌ Gagal membaca gambar: {comparison_path}")
        return None
    
    # Split menjadi 3 bagian vertikal
    height = img.shape[0]
    section_height = height // 3
    
    degraded = img[0:section_height, :]
    ground_truth = img[section_height:2*section_height, :]
    restored = img[2*section_height:, :]
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"📊 ANALISIS: {os.path.basename(comparison_path)}")
        print(f"{'='*80}")
    
    sections = {
        'Degraded': degraded,
        'Ground Truth': ground_truth,
        'Restored': restored
    }
    
    results = {}
    for name, section in sections.items():
        if verbose:
            print(f"\n🔍 {name}")
            print(f"   Dimensi: {section.shape}")
        
        min_val = int(np.min(section))
        max_val = int(np.max(section))
        mean_val = float(np.mean(section))
        median_val = float(np.median(section))
        
        # Analisis text region (pixel < 100 = hitam)
        text_mask = section < 100
        text_region = section[text_mask]
        
        # Analisis background (pixel > 200 = putih)
        bg_mask = section > 200
        
        # Analisis gray area (100-200)
        gray_mask = (section >= 100) & (section <= 200)
        
        text_count = int(np.sum(text_mask))
        bg_count = int(np.sum(bg_mask))
        gray_count = int(np.sum(gray_mask))
        total = section.size
        
        text_percent = (text_count / total) * 100
        bg_percent = (bg_count / total) * 100
        gray_percent = (gray_count / total) * 100
        
        if verbose:
            print(f"   Pixel Range: [{min_val}, {max_val}]")
            print(f"   Mean Intensity: {mean_val:.2f}")
            print(f"   Median Intensity: {median_val:.2f}")
        
        if len(text_region) > 0:
            text_mean = float(np.mean(text_region))
            text_min = int(np.min(text_region))
            text_max = int(np.max(text_region))
            text_median = float(np.median(text_region))
            
            if verbose:
                print(f"   Text Region (pixel < 100):")
                print(f"      - Percentage: {text_percent:.2f}%")
                print(f"      - Mean: {text_mean:.2f} (ideal: 0-50 untuk hitam pekat)")
                print(f"      - Median: {text_median:.2f}")
                print(f"      - Range: [{text_min}, {text_max}]")
        else:
            if verbose:
                print(f"   Text Region: ⚠️ TIDAK ADA PIXEL HITAM")
            text_mean = None
            text_min = None
            text_max = None
            text_median = None
        
        if verbose:
            print(f"   Background (> 200): {bg_percent:.2f}%")
            print(f"   Gray Area (100-200): {gray_percent:.2f}%")
        
        results[name] = {
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'median': median_val,
            'text_mean': text_mean,
            'text_median': text_median,
            'text_min': text_min,
            'text_max': text_max,
            'text_percent': text_percent,
            'text_count': text_count,
            'bg_percent': bg_percent,
            'bg_count': bg_count,
            'gray_percent': gray_percent,
            'gray_count': gray_count,
            'total_pixels': total
        }
    
    # Bandingkan restored vs ground truth
    if verbose:
        print(f"\n📈 PERBANDINGAN RESTORED vs GROUND TRUTH:")
        
        gt_text_mean = results['Ground Truth']['text_mean']
        restored_text_mean = results['Restored']['text_mean']
        
        if restored_text_mean is not None and gt_text_mean is not None:
            diff_mean = restored_text_mean - gt_text_mean
            diff_percent = (diff_mean / gt_text_mean) * 100 if gt_text_mean > 0 else 0
            
            print(f"   Ground Truth Text Mean: {gt_text_mean:.2f}")
            print(f"   Restored Text Mean: {restored_text_mean:.2f}")
            print(f"   Selisih: {diff_mean:+.2f} ({diff_percent:+.1f}%)")
            print(f"   (Positif = Restored lebih terang/kurang hitam)")
            
            if diff_mean > 20:
                print(f"   ⚠️ MASALAH SIGNIFIKAN: Text restored {diff_mean:.1f} lebih terang dari GT")
            elif diff_mean > 10:
                print(f"   ⚡ PERHATIAN: Text restored sedikit lebih terang dari GT")
            elif diff_mean > 5:
                print(f"   ℹ️ Text restored sedikit berbeda dari GT (masih acceptable)")
            else:
                print(f"   ✅ Text restored mendekati GT")
        else:
            print(f"   ⚠️ Tidak bisa membandingkan (ada yang tidak memiliki text region)")
    
    return results

def analyze_multiple_samples(sample_dir, pattern="comparison_epoch_0050_sample_*.png", max_samples=5):
    """Analisis multiple samples dan buat summary statistik"""
    sample_dir = Path(sample_dir)
    comparison_files = sorted(sample_dir.glob(pattern))
    
    if not comparison_files:
        print(f"❌ Tidak ada file ditemukan dengan pattern: {pattern}")
        return None
    
    print(f"✅ Ditemukan {len(comparison_files)} file comparison")
    print(f"📊 Menganalisis {min(max_samples, len(comparison_files))} sample...\n")
    
    all_results = []
    
    for i, comp_file in enumerate(comparison_files[:max_samples]):
        results = analyze_comparison_image(str(comp_file), verbose=True)
        if results:
            all_results.append(results)
        
        if i < min(max_samples, len(comparison_files)) - 1:
            print("\n" + "-"*80)
    
    # Buat summary statistik
    if all_results:
        print(f"\n{'='*80}")
        print(f"📊 SUMMARY STATISTIK ({len(all_results)} samples)")
        print(f"{'='*80}")
        
        for section_name in ['Degraded', 'Ground Truth', 'Restored']:
            text_means = [r[section_name]['text_mean'] for r in all_results if r[section_name]['text_mean'] is not None]
            
            if text_means:
                avg_text_mean = np.mean(text_means)
                std_text_mean = np.std(text_means)
                min_text_mean = np.min(text_means)
                max_text_mean = np.max(text_means)
                
                print(f"\n{section_name}:")
                print(f"   Text Mean Intensity (rata-rata): {avg_text_mean:.2f} ± {std_text_mean:.2f}")
                print(f"   Text Mean Range: [{min_text_mean:.2f}, {max_text_mean:.2f}]")
        
        # Analisis perbandingan
        gt_means = [r['Ground Truth']['text_mean'] for r in all_results if r['Ground Truth']['text_mean'] is not None]
        restored_means = [r['Restored']['text_mean'] for r in all_results if r['Restored']['text_mean'] is not None]
        
        if gt_means and restored_means:
            avg_gt = np.mean(gt_means)
            avg_restored = np.mean(restored_means)
            diff = avg_restored - avg_gt
            diff_percent = (diff / avg_gt) * 100 if avg_gt > 0 else 0
            
            print(f"\n🎯 KESIMPULAN:")
            print(f"   GT Average Text Intensity: {avg_gt:.2f}")
            print(f"   Restored Average Text Intensity: {avg_restored:.2f}")
            print(f"   Selisih Rata-rata: {diff:+.2f} ({diff_percent:+.1f}%)")
            
            if diff > 20:
                print(f"   ⚠️ MASALAH: Model menghasilkan text yang terlalu terang")
                print(f"   💡 Rekomendasi: Perlu post-processing untuk menghitamkan text")
            elif diff > 10:
                print(f"   ⚡ PERHATIAN: Text sedikit lebih terang dari seharusnya")
                print(f"   💡 Rekomendasi: Pertimbangkan minor adjustment")
            else:
                print(f"   ✅ Text intensity sudah baik")
    
    return all_results

def main():
    """Main function"""
    print(f"\n{'='*80}")
    print(f"🔬 ANALISIS INTENSITAS PIXEL - BEST MODEL")
    print(f"{'='*80}")
    
    # Path ke direktori output
    sample_dir = "/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/dual_modal_gan/outputs/samples_full_training_production_v1"
    
    # Analisis samples dari epoch terakhir (epoch 50)
    results = analyze_multiple_samples(
        sample_dir=sample_dir,
        pattern="comparison_epoch_0050_sample_*.png",
        max_samples=5
    )
    
    if results:
        print(f"\n{'='*80}")
        print(f"📝 INTERPRETASI NILAI INTENSITY:")
        print(f"   - Pixel 0 = Hitam murni (ideal untuk text)")
        print(f"   - Pixel 0-50 = Hitam pekat (sangat baik)")
        print(f"   - Pixel 50-100 = Hitam keabuan (masih acceptable)")
        print(f"   - Pixel 100-200 = Abu-abu (kurang ideal untuk text)")
        print(f"   - Pixel 200-255 = Putih/background")
        print(f"{'='*80}\n")
        
        # Simpan hasil ke JSON untuk referensi
        output_file = Path(sample_dir) / "intensity_analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Hasil disimpan ke: {output_file}")
    
    return results

if __name__ == "__main__":
    main()
