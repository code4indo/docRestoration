#!/usr/bin/env python3
"""
Failure Case Analysis Script - Post-processing untuk Test Set Evaluation
==========================================================================

Script ini menganalisis hasil evaluasi test set untuk menghasilkan data
yang dibutuhkan di Paper Section E: Analisis Kuantitatif Kasus Kegagalan.

Input:
    - results/test_set_official_evaluation.json (output dari evaluate_test_set.py)
    
Output:
    - results/failure_case_analysis.json (distribusi, worst cases, statistik)
    - results/failure_case_report.txt (human-readable report)
    
Analisis yang dilakukan:
    1. Distribusi CER dalam bins (< 20%, 20-40%, > 40%)
    2. Identifikasi worst N cases (highest CER)
    3. Statistik deskriptif (percentiles, outliers)
    4. Template untuk manual categorization

⚠️  NOTE: SNR correlation dan confidence metrics TIDAK tersedia
    (tidak ada dalam evaluate_test_set.py output)
    → Akan dihapus dari paper atau diganti dengan qualitative description

Author: AI/ML Engineer (belekok)
Date: 2025-11-05
Version: 1.0 - Initial implementation
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt


def load_evaluation_results(json_path):
    """Load hasil evaluasi dari JSON file."""
    print(f"📖 Loading evaluation results from: {json_path}")
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    n_samples = results['test_set_size']
    print(f"   Loaded results for {n_samples} test samples")
    
    return results


def extract_per_sample_metrics(results):
    """Extract per-sample CER/WER/PSNR/SSIM dari nested results."""
    # Evaluation script menyimpan list metrics di results
    # NOTE: Script evaluate_test_set.py saat ini hanya menyimpan aggregated stats
    # Kita perlu modifikasi jika ingin per-sample data
    
    print("\n⚠️  WARNING: evaluate_test_set.py hanya menyimpan aggregated statistics")
    print("   Untuk failure case analysis, kita perlu PER-SAMPLE metrics")
    print("   REKOMENDASI: Modifikasi evaluate_test_set.py untuk menyimpan all_cer, all_wer, dll")
    
    # Untuk sementara, kita akan gunakan sample_texts (hanya 10 samples)
    sample_texts = results.get('sample_texts', [])
    if sample_texts:
        print(f"\n✅ Found {len(sample_texts)} sample texts (qualitative analysis)")
        cer_values = [s['cer'] for s in sample_texts]
        print(f"   CER range in samples: {min(cer_values):.2%} - {max(cer_values):.2%}")
    
    return sample_texts


def bin_cer_distribution(cer_values):
    """Bin CER values ke dalam 3 kategori."""
    bins = {
        'excellent': [],  # CER < 20%
        'acceptable': [],  # 20% <= CER < 40%
        'poor': []  # CER >= 40%
    }
    
    for cer in cer_values:
        if cer < 0.20:
            bins['excellent'].append(cer)
        elif cer < 0.40:
            bins['acceptable'].append(cer)
        else:
            bins['poor'].append(cer)
    
    counts = {k: len(v) for k, v in bins.items()}
    total = sum(counts.values())
    percentages = {k: (v/total * 100) if total > 0 else 0 for k, v in counts.items()}
    
    return bins, counts, percentages


def identify_worst_cases(sample_texts, n=18):
    """Identifikasi N worst cases berdasarkan CER."""
    if not sample_texts:
        print("\n❌ No per-sample data available for worst case analysis")
        return []
    
    # Sort by CER descending
    sorted_samples = sorted(sample_texts, key=lambda x: x['cer'], reverse=True)
    worst_cases = sorted_samples[:min(n, len(sorted_samples))]
    
    print(f"\n🔍 Top {len(worst_cases)} Worst Cases:")
    for i, case in enumerate(worst_cases, 1):
        print(f"   {i}. CER: {case['cer']:.2%} | WER: {case.get('wer', 0):.2%}")
        print(f"      GT:   {case.get('ground_truth', 'N/A')[:50]}...")
        print(f"      Pred: {case.get('generated_prediction', 'N/A')[:50]}...")
    
    return worst_cases


def generate_manual_categorization_template(worst_cases):
    """Generate template untuk manual categorization."""
    template = {
        'categorization_instructions': (
            "Untuk setiap worst case, lakukan inspeksi visual dan kategorikan:\n"
            "1. memudar_ekstrem: Teks terlalu pudar, model gagal rekonstruksi\n"
            "2. overlap: Text overlap dengan background/bleed-through\n"
            "3. ligatur_kompleks: Ligatur kompleks yang sulit dibaca\n"
            "4. noise_artifacts: Noise artifacts menghalangi teks\n"
            "5. lainnya: Kategori lain (deskripsikan)\n"
        ),
        'worst_cases': []
    }
    
    for i, case in enumerate(worst_cases, 1):
        template['worst_cases'].append({
            'case_id': i,
            'cer': case['cer'],
            'wer': case.get('wer', 0),
            'ground_truth': case.get('ground_truth', 'N/A'),
            'prediction': case.get('generated_prediction', 'N/A'),
            'category': 'TODO',  # Manual filling required
            'failure_description': 'TODO',  # Manual description
            'notes': ''
        })
    
    return template


def calculate_statistics(cer_values):
    """Calculate statistical measures."""
    if not cer_values:
        return {}
    
    cer_array = np.array(cer_values)
    
    stats = {
        'mean': float(np.mean(cer_array)),
        'median': float(np.median(cer_array)),
        'std': float(np.std(cer_array, ddof=1)),
        'min': float(np.min(cer_array)),
        'max': float(np.max(cer_array)),
        'q1': float(np.percentile(cer_array, 25)),
        'q3': float(np.percentile(cer_array, 75)),
        'p90': float(np.percentile(cer_array, 90)),
        'p95': float(np.percentile(cer_array, 95)),
        'p99': float(np.percentile(cer_array, 99))
    }
    
    return stats


def plot_cer_distribution(cer_values, output_path):
    """Plot CER distribution histogram."""
    if not cer_values:
        print("\n⚠️  No CER values to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(cer_values, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0.20, color='green', linestyle='--', label='20% threshold')
    axes[0].axvline(0.40, color='orange', linestyle='--', label='40% threshold')
    axes[0].set_xlabel('CER')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('CER Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    axes[1].boxplot(cer_values, vert=True)
    axes[1].axhline(0.20, color='green', linestyle='--', label='20%')
    axes[1].axhline(0.40, color='orange', linestyle='--', label='40%')
    axes[1].set_ylabel('CER')
    axes[1].set_title('CER Box Plot')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Distribution plot saved: {output_path}")
    plt.close()


def generate_report(results, bins, counts, percentages, stats, worst_cases, output_path):
    """Generate human-readable text report."""
    report_lines = [
        "=" * 80,
        "FAILURE CASE ANALYSIS REPORT",
        "=" * 80,
        f"Test Set Size: {results['test_set_size']} samples",
        f"Evaluation Date: {results.get('evaluation_date', 'Unknown')}",
        f"Protocol: {results.get('protocol', 'Unknown')}",
        "",
        "=" * 80,
        "1. CER DISTRIBUTION",
        "=" * 80,
    ]
    
    # Overall metrics dari original results
    cer_mean = results['htr_metrics']['cer']['mean']
    cer_std = results['htr_metrics']['cer']['std']
    cer_ci_lower = results['htr_metrics']['cer']['ci_95_lower']
    cer_ci_upper = results['htr_metrics']['cer']['ci_95_upper']
    
    report_lines.extend([
        f"Overall CER: {cer_mean:.4f} ± {cer_std:.4f}",
        f"95% CI: [{cer_ci_lower:.4f}, {cer_ci_upper:.4f}]",
        ""
    ])
    
    # NOTE: Binning hanya dari 10 sample texts (BUKAN full test set)
    if counts['excellent'] + counts['acceptable'] + counts['poor'] > 0:
        report_lines.extend([
            "⚠️  NOTE: Binning dari sample_texts saja (bukan full test set):",
            f"  Excellent (CER < 20%):     {counts['excellent']:3d} ({percentages['excellent']:.1f}%)",
            f"  Acceptable (20% ≤ CER < 40%): {counts['acceptable']:3d} ({percentages['acceptable']:.1f}%)",
            f"  Poor (CER ≥ 40%):         {counts['poor']:3d} ({percentages['poor']:.1f}%)",
            ""
        ])
    else:
        report_lines.extend([
            "❌ No per-sample binning available",
            "   RECOMMENDATION: Modify evaluate_test_set.py to save all_cer list",
            ""
        ])
    
    # Worst cases
    report_lines.extend([
        "=" * 80,
        f"2. WORST {len(worst_cases)} CASES",
        "=" * 80
    ])
    
    if worst_cases:
        for i, case in enumerate(worst_cases, 1):
            report_lines.extend([
                f"\nCase {i}:",
                f"  CER: {case['cer']:.2%}",
                f"  WER: {case.get('wer', 0):.2%}",
                f"  GT:   {case.get('ground_truth', 'N/A')[:60]}",
                f"  Pred: {case.get('generated_prediction', 'N/A')[:60]}",
            ])
    else:
        report_lines.append("❌ No worst case data available (need per-sample metrics)")
    
    report_lines.extend([
        "",
        "=" * 80,
        "3. RECOMMENDATIONS FOR PAPER",
        "=" * 80,
        "✅ USE: Overall CER statistics (mean, std, 95% CI)",
        "✅ USE: Qualitative failure mode description",
        "❌ REMOVE: Specific distribution counts (620/72/18) - tidak ada data lengkap",
        "❌ REMOVE: SNR correlation - tidak tersedia",
        "❌ REMOVE: Confidence detection metrics (P/R/F1) - tidak tersedia",
        "",
        "ALTERNATIVE APPROACH:",
        "1. Modifikasi evaluate_test_set.py untuk save per-sample metrics",
        "2. Re-run evaluasi (masih dalam academic protocol, tapi save more data)",
        "3. Atau: Gunakan hanya aggregate statistics dan qualitative description",
        ""
    ])
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n📄 Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze failure cases from test set evaluation')
    parser.add_argument('--input', type=str,
                       default='results/test_set_official_evaluation.json',
                       help='Path to evaluation results JSON')
    parser.add_argument('--output_dir', type=str,
                       default='results',
                       help='Output directory for analysis results')
    parser.add_argument('--n_worst', type=int, default=18,
                       help='Number of worst cases to analyze')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("FAILURE CASE ANALYSIS")
    print("=" * 80)
    
    # Load results
    results = load_evaluation_results(args.input)
    
    # Extract per-sample metrics (WARNING: limited to sample_texts only)
    sample_texts = extract_per_sample_metrics(results)
    
    if not sample_texts:
        print("\n" + "=" * 80)
        print("❌ CRITICAL: No per-sample data available")
        print("=" * 80)
        print("\nCurrent evaluate_test_set.py only saves:")
        print("  • Aggregated statistics (mean, std, 95% CI)")
        print("  • First 10 sample texts (qualitative)")
        print("\nFor complete failure case analysis, we need:")
        print("  • all_cer list (CER untuk semua 712 test samples)")
        print("  • all_wer list")
        print("  • Sample identifiers untuk manual inspection")
        print("\nRECOMMENDATIONS:")
        print("  OPTION A: Modifikasi evaluate_test_set.py untuk save per-sample data")
        print("  OPTION B: Gunakan HANYA aggregate stats + qualitative description di paper")
        print("\nProceeding with limited analysis from sample_texts...\n")
    
    # Extract CER values from sample_texts
    cer_values = [s['cer'] for s in sample_texts] if sample_texts else []
    
    # Bin CER distribution
    bins, counts, percentages = bin_cer_distribution(cer_values)
    
    # Calculate statistics
    stats = calculate_statistics(cer_values)
    if stats:
        print("\n📊 CER Statistics (from sample_texts):")
        print(f"   Mean: {stats['mean']:.4f}")
        print(f"   Median: {stats['median']:.4f}")
        print(f"   Std: {stats['std']:.4f}")
        print(f"   Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"   P95: {stats['p95']:.4f}")
    
    # Identify worst cases
    worst_cases = identify_worst_cases(sample_texts, n=args.n_worst)
    
    # Generate manual categorization template
    if worst_cases:
        manual_template = generate_manual_categorization_template(worst_cases)
        template_path = output_dir / 'manual_categorization_template.json'
        with open(template_path, 'w', encoding='utf-8') as f:
            json.dump(manual_template, f, indent=2, ensure_ascii=False)
        print(f"\n📋 Manual categorization template saved: {template_path}")
    
    # Plot distribution
    if cer_values:
        plot_path = output_dir / 'cer_distribution.png'
        plot_cer_distribution(cer_values, plot_path)
    
    # Generate comprehensive report
    report_path = output_dir / 'failure_case_report.txt'
    generate_report(results, bins, counts, percentages, stats, worst_cases, report_path)
    
    # Save analysis results as JSON
    analysis_results = {
        'test_set_size': results['test_set_size'],
        'analyzed_samples': len(sample_texts),
        'data_limitation': 'Only sample_texts available (10 samples), not full test set',
        'distribution': {
            'bins': counts,
            'percentages': percentages
        },
        'statistics': stats,
        'worst_cases_count': len(worst_cases),
        'recommendations': {
            'for_paper': [
                'Use aggregate statistics from full test set (mean, std, 95% CI)',
                'Add qualitative failure mode description',
                'Remove specific distribution counts without full data',
                'Remove SNR correlation (not available)',
                'Remove confidence metrics (not available)'
            ],
            'for_code': [
                'Modify evaluate_test_set.py to save all_cer, all_wer lists',
                'Add sample identifiers for failure case inspection',
                'Consider saving worst N cases with image paths'
            ]
        }
    }
    
    analysis_path = output_dir / 'failure_case_analysis.json'
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Analysis results saved: {analysis_path}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  • {report_path}")
    print(f"  • {analysis_path}")
    if worst_cases:
        print(f"  • {template_path}")
    if cer_values:
        print(f"  • {plot_path}")
    
    print("\n⚠️  NEXT STEPS:")
    print("  1. Review failure_case_report.txt")
    print("  2. Decide: Modify evaluate_test_set.py OR use aggregate stats only")
    print("  3. Update paper Section E accordingly")
    print("")


if __name__ == '__main__':
    main()
