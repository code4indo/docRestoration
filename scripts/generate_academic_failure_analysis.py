#!/usr/bin/env python3
"""
Generate Academically Valid Failure Analysis - V.7.2 Revision
==============================================================

Script ini menghasilkan:
1. Tabel distribusi CER yang terverifikasi
2. Tabel karakteristik error karakter
3. Visualisasi distribusi pola kegagalan
4. Data untuk revisi thesis

Author: ML Engineer
Date: 2025-12-10
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter, defaultdict
from pathlib import Path
import editdistance

# Paths
TEST_PREDICTIONS_RESTORED = "dual_modal_gan/analysis/test_predictions_restored.json"
CHARACTER_ERRORS_RESTORED = "dual_modal_gan/analysis/character_errors_restored.json"
OUTPUT_DIR = "Paper/data_dukung"

def load_data():
    """Load all required data"""
    # Load predictions
    with open(TEST_PREDICTIONS_RESTORED, 'r') as f:
        pred_data = json.load(f)
    
    predictions = pred_data.get('predictions', [])
    
    # Calculate CER for each sample
    for pred in predictions:
        gt = pred.get('gt_text', '')
        pred_text = pred.get('pred_text_restored', '')
        if len(gt) > 0:
            cer = editdistance.eval(gt, pred_text) / len(gt)
        else:
            cer = 0 if len(pred_text) == 0 else 1.0
        pred['cer'] = cer
    
    # Load character errors
    char_errors = None
    if os.path.exists(CHARACTER_ERRORS_RESTORED):
        with open(CHARACTER_ERRORS_RESTORED, 'r') as f:
            char_errors = json.load(f)
    
    return predictions, char_errors

def categorize_by_cer(predictions):
    """Categorize samples by CER thresholds"""
    categories = {
        'excellent': {'samples': [], 'threshold': '< 15%', 'cer_list': []},
        'moderate': {'samples': [], 'threshold': '15-50%', 'cer_list': []},
        'high': {'samples': [], 'threshold': '50-90%', 'cer_list': []},
        'extreme': {'samples': [], 'threshold': '≥ 90%', 'cer_list': []}
    }
    
    for pred in predictions:
        cer = pred['cer']
        if cer < 0.15:
            categories['excellent']['samples'].append(pred)
            categories['excellent']['cer_list'].append(cer)
        elif cer < 0.50:
            categories['moderate']['samples'].append(pred)
            categories['moderate']['cer_list'].append(cer)
        elif cer < 0.90:
            categories['high']['samples'].append(pred)
            categories['high']['cer_list'].append(cer)
        else:
            categories['extreme']['samples'].append(pred)
            categories['extreme']['cer_list'].append(cer)
    
    # Calculate statistics
    for cat in categories.values():
        cat['count'] = len(cat['samples'])
        cat['percentage'] = len(cat['samples']) / len(predictions) * 100
        cat['mean_cer'] = np.mean(cat['cer_list']) * 100 if cat['cer_list'] else 0
        cat['std_cer'] = np.std(cat['cer_list']) * 100 if len(cat['cer_list']) > 1 else 0
    
    return categories

def analyze_content_type(predictions):
    """Analyze content type for high-error samples"""
    content_analysis = defaultdict(list)
    
    for pred in predictions:
        gt = pred.get('gt_text', '')
        cer = pred['cer']
        
        # Detect content type
        if not gt:
            content_type = 'kosong'
        elif any(c in gt for c in ['ƒ', '§', '€', '$']):
            content_type = 'simbol_khusus'
        elif sum(1 for c in gt if c.isdigit()) / max(len(gt), 1) > 0.3:
            content_type = 'numerik_dominan'
        elif '. . .' in gt or '...' in gt:
            content_type = 'pola_titik'
        else:
            content_type = 'teks_alfabet'
        
        content_analysis[content_type].append({
            'cer': cer,
            'gt': gt,
            'pred': pred.get('pred_text_restored', '')
        })
    
    return content_analysis

def analyze_character_errors(char_errors):
    """Analyze character-level error patterns"""
    if not char_errors:
        return None
    
    errors = char_errors.get('errors', [])
    total_errors = len(errors)
    
    # Position distribution
    position_counts = Counter(e.get('word_position', 'unknown') for e in errors)
    
    # Ligature errors
    ligature_count = sum(1 for e in errors if e.get('is_ligature', False))
    
    # Punctuation errors
    punct_count = sum(1 for e in errors if e.get('is_punctuation', False))
    
    # Capital errors
    capital_count = sum(1 for e in errors if e.get('is_capital', False))
    
    return {
        'total_errors': total_errors,
        'position_distribution': dict(position_counts),
        'ligature_errors': ligature_count,
        'ligature_percentage': ligature_count / total_errors * 100 if total_errors > 0 else 0,
        'punctuation_errors': punct_count,
        'punctuation_percentage': punct_count / total_errors * 100 if total_errors > 0 else 0,
        'capital_errors': capital_count,
        'capital_percentage': capital_count / total_errors * 100 if total_errors > 0 else 0
    }

def create_visualization(categories, char_analysis, output_path):
    """Create academically valid visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: CER Distribution Bar Chart
    cat_names = ['Rendah\n(<15%)', 'Moderat\n(15-50%)', 'Tinggi\n(50-90%)', 'Ekstrem\n(≥90%)']
    counts = [categories['excellent']['count'], categories['moderate']['count'],
              categories['high']['count'], categories['extreme']['count']]
    percentages = [categories['excellent']['percentage'], categories['moderate']['percentage'],
                   categories['high']['percentage'], categories['extreme']['percentage']]
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
    
    bars = axes[0].bar(cat_names, counts, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Jumlah Sampel', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribusi Tingkat CER pada Set Uji (n=712)', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    # Right: Character Error Characteristics (Pie Chart)
    if char_analysis:
        # Position distribution
        positions = char_analysis['position_distribution']
        pos_labels = []
        pos_values = []
        pos_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for pos, color in zip(['middle', 'start', 'end'], pos_colors):
            if pos in positions:
                count = positions[pos]
                pct = count / char_analysis['total_errors'] * 100
                pos_labels.append(f'{pos.capitalize()}\n({pct:.1f}%)')
                pos_values.append(count)
        
        axes[1].pie(pos_values, labels=pos_labels, colors=pos_colors[:len(pos_values)],
                   autopct='', startangle=90, explode=[0.02]*len(pos_values))
        axes[1].set_title(f'Distribusi Posisi Error Karakter\n(n={char_analysis["total_errors"]:,} error)', 
                         fontsize=13, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'Data tidak tersedia', ha='center', va='center', fontsize=14)
        axes[1].set_title('Karakteristik Error Karakter', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, format='png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved: {output_path}")
    print(f"✅ Preview saved: {png_path}")

def generate_latex_table(categories, char_analysis):
    """Generate LaTeX table code"""
    
    # Table 1: CER Distribution
    table1 = """\\begin{table}[!htbp]
    \\renewcommand{\\arraystretch}{1.15}
    \\caption[Distribusi Tingkat CER]{Distribusi Tingkat CER pada \\textit{Set} Uji ($n$=712)}
    \\label{table:cer-distribution}
    \\centering
    \\small
    \\begin{tabular}{|l|c|c|c|}
        \\hline
        Kategori CER & Jumlah & \\% Total & CER Rata-rata (\\%) \\\\
        \\hline
        Rendah ($<$15\\%) & %d & %.1f & %.1f $\\pm$ %.1f \\\\
        Moderat (15--50\\%) & %d & %.1f & %.1f $\\pm$ %.1f \\\\
        Tinggi (50--90\\%) & %d & %.1f & %.1f $\\pm$ %.1f \\\\
        Ekstrem ($\\geq$90\\%) & %d & %.1f & %.1f $\\pm$ %.1f \\\\
        \\hline
        \\textbf{Total} & \\textbf{712} & \\textbf{100,0} & \\textbf{%.1f} \\\\
        \\hline
    \\end{tabular}
\\end{table}""" % (
        categories['excellent']['count'], categories['excellent']['percentage'],
        categories['excellent']['mean_cer'], categories['excellent']['std_cer'],
        categories['moderate']['count'], categories['moderate']['percentage'],
        categories['moderate']['mean_cer'], categories['moderate']['std_cer'],
        categories['high']['count'], categories['high']['percentage'],
        categories['high']['mean_cer'], categories['high']['std_cer'],
        categories['extreme']['count'], categories['extreme']['percentage'],
        categories['extreme']['mean_cer'], categories['extreme']['std_cer'],
        np.mean([p['cer'] for cat in categories.values() for p in cat['samples']]) * 100
    )
    
    # Table 2: Character Error Characteristics
    if char_analysis:
        pos = char_analysis['position_distribution']
        total = char_analysis['total_errors']
        table2 = """\\begin{table}[!htbp]
    \\renewcommand{\\arraystretch}{1.15}
    \\caption[Karakteristik Error Karakter]{Karakteristik Error Karakter pada Citra Terrestorasi ($n$=%d error)}
    \\label{table:char-error-characteristics}
    \\centering
    \\small
    \\begin{tabular}{|l|c|c|}
        \\hline
        Karakteristik & Jumlah & \\%% Total \\\\
        \\hline
        \\multicolumn{3}{|l|}{\\textit{Posisi dalam Kata}} \\\\
        \\hline
        \\quad Tengah & %d & %.1f \\\\
        \\quad Awal & %d & %.1f \\\\
        \\quad Akhir & %d & %.1f \\\\
        \\hline
        \\multicolumn{3}{|l|}{\\textit{Properti Karakter}} \\\\
        \\hline
        \\quad Dalam konteks ligatur & %d & %.1f \\\\
        \\quad Tanda baca & %d & %.1f \\\\
        \\quad Huruf kapital & %d & %.1f \\\\
        \\hline
    \\end{tabular}
\\end{table}""" % (
            total,
            pos.get('middle', 0), pos.get('middle', 0) / total * 100,
            pos.get('start', 0), pos.get('start', 0) / total * 100,
            pos.get('end', 0), pos.get('end', 0) / total * 100,
            char_analysis['ligature_errors'], char_analysis['ligature_percentage'],
            char_analysis['punctuation_errors'], char_analysis['punctuation_percentage'],
            char_analysis['capital_errors'], char_analysis['capital_percentage']
        )
    else:
        table2 = "% Character error data not available"
    
    return table1, table2

def main():
    print("=" * 80)
    print("GENERATING ACADEMICALLY VALID FAILURE ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\n[1/5] Loading data...")
    predictions, char_errors = load_data()
    print(f"      Loaded {len(predictions)} samples")
    
    # Categorize by CER
    print("\n[2/5] Categorizing by CER...")
    categories = categorize_by_cer(predictions)
    for name, cat in categories.items():
        print(f"      {name}: {cat['count']} samples ({cat['percentage']:.1f}%), mean CER: {cat['mean_cer']:.1f}%")
    
    # Analyze content types
    print("\n[3/5] Analyzing content types...")
    content_analysis = analyze_content_type(predictions)
    for content_type, samples in content_analysis.items():
        mean_cer = np.mean([s['cer'] for s in samples]) * 100
        print(f"      {content_type}: {len(samples)} samples, mean CER: {mean_cer:.1f}%")
    
    # Analyze character errors
    print("\n[4/5] Analyzing character errors...")
    char_analysis = analyze_character_errors(char_errors)
    if char_analysis:
        print(f"      Total errors: {char_analysis['total_errors']}")
        print(f"      Ligature context: {char_analysis['ligature_percentage']:.1f}%")
        print(f"      Punctuation: {char_analysis['punctuation_percentage']:.1f}%")
    
    # Create visualization
    print("\n[5/5] Creating visualization...")
    output_path = os.path.join(OUTPUT_DIR, "fig_failure_distribution_revised.pdf")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    create_visualization(categories, char_analysis, output_path)
    
    # Generate LaTeX tables
    print("\n" + "=" * 80)
    print("LATEX TABLE CODE")
    print("=" * 80)
    table1, table2 = generate_latex_table(categories, char_analysis)
    print("\n--- Table 1: CER Distribution ---")
    print(table1)
    print("\n--- Table 2: Character Error Characteristics ---")
    print(table2)
    
    # Save results
    results = {
        'cer_categories': {
            name: {
                'count': cat['count'],
                'percentage': cat['percentage'],
                'mean_cer': cat['mean_cer'],
                'std_cer': cat['std_cer']
            } for name, cat in categories.items()
        },
        'character_analysis': char_analysis,
        'content_analysis': {
            ct: {'count': len(samples), 'mean_cer': np.mean([s['cer'] for s in samples]) * 100}
            for ct, samples in content_analysis.items()
        }
    }
    
    results_path = "dual_modal_gan/analysis/failure_analysis_academic.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved: {results_path}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
