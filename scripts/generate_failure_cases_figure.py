#!/usr/bin/env python3
"""
Generate failure cases figure for IEEE Journal paper
Menampilkan contoh representatif dari setiap kategori pola kegagalan

Author: Research Team
Date: 2025-11-05
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import cv2

# Configuration
DATA_FILE = "results/test_set_detailed_evaluation.json"
OUTPUT_DIR = "Paper/data_dukung"
OUTPUT_FILE = "fig_failure_cases.pdf"

# Sample IDs for each category (verified from analysis)
EXAMPLES = {
    'ligatur': {
        'id': 9,
        'title': '(a) Complex Paleographic Ligatures',
        'gt': 'jongst verleden jar zal mogen vol,',
        'pred': 'Jongst verleeden saar zal moogen vo,l,',
        'cer': 0.176,
        'highlight': ['verleden→verleeden', 'jar→saar']
    },
    'memudar': {
        'id': 0,
        'title': '(b) Extreme Text Fading',
        'gt': 'nodig . . . . . . . . . . . ƒ 1460 . - -',
        'pred': '1',
        'cer': 0.975,
        'highlight': ['Prediction: "1" (97.5% error)']
    },
    'simbol': {
        'id': 684,
        'title': '(c) Numeric & Symbol Artifacts',
        'gt': '§ 73:',
        'pred': '537201',
        'cer': 1.000,
        'highlight': ['§→537201']
    },
    'normal': {
        'id': 8,
        'title': '(d) Normal Case (Low CER)',
        'gt': 'Terdaan doarvan onse gesanten',
        'pred': 'Terdaan daarvan onse gesanten',
        'cer': 0.034,
        'highlight': ['doarvan→daarvan']
    }
}

def load_data():
    """Load test set evaluation data"""
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    return data

def find_image_path(sample_id):
    """Find image file for given sample ID"""
    # Try multiple possible locations
    base_dirs = [
        "DataUjiKuantitatifSintetis/metadata",
        "data/test/degraded",
        "data/test/ground_truth"
    ]
    
    patterns = [
        f"sample_{sample_id:02d}_*.png",
        f"sample_{sample_id:04d}_*.png",
        f"{sample_id:04d}.png"
    ]
    
    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            continue
            
        for pattern in patterns:
            matches = list(base_path.glob(pattern))
            if matches:
                return str(matches[0])
    
    return None

def create_text_comparison_subplot(ax, example, sample_data):
    """Create text comparison visualization with larger, clearer text"""
    ax.axis('off')
    
    # Title - LARGER
    ax.text(0.5, 0.96, example['title'], 
            ha='center', va='top', fontsize=13, fontweight='bold',
            transform=ax.transAxes)
    
    # Ground Truth - LARGER
    ax.text(0.03, 0.78, 'GT:', fontsize=11, fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.13, 0.78, f'"{example["gt"]}"', fontsize=10.5,
            transform=ax.transAxes, style='italic', wrap=True)
    
    # Prediction - LARGER
    ax.text(0.03, 0.58, 'Pred:', fontsize=11, fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.13, 0.58, f'"{example["pred"]}"', fontsize=10.5,
            transform=ax.transAxes, color='darkred', wrap=True)
    
    # Metrics - LARGER with better spacing
    cer_color = 'darkred' if example['cer'] > 0.5 else 'darkorange' if example['cer'] > 0.15 else 'darkgreen'
    ax.text(0.03, 0.38, f'CER: {example["cer"]*100:.1f}%', fontsize=11,
            fontweight='bold', color=cer_color, transform=ax.transAxes)
    
    if 'psnr' in sample_data:
        ax.text(0.35, 0.38, f'PSNR: {sample_data["psnr"]:.1f} dB', 
                fontsize=10, transform=ax.transAxes)
    
    # Highlights - LARGER
    y_pos = 0.20
    for highlight in example['highlight']:
        ax.text(0.03, y_pos, f'• {highlight}', fontsize=9.5,
                color='navy', transform=ax.transAxes, fontweight='medium')
        y_pos -= 0.12

def create_failure_cases_figure():
    """Create comprehensive failure cases figure with larger, clearer layout"""
    print("Loading data...")
    data = load_data()
    sample_texts = data['sample_texts']
    
    # Create LARGER figure for better readability
    fig = plt.figure(figsize=(16, 12))
    
    # Title - LARGER
    fig.suptitle('Categorization of Failure Patterns on Test Set (n=712)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Create subplots - 2x2 grid with MORE SPACING
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for idx, (category, example) in enumerate(EXAMPLES.items()):
        row, col = positions[idx]
        # Use GridSpec for better control
        ax = plt.subplot2grid((2, 2), (row, col), fig=fig)
        
        # Get sample data
        sample_data = sample_texts[example['id']]
        
        # Create text comparison
        create_text_comparison_subplot(ax, example, sample_data)
    
    # Add summary statistics box - LARGER
    fig.text(0.5, 0.015, 
             'Data: Test set evaluation (n=712) | Model: Production v3 (epoch 44) | '
             'Distribution: Ligatures 62.9%, Numeric/Symbols 1.8%, Faded Text 0.4%',
             ha='center', fontsize=10, style='italic', color='gray')
    
    # Better spacing - MORE COMPACT layout
    plt.tight_layout(rect=[0, 0.025, 1, 0.97], h_pad=3.5, w_pad=2.5)
    
    # Save
    output_path = Path(OUTPUT_DIR) / OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved to: {output_path}")
    
    # Also save as HIGH-RES PNG for preview
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, format='png', dpi=200, bbox_inches='tight')
    print(f"✅ Preview saved to: {png_path}")
    
def create_distribution_bar_chart():
    """Create bar chart showing distribution of failure patterns with larger elements"""
    categories = ['Paleographic\nLigatures', 'Others\n(CER>0.5)', 
                  'Numeric &\nSymbols', 'Faded\nText', 'Short\nPredictions']
    counts = [448, 67, 13, 3, 3]
    percentages = [62.9, 9.4, 1.8, 0.4, 0.4]
    cer_avg = [35.2, 78.3, 92.1, 100.0, 97.5]
    
    # LARGER figure for better readability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart - Counts with LARGER bars and text
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
    bars1 = ax1.bar(categories, counts, color=colors, alpha=0.75, 
                    edgecolor='black', linewidth=1.5, width=0.7)
    ax1.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax1.set_title('Distribution of Failure Patterns (n=712)', fontsize=15, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.2)
    ax1.tick_params(axis='both', labelsize=12)
    
    # Add value labels - LARGER
    for bar, count, pct in zip(bars1, counts, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{count}\n({pct}%)', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # Bar chart - Average CER with LARGER bars and text
    colors_cer = ['#2ca02c' if c < 40 else '#ff7f0e' if c < 80 else '#d62728' for c in cer_avg]
    bars2 = ax2.bar(categories, cer_avg, color=colors_cer, alpha=0.75, 
                    edgecolor='black', linewidth=1.5, width=0.7)
    ax2.set_ylabel('Average CER (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Failure Severity by Category', fontsize=15, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.2)
    ax2.set_ylim([0, 110])
    ax2.tick_params(axis='both', labelsize=12)
    
    # Add value labels - LARGER
    for bar, cer in zip(bars2, cer_avg):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{cer:.1f}%', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # Better spacing
    plt.tight_layout(pad=2.0)
    
    # Save
    output_path = Path(OUTPUT_DIR) / "fig_failure_distribution.pdf"
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"✅ Distribution chart saved to: {output_path}")
    
    # HIGH-RES PNG preview
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, format='png', dpi=200, bbox_inches='tight')
    print(f"✅ Preview saved to: {png_path}")
    
    plt.close()


if __name__ == "__main__":
    print("="*80)
    print("GENERATING FAILURE CASES VISUALIZATION FOR IEEE JOURNAL PAPER")
    print("="*80)
    
    try:
        # Create main failure cases figure
        print("\n[1/2] Creating failure cases comparison figure...")
        create_failure_cases_figure()
        
        # Create distribution chart
        print("\n[2/2] Creating distribution bar chart...")
        create_distribution_bar_chart()
        
        print("\n" + "="*80)
        print("✅ ALL FIGURES GENERATED SUCCESSFULLY")
        print("="*80)
        print("\nFiles created:")
        print("  - Paper/data_dukung/fig_failure_cases.pdf")
        print("  - Paper/data_dukung/fig_failure_cases.png (preview)")
        print("  - Paper/data_dukung/fig_failure_distribution.pdf")
        print("  - Paper/data_dukung/fig_failure_distribution.png (preview)")
        print("\nYou can now reference these in your LaTeX document:")
        print("  \\includegraphics[width=\\textwidth]{../data_dukung/fig_failure_cases.pdf}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
