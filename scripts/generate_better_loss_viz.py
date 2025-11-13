#!/usr/bin/env python3
"""
Generate Better Loss Visualization: Side-by-side Bar Chart
Menunjukkan "Before vs After" Inverse Scaling - LEBIH INFORMATIF dari Pie Chart

Keunggulan:
1. Menunjukkan mekanisme inverse scaling (raw magnitude vs weighted contribution)
2. Mudah membandingkan nilai kecil (Pixel, RecFeat)
3. Memvisualisasikan novelty penelitian dengan jelas
4. Lebih scientific dan sesuai standar paper ML
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_side_by_side_bars():
    """Generate side-by-side bar chart: Raw Magnitude vs Weighted Contribution"""
    
    components = ['CTC', 'Perceptual', 'Adversarial', 'Pixel', 'RecFeat']
    
    # Data dari paper (empirical analysis) - Updated untuk konsistensi
    raw_magnitude = [392.01, 24.92, 0.89, 0.012, 0.010]  # Raw loss values
    weights = [0.15, 1.0, 3.0, 50.0, 8.0]  # Inverse scaling weights
    weighted_contribution = [67.5, 28.6, 3.1, 0.7, 0.1]  # Effective contribution (%) - normalized to 100%
    
    # Setup figure dengan 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    x_pos = np.arange(len(components))
    
    # === LEFT: Raw Magnitude (Before Inverse Scaling) ===
    bars1 = ax1.barh(x_pos, raw_magnitude, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_yticks(x_pos)
    ax1.set_yticklabels(components, fontsize=16, fontweight='bold')
    ax1.set_xlabel('Raw Loss Magnitude (Unweighted)', fontsize=16, fontweight='bold')
    ax1.set_title('BEFORE Inverse Scaling\n(Imbalanced)', 
                  fontsize=18, fontweight='bold', pad=20, color='#c0392b')
    ax1.set_xscale('log')  # Log scale karena range 10^-2 to 10^2
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Annotate values
    for i, (bar, val) in enumerate(zip(bars1, raw_magnitude)):
        ax1.text(val * 1.3, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}',
                va='center', ha='left', fontsize=14, fontweight='bold')
    
    # Highlight magnitude difference
    ax1.text(0.5, -0.15, 'Range: 4 orders of magnitude (10⁻² to 10²)',
            transform=ax1.transAxes, ha='center', fontsize=12, 
            style='italic', color='#c0392b', fontweight='bold')
    
    # === RIGHT: Weighted Contribution (After Inverse Scaling) ===
    bars2 = ax2.barh(x_pos, weighted_contribution, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(components, fontsize=16, fontweight='bold')
    ax2.set_xlabel('Effective Contribution (%)', fontsize=16, fontweight='bold')
    ax2.set_title('AFTER Inverse Scaling\n(Balanced by Design)', 
                  fontsize=18, fontweight='bold', pad=20, color='#27ae60')
    ax2.set_xlim(0, 70)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Annotate values and weights
    for i, (bar, val, weight) in enumerate(zip(bars2, weighted_contribution, weights)):
        ax2.text(val + 1.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}% (w={weight:.2f})',
                va='center', ha='left', fontsize=14, fontweight='bold')
    
    # Highlight target distribution
    ax2.text(0.5, -0.15, 'CTC (67.5%) + Perceptual (28.6%) = 96.1% → HTR-oriented',
            transform=ax2.transAxes, ha='center', fontsize=12, 
            style='italic', color='#27ae60', fontweight='bold')
    
    # Main title
    fig.suptitle('Inverse Scaling Principle: Balancing Loss Components\nfor HTR-Oriented Document Restoration',
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save
    output_dir = Path("dual_modal_gan/docs/loss_weights_justification")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'loss_contribution_sidebyside.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    
    print(f"✅ Saved: {output_path}")
    print(f"✅ Saved: {output_path.with_suffix('.pdf')}")
    
    plt.close()


def generate_stacked_comparison():
    """Generate stacked bar untuk menunjukkan transformation lebih jelas"""
    
    components = ['CTC', 'Perceptual', 'Adversarial', 'Pixel', 'RecFeat']
    
    # Normalized untuk perbandingan apple-to-apple
    raw_normalized = [91.2, 5.8, 0.2, 0.003, 0.002]  # % dari total raw
    weighted_contribution = [64.7, 27.4, 2.9, 0.7, 0.1]  # % setelah weighting
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    x = ['Before\nInverse Scaling\n(Raw)', 'After\nInverse Scaling\n(Weighted)']
    width = 0.6
    
    # Stacked bars
    bottom_raw = 0
    bottom_weighted = 0
    
    for i, comp in enumerate(components):
        # Before
        ax.bar(0, raw_normalized[i], width, bottom=bottom_raw, 
               color=colors[i], label=comp, edgecolor='black', linewidth=2, alpha=0.85)
        
        # Annotate if visible
        if raw_normalized[i] > 2:
            ax.text(0, bottom_raw + raw_normalized[i]/2, 
                   f'{comp}\n{raw_normalized[i]:.1f}%',
                   ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        bottom_raw += raw_normalized[i]
        
        # After
        ax.bar(1, weighted_contribution[i], width, bottom=bottom_weighted, 
               color=colors[i], edgecolor='black', linewidth=2, alpha=0.85)
        
        # Annotate
        if weighted_contribution[i] > 1:
            ax.text(1, bottom_weighted + weighted_contribution[i]/2, 
                   f'{comp}\n{weighted_contribution[i]:.1f}%',
                   ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        bottom_weighted += weighted_contribution[i]
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(x, fontsize=16, fontweight='bold')
    ax.set_ylabel('Contribution (%)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_title('Transformation Through Inverse Scaling:\nRebalancing Loss Components for HTR Priority',
                fontsize=20, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='upper right', fontsize=13, framealpha=0.95, edgecolor='black')
    
    # Annotations
    ax.annotate('', xy=(0.5, 50), xytext=(0.05, 50),
                arrowprops=dict(arrowstyle='->', lw=3, color='#2c3e50'))
    ax.text(0.27, 52, 'Inverse Scaling\nw ∝ 1/magnitude', 
           ha='center', fontsize=12, fontweight='bold', 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("dual_modal_gan/docs/loss_weights_justification")
    output_path = output_dir / 'loss_contribution_stacked.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    
    print(f"✅ Saved: {output_path}")
    print(f"✅ Saved: {output_path.with_suffix('.pdf')}")
    
    plt.close()


def generate_all_visualizations():
    """Generate semua visualisasi alternatif"""
    
    print("=" * 70)
    print("GENERATE BETTER LOSS VISUALIZATIONS")
    print("(More Informative than Pie Chart)")
    print("=" * 70)
    print()
    
    print("📊 Generating Side-by-Side Bar Chart (Raw vs Weighted)...")
    generate_side_by_side_bars()
    print()
    
    print("📊 Generating Stacked Bar Chart (Transformation)...")
    generate_stacked_comparison()
    print()
    
    print("=" * 70)
    print("✅ COMPLETED - 2 Alternative Visualizations")
    print("=" * 70)
    print()
    print("📈 REKOMENDASI:")
    print("  1. Side-by-Side Bar Chart → BEST untuk menunjukkan inverse scaling")
    print("  2. Stacked Bar Chart → BEST untuk menunjukkan transformation")
    print("  3. Pie Chart → Hanya untuk final contribution (kurang informatif)")
    print()
    print("💡 Untuk thesis/paper, gunakan Side-by-Side Bar Chart")
    print("   Lebih scientific dan menunjukkan novelty dengan jelas!")


if __name__ == "__main__":
    generate_all_visualizations()
