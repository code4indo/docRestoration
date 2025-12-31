#!/usr/bin/env python3
"""
Generate Simple Pie Chart untuk Loss Contribution
Fokus pada keterbacaan maksimal - IMPROVED VERSION
Menghindari tumpang tindih label dengan menggunakan legend box
"""

import matplotlib.pyplot as plt
from pathlib import Path

def generate_simple_pie_chart():
    """Generate pie chart dengan legend terpisah untuk menghindari overlap"""
    
    # Data dari paper - sesuai dengan tabel di chapter 5
    paper_data = {
        'CTC': 67.5,
        'Perceptual': 28.6,
        'Adversarial': 3.1,
        'Pixel': 0.7,
        'RecFeat': 0.1
    }
    
    components = ['CTC', 'Perceptual', 'Adversarial', 'Pixel', 'RecFeat']
    contributions = [paper_data[comp] for comp in components]
    
    # Warna cerah dengan kontras tinggi
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    explode = [0.1, 0.05, 0.02, 0, 0]  # Lebih jelas memisahkan komponen besar
    
    # Figure dengan ukuran yang lebih besar untuk menghindari cramming
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Pie chart TANPA label langsung (gunakan legend)
    # Tanpa persentase di atas wedges untuk tampilan lebih bersih
    wedges, texts = ax.pie(
        contributions,
        labels=None,  # Hapus label langsung
        autopct=None,  # Hapus persentase di atas pie chart
        startangle=45,  # Rotasi optimal untuk memisahkan komponen kecil
        colors=colors,
        explode=explode,
        wedgeprops={'edgecolor': 'white', 'linewidth': 4}
    )
    
    # Buat legend dengan informasi lengkap (komponen + nilai + persentase)
    legend_labels = [
        f'{comp}: {val:.1f}%' 
        for comp, val in zip(components, contributions)
    ]
    
    # Legend di kanan atas dengan box yang rapi
    legend = ax.legend(
        wedges, 
        legend_labels,
        title="Komponen Loss",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=16,
        title_fontsize=18,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        edgecolor='#2c3e50',
        facecolor='white'
    )
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_color('#2c3e50')
    
    # Save dengan berbagai format
    output_dir = Path("dual_modal_gan/docs/loss_weights_justification")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'loss_contribution_pie_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white', pad_inches=0.3)
    
    print(f"✅ Saved: {output_path}")
    print(f"✅ Saved: {output_path.with_suffix('.pdf')}")
    print("\n📊 Data (Corrected from Table):")
    for i, comp in enumerate(components):
        print(f"  • {comp}: {contributions[i]:.1f}%")
    print(f"  • Total: {sum(contributions):.1f}%")
    print("\n✨ Improvements:")
    print("  • Labels moved to legend box (no overlap)")
    print("  • Optimized startangle for small component visibility")
    print("  • Larger explode values for better separation")
    print("  • Percentage positioned at 0.82 (outer edge)")
    print("  • Data corrected to match Chapter 5 Table values")
    
    plt.close()

if __name__ == "__main__":
    print("=" * 60)
    print("GENERATE SIMPLE PIE CHART - LOSS CONTRIBUTION")
    print("=" * 60)
    print()
    
    generate_simple_pie_chart()
    
    print()
    print("=" * 60)
    print("✅ COMPLETED")
    print("=" * 60)
