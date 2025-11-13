#!/usr/bin/env python3
"""
Generate Simple Pie Chart untuk Loss Contribution
Fokus pada keterbacaan maksimal
"""

import matplotlib.pyplot as plt
from pathlib import Path

def generate_simple_pie_chart():
    """Generate pie chart sederhana dengan keterbacaan maksimal"""
    
    # Data dari paper
    paper_data = {
        'CTC': 64.7,
        'Perceptual': 27.4,
        'Adversarial': 2.9,
        'Pixel': 0.7,
        'RecFeat': 0.1
    }
    
    components = ['CTC', 'Perceptual', 'Adversarial', 'Pixel', 'RecFeat']
    contributions = [paper_data[comp] for comp in components]
    
    # Warna cerah dengan kontras tinggi
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    explode = [0.08, 0.04, 0, 0, 0]  # Highlight CTC dan Perceptual
    
    # Figure dengan ukuran optimal
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Pie chart sederhana
    wedges, texts, autotexts = ax.pie(
        contributions,
        labels=components,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=explode,
        textprops={'fontsize': 18, 'fontweight': 'bold'},
        pctdistance=0.75,
        labeldistance=1.1,
        wedgeprops={'edgecolor': 'white', 'linewidth': 3}
    )
    
    # Style labels (nama komponen) - hitam tebal
    for text in texts:
        text.set_fontsize(22)
        text.set_fontweight('bold')
        text.set_color('#2c3e50')
    
    # Style persentase - putih dengan background hitam
    for autotext in autotexts:
        autotext.set_fontsize(20)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
        autotext.set_bbox(dict(
            boxstyle="round,pad=0.6",
            facecolor='black',
            edgecolor='white',
            linewidth=2,
            alpha=0.9
        ))
    
    # Title sederhana
    ax.set_title('Kontribusi Efektif Setiap Komponen Loss',
                fontsize=26, fontweight='bold', pad=40, color='#2c3e50')
    
    # Save
    output_dir = Path("dual_modal_gan/docs/loss_weights_justification")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'loss_contribution_pie_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    
    print(f"✅ Saved: {output_path}")
    print(f"✅ Saved: {output_path.with_suffix('.pdf')}")
    print("\n📊 Data:")
    for i, comp in enumerate(components):
        print(f"  • {comp}: {contributions[i]:.1f}%")
    print(f"  • Total: {sum(contributions):.1f}%")
    print("\n✨ Design: Simple, clean, maximum readability")
    
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
