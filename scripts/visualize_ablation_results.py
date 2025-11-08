#!/usr/bin/env python3
"""
Generate publication-quality ablation study visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Ablation results (extracted from training logs)
ablation_data = {
    '01_PIXEL_ONLY': {
        'psnr': 24.59,
        'psnr_std': 4.10,
        'ssim': 0.9614,
        'ssim_std': 0.0278,
        'cer': 1.0000,  # CER disabled (no recognizer)
        'components': ['Pixel (L1)']
    },
    '02_PIXEL+ADV': {
        'psnr': 24.86,
        'psnr_std': 4.21,
        'ssim': 0.9626,
        'ssim_std': 0.0279,
        'cer': 1.0000,  # CER disabled
        'components': ['Pixel (L1)', 'Adversarial']
    },
    '03_PIXEL+ADV+PERC': {
        'psnr': 24.55,
        'psnr_std': 4.69,
        'ssim': 0.9630,
        'ssim_std': 0.0273,
        'cer': 1.0000,  # CER disabled
        'components': ['Pixel (L1)', 'Adversarial', 'Perceptual']
    },
    '04_PIXEL+ADV+PERC+CTC': {
        'psnr': 24.76,
        'psnr_std': 4.71,
        'ssim': 0.9627,
        'ssim_std': 0.0294,
        'cer': 0.2966,  # CER enabled
        'components': ['Pixel (L1)', 'Adversarial', 'Perceptual', 'CTC']
    },
    '05_FULL': {
        'psnr': 24.75,
        'psnr_std': 4.60,
        'ssim': 0.9629,
        'ssim_std': 0.0312,
        'cer': 0.3016,  # CER enabled
        'components': ['Pixel (L1)', 'Adversarial', 'Perceptual', 'CTC', 'RecFeat']
    }
}

# Create output directory
output_dir = Path('visualization/ablation_study')
output_dir.mkdir(parents=True, exist_ok=True)

# Prepare data for plotting
exp_names = ['Pixel\nOnly', 'Pixel\n+Adv', 'Pixel+Adv\n+Perc', 'Pixel+Adv\n+Perc+CTC', 'Full\nModel']
psnr_values = [ablation_data[k]['psnr'] for k in ablation_data.keys()]
psnr_stds = [ablation_data[k]['psnr_std'] for k in ablation_data.keys()]
ssim_values = [ablation_data[k]['ssim'] for k in ablation_data.keys()]
ssim_stds = [ablation_data[k]['ssim_std'] for k in ablation_data.keys()]
cer_values = [ablation_data[k]['cer'] for k in ablation_data.keys()]

# Calculate improvements
psnr_improvements = [0] + [psnr_values[i] - psnr_values[0] for i in range(1, len(psnr_values))]
ssim_improvements = [0] + [(ssim_values[i] - ssim_values[0]) * 100 for i in range(1, len(ssim_values))]

# 1. Create multi-panel comparison figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Ablation Study: Incremental Loss Component Contribution', 
             fontsize=16, fontweight='bold', y=0.995)

# Panel A: PSNR comparison
ax = axes[0, 0]
x = np.arange(len(exp_names))
bars = ax.bar(x, psnr_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#06A77D'], 
              alpha=0.8, edgecolor='black', linewidth=1.5)
ax.errorbar(x, psnr_values, yerr=psnr_stds, fmt='none', color='black', 
            capsize=5, capthick=2, linewidth=1.5)

# Add value labels on bars
for i, (bar, val, std) in enumerate(zip(bars, psnr_values, psnr_stds)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.3,
            f'{val:.2f}±{std:.2f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
ax.set_xlabel('Loss Configuration', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(exp_names, fontsize=10)
ax.set_ylim([0, max(psnr_values) + max(psnr_stds) + 3])
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_title('(a) Peak Signal-to-Noise Ratio', fontsize=11, fontweight='bold', pad=10)

# Panel B: SSIM comparison
ax = axes[0, 1]
bars = ax.bar(x, ssim_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#06A77D'], 
              alpha=0.8, edgecolor='black', linewidth=1.5)
ax.errorbar(x, ssim_values, yerr=ssim_stds, fmt='none', color='black', 
            capsize=5, capthick=2, linewidth=1.5)

for i, (bar, val, std) in enumerate(zip(bars, ssim_values, ssim_stds)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.003,
            f'{val:.4f}±{std:.4f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('SSIM', fontsize=12, fontweight='bold')
ax.set_xlabel('Loss Configuration', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(exp_names, fontsize=10)
ax.set_ylim([0.90, 1.0])
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_title('(b) Structural Similarity Index', fontsize=11, fontweight='bold', pad=10)

# Panel C: PSNR incremental improvement
ax = axes[1, 0]
colors_improvement = ['gray', '#A23B72', '#F18F01', '#C73E1D', '#06A77D']
bars = ax.bar(x, psnr_improvements, color=colors_improvement, 
              alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (bar, val) in enumerate(zip(bars, psnr_improvements)):
    if val != 0:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02 if height > 0 else height - 0.1,
                f'{val:+.2f} dB',
                ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=9, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_ylabel('PSNR Improvement vs Baseline (dB)', fontsize=12, fontweight='bold')
ax.set_xlabel('Loss Configuration', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(exp_names, fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_title('(c) Incremental PSNR Gain', fontsize=11, fontweight='bold', pad=10)

# Panel D: SSIM incremental improvement (in percentage)
ax = axes[1, 1]
bars = ax.bar(x, ssim_improvements, color=colors_improvement, 
              alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (bar, val) in enumerate(zip(bars, ssim_improvements)):
    if val != 0:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.03 if height > 0 else height - 0.05,
                f'{val:+.2f}%',
                ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=9, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_ylabel('SSIM Improvement vs Baseline (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Loss Configuration', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(exp_names, fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_title('(d) Incremental SSIM Gain', fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig(output_dir / 'ablation_comparison_4panel.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig(output_dir / 'ablation_comparison_4panel.png', 
            dpi=300, bbox_inches='tight', format='png')
plt.close()

print(f"✅ 4-panel ablation comparison saved")

# 2. Create LaTeX table
latex_table = r"""\begin{table*}[t]
\centering
\caption{Studi Ablasi: Kontribusi Inkremental Komponen Fungsi Loss terhadap Performa Restorasi}
\label{tab:ablation_study}
\begin{tabular}{l|cccc|cc|cc}
\hline
\multirow{2}{*}{\textbf{Konfigurasi}} & \multicolumn{4}{c|}{\textbf{Komponen Loss}} & \multicolumn{2}{c|}{\textbf{Metrik Kualitas}} & \multicolumn{2}{c}{\textbf{Metrik HTR}} \\
\cline{2-9}
& Pixel & Adv & Perc & CTC & PSNR (dB) $\uparrow$ & SSIM $\uparrow$ & CER $\downarrow$ & $\Delta$CER \\
\hline
"""

# Add data rows
for i, (exp_id, data) in enumerate(ablation_data.items()):
    components = data['components']
    row = f"({i+1}) {exp_id.replace('_', ' ')}"
    
    # Check marks for components
    marks = []
    marks.append('✓' if 'Pixel (L1)' in components else '—')
    marks.append('✓' if 'Adversarial' in components else '—')
    marks.append('✓' if 'Perceptual' in components else '—')
    marks.append('✓' if 'CTC' in components else '—')
    
    # Metrics
    psnr_str = f"{data['psnr']:.2f} $\\pm$ {data['psnr_std']:.2f}"
    ssim_str = f"{data['ssim']:.4f} $\\pm$ {data['ssim_std']:.4f}"
    
    if data['cer'] < 1.0:
        cer_str = f"{data['cer']:.2%}"
        delta_cer = f"{(data['cer'] - ablation_data['04_PIXEL+ADV+PERC+CTC']['cer']):.2%}" if i > 3 else "—"
    else:
        cer_str = "N/A"
        delta_cer = "—"
    
    latex_table += f"{row} & {marks[0]} & {marks[1]} & {marks[2]} & {marks[3]} & {psnr_str} & {ssim_str} & {cer_str} & {delta_cer} \\\\\n"

latex_table += r"""\hline
\end{tabular}
\begin{tablenotes}
\small
\item \textbf{Catatan:} Semua eksperimen dilatih selama 15 epoch dengan dataset yang sama (3317 sampel train, 710 sampel validasi). 
\item Konfigurasi (1)-(3) dilatih tanpa recognizer ($\mathcal{L}_{CTC}=0$), sehingga CER tidak tersedia.
\item Konfigurasi (4) menambahkan CTC loss untuk pengenalan karakter. 
\item Konfigurasi (5) adalah model lengkap dengan recognizer feature loss ($\mathcal{L}_{RecFeat}$).
\item $\uparrow$ menunjukkan semakin tinggi semakin baik, $\downarrow$ menunjukkan semakin rendah semakin baik.
\end{tablenotes}
\end{table*}
"""

# Save LaTeX table
with open(output_dir / 'ablation_table.tex', 'w') as f:
    f.write(latex_table)

print(f"✅ LaTeX ablation table saved")

# 3. Create summary statistics table
summary_df = pd.DataFrame({
    'Experiment': [k.replace('_', ' ') for k in ablation_data.keys()],
    'PSNR (dB)': [f"{ablation_data[k]['psnr']:.2f} ± {ablation_data[k]['psnr_std']:.2f}" for k in ablation_data.keys()],
    'SSIM': [f"{ablation_data[k]['ssim']:.4f} ± {ablation_data[k]['ssim_std']:.4f}" for k in ablation_data.keys()],
    'CER': [f"{ablation_data[k]['cer']:.2%}" if ablation_data[k]['cer'] < 1.0 else "N/A" for k in ablation_data.keys()],
    'Components': [' + '.join(ablation_data[k]['components']) for k in ablation_data.keys()]
})

summary_df.to_csv(output_dir / 'ablation_summary.csv', index=False)
print(f"✅ CSV summary saved")

# Print summary to console
print("\n" + "="*80)
print("ABLATION STUDY RESULTS SUMMARY")
print("="*80)
print(summary_df.to_string(index=False))
print("="*80)

# 4. Create CER comparison (for experiments 4 and 5 only)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
cer_exp_names = ['Pixel+Adv+Perc+CTC', 'Full Model']
cer_exp_values = [ablation_data['04_PIXEL+ADV+PERC+CTC']['cer'], 
                  ablation_data['05_FULL']['cer']]

x = np.arange(len(cer_exp_names))
bars = ax.bar(x, cer_exp_values, color=['#C73E1D', '#06A77D'], 
              alpha=0.8, edgecolor='black', linewidth=1.5)

for bar, val in zip(bars, cer_exp_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{val:.2%}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Character Error Rate (CER)', fontsize=12, fontweight='bold')
ax.set_xlabel('Loss Configuration', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cer_exp_names, fontsize=11)
ax.set_ylim([0, max(cer_exp_values) * 1.2])
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_title('CER Comparison: Impact of Recognizer Feature Loss', 
             fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'ablation_cer_comparison.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig(output_dir / 'ablation_cer_comparison.png', 
            dpi=300, bbox_inches='tight', format='png')
plt.close()

print(f"✅ CER comparison figure saved")

print(f"\n📁 All ablation study visualizations saved to: {output_dir}")
print(f"   - ablation_comparison_4panel.pdf (main figure)")
print(f"   - ablation_table.tex (LaTeX table)")
print(f"   - ablation_summary.csv (raw data)")
print(f"   - ablation_cer_comparison.pdf (CER analysis)")
