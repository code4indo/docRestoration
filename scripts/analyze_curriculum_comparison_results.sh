#!/bin/bash

# Script untuk menganalisis hasil eksperimen curriculum learning comparison
# Eksekusi: bash analyze_curriculum_comparison_results.sh

echo "📊 ANALISIS HASIL EKSPERIMEN CURRICULUM LEARNING"
echo "================================================"
echo ""

# Define paths
WITH_CURRICULUM_DIR="dual_modal_gan/checkpoints/experiment_curriculum_with"
NO_CURRICULUM_DIR="dual_modal_gan/checkpoints/experiment_curriculum_no"

WITH_CURRICULUM_METRICS="$WITH_CURRICULUM_DIR/metrics/training_metrics.json"
NO_CURRICULUM_METRICS="$NO_CURRICULUM_DIR/metrics/training_metrics.json"

# Check if directories exist
if [ ! -d "$WITH_CURRICULUM_DIR" ]; then
    echo "❌ Error: Directory $WITH_CURRICULUM_DIR tidak ditemukan!"
    echo "Pastikan eksperimen DENGAN curriculum learning sudah selesai."
    exit 1
fi

if [ ! -d "$NO_CURRICULUM_DIR" ]; then
    echo "❌ Error: Directory $NO_CURRICULUM_DIR tidak ditemukan!"
    echo "Pastikan eksperimen TANPA curriculum learning sudah selesai."
    exit 1
fi

# Check if metrics files exist
if [ ! -f "$WITH_CURRICULUM_METRICS" ]; then
    echo "❌ Error: File $WITH_CURRICULUM_METRICS tidak ditemukan!"
    exit 1
fi

if [ ! -f "$NO_CURRICULUM_METRICS" ]; then
    echo "❌ Error: File $NO_CURRICULUM_METRICS tidak ditemukan!"
    exit 1
fi

echo "✅ Semua file ditemukan!"
echo ""

# Create analysis directory
ANALYSIS_DIR="analysis_curriculum_comparison"
mkdir -p "$ANALYSIS_DIR"

echo "🔍 MELAKUKAN ANALISIS..."
echo ""

# Run Python analysis script
python3 - << EOF
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Load metrics data
with open('$WITH_CURRICULUM_METRICS', 'r') as f:
    with_data = json.load(f)

with open('$NO_CURRICULUM_METRICS', 'r') as f:
    no_data = json.load(f)

print("📊 HASIL ANALISIS EKSPERIMEN CURRICULUM LEARNING")
print("=" * 60)
print()

# 1. Basic Information
print("1. INFORMASI DASAR:")
print("-" * 30)
print(f"Dengan Curriculum:")
print(f"  - Total Epochs: {len(with_data['epochs'])}")
print(f"  - Start Time: {with_data['start_time']}")
print(f"  - Config: {with_data.get('hyperparameters', {}).get('precision', 'Unknown')}")

print(f"\nTanpa Curriculum:")
print(f"  - Total Epochs: {len(no_data['epochs'])}")
print(f"  - Start Time: {no_data['start_time']}")
print(f"  - Config: {no_data.get('hyperparameters', {}).get('precision', 'Unknown')}")
print()

# 2. Extract training metrics
def extract_metrics(epochs_data, metric_name):
    values = []
    for epoch in epochs_data:
        if metric_name in epoch.get('losses', {}):
            values.extend(epoch['losses'][metric_name])
        elif metric_name in epoch:
            values.append(epoch[metric_name])
    return values

def get_final_metrics(epochs_data):
    # Get final epoch validation metrics
    if epochs_data:
        final_epoch = epochs_data[-1]
        if 'validation' in final_epoch:
            return final_epoch['validation']
    return {}

print("2. ANALISIS STABILITAS LOSS:")
print("-" * 30)

# Calculate variance for generator loss
with_g_loss = extract_metrics(with_data['epochs'], 'g_loss')
no_g_loss = extract_metrics(no_data['epochs'], 'g_loss')

if with_g_loss and no_g_loss:
    with_variance = np.var(with_g_loss)
    no_variance = np.var(no_g_loss)
    
    print(f"Generator Loss Variance:")
    print(f"  - Dengan Curriculum: {with_variance:,.0f}")
    print(f"  - Tanpa Curriculum: {no_variance:,.0f}")
    print(f"  - Perbedaan: {no_variance/with_variance:.2f}x lebih tinggi (tanpa curriculum)")
    
    # Calculate percentage reduction
    reduction = (1 - with_variance/no_variance) * 100
    print(f"  - Reduction dengan Curriculum: {reduction:.1f}%")
else:
    print("  - Data loss tidak lengkap untuk analisis varians")

print()

# 3. Final Performance Metrics
print("3. FINAL PERFORMANCE METRICS:")
print("-" * 30)

with_final = get_final_metrics(with_data['epochs'])
no_final = get_final_metrics(no_data['epochs'])

if with_final and no_final:
    metrics = ['psnr', 'cer', 'ssim']
    for metric in metrics:
        if metric in with_final and metric in no_final:
            with_val = with_final[metric]
            no_val = no_final[metric]
            
            if metric == 'cer':  # Lower is better
                diff = no_val - with_val
                change = "peningkatan" if diff > 0 else "degradasi"
                print(f"{metric.upper()}:")
                print(f"  - Dengan Curriculum: {with_val:.4f}")
                print(f"  - Tanpa Curriculum: {no_val:.4f}")
                print(f"  - Perbedaan: {diff:+.4f} ({change} {abs(diff)/no_val*100:.1f}%)")
            else:  # Higher is better
                diff = with_val - no_val
                change = "peningkatan" if diff > 0 else "degradasi"
                print(f"{metric.upper()}:")
                print(f"  - Dengan Curriculum: {with_val:.4f}")
                print(f"  - Tanpa Curriculum: {no_val:.4f}")
                print(f"  - Perbedaan: {diff:+.4f} ({change} {abs(diff)/no_val*100:.1f}%)")
            print()
else:
    print("  - Data final metrics tidak lengkap")
    print()

# 4. Convergence Analysis
print("4. ANALISIS KONVERGENSI:")
print("-" * 30)

# Analyze training loss convergence
def analyze_convergence(g_losses):
    if not g_losses:
        return None, None
    
    # Calculate moving average
    window = 10
    if len(g_losses) >= window:
        moving_avg = np.convolve(g_losses, np.ones(window)/window, mode='valid')
        # Find where convergence happens (variance of moving avg becomes small)
        final_variance = np.var(moving_avg[-10:]) if len(moving_avg) >= 10 else np.var(moving_avg)
        return moving_avg, final_variance
    return g_losses, np.var(g_losses)

with_ma, with_conv = analyze_convergence(with_g_loss)
no_ma, no_conv = analyze_convergence(no_g_loss)

if with_conv is not None and no_conv is not None:
    print(f"Konvergensi (variance moving average):")
    print(f"  - Dengan Curriculum: {with_conv:,.0f}")
    print(f"  - Tanpa Curriculum: {no_conv:,.0f}")
    
    if with_conv < no_conv:
        improvement = ((no_conv - with_conv) / no_conv) * 100
        print(f"  - Curriculum learning menghasilkan konvergensi {improvement:.1f}% lebih stabil")
    print()

# 5. Generate comparison table
print("5. TABEL PERBANDINGAN (UNTUK CHAPTER 5):")
print("-" * 30)

comparison_data = {
    'Metrik': [],
    'Curriculum': [],
    'Tanpa Curriculum': [],
    'Perbedaan': []
}

if with_final and no_final:
    metrics_map = {
        'psnr': ('PSNR (dB)', True),  # Higher is better
        'ssim': ('SSIM', True),
        'cer': ('CER (%)', False),    # Lower is better
    }
    
    for metric_key, (metric_name, higher_better) in metrics_map.items():
        if metric_key in with_final and metric_key in no_final:
            with_val = with_final[metric_key]
            no_val = no_final[metric_key]
            
            if higher_better:
                diff = with_val - no_val
                perc_change = (diff / no_val) * 100
                comparison_data['Metrik'].append(metric_name)
                comparison_data['Curriculum'].append(f"{with_val:.2f}")
                comparison_data['Tanpa Curriculum'].append(f"{no_val:.2f}")
                comparison_data['Perbedaan'].append(f"+{diff:.2f} (+{perc_change:.1f}%)")
            else:
                diff = no_val - with_val
                perc_change = (diff / no_val) * 100
                comparison_data['Metrik'].append(metric_name)
                comparison_data['Curriculum'].append(f"{with_val:.2f}")
                comparison_data['Tanpa Curriculum'].append(f"{no_val:.2f}")
                comparison_data['Perbedaan'].append(f"-{diff:.2f} (-{perc_change:.1f}%)")

# Add stability metrics
if with_g_loss and no_g_loss:
    comparison_data['Metrik'].append('Stabilitas Loss (σ²)')
    comparison_data['Curriculum'].append(f"{with_variance:,.0f}")
    comparison_data['Tanpa Curriculum'].append(f"{no_variance:,.0f}")
    reduction = (1 - with_variance/no_variance) * 100
    comparison_data['Perbedaan'].append(f"-{reduction:.1f}%")

# Save to CSV
df = pd.DataFrame(comparison_data)
csv_path = f'{'$ANALYSIS_DIR'}/curriculum_comparison_results.csv'
df.to_csv(csv_path, index=False)
print(f"✅ Tabel disimpan ke: {csv_path}")
print()

print(df.to_string(index=False))
print()

# 6. Generate plots
print("6. GENERATING VISUALIZATIONS...")
print("-" * 30)

# Plot 1: Training Loss Comparison
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(with_g_loss, label='Dengan Curriculum', alpha=0.8)
plt.plot(no_g_loss, label='Tanpa Curriculum', alpha=0.8)
plt.title('Generator Loss Comparison')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Moving Average Convergence
if with_ma is not None and no_ma is not None:
    plt.subplot(2, 2, 2)
    x_with = np.arange(len(with_ma))
    x_no = np.arange(len(no_ma))
    plt.plot(x_with, with_ma, label='Dengan Curriculum', alpha=0.8)
    plt.plot(x_no, no_ma, label='Tanpa Curriculum', alpha=0.8)
    plt.title('Convergence Comparison (Moving Average)')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss (Moving Average)')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Plot 3: Final Metrics Bar Chart
if with_final and no_final:
    plt.subplot(2, 2, 3)
    metrics_to_plot = ['psnr', 'ssim', 'cer']
    metric_names = ['PSNR (dB)', 'SSIM', 'CER (%)']
    with_values = [with_final[m] for m in metrics_to_plot if m in with_final]
    no_values = [no_final[m] for m in metrics_to_plot if m in no_final]
    
    x = np.arange(len(metric_names[:len(with_values)]))
    width = 0.35
    
    plt.bar(x - width/2, with_values, width, label='Dengan Curriculum', alpha=0.8)
    plt.bar(x + width/2, no_values, width, label='Tanpa Curriculum', alpha=0.8)
    plt.title('Final Performance Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.xticks(x, metric_names[:len(with_values)], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

# Plot 4: Stability Comparison
plt.subplot(2, 2, 4)
stability_data = ['Dengan Curriculum', 'Tanpa Curriculum']
variance_data = [with_variance, no_variance] if 'with_variance' in locals() and 'no_variance' in locals() else [0, 0]

if variance_data[0] != 0 or variance_data[1] != 0:
    plt.bar(stability_data, variance_data, alpha=0.8, color=['blue', 'red'])
    plt.title('Training Stability (Loss Variance)')
    plt.ylabel('Loss Variance')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = f'{'$ANALYSIS_DIR'}/curriculum_comparison_analysis.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Plots disimpan ke: {plot_path}")
print()

# 7. Summary for Chapter 5
print("7. RINGKASAN UNTUK CHAPTER 5:")
print("-" * 30)

summary = f"""
ANALISIS EKSPERIMEN CURRICULUM LEARNING VS TANPA CURRICULUM LEARNING

HASIL UTAMA:
1. Stabilitas Loss:
   - Dengan Curriculum: {with_variance:,.0f}
   - Tanpa Curriculum: {no_variance:,.0f}
   - Improvement: {(1 - with_variance/no_variance)*100:.1f}% lebih stabil

2. Performance (Final Validation):
   - PSNR: {with_final.get('psnr', 'N/A')} vs {no_final.get('psnr', 'N/A')}
   - CER: {with_final.get('cer', 'N/A')} vs {no_final.get('cer', 'N/A')}
   - SSIM: {with_final.get('ssim', 'N/A')} vs {no_final.get('ssim', 'N/A')}

3. Kesimpulan:
   Curriculum learning terbukti {"menghasilkan konvergensi yang lebih stabil" if with_variance < no_variance else "tidak memberikan peningkatan signifikan"} dan {"performance yang lebih baik" if with_final.get('psnr', 0) > no_final.get('psnr', 0) else "performance serupa"}.
"""

print(summary)

# Save summary
summary_path = f'{'$ANALYSIS_DIR'}/curriculum_analysis_summary.txt'
with open(summary_path, 'w') as f:
    f.write(summary)

print(f"✅ Ringkasan disimpan ke: {summary_path}")
print()

print("🎉 ANALISIS SELESAI!")
print("===================")
print(f"📊 Files yang dihasilkan:")
print(f"  - {csv_path}")
print(f"  - {plot_path}")
print(f"  - {summary_path}")
print()
print("📝 UNTUK UPDATE CHAPTER 5:")
print("1. Ganti data di Tabel 4 dengan data dari CSV file")
print("2. Update teks analisis berdasarkan findings aktual")
print("3. Hapus semua data simulasi/hipotetis")
print("4. Tambahkan referensi ke eksperimen ini")

EOF

echo ""
echo "✅ ANALISIS SELESAI!"
echo "==================="
