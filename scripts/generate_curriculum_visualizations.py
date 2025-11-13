#!/usr/bin/env python3
"""
Script untuk membuat visualisasi analisis Curriculum Learning vs Non-Curriculum Learning
untuk mendukung penulisan Chapter 5: Hasil dan Pembahasan
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for academic publications
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_training_metrics(file_path):
    """Load training metrics from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    epochs = data['epochs']
    results = {
        'epochs': [],
        'psnr': [],
        'ssim': [],
        'cer': [],
        'wer': [],
        'total_loss': [],
        'pixel_loss': [],
        'adv_loss': [],
        'rec_feat_loss': [],
        'ctc_loss': [],
        'perceptual_loss': [],
        'g_grad_norm': [],
        'd_grad_norm': [],
        'curriculum_phase': [],
        'current_ctc_weight': []
    }
    
    for epoch_data in epochs:
        epoch = epoch_data['epoch']
        val = epoch_data.get('validation', {})
        train = epoch_data.get('training_losses', {})
        
        results['epochs'].append(epoch)
        results['psnr'].append(val.get('psnr', 0))
        results['ssim'].append(val.get('ssim', 0))
        results['cer'].append(val.get('cer', 0))
        results['wer'].append(val.get('wer', 0))
        results['total_loss'].append(train.get('total_loss', 0))
        results['pixel_loss'].append(train.get('pixel_loss', 0))
        results['adv_loss'].append(train.get('adv_loss', 0))
        results['rec_feat_loss'].append(train.get('rec_feat_loss', 0))
        results['ctc_loss'].append(train.get('ctc_loss', 0))
        results['perceptual_loss'].append(train.get('perceptual_loss', 0))
        results['g_grad_norm'].append(train.get('gradient_norm', {}).get('generator_mean', 0))
        results['d_grad_norm'].append(train.get('gradient_norm', {}).get('discriminator_mean', 0))
        
        # Extract actual CTC weight from experimental data
        results['current_ctc_weight'].append(epoch_data.get('current_ctc_weight', 0))
        
        # Determine curriculum phase
        if epoch <= 30:
            phase = 'Warmup + Transition' if epoch <= 10 else 'CTC Transition'
        else:
            phase = 'Full Training'
        results['curriculum_phase'].append(phase)
    
    return pd.DataFrame(results)

def create_comparison_visualization():
    """Create comprehensive comparison visualization"""
    
    # Load data
    curriculum_path = "dual_modal_gan/checkpoints/experiment_curriculum_with/metrics/training_metrics_fp32_final.json"
    no_curriculum_path = "dual_modal_gan/checkpoints/experiment_curriculum_no/metrics/training_metrics_fp32_final.json"
    
    df_curriculum = load_training_metrics(curriculum_path)
    df_no_curriculum = load_training_metrics(no_curriculum_path)
    
    # Add experiment identifier
    df_curriculum['experiment'] = 'Curriculum Learning'
    df_no_curriculum['experiment'] = 'Non-Curriculum Learning'
    
    # Combine dataframes
    df_combined = pd.concat([df_curriculum, df_no_curriculum], ignore_index=True)
    
    # Create the main comparison figure
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Performance Metrics Comparison (PSNR, CER)
    ax1 = plt.subplot(4, 3, 1)
    sns.lineplot(data=df_combined, x='epochs', y='psnr', hue='experiment', ax=ax1, linewidth=2)
    ax1.set_title('Perbandingan PSNR selama Training', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('PSNR (dB)')
    ax1.legend(title='Pendekatan')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(4, 3, 2)
    sns.lineplot(data=df_combined, x='epochs', y='cer', hue='experiment', ax=ax2, linewidth=2)
    ax2.set_title('Perbandingan CER selama Training', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Character Error Rate')
    ax2.legend(title='Pendekatan')
    ax2.grid(True, alpha=0.3)
    
    # 2. Loss Components Comparison
    ax3 = plt.subplot(4, 3, 3)
    sns.lineplot(data=df_combined, x='epochs', y='total_loss', hue='experiment', ax=ax3, linewidth=2)
    ax3.set_title('Total Loss Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Total Loss')
    ax3.legend(title='Pendekatan')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 3. Individual Loss Components for Curriculum
    ax4 = plt.subplot(4, 3, 4)
    loss_components = ['pixel_loss', 'adv_loss', 'rec_feat_loss', 'ctc_loss', 'perceptual_loss']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, component in enumerate(loss_components):
        sns.lineplot(data=df_curriculum, x='epochs', y=component, 
                    label=component.replace('_', ' ').title(), 
                    ax=ax4, linewidth=2, color=colors[i])
    ax4.set_title('Komponen Loss - Curriculum Learning', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss Value')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # 4. Individual Loss Components for Non-Curriculum
    ax5 = plt.subplot(4, 3, 5)
    for i, component in enumerate(loss_components):
        sns.lineplot(data=df_no_curriculum, x='epochs', y=component, 
                    label=component.replace('_', ' ').title(), 
                    ax=ax5, linewidth=2, color=colors[i])
    ax5.set_title('Komponen Loss - Non-Curriculum Learning', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss Value')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # 5. Gradient Norms Comparison
    ax6 = plt.subplot(4, 3, 6)
    sns.lineplot(data=df_combined, x='epochs', y='g_grad_norm', hue='experiment', ax=ax6, linewidth=2)
    ax6.set_title('Generator Gradient Norm', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Gradient Norm')
    ax6.legend(title='Pendekatan')
    ax6.grid(True, alpha=0.3)
    
    # 6. SSIM Comparison
    ax7 = plt.subplot(4, 3, 7)
    sns.lineplot(data=df_combined, x='epochs', y='ssim', hue='experiment', ax=ax7, linewidth=2)
    ax7.set_title('Perbandingan SSIM selama Training', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('SSIM')
    ax7.legend(title='Pendekatan')
    ax7.grid(True, alpha=0.3)
    
    # 7. WER Comparison
    ax8 = plt.subplot(4, 3, 8)
    sns.lineplot(data=df_combined, x='epochs', y='wer', hue='experiment', ax=ax8, linewidth=2)
    ax8.set_title('Perbandingan WER selama Training', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('Word Error Rate')
    ax8.legend(title='Pendekatan')
    ax8.grid(True, alpha=0.3)
    
    # 8. Loss Variance Analysis
    ax9 = plt.subplot(4, 3, 9)
    # Calculate rolling variance for the last 10 epochs
    curriculum_var = df_curriculum['total_loss'].rolling(10).var()
    no_curriculum_var = df_no_curriculum['total_loss'].rolling(10).var()
    
    ax9.plot(df_curriculum['epochs'], curriculum_var, label='Curriculum Learning', linewidth=2)
    ax9.plot(df_no_curriculum['epochs'], no_curriculum_var, label='Non-Curriculum Learning', linewidth=2)
    ax9.set_title('Rolling Variance of Total Loss (Window=10)', fontsize=14, fontweight='bold')
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('Variance')
    ax9.legend(title='Pendekatan')
    ax9.grid(True, alpha=0.3)
    
    # 9. Final Performance Comparison (Bar Chart)
    ax10 = plt.subplot(4, 3, 10)
    final_metrics = ['PSNR', 'SSIM', 'CER', 'WER']
    curriculum_final = [df_curriculum['psnr'].iloc[-1], df_curriculum['ssim'].iloc[-1], 
                       df_curriculum['cer'].iloc[-1], df_curriculum['wer'].iloc[-1]]
    no_curriculum_final = [df_no_curriculum['psnr'].iloc[-1], df_no_curriculum['ssim'].iloc[-1],
                          df_no_curriculum['cer'].iloc[-1], df_no_curriculum['wer'].iloc[-1]]
    
    x = np.arange(len(final_metrics))
    width = 0.35
    
    bars1 = ax10.bar(x - width/2, curriculum_final, width, label='Curriculum Learning', alpha=0.8)
    bars2 = ax10.bar(x + width/2, no_curriculum_final, width, label='Non-Curriculum Learning', alpha=0.8)
    
    ax10.set_title('Perbandingan Metrik Akhir', fontsize=14, fontweight='bold')
    ax10.set_xlabel('Metrik')
    ax10.set_ylabel('Nilai')
    ax10.set_xticks(x)
    ax10.set_xticklabels(final_metrics)
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax10.annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax10.annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)
    
    # 10. Convergence Speed Analysis
    ax11 = plt.subplot(4, 3, 11)
    # Find peak performance epochs
    curriculum_peak_psnr = df_curriculum.loc[df_curriculum['psnr'].idxmax(), 'epochs']
    no_curriculum_peak_psnr = df_no_curriculum.loc[df_no_curriculum['psnr'].idxmax(), 'epochs']
    
    convergence_data = {
        'Metric': ['Peak PSNR', 'Peak CER', 'Peak SSIM'],
        'Curriculum Learning': [
            curriculum_peak_psnr,
            df_curriculum.loc[df_curriculum['cer'].idxmin(), 'epochs'],
            df_curriculum.loc[df_curriculum['ssim'].idxmax(), 'epochs']
        ],
        'Non-Curriculum Learning': [
            no_curriculum_peak_psnr,
            df_no_curriculum.loc[df_no_curriculum['cer'].idxmin(), 'epochs'],
            df_no_curriculum.loc[df_no_curriculum['ssim'].idxmax(), 'epochs']
        ]
    }
    
    convergence_df = pd.DataFrame(convergence_data)
    convergence_melted = convergence_df.melt(id_vars=['Metric'], var_name='Approach', value_name='Epoch')
    
    sns.barplot(data=convergence_melted, x='Metric', y='Epoch', hue='Approach', ax=ax11)
    ax11.set_title('Kecepatan Konvergensi (Epoch untuk Peak Performance)', fontsize=14, fontweight='bold')
    ax11.set_ylabel('Epoch')
    ax11.legend(title='Pendekatan')
    ax11.grid(True, alpha=0.3)
    
    # 11. Statistical Summary Table (as text)
    ax12 = plt.subplot(4, 3, 12)
    ax12.axis('off')
    
    # Create summary statistics
    summary_stats = {
        'Metric': ['PSNR', 'CER', 'SSIM', 'WER'],
        'Curriculum Mean': [f"{df_curriculum['psnr'].mean():.2f}", 
                          f"{df_curriculum['cer'].mean():.3f}",
                          f"{df_curriculum['ssim'].mean():.3f}", 
                          f"{df_curriculum['wer'].mean():.3f}"],
        'Curriculum Std': [f"{df_curriculum['psnr'].std():.2f}", 
                         f"{df_curriculum['cer'].std():.3f}",
                         f"{df_curriculum['ssim'].std():.3f}", 
                         f"{df_curriculum['wer'].std():.3f}"],
        'Non-Curriculum Mean': [f"{df_no_curriculum['psnr'].mean():.2f}", 
                              f"{df_no_curriculum['cer'].mean():.3f}",
                              f"{df_no_curriculum['ssim'].mean():.3f}", 
                              f"{df_no_curriculum['wer'].mean():.3f}"],
        'Non-Curriculum Std': [f"{df_no_curriculum['psnr'].std():.2f}", 
                             f"{df_no_curriculum['cer'].std():.3f}",
                             f"{df_no_curriculum['ssim'].std():.3f}", 
                             f"{df_no_curriculum['wer'].std():.3f}"]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Create table
    table_text = "Statistik Deskriptif\n" + "="*50 + "\n"
    for _, row in summary_df.iterrows():
        table_text += f"{row['Metric']:>8} | C: {row['Curriculum Mean']}±{row['Curriculum Std']} | NC: {row['Non-Curriculum Mean']}±{row['Non-Curriculum Std']}\n"
    
    ax12.text(0.05, 0.95, table_text, transform=ax12.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('dual_modal_gan/docs/curriculum_learning_comprehensive_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Comprehensive comparison visualization saved!")

def create_detailed_loss_analysis():
    """Create detailed analysis of individual loss components"""
    
    # Load data
    curriculum_path = "dual_modal_gan/checkpoints/experiment_curriculum_with/metrics/training_metrics_fp32_final.json"
    no_curriculum_path = "dual_modal_gan/checkpoints/experiment_curriculum_no/metrics/training_metrics_fp32_final.json"
    
    df_curriculum = load_training_metrics(curriculum_path)
    df_no_curriculum = load_training_metrics(no_curriculum_path)
    
    # Create detailed loss analysis figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    loss_components = ['pixel_loss', 'adv_loss', 'rec_feat_loss', 'ctc_loss', 'perceptual_loss']
    titles = ['Pixel Loss', 'Adversarial Loss', 'Recognition Feature Loss', 
              'CTC Loss', 'Perceptual Loss']
    
    for i, (component, title) in enumerate(zip(loss_components, titles)):
        ax = axes[i]
        
        # Plot both experiments
        sns.lineplot(data=df_curriculum, x='epochs', y=component, 
                    label='Curriculum Learning', ax=ax, linewidth=2)
        sns.lineplot(data=df_no_curriculum, x='epochs', y=component, 
                    label='Non-Curriculum Learning', ax=ax, linewidth=2)
        
        ax.set_title(f'{title} Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Add phase indicators for curriculum
        if i == 0:  # Only add to first plot
            ax.axvspan(1, 10, alpha=0.2, color='green', label='Warmup Phase')
            ax.axvspan(11, 30, alpha=0.2, color='orange', label='Transition Phase')
            ax.axvspan(31, 50, alpha=0.2, color='blue', label='Full Training')
    
    # Last subplot for gradient analysis
    ax = axes[5]
    sns.lineplot(data=df_curriculum, x='epochs', y='g_grad_norm', 
                label='Gen Grad (Curriculum)', ax=ax, linewidth=2)
    sns.lineplot(data=df_no_curriculum, x='epochs', y='g_grad_norm', 
                label='Gen Grad (Non-Curriculum)', ax=ax, linewidth=2)
    sns.lineplot(data=df_curriculum, x='epochs', y='d_grad_norm', 
                label='Disc Grad (Curriculum)', ax=ax, linewidth=2, linestyle='--')
    sns.lineplot(data=df_no_curriculum, x='epochs', y='d_grad_norm', 
                label='Disc Grad (Non-Curriculum)', ax=ax, linewidth=2, linestyle='--')
    
    ax.set_title('Gradient Norms Analysis', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('dual_modal_gan/docs/detailed_loss_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Detailed loss analysis visualization saved!")

def create_curriculum_phases_visualization():
    """Create visualization showing curriculum phases"""
    
    # Load curriculum data
    curriculum_path = "dual_modal_gan/checkpoints/experiment_curriculum_with/metrics/training_metrics_fp32_final.json"
    df_curriculum = load_training_metrics(curriculum_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. CTC Weight Schedule - FIXED: Use actual data from JSON
    ax1 = axes[0, 0]
    epochs = df_curriculum['epochs']
    
    # Use actual CTC weights from experimental data
    ctc_weights = df_curriculum['current_ctc_weight'].tolist()
    
    ax1.plot(epochs, ctc_weights, linewidth=3, color='red')
    ax1.axvspan(1, 10, alpha=0.2, color='green', label='Warmup Phase')
    ax1.axvspan(11, 30, alpha=0.2, color='orange', label='Transition Phase')
    ax1.axvspan(31, 50, alpha=0.2, color='blue', label='Full Training')
    ax1.set_title('Curriculum Learning: CTC Weight Schedule', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('CTC Weight')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss components by phase
    ax2 = axes[0, 1]
    phase_colors = {'Warmup + Transition': 'green', 'CTC Transition': 'orange', 'Full Training': 'blue'}
    
    for phase in df_curriculum['curriculum_phase'].unique():
        phase_data = df_curriculum[df_curriculum['curriculum_phase'] == phase]
        ax2.scatter(phase_data['epochs'], phase_data['total_loss'], 
                   c=phase_colors[phase], label=phase, alpha=0.7, s=30)
    
    ax2.set_title('Total Loss by Curriculum Phase', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Total Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Performance metrics by phase
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(epochs, df_curriculum['psnr'], 'b-', linewidth=2, label='PSNR')
    line2 = ax3_twin.plot(epochs, df_curriculum['cer'], 'r-', linewidth=2, label='CER')
    
    ax3.axvspan(1, 10, alpha=0.2, color='green')
    ax3.axvspan(11, 30, alpha=0.2, color='orange')
    ax3.axvspan(31, 50, alpha=0.2, color='blue')
    
    ax3.set_title('Performance Metrics by Curriculum Phase', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('PSNR (dB)', color='b')
    ax3_twin.set_ylabel('CER', color='r')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    ax3.grid(True, alpha=0.3)
    
    # 4. Loss contribution breakdown
    ax4 = axes[1, 1]
    
    # Calculate relative contribution of each loss component
    pixel_contrib = df_curriculum['pixel_loss'] / df_curriculum['total_loss']
    adv_contrib = df_curriculum['adv_loss'] / df_curriculum['total_loss']
    rec_contrib = df_curriculum['rec_feat_loss'] / df_curriculum['total_loss']
    ctc_contrib = df_curriculum['ctc_loss'] / df_curriculum['total_loss']
    perc_contrib = df_curriculum['perceptual_loss'] / df_curriculum['total_loss']
    
    ax4.stackplot(epochs, pixel_contrib, adv_contrib, rec_contrib, ctc_contrib, perc_contrib,
                  labels=['Pixel', 'Adversarial', 'Recognition', 'CTC', 'Perceptual'],
                  alpha=0.8)
    ax4.axvspan(1, 10, alpha=0.2, color='green')
    ax4.axvspan(11, 30, alpha=0.2, color='orange')
    ax4.axvspan(31, 50, alpha=0.2, color='blue')
    
    ax4.set_title('Loss Component Contributions Over Time', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Relative Contribution')
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dual_modal_gan/docs/curriculum_phases_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Curriculum phases visualization saved!")

def main():
    """Main function to generate all visualizations"""
    print("🎨 Generating Curriculum Learning Visualizations...")
    
    # Create docs directory if it doesn't exist
    Path("dual_modal_gan/docs").mkdir(parents=True, exist_ok=True)
    
    # Generate all visualizations
    create_comparison_visualization()
    create_detailed_loss_analysis()
    create_curriculum_phases_visualization()
    
    print("\n✅ All visualizations generated successfully!")
    print("📁 Saved in: dual_modal_gan/docs/")
    print("   - curriculum_learning_comprehensive_analysis.png")
    print("   - detailed_loss_analysis.png") 
    print("   - curriculum_phases_analysis.png")

if __name__ == "__main__":
    main()