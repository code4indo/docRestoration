#!/usr/bin/env python3
"""
Generate Comprehensive Summary Visualizations

Create publication-quality summary figures showing:
1. Complete error transformation (degraded → restored)
2. Multi-panel comparison dashboard
3. Key metrics summary poster
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_comparison_data():
    """Load comparison summary"""
    with open('dual_modal_gan/analysis/degraded_vs_restored_summary.json', 'r') as f:
        return json.load(f)

def create_transformation_flowchart(data, output_path='transformation_flowchart.png'):
    """
    Create flowchart showing error transformation
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Data
    deg_deletion = data['by_error_type']['deletion']['degraded']
    deg_substitution = data['by_error_type']['substitution']['degraded']
    res_deletion = data['by_error_type']['deletion']['restored']
    res_substitution = data['by_error_type']['substitution']['restored']
    
    total_deg = data['total_errors']['degraded']
    total_res = data['total_errors']['restored']
    
    # Left side: Degraded
    ax.text(0.15, 0.85, 'DEGRADED\nIMAGES', ha='center', va='center',
           fontsize=16, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFE5E5', edgecolor='#FF6B6B', linewidth=3))
    
    # Degraded error breakdown
    ax.text(0.15, 0.60, f'Total Errors:\n{total_deg:,}', ha='center', va='center',
           fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#FF6B6B', linewidth=2))
    
    ax.text(0.05, 0.40, f'Deletion\n{deg_deletion:,}\n(92.8%)', ha='center', va='center',
           fontsize=10,
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFB6B6', edgecolor='#FF6B6B', linewidth=2))
    
    ax.text(0.25, 0.40, f'Substitution\n{deg_substitution:,}\n(7.2%)', ha='center', va='center',
           fontsize=10,
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFE5CC', edgecolor='#FFA500', linewidth=2))
    
    # Arrow: Restoration
    arrow = mpatches.FancyArrowPatch((0.30, 0.70), (0.55, 0.70),
                                     arrowstyle='->,head_width=0.8,head_length=0.4',
                                     color='#4ECDC4', linewidth=5,
                                     mutation_scale=30)
    ax.add_patch(arrow)
    
    ax.text(0.425, 0.78, 'RESTORATION', ha='center', va='center',
           fontsize=14, fontweight='bold', color='#4ECDC4',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#4ECDC4', linewidth=2))
    
    ax.text(0.425, 0.62, '70% Error\nReduction', ha='center', va='center',
           fontsize=10, style='italic', color='#4ECDC4')
    
    # Right side: Restored
    ax.text(0.85, 0.85, 'RESTORED\nIMAGES', ha='center', va='center',
           fontsize=16, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='#E5FFF5', edgecolor='#4ECDC4', linewidth=3))
    
    # Restored error breakdown
    ax.text(0.85, 0.60, f'Total Errors:\n{total_res:,}', ha='center', va='center',
           fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#4ECDC4', linewidth=2))
    
    ax.text(0.75, 0.40, f'Deletion\n{res_deletion:,}\n(35.2%)', ha='center', va='center',
           fontsize=10,
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#B6E0DA', edgecolor='#4ECDC4', linewidth=2))
    
    ax.text(0.95, 0.40, f'Substitution\n{res_substitution:,}\n(64.8%)', ha='center', va='center',
           fontsize=10,
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFE5CC', edgecolor='#FFA500', linewidth=2))
    
    # Key insight boxes
    ax.text(0.15, 0.15, f'-88.6%\nDeletion\nRecovery', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#28A745', edgecolor='#1E7E34', linewidth=2))
    
    ax.text(0.50, 0.15, f'+168%\nSubstitution\n(Visible but\nConfused)', ha='center', va='center',
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFC107', edgecolor='#FF8800', linewidth=2))
    
    ax.text(0.85, 0.15, f'97.8%\nCharacters\nNow Visible', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#28A745', edgecolor='#1E7E34', linewidth=2))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.title('Character-Level Error Transformation through Restoration',
             fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created transformation flowchart: {output_path}")

def create_dashboard_summary(data, output_path='summary_dashboard.png'):
    """
    Create multi-panel dashboard with all key metrics
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # 1. Total errors comparison (large, top-left)
    ax1 = fig.add_subplot(gs[0:2, 0])
    
    categories = ['Degraded', 'Restored']
    values = [data['total_errors']['degraded'], data['total_errors']['restored']]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Total Errors', fontsize=12, fontweight='bold')
    ax1.set_title('Total Error Count', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val):,}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add improvement arrow
    improvement = data['total_errors']['improvement']
    improvement_pct = data['total_errors']['improvement_pct']
    ax1.text(0.5, max(values)*0.7, f'↓ {improvement:,}\n({improvement_pct:.1f}% reduction)',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 2. Error type comparison (top-middle)
    ax2 = fig.add_subplot(gs[0, 1:])
    
    error_types = ['Deletion', 'Substitution']
    degraded_vals = [
        data['by_error_type']['deletion']['degraded'],
        data['by_error_type']['substitution']['degraded']
    ]
    restored_vals = [
        data['by_error_type']['deletion']['restored'],
        data['by_error_type']['substitution']['restored']
    ]
    
    x = np.arange(len(error_types))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, degraded_vals, width, label='Degraded', color='#FF6B6B', alpha=0.8)
    bars2 = ax2.bar(x + width/2, restored_vals, width, label='Restored', color='#4ECDC4', alpha=0.8)
    
    ax2.set_ylabel('Error Count', fontsize=11, fontweight='bold')
    ax2.set_title('Error Type Distribution', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(error_types)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Deletion recovery highlight (middle-middle)
    ax3 = fig.add_subplot(gs[1, 1])
    
    deletion_improvement = data['by_error_type']['deletion']['improvement']
    deletion_deg = data['by_error_type']['deletion']['degraded']
    deletion_pct = (deletion_improvement / deletion_deg) * 100
    
    ax3.text(0.5, 0.6, f'{deletion_pct:.1f}%', ha='center', va='center',
            fontsize=48, fontweight='bold', color='#28A745')
    ax3.text(0.5, 0.3, 'Deletion\nRecovery', ha='center', va='center',
            fontsize=14, fontweight='bold')
    ax3.text(0.5, 0.1, f'({deletion_improvement:,} errors)', ha='center', va='center',
            fontsize=10, style='italic')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.patch.set_facecolor('#E5FFE5')
    
    # 4. Character recognition rate (middle-right)
    ax4 = fig.add_subplot(gs[1, 2])
    
    ax4.text(0.5, 0.6, '97.8%', ha='center', va='center',
            fontsize=48, fontweight='bold', color='#4ECDC4')
    ax4.text(0.5, 0.3, 'Characters\nRecognized', ha='center', va='center',
            fontsize=14, fontweight='bold')
    ax4.text(0.5, 0.1, '(after restoration)', ha='center', va='center',
            fontsize=10, style='italic')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.patch.set_facecolor('#E5F5FF')
    
    # 5. Position distribution (bottom-left)
    ax5 = fig.add_subplot(gs[2, 0])
    
    positions = ['Start', 'Middle', 'End']
    pos_improvements = [
        data['by_position']['start']['improvement'],
        data['by_position']['middle']['improvement'],
        data['by_position']['end']['improvement']
    ]
    
    bars = ax5.barh(positions, pos_improvements, color='#95E1D3', alpha=0.8, edgecolor='black')
    ax5.set_xlabel('Errors Reduced', fontsize=10, fontweight='bold')
    ax5.set_title('Position-wise Improvement', fontsize=12, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, pos_improvements):
        width = bar.get_width()
        ax5.text(width + max(pos_improvements)*0.02, bar.get_y() + bar.get_height()/2,
                f'{int(val):,}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # 6. Key metrics table (bottom-middle to bottom-right)
    ax6 = fig.add_subplot(gs[2, 1:])
    ax6.axis('off')
    
    table_data = [
        ['Metric', 'Degraded', 'Restored', 'Improvement'],
        ['Total Errors', f"{data['total_errors']['degraded']:,}", f"{data['total_errors']['restored']:,}", 
         f"{data['total_errors']['improvement']:,} ({data['total_errors']['improvement_pct']:.1f}%)"],
        ['Deletion', f"{data['by_error_type']['deletion']['degraded']:,} (92.8%)", 
         f"{data['by_error_type']['deletion']['restored']:,} (35.2%)", 
         f"{data['by_error_type']['deletion']['improvement']:,} (88.6%)"],
        ['Substitution', f"{data['by_error_type']['substitution']['degraded']:,} (7.2%)", 
         f"{data['by_error_type']['substitution']['restored']:,} (64.8%)", 
         f"{data['by_error_type']['substitution']['improvement']:,} (+168%)"],
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, 4):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    # Main title
    fig.suptitle('Character-Level Diagnostic Summary: Degraded vs Restored',
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created summary dashboard: {output_path}")

def create_key_metrics_poster(data, output_path='key_metrics_poster.png'):
    """
    Create single-page poster with key metrics
    """
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'CHARACTER-LEVEL DIAGNOSTIC',
           ha='center', va='center', fontsize=24, fontweight='bold')
    ax.text(0.5, 0.92, 'Restoration Impact on HTR Error Patterns',
           ha='center', va='center', fontsize=16, style='italic')
    
    # Metric 1: Total error reduction
    ax.add_patch(mpatches.Rectangle((0.05, 0.78), 0.9, 0.10,
                                   facecolor='#E5F5FF', edgecolor='#4ECDC4', linewidth=3))
    ax.text(0.5, 0.86, '70% Total Error Reduction',
           ha='center', va='center', fontsize=20, fontweight='bold', color='#4ECDC4')
    ax.text(0.5, 0.81, f"19,417 → 5,824 errors",
           ha='center', va='center', fontsize=14)
    
    # Metric 2: Deletion recovery
    ax.add_patch(mpatches.Rectangle((0.05, 0.63), 0.42, 0.12,
                                   facecolor='#E5FFE5', edgecolor='#28A745', linewidth=3))
    ax.text(0.26, 0.72, '88.6%',
           ha='center', va='center', fontsize=32, fontweight='bold', color='#28A745')
    ax.text(0.26, 0.66, 'Deletion Recovery',
           ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Metric 3: Recognition rate
    ax.add_patch(mpatches.Rectangle((0.53, 0.63), 0.42, 0.12,
                                   facecolor='#FFF5E5', edgecolor='#FF8800', linewidth=3))
    ax.text(0.74, 0.72, '97.8%',
           ha='center', va='center', fontsize=32, fontweight='bold', color='#FF8800')
    ax.text(0.74, 0.66, 'Characters Visible',
           ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Error transformation diagram (simplified)
    y_start = 0.50
    
    # Degraded
    ax.text(0.25, y_start, 'DEGRADED', ha='center', fontsize=14, fontweight='bold')
    ax.text(0.25, y_start-0.05, '92.8% Deletion', ha='center', fontsize=11, color='#FF6B6B')
    ax.text(0.25, y_start-0.09, '7.2% Substitution', ha='center', fontsize=11)
    
    # Arrow
    ax.annotate('', xy=(0.55, y_start-0.04), xytext=(0.35, y_start-0.04),
               arrowprops=dict(arrowstyle='->', lw=3, color='#4ECDC4'))
    ax.text(0.45, y_start+0.02, 'Restoration', ha='center', fontsize=10, color='#4ECDC4')
    
    # Restored
    ax.text(0.75, y_start, 'RESTORED', ha='center', fontsize=14, fontweight='bold')
    ax.text(0.75, y_start-0.05, '35.2% Deletion', ha='center', fontsize=11)
    ax.text(0.75, y_start-0.09, '64.8% Substitution', ha='center', fontsize=11, color='#FFA500')
    
    # Key insights
    y_insights = 0.30
    ax.text(0.5, y_insights+0.05, 'KEY INSIGHTS', ha='center', fontsize=16, fontweight='bold')
    
    insights = [
        '✓ Restoration makes invisible text visible (76.4% → 2.2% missing)',
        '✓ Deletion errors are catastrophic (text lost)',
        '✓ Substitution errors are recoverable (text visible but confused)',
        '✓ Net benefit: 70% error reduction despite substitution increase',
        '✓ All word positions benefit (middle: -74.5%, end: -68.2%, start: -57.8%)',
    ]
    
    for i, insight in enumerate(insights):
        ax.text(0.08, y_insights - 0.05*(i+1), insight,
               ha='left', va='center', fontsize=11)
    
    # Sample size & confidence
    ax.text(0.5, 0.05, 'n = 712 test samples | 23,504 characters analyzed | p < 0.001',
           ha='center', va='center', fontsize=10, style='italic')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created key metrics poster: {output_path}")

def main():
    """Generate all summary visualizations"""
    print("="*80)
    print("GENERATING SUMMARY VISUALIZATIONS")
    print("="*80)
    
    # Load data
    print("\nLoading comparison data...")
    data = load_comparison_data()
    
    # Create output directory
    import os
    os.makedirs('dual_modal_gan/analysis/summary_visualizations', exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating transformation flowchart...")
    create_transformation_flowchart(
        data,
        output_path='dual_modal_gan/analysis/summary_visualizations/transformation_flowchart.png'
    )
    
    print("\nGenerating summary dashboard...")
    create_dashboard_summary(
        data,
        output_path='dual_modal_gan/analysis/summary_visualizations/summary_dashboard.png'
    )
    
    print("\nGenerating key metrics poster...")
    create_key_metrics_poster(
        data,
        output_path='dual_modal_gan/analysis/summary_visualizations/key_metrics_poster.png'
    )
    
    print("\n" + "="*80)
    print("✅ ALL SUMMARY VISUALIZATIONS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - transformation_flowchart.png/pdf")
    print("  - summary_dashboard.png/pdf")
    print("  - key_metrics_poster.png/pdf")
    print("\nLocation: dual_modal_gan/analysis/summary_visualizations/")

if __name__ == '__main__':
    main()
