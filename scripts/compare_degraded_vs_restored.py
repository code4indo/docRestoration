#!/usr/bin/env python3
"""
Compare Character-Level Errors: Degraded vs Restored

Analyze how restoration changes error patterns:
- Error type distribution (deletion, substitution, insertion)
- Character confusion patterns
- Position-dependent changes
- Overall improvement metrics

Output: Comprehensive comparison showing restoration impact
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

sys.path.append('/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/scripts')
from extract_character_errors_simple import extract_character_errors_from_text_pairs

def main():
    print("="*80)
    print("DEGRADED VS RESTORED: CHARACTER-LEVEL COMPARISON")
    print("="*80)
    
    # Load restored predictions
    print("\nLoading restored predictions...")
    with open('dual_modal_gan/analysis/test_predictions_restored.json', 'r') as f:
        data = json.load(f)
    
    predictions = data['predictions']
    print(f"Loaded {len(predictions)} samples")
    
    # Extract GT and predictions
    gt_texts = [p['gt_text'] for p in predictions]
    pred_degraded = [p['pred_text_degraded'] for p in predictions]
    pred_restored = [p['pred_text_restored'] for p in predictions]
    
    # Extract character-level errors for DEGRADED
    print("\n" + "="*80)
    print("ANALYZING DEGRADED PREDICTIONS...")
    print("="*80)
    
    errors_degraded = extract_character_errors_from_text_pairs(
        gt_texts,
        pred_degraded,
        output_path='dual_modal_gan/analysis/character_errors_degraded.json'
    )
    
    # Extract character-level errors for RESTORED
    print("\n" + "="*80)
    print("ANALYZING RESTORED PREDICTIONS...")
    print("="*80)
    
    errors_restored = extract_character_errors_from_text_pairs(
        gt_texts,
        pred_restored,
        output_path='dual_modal_gan/analysis/character_errors_restored.json'
    )
    
    # Comparison Analysis
    print("\n" + "="*80)
    print("COMPARISON ANALYSIS")
    print("="*80)
    
    # Error type distribution
    degraded_types = Counter(e['error_type'] for e in errors_degraded)
    restored_types = Counter(e['error_type'] for e in errors_restored)
    
    print("\nError Type Distribution:")
    print(f"\n{'Type':<15} {'Degraded':<12} {'Restored':<12} {'Improvement':<15}")
    print("-"*60)
    
    for err_type in ['deletion', 'substitution', 'insertion']:
        deg_count = degraded_types.get(err_type, 0)
        res_count = restored_types.get(err_type, 0)
        improvement = deg_count - res_count
        improvement_pct = (improvement / deg_count * 100) if deg_count > 0 else 0
        
        print(f"{err_type:<15} {deg_count:>6,} ({deg_count/len(errors_degraded)*100:4.1f}%)  "
              f"{res_count:>6,} ({res_count/len(errors_restored)*100:4.1f}%)  "
              f"{improvement:>6,} ({improvement_pct:+5.1f}%)")
    
    print(f"\n{'TOTAL':<15} {len(errors_degraded):>6,}       {len(errors_restored):>6,}       "
          f"{len(errors_degraded) - len(errors_restored):>6,} "
          f"({(len(errors_degraded) - len(errors_restored))/len(errors_degraded)*100:+5.1f}%)")
    
    # Position distribution
    degraded_positions = Counter(e['word_position'] for e in errors_degraded)
    restored_positions = Counter(e['word_position'] for e in errors_restored)
    
    print("\n" + "="*80)
    print("Position Distribution:")
    print(f"\n{'Position':<15} {'Degraded':<12} {'Restored':<12} {'Improvement':<15}")
    print("-"*60)
    
    for pos in ['start', 'middle', 'end']:
        deg_count = degraded_positions.get(pos, 0)
        res_count = restored_positions.get(pos, 0)
        improvement = deg_count - res_count
        
        print(f"{pos:<15} {deg_count:>6,} ({deg_count/len(errors_degraded)*100:4.1f}%)  "
              f"{res_count:>6,} ({res_count/len(errors_restored)*100:4.1f}%)  "
              f"{improvement:>6,}")
    
    # Visualize comparison
    create_comparison_visualizations(
        errors_degraded, errors_restored,
        output_dir='dual_modal_gan/analysis/degraded_vs_restored'
    )
    
    # Save summary
    summary = {
        'total_samples': len(predictions),
        'total_errors': {
            'degraded': len(errors_degraded),
            'restored': len(errors_restored),
            'improvement': len(errors_degraded) - len(errors_restored),
            'improvement_pct': (len(errors_degraded) - len(errors_restored)) / len(errors_degraded) * 100
        },
        'by_error_type': {
            err_type: {
                'degraded': degraded_types.get(err_type, 0),
                'restored': restored_types.get(err_type, 0),
                'improvement': degraded_types.get(err_type, 0) - restored_types.get(err_type, 0)
            }
            for err_type in ['deletion', 'substitution', 'insertion']
        },
        'by_position': {
            pos: {
                'degraded': degraded_positions.get(pos, 0),
                'restored': restored_positions.get(pos, 0),
                'improvement': degraded_positions.get(pos, 0) - restored_positions.get(pos, 0)
            }
            for pos in ['start', 'middle', 'end']
        }
    }
    
    with open('dual_modal_gan/analysis/degraded_vs_restored_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE!")
    print("="*80)
    print("\nOutput files:")
    print("  - character_errors_degraded.json")
    print("  - character_errors_restored.json")
    print("  - degraded_vs_restored_summary.json")
    print("  - degraded_vs_restored/ (visualizations)")

def create_comparison_visualizations(errors_degraded, errors_restored, output_dir):
    """Create comparison visualizations"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Error type comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    error_types = ['deletion', 'substitution']
    degraded_counts = [
        sum(1 for e in errors_degraded if e['error_type'] == et)
        for et in error_types
    ]
    restored_counts = [
        sum(1 for e in errors_restored if e['error_type'] == et)
        for et in error_types
    ]
    
    x = np.arange(len(error_types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, degraded_counts, width, label='Degraded', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, restored_counts, width, label='Restored', color='#4ECDC4', alpha=0.8)
    
    ax.set_ylabel('Error Count', fontsize=12, fontweight='bold')
    ax.set_title('Error Type Distribution: Degraded vs Restored', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([et.title() for et in error_types])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_type_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/error_type_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved: {output_dir}/error_type_comparison.png")
    
    # 2. Overall improvement chart
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Total Errors']
    degraded_total = [len(errors_degraded)]
    restored_total = [len(errors_restored)]
    improvement = [len(errors_degraded) - len(errors_restored)]
    
    x = np.arange(len(categories))
    width = 0.25
    
    bars1 = ax.bar(x - width, degraded_total, width, label='Degraded', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x, restored_total, width, label='Restored', color='#4ECDC4', alpha=0.8)
    bars3 = ax.bar(x + width, improvement, width, label='Improvement', color='#95E1D3', alpha=0.8)
    
    ax.set_ylabel('Error Count', fontsize=12, fontweight='bold')
    ax.set_title('Overall Error Reduction through Restoration', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add percentage improvement annotation
    improvement_pct = (improvement[0] / degraded_total[0]) * 100
    ax.text(0, max(degraded_total[0], restored_total[0]) * 0.9,
           f'{improvement_pct:.1f}% Reduction',
           ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/overall_improvement.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/overall_improvement.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_dir}/overall_improvement.png")

if __name__ == '__main__':
    main()
