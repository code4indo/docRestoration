#!/usr/bin/env python3
"""
Phase 3: Position-Dependent Error Analysis

Analyze how error rates vary by:
1. Position in word (start/middle/end)
2. Normalized position (0-1 scale for continuous analysis)
3. Bigram/trigram context patterns
4. Statistical significance tests

Output:
- Position-dependent error rate curves
- Context pattern analysis
- Statistical test results
- Publication-quality visualizations
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from scipy import stats
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_character_errors(json_path):
    """Load character errors from JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['errors'], data.get('metadata', {})

def analyze_position_dependency(errors):
    """
    Analyze error rates by categorical position (start/middle/end)
    
    Returns:
        position_stats: Dict with error counts and rates per position
    """
    position_counts = Counter(e['word_position'] for e in errors)
    total_errors = len(errors)
    
    # Also count by error type per position
    position_error_types = defaultdict(lambda: defaultdict(int))
    for error in errors:
        pos = error['word_position']
        err_type = error['error_type']
        position_error_types[pos][err_type] += 1
    
    stats_dict = {}
    for pos in ['start', 'middle', 'end']:
        count = position_counts.get(pos, 0)
        stats_dict[pos] = {
            'count': count,
            'percentage': count / total_errors * 100 if total_errors > 0 else 0,
            'error_types': dict(position_error_types[pos])
        }
    
    return stats_dict

def plot_position_distribution(position_stats, output_path='position_distribution.png'):
    """
    Plot bar chart of error distribution by position
    """
    positions = ['start', 'middle', 'end']
    counts = [position_stats[pos]['count'] for pos in positions]
    percentages = [position_stats[pos]['percentage'] for pos in positions]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute counts
    bars1 = ax1.bar(positions, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Error Count', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Word Position', fontsize=12, fontweight='bold')
    ax1.set_title('Error Distribution by Word Position (Absolute)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars1, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count):,}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Percentages
    bars2 = ax2.bar(positions, percentages, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Percentage of Errors (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Word Position', fontsize=12, fontweight='bold')
    ax2.set_title('Error Distribution by Word Position (Relative)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for bar, pct in zip(bars2, percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved position distribution: {output_path}")

def analyze_normalized_position(errors):
    """
    Analyze errors by normalized position (0-1) in word
    Bin into 10 buckets for smooth curve
    """
    # Calculate normalized position for each error
    normalized_positions = []
    for error in errors:
        if error['full_word'] and len(error['full_word']) > 0:
            # Normalize position to [0, 1]
            norm_pos = error['position_in_word'] / max(len(error['full_word']) - 1, 1)
            normalized_positions.append(norm_pos)
    
    # Bin into 10 buckets
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist, _ = np.histogram(normalized_positions, bins=bins)
    
    return bin_centers, hist, normalized_positions

def plot_normalized_position_curve(bin_centers, hist, output_path='position_curve.png'):
    """
    Plot smooth curve of error rate vs normalized position
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot as line with markers
    ax.plot(bin_centers, hist, 'o-', linewidth=2.5, markersize=8, 
            color='#FF6B6B', label='Error Count', markerfacecolor='white', 
            markeredgewidth=2, markeredgecolor='#FF6B6B')
    
    # Fill area under curve
    ax.fill_between(bin_centers, hist, alpha=0.3, color='#FF6B6B')
    
    ax.set_xlabel('Normalized Position in Word (0=Start, 1=End)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Count', fontsize=12, fontweight='bold')
    ax.set_title('Error Distribution by Normalized Word Position', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add vertical lines for reference
    ax.axvline(x=0.33, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0.67, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.165, ax.get_ylim()[1]*0.95, 'Start', ha='center', fontsize=10, alpha=0.7)
    ax.text(0.5, ax.get_ylim()[1]*0.95, 'Middle', ha='center', fontsize=10, alpha=0.7)
    ax.text(0.835, ax.get_ylim()[1]*0.95, 'End', ha='center', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved position curve: {output_path}")

def analyze_context_patterns(errors, top_k=20):
    """
    Analyze bigram and trigram context patterns
    
    Returns:
        bigram_errors: Counter of (char_before, error_char) pairs
        trigram_errors: Counter of (char_before2, char_before1, error_char) triples
    """
    bigram_errors = Counter()
    trigram_errors = Counter()
    
    for error in errors:
        if error['error_type'] != 'substitution':
            continue  # Focus on substitutions for context
        
        gt_char = error['gt_char']
        pred_char = error['pred_char']
        context_before = error['context_before']
        
        # Bigram: (char_before, gt_char) → pred_char
        if len(context_before) >= 1:
            char_before = context_before[-1]
            bigram_key = (char_before, gt_char, pred_char)
            bigram_errors[bigram_key] += 1
        
        # Trigram: (char_before2, char_before1, gt_char) → pred_char
        if len(context_before) >= 2:
            char_before2 = context_before[-2]
            char_before1 = context_before[-1]
            trigram_key = (char_before2, char_before1, gt_char, pred_char)
            trigram_errors[trigram_key] += 1
    
    return bigram_errors, trigram_errors

def plot_top_bigram_contexts(bigram_errors, output_path='bigram_contexts.png', top_k=15):
    """
    Plot top bigram context patterns
    """
    top_bigrams = bigram_errors.most_common(top_k)
    
    # Format labels
    labels = [f"{b[0]}{b[1]}→{b[2]}" for b, _ in top_bigrams]
    counts = [count for _, count in top_bigrams]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(range(len(labels)), counts, color='#4ECDC4', alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bigram Pattern (Context + Error)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top-{top_k} Bigram Context Error Patterns', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width + max(counts)*0.01, i, f'{int(count)}',
                va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved bigram contexts: {output_path}")

def statistical_tests(position_stats):
    """
    Perform statistical tests on position-dependent errors
    
    Tests:
    1. Chi-square test: Are errors uniformly distributed across positions?
    2. Effect size: Cramer's V
    """
    positions = ['start', 'middle', 'end']
    observed = [position_stats[pos]['count'] for pos in positions]
    
    # Chi-square test for uniform distribution
    total = sum(observed)
    expected = [total / 3] * 3  # Uniform expectation
    
    chi2, p_value = stats.chisquare(observed, expected)
    
    # Cramer's V (effect size)
    cramers_v = np.sqrt(chi2 / total)
    
    results = {
        'chi_square': {
            'statistic': float(chi2),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),  # Convert to Python bool
            'interpretation': 'Non-uniform distribution' if p_value < 0.05 else 'Uniform distribution'
        },
        'effect_size': {
            'cramers_v': float(cramers_v),
            'interpretation': 'Large' if cramers_v > 0.5 else ('Medium' if cramers_v > 0.3 else 'Small')
        }
    }
    
    return results

def main(errors_json_path, output_dir='dual_modal_gan/analysis/position_analysis'):
    """
    Main Phase 3 analysis pipeline
    """
    print("="*80)
    print("PHASE 3: POSITION-DEPENDENT ERROR ANALYSIS")
    print("="*80)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load errors
    print(f"\nLoading errors from: {errors_json_path}")
    errors, metadata = load_character_errors(errors_json_path)
    print(f"Total errors loaded: {len(errors):,}")
    
    # 1. Categorical position analysis
    print("\n" + "="*80)
    print("1. CATEGORICAL POSITION ANALYSIS (Start/Middle/End)")
    print("="*80)
    
    position_stats = analyze_position_dependency(errors)
    
    print("\nPosition Distribution:")
    for pos in ['start', 'middle', 'end']:
        stats = position_stats[pos]
        print(f"  {pos.title():8s}: {stats['count']:6,} ({stats['percentage']:5.1f}%)")
        for err_type, count in stats['error_types'].items():
            print(f"    - {err_type:15s}: {count:6,}")
    
    # Plot position distribution
    plot_position_distribution(
        position_stats,
        output_path=f"{output_dir}/position_distribution.png"
    )
    
    # 2. Normalized position analysis
    print("\n" + "="*80)
    print("2. NORMALIZED POSITION ANALYSIS (0-1 Scale)")
    print("="*80)
    
    bin_centers, hist, norm_positions = analyze_normalized_position(errors)
    
    print(f"\nAnalyzed {len(norm_positions):,} errors with position data")
    print(f"Position range: [{min(norm_positions):.2f}, {max(norm_positions):.2f}]")
    print(f"Distribution across 10 bins:")
    for i, (center, count) in enumerate(zip(bin_centers, hist)):
        print(f"  Bin {i+1} (pos={center:.2f}): {int(count):6,} errors")
    
    # Plot normalized position curve
    plot_normalized_position_curve(
        bin_centers,
        hist,
        output_path=f"{output_dir}/position_curve.png"
    )
    
    # 3. Context pattern analysis
    print("\n" + "="*80)
    print("3. CONTEXT PATTERN ANALYSIS (Bigrams/Trigrams)")
    print("="*80)
    
    bigram_errors, trigram_errors = analyze_context_patterns(errors)
    
    print(f"\nBigram patterns found: {len(bigram_errors):,}")
    print(f"Trigram patterns found: {len(trigram_errors):,}")
    
    print(f"\nTop-10 Bigram Context Patterns:")
    for i, ((char_before, gt_char, pred_char), count) in enumerate(bigram_errors.most_common(10), 1):
        print(f"  {i:2d}. '{char_before}{gt_char}' → '{char_before}{pred_char}': {count:3d} times")
    
    # Plot bigram contexts
    plot_top_bigram_contexts(
        bigram_errors,
        output_path=f"{output_dir}/bigram_contexts.png",
        top_k=15
    )
    
    # 4. Statistical tests
    print("\n" + "="*80)
    print("4. STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    
    stat_results = statistical_tests(position_stats)
    
    print(f"\nChi-Square Test (Uniform Distribution):")
    print(f"  χ² = {stat_results['chi_square']['statistic']:.2f}")
    print(f"  p-value = {stat_results['chi_square']['p_value']:.4e}")
    print(f"  Significant: {stat_results['chi_square']['significant']}")
    print(f"  Interpretation: {stat_results['chi_square']['interpretation']}")
    
    print(f"\nEffect Size (Cramér's V):")
    print(f"  V = {stat_results['effect_size']['cramers_v']:.3f}")
    print(f"  Interpretation: {stat_results['effect_size']['interpretation']} effect")
    
    # Save results
    results = {
        'position_stats': position_stats,
        'normalized_position': {
            'bin_centers': bin_centers.tolist(),
            'histogram': hist.tolist()
        },
        'bigram_patterns': {
            f"{b[0]}{b[1]}→{b[2]}": count 
            for (b, count) in bigram_errors.most_common(20)
        },
        'statistical_tests': stat_results
    }
    
    results_path = f"{output_dir}/position_analysis_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved results: {results_path}")
    
    # Summary
    print("\n" + "="*80)
    print("✅ PHASE 3 COMPLETE!")
    print("="*80)
    print(f"\nOutput files in: {output_dir}/")
    print(f"  - position_distribution.png (categorical)")
    print(f"  - position_curve.png (normalized)")
    print(f"  - bigram_contexts.png (context patterns)")
    print(f"  - position_analysis_results.json (all data)")
    print(f"\nAll figures available in PNG + PDF format")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        errors_json = sys.argv[1]
    else:
        errors_json = 'dual_modal_gan/analysis/character_errors_full.json'
    
    main(errors_json)
