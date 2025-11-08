"""
H2A Statistical Analysis Script V2 - Dual-Modal vs Single-Modal Comparison

Improved version yang menerima pre-extracted CER data dari JSON files.

Statistical Framework:
1. Load per-sample CER dari JSON files (hasil h2a_evaluate_model.py)
2. Calculate descriptive statistics
3. Paired t-test: H0: μ_dual = μ_single vs H1: μ_dual < μ_single
4. Effect size: Cohen's d > 0.5 (medium effect)
5. Power analysis: ≥80% statistical power

Author: Claude Code (AI/ML Engineer)
Purpose: H2A Hypothesis Validation - Statistical Proof
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_cer_from_json(json_path):
    """
    Load per-sample CER data dari JSON file (hasil h2a_evaluate_model.py).
    
    Args:
        json_path: Path ke JSON file dengan per-sample CER
    
    Returns:
        tuple: (cer_values_array, metadata_dict)
    """
    print(f"📂 Loading CER data from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cer_values = np.array(data['per_sample_cer'])
    metadata = data['metadata']
    statistics = data['statistics']
    
    print(f"   ✅ Loaded {len(cer_values)} CER values")
    print(f"   📊 Mean: {statistics['mean']:.4f} ± {statistics['std']:.4f}")
    print(f"   📊 Discriminator: {metadata['discriminator_type']}")
    
    return cer_values, metadata, statistics


def calculate_cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size.
    
    Args:
        group1, group2: Two groups of measurements (arrays)
    
    Returns:
        float: Cohen's d value
    """
    # Pooled standard deviation
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    
    # Cohen's d (negative = group1 < group2)
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return d


def interpret_effect_size(d):
    """
    Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
    
    Returns:
        str: Effect size interpretation
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def estimate_statistical_power(n, effect_size, alpha=0.05):
    """
    Estimate statistical power untuk paired t-test.
    
    Simplified estimation based on sample size and effect size.
    
    Args:
        n: Sample size
        effect_size: Cohen's d
        alpha: Significance level
    
    Returns:
        float: Estimated power
    """
    # Simplified power estimation
    # For paired t-test, power increases with n and |effect_size|
    # This is a rough approximation
    
    if n < 30:
        base_power = 0.3
    elif n < 100:
        base_power = 0.5
    elif n < 300:
        base_power = 0.65
    else:
        base_power = 0.75
    
    # Adjust by effect size
    abs_d = abs(effect_size)
    if abs_d < 0.2:
        power = base_power * 0.5
    elif abs_d < 0.5:
        power = base_power * 0.75
    elif abs_d < 0.8:
        power = base_power * 0.95
    else:
        power = min(0.99, base_power * 1.1)
    
    return power


def perform_statistical_analysis(dual_cer, single_cer):
    """
    Perform comprehensive statistical analysis untuk H2A experiment.
    
    Args:
        dual_cer: CER values dari dual-modal discriminator
        single_cer: CER values dari single-modal discriminator
    
    Returns:
        dict: Statistical analysis results
    """
    
    print(f"\n{'='*80}")
    print(f"🔬 STATISTICAL ANALYSIS - H2A HYPOTHESIS")
    print(f"{'='*80}\n")
    
    # Descriptive statistics
    dual_mean = np.mean(dual_cer)
    dual_std = np.std(dual_cer, ddof=1)
    dual_median = np.median(dual_cer)
    
    single_mean = np.mean(single_cer)
    single_std = np.std(single_cer, ddof=1)
    single_median = np.median(single_cer)
    
    print(f"📊 DESCRIPTIVE STATISTICS")
    print(f"{'─'*80}")
    print(f"   Dual-Modal (Treatment Group):")
    print(f"      Mean CER:   {dual_mean:.4f} ± {dual_std:.4f}")
    print(f"      Median CER: {dual_median:.4f}")
    print(f"      Range:      [{np.min(dual_cer):.4f}, {np.max(dual_cer):.4f}]")
    print(f"      n =         {len(dual_cer)}")
    
    print(f"\n   Single-Modal (Control Group):")
    print(f"      Mean CER:   {single_mean:.4f} ± {single_std:.4f}")
    print(f"      Median CER: {single_median:.4f}")
    print(f"      Range:      [{np.min(single_cer):.4f}, {np.max(single_cer):.4f}]")
    print(f"      n =         {len(single_cer)}")
    
    # Check if samples are paired
    n_dual = len(dual_cer)
    n_single = len(single_cer)
    is_paired = (n_dual == n_single)
    
    print(f"\n   Sample Configuration:")
    print(f"      Paired Design: {'✅ YES' if is_paired else '❌ NO'}")
    if not is_paired:
        print(f"      ⚠️  Sample sizes differ: n_dual={n_dual}, n_single={n_single}")
        print(f"      ⚠️  Using minimum: n={min(n_dual, n_single)}")
        # Trim to same size
        min_n = min(n_dual, n_single)
        dual_cer = dual_cer[:min_n]
        single_cer = single_cer[:min_n]
    
    # Effect size
    effect_size = calculate_cohens_d(dual_cer, single_cer)
    effect_interpretation = interpret_effect_size(effect_size)
    
    print(f"\n{'='*80}")
    print(f"📏 EFFECT SIZE ANALYSIS")
    print(f"{'─'*80}")
    print(f"   Cohen's d:      {effect_size:.4f}")
    print(f"   Interpretation: {effect_interpretation} effect")
    print(f"   Direction:      {'Dual-Modal BETTER (lower CER) ✅' if effect_size < 0 else 'Single-Modal BETTER ❌'}")
    print(f"   Target:         |d| > 0.5 (medium effect)")
    
    if abs(effect_size) > 0.5:
        print(f"   Status:         ✅ ACHIEVED (|d|={abs(effect_size):.4f} > 0.5)")
    else:
        print(f"   Status:         ❌ NOT ACHIEVED (|d|={abs(effect_size):.4f} < 0.5)")
    
    # Statistical significance test
    print(f"\n{'='*80}")
    print(f"🔬 STATISTICAL SIGNIFICANCE TEST")
    print(f"{'─'*80}")
    print(f"   Hypotheses:")
    print(f"      H0: μ_dual = μ_single (no difference)")
    print(f"      H1: μ_dual < μ_single (dual-modal better)")
    
    # Paired t-test (one-tailed)
    t_stat, p_value_two_tailed = ttest_rel(dual_cer, single_cer)
    p_value = p_value_two_tailed / 2  # One-tailed test
    
    # Adjust p-value direction (we expect dual < single, so t should be negative)
    if t_stat > 0:
        p_value = 1 - p_value
    
    test_type = "Paired t-test (one-tailed)"
    print(f"\n   Test Type:      {test_type}")
    print(f"   t-statistic:    {t_stat:.4f}")
    print(f"   p-value:        {p_value:.6f}")
    
    # Significance level
    alpha = 0.05
    is_significant = p_value < alpha
    
    print(f"   Alpha level:    {alpha}")
    print(f"   Significant:    {'✅ YES (p < α)' if is_significant else '❌ NO (p ≥ α)'}")
    
    # Power analysis
    power_estimate = estimate_statistical_power(len(dual_cer), effect_size, alpha)
    print(f"\n   Statistical Power:")
    print(f"      Estimated:   {power_estimate:.2f}")
    print(f"      Target:      ≥0.80")
    print(f"      Status:      {'✅ ADEQUATE' if power_estimate >= 0.80 else '⚠️  LOW POWER'}")
    
    # Practical significance
    cer_reduction = single_mean - dual_mean
    cer_reduction_percent = (cer_reduction / single_mean) * 100
    
    print(f"\n{'='*80}")
    print(f"🎯 PRACTICAL SIGNIFICANCE")
    print(f"{'─'*80}")
    print(f"   Absolute CER Reduction: {cer_reduction:.4f}")
    print(f"   Relative CER Reduction: {cer_reduction_percent:.2f}%")
    
    if cer_reduction > 0:
        print(f"   Interpretation:         Dual-modal reduces CER by {cer_reduction_percent:.2f}% ✅")
    else:
        print(f"   Interpretation:         Dual-modal INCREASES CER (worse!) ❌")
    
    # Hypothesis conclusion
    print(f"\n{'='*80}")
    print(f"🏁 HYPOTHESIS CONCLUSION")
    print(f"{'='*80}")
    
    # Criteria untuk H2A support:
    # 1. Statistical significance (p < 0.05)
    # 2. Effect size medium or large (|d| > 0.5)
    # 3. Direction correct (dual < single)
    
    all_criteria_met = (
        is_significant and 
        abs(effect_size) > 0.5 and 
        effect_size < 0  # dual < single
    )
    
    if all_criteria_met:
        print(f"   ✅ H2A: FULLY SUPPORTED")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   ✓ Statistical significance: p = {p_value:.6f} < {alpha}")
        print(f"   ✓ Effect size: {effect_interpretation} (d = {effect_size:.4f})")
        print(f"   ✓ Direction: Dual-modal has LOWER CER ✅")
        print(f"   ✓ Practical impact: {cer_reduction_percent:.2f}% CER reduction")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"\n   🎯 CONCLUSION:")
        print(f"      Dual-Modal discriminator (CNN+LSTM) significantly")
        print(f"      outperforms Single-Modal (CNN-only) for HTR tasks.")
        print(f"      The effect is {effect_interpretation} and practically meaningful.")
        
    elif is_significant:
        print(f"   ⚠️  H2A: PARTIALLY SUPPORTED")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   ✓ Statistical significance: p = {p_value:.6f} < {alpha}")
        
        if abs(effect_size) <= 0.5:
            print(f"   ✗ Effect size: {effect_interpretation} (d = {effect_size:.4f}) - BELOW MEDIUM THRESHOLD")
        
        if effect_size >= 0:
            print(f"   ✗ Direction: Wrong direction (dual-modal NOT better)")
        
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
    else:
        print(f"   ❌ H2A: NOT SUPPORTED")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   ✗ No statistical significance: p = {p_value:.6f} ≥ {alpha}")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"\n   🔍 INTERPRETATION:")
        print(f"      Insufficient evidence to conclude that dual-modal")
        print(f"      discriminator is superior to single-modal.")
    
    print(f"\n{'='*80}\n")
    
    return {
        'descriptive_stats': {
            'dual_modal': {
                'mean': float(dual_mean),
                'std': float(dual_std),
                'median': float(dual_median),
                'min': float(np.min(dual_cer)),
                'max': float(np.max(dual_cer)),
                'n': int(len(dual_cer))
            },
            'single_modal': {
                'mean': float(single_mean),
                'std': float(single_std),
                'median': float(single_median),
                'min': float(np.min(single_cer)),
                'max': float(np.max(single_cer)),
                'n': int(len(single_cer))
            }
        },
        'effect_size': {
            'cohens_d': float(effect_size),
            'interpretation': effect_interpretation,
            'target_met': abs(effect_size) > 0.5,
            'direction': 'dual_better' if effect_size < 0 else 'single_better'
        },
        'statistical_test': {
            'test_type': test_type,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'p_value_two_tailed': float(p_value_two_tailed),
            'alpha': alpha,
            'significant': bool(is_significant)
        },
        'practical_significance': {
            'cer_reduction': float(cer_reduction),
            'cer_reduction_percent': float(cer_reduction_percent)
        },
        'power_analysis': {
            'estimated_power': float(power_estimate),
            'adequate': bool(power_estimate >= 0.80)
        },
        'hypothesis_conclusion': {
            'fully_supported': bool(all_criteria_met),
            'partially_supported': bool(is_significant and not all_criteria_met),
            'not_supported': bool(not is_significant)
        }
    }


def create_visualization(dual_cer, single_cer, results, output_dir):
    """
    Create comprehensive visualization untuk statistical analysis.
    
    Args:
        dual_cer, single_cer: CER values arrays
        results: Statistical analysis results dict
        output_dir: Directory untuk save plots
    """
    print(f"📊 Creating visualizations...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('H2A Experiment: Dual-Modal vs Single-Modal Discriminator Comparison',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Distribution comparison (histogram)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(dual_cer, bins=40, alpha=0.6, label='Dual-Modal', color='#2E86AB', density=True, edgecolor='black')
    ax1.hist(single_cer, bins=40, alpha=0.6, label='Single-Modal', color='#A23B72', density=True, edgecolor='black')
    ax1.axvline(np.mean(dual_cer), color='#2E86AB', linestyle='--', linewidth=2, label=f'Dual Mean: {np.mean(dual_cer):.4f}')
    ax1.axvline(np.mean(single_cer), color='#A23B72', linestyle='--', linewidth=2, label=f'Single Mean: {np.mean(single_cer):.4f}')
    ax1.set_xlabel('Character Error Rate (CER)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('CER Distribution Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax2 = fig.add_subplot(gs[0, 2])
    box_data = [dual_cer, single_cer]
    bp = ax2.boxplot(box_data, labels=['Dual-Modal', 'Single-Modal'],
                     patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('#2E86AB')
    bp['boxes'][1].set_facecolor('#A23B72')
    ax2.set_ylabel('CER', fontsize=11)
    ax2.set_title('CER Distribution\n(Box Plot)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Q-Q plots untuk normality check
    from scipy.stats import probplot
    
    ax3 = fig.add_subplot(gs[1, 0])
    probplot(dual_cer, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot: Dual-Modal\n(Normality Check)', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    probplot(single_cer, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot: Single-Modal\n(Normality Check)', fontsize=10, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 4. Effect size visualization
    ax5 = fig.add_subplot(gs[1, 2])
    effect_sizes = ['Small\n(0.2)', 'Medium\n(0.5)', 'Large\n(0.8)', 'Observed']
    effect_values = [0.2, 0.5, 0.8, abs(results['effect_size']['cohens_d'])]
    colors = ['#CCCCCC', '#CCCCCC', '#CCCCCC', 
              '#2E86AB' if results['effect_size']['target_met'] else '#F18F01']
    
    bars = ax5.bar(range(len(effect_sizes)), effect_values, color=colors, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel("Cohen's d (absolute)", fontsize=11)
    ax5.set_title('Effect Size Analysis', fontsize=11, fontweight='bold')
    ax5.set_xticks(range(len(effect_sizes)))
    ax5.set_xticklabels(effect_sizes, fontsize=9)
    ax5.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Medium Threshold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, effect_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. Summary statistics table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Create summary table
    summary_data = [
        ['Metric', 'Dual-Modal', 'Single-Modal', 'Difference'],
        ['─'*20, '─'*15, '─'*15, '─'*15],
        ['Mean CER', 
         f"{results['descriptive_stats']['dual_modal']['mean']:.4f}",
         f"{results['descriptive_stats']['single_modal']['mean']:.4f}",
         f"{results['practical_significance']['cer_reduction']:.4f}"],
        ['Std Dev', 
         f"{results['descriptive_stats']['dual_modal']['std']:.4f}",
         f"{results['descriptive_stats']['single_modal']['std']:.4f}",
         '─'],
        ['Median CER',
         f"{results['descriptive_stats']['dual_modal']['median']:.4f}",
         f"{results['descriptive_stats']['single_modal']['median']:.4f}",
         '─'],
        ['Sample Size',
         f"{results['descriptive_stats']['dual_modal']['n']}",
         f"{results['descriptive_stats']['single_modal']['n']}",
         '─'],
        ['─'*20, '─'*15, '─'*15, '─'*15],
        ['Statistical Test', f"t = {results['statistical_test']['t_statistic']:.4f}", 
         f"p = {results['statistical_test']['p_value']:.6f}",
         f"Sig: {'YES ✅' if results['statistical_test']['significant'] else 'NO ❌'}"],
        ['Effect Size', 
         f"Cohen's d = {results['effect_size']['cohens_d']:.4f}",
         f"({results['effect_size']['interpretation']})",
         f"Target met: {'YES ✅' if results['effect_size']['target_met'] else 'NO ❌'}"],
        ['CER Reduction',
         f"{results['practical_significance']['cer_reduction_percent']:.2f}%",
         '',
         ''],
        ['─'*20, '─'*15, '─'*15, '─'*15],
        ['H2A Hypothesis',
         'FULLY SUPPORTED ✅' if results['hypothesis_conclusion']['fully_supported'] 
         else ('PARTIALLY SUPPORTED ⚠️' if results['hypothesis_conclusion']['partially_supported']
               else 'NOT SUPPORTED ❌'),
         '', '']
    ]
    
    table = ax6.table(cellText=summary_data, cellLoc='left', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style conclusion row
    conclusion_row = len(summary_data) - 1
    for i in range(4):
        if results['hypothesis_conclusion']['fully_supported']:
            table[(conclusion_row, i)].set_facecolor('#90EE90')
        elif results['hypothesis_conclusion']['partially_supported']:
            table[(conclusion_row, i)].set_facecolor('#FFD700')
        else:
            table[(conclusion_row, i)].set_facecolor('#FFB6C6')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'h2a_statistical_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Visualization saved: {plot_path}")


def save_analysis_report(results, dual_metadata, single_metadata, output_dir):
    """
    Save detailed analysis report to JSON.
    
    Args:
        results: Statistical analysis results
        dual_metadata: Metadata dari dual-modal evaluation
        single_metadata: Metadata dari single-modal evaluation
        output_dir: Directory untuk save report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f'h2a_analysis_report_{timestamp}.json')
    
    # Combine all results
    full_report = {
        'metadata': {
            'experiment': 'H2A Hypothesis Validation',
            'comparison': 'Dual-Modal vs Single-Modal Discriminator',
            'metric': 'Character Error Rate (CER)',
            'hypothesis': 'Dual-Modal (CNN+LSTM) menghasilkan CER lebih rendah dibanding Single-Modal (CNN-only)',
            'analysis_date': timestamp,
            'dual_modal_checkpoint': dual_metadata['checkpoint_dir'],
            'single_modal_checkpoint': single_metadata['checkpoint_dir'],
            'statistical_framework': {
                'test_type': 'Paired t-test (one-tailed)',
                'alpha_level': 0.05,
                'effect_size_target': "Cohen's d > 0.5 (medium effect)",
                'power_target': '≥80%'
            }
        },
        'results': results
    }
    
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2)
    
    print(f"   ✅ Analysis report saved: {report_path}")
    
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description='H2A Statistical Analysis V2: Dual-Modal vs Single-Modal Comparison'
    )
    parser.add_argument('--dual_modal_cer', type=str, required=True,
                       help='Path to dual-modal CER JSON file (dari h2a_evaluate_model.py)')
    parser.add_argument('--single_modal_cer', type=str, required=True,
                       help='Path to single-modal CER JSON file (dari h2a_evaluate_model.py)')
    parser.add_argument('--output_dir', type=str, default='./h2a_results',
                       help='Directory untuk save analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"🚀 H2A STATISTICAL ANALYSIS V2 - HYPOTHESIS VALIDATION")
    print(f"{'='*80}")
    print(f"Experiment: Dual-Modal vs Single-Modal Discriminator")
    print(f"Hypothesis: Dual-Modal → Lower CER (Better Text Readability)")
    print(f"{'='*80}\n")
    
    try:
        # Load CER data
        dual_cer, dual_metadata, dual_stats = load_cer_from_json(args.dual_modal_cer)
        print()
        single_cer, single_metadata, single_stats = load_cer_from_json(args.single_modal_cer)
        
        # Ensure same sample size (paired design)
        min_samples = min(len(dual_cer), len(single_cer))
        if len(dual_cer) != len(single_cer):
            print(f"\n⚠️  WARNING: Sample sizes differ!")
            print(f"   Dual-Modal:   {len(dual_cer)} samples")
            print(f"   Single-Modal: {len(single_cer)} samples")
            print(f"   Using first {min_samples} samples from each for paired analysis")
            dual_cer = dual_cer[:min_samples]
            single_cer = single_cer[:min_samples]
        
        print(f"\n✅ FINAL SAMPLE SIZE: n={min_samples} per group (paired design)")
        
        # Perform statistical analysis
        results = perform_statistical_analysis(dual_cer, single_cer)
        
        # Create visualization
        create_visualization(dual_cer, single_cer, results, args.output_dir)
        
        # Save analysis report
        report_path = save_analysis_report(results, dual_metadata, single_metadata, args.output_dir)
        
        # Print final conclusion
        print(f"\n{'='*80}")
        print(f"🏆 FINAL CONCLUSION")
        print(f"{'='*80}")
        
        if results['hypothesis_conclusion']['fully_supported']:
            print(f"   ✅ H2A HYPOTHESIS: FULLY SUPPORTED")
            print(f"   🎯 Dual-Modal discriminator significantly outperforms Single-Modal")
            print(f"   📈 Effect size: {results['effect_size']['interpretation']} (d={results['effect_size']['cohens_d']:.4f})")
            print(f"   📊 Statistical significance: p={results['statistical_test']['p_value']:.6f} < 0.05")
            print(f"   💡 Practical impact: {results['practical_significance']['cer_reduction_percent']:.2f}% CER reduction")
        elif results['hypothesis_conclusion']['partially_supported']:
            print(f"   ⚠️  H2A HYPOTHESIS: PARTIALLY SUPPORTED")
            print(f"   📊 Statistically significant but effect size below target")
        else:
            print(f"   ❌ H2A HYPOTHESIS: NOT SUPPORTED")
            print(f"   🔍 Insufficient evidence untuk conclude dual-modal superiority")
        
        print(f"{'='*80}\n")
        
        print(f"✅ Analysis completed successfully!")
        print(f"📁 Results saved in: {args.output_dir}")
        print(f"📊 Report: {report_path}")
        print(f"📈 Visualization: {os.path.join(args.output_dir, 'h2a_statistical_analysis.png')}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
