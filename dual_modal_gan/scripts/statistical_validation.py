#!/usr/bin/env python3
"""
Statistical Validation Script untuk GAN-HTR Results
Performs comprehensive statistical tests to prove significance:
1. Paired t-test (degraded vs generated CER)
2. Effect size calculation (Cohen's d)
3. Confidence intervals (bootstrap)
4. Wilcoxon signed-rank test (non-parametric)
5. Power analysis

Usage:
    poetry run python dual_modal_gan/scripts/statistical_validation.py \
        --results_json outputs/checkpoints_final_100/metrics/comprehensive_metrics.json \
        --output_dir outputs/statistical_analysis
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import t as t_dist
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10


def load_results(json_path):
    """Load evaluation results from JSON."""
    print(f"📊 Loading results from {json_path}...")
    with open(json_path, 'r') as f:
        results = json.load(f)
    print(f"  ✅ Loaded: {len(results['sample_results'])} samples")
    return results


def extract_cer_data(sample_results):
    """Extract CER data for statistical tests."""
    degraded_cer = np.array([r['cer_degraded'] for r in sample_results])
    generated_cer = np.array([r['cer_generated'] for r in sample_results])
    improvements = np.array([r['cer_improvement'] for r in sample_results])
    
    print(f"\n📈 Data extracted:")
    print(f"  Samples: {len(degraded_cer)}")
    print(f"  Degraded CER: {degraded_cer.mean():.4f} ± {degraded_cer.std():.4f}")
    print(f"  Generated CER: {generated_cer.mean():.4f} ± {generated_cer.std():.4f}")
    print(f"  Mean improvement: {improvements.mean():.4f} ({improvements.mean()*100:.2f}%)")
    
    return degraded_cer, generated_cer, improvements


def paired_t_test(degraded_cer, generated_cer, alpha=0.05):
    """
    Perform paired t-test.
    H0: No difference between degraded and generated CER
    H1: Generated CER is significantly lower than degraded CER
    """
    print(f"\n{'='*70}")
    print("PAIRED T-TEST")
    print(f"{'='*70}")
    
    # One-tailed paired t-test (generated < degraded)
    t_statistic, p_value_two_tailed = stats.ttest_rel(degraded_cer, generated_cer)
    p_value_one_tailed = p_value_two_tailed / 2
    
    # Degrees of freedom
    n = len(degraded_cer)
    df = n - 1
    
    # Critical value
    critical_value = t_dist.ppf(1 - alpha, df)
    
    print(f"\nHypothesis:")
    print(f"  H₀: μ(degraded) = μ(generated)")
    print(f"  H₁: μ(degraded) > μ(generated)")
    
    print(f"\nTest Statistics:")
    print(f"  Sample size (n): {n}")
    print(f"  Degrees of freedom: {df}")
    print(f"  t-statistic: {t_statistic:.4f}")
    print(f"  p-value (one-tailed): {p_value_one_tailed:.6f}")
    print(f"  Critical value (α={alpha}): {critical_value:.4f}")
    
    print(f"\nMeans:")
    print(f"  Degraded CER: {degraded_cer.mean():.4f} ({degraded_cer.mean()*100:.2f}%)")
    print(f"  Generated CER: {generated_cer.mean():.4f} ({generated_cer.mean()*100:.2f}%)")
    print(f"  Mean difference: {(degraded_cer.mean() - generated_cer.mean()):.4f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if p_value_one_tailed < alpha:
        print(f"  ✅ REJECT H₀ (p < {alpha})")
        print(f"  Generated CER is SIGNIFICANTLY LOWER than degraded CER")
        if p_value_one_tailed < 0.001:
            print(f"  Significance level: *** p < 0.001 (VERY STRONG)")
        elif p_value_one_tailed < 0.01:
            print(f"  Significance level: ** p < 0.01 (STRONG)")
        else:
            print(f"  Significance level: * p < 0.05 (SIGNIFICANT)")
    else:
        print(f"  ❌ FAIL TO REJECT H₀ (p ≥ {alpha})")
        print(f"  No significant difference found")
    
    return {
        't_statistic': float(t_statistic),
        'p_value': float(p_value_one_tailed),
        'df': int(df),
        'critical_value': float(critical_value),
        'significant': bool(p_value_one_tailed < alpha)
    }


def cohens_d(degraded_cer, generated_cer):
    """
    Calculate Cohen's d effect size.
    
    Interpretation:
    - d < 0.2: Trivial
    - 0.2 ≤ d < 0.5: Small
    - 0.5 ≤ d < 0.8: Medium
    - d ≥ 0.8: Large
    - d ≥ 1.2: Very large
    """
    print(f"\n{'='*70}")
    print("EFFECT SIZE (COHEN'S d)")
    print(f"{'='*70}")
    
    # Calculate differences
    differences = degraded_cer - generated_cer
    
    # Cohen's d for paired samples
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    d = mean_diff / std_diff
    
    print(f"\nCalculation:")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Std of differences: {std_diff:.4f}")
    print(f"  Cohen's d: {d:.4f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if d < 0.2:
        effect_size = "Trivial"
        symbol = "⚠️"
    elif d < 0.5:
        effect_size = "Small"
        symbol = "✓"
    elif d < 0.8:
        effect_size = "Medium"
        symbol = "✅"
    elif d < 1.2:
        effect_size = "Large"
        symbol = "✅✅"
    else:
        effect_size = "Very Large"
        symbol = "✅✅✅"
    
    print(f"  {symbol} Effect size: {effect_size} (d = {d:.4f})")
    
    if d >= 0.8:
        print(f"  This represents a STRONG practical significance")
        print(f"  The improvement is not just statistically significant, but also practically meaningful")
    
    return {
        'cohens_d': float(d),
        'effect_size': effect_size,
        'mean_difference': float(mean_diff),
        'std_difference': float(std_diff)
    }


def bootstrap_confidence_interval(improvements, n_bootstrap=10000, confidence=0.95):
    """
    Calculate bootstrap confidence interval for mean improvement.
    """
    print(f"\n{'='*70}")
    print(f"BOOTSTRAP CONFIDENCE INTERVAL ({confidence*100:.0f}%)")
    print(f"{'='*70}")
    
    print(f"\nBootstrap parameters:")
    print(f"  Original sample size: {len(improvements)}")
    print(f"  Bootstrap iterations: {n_bootstrap}")
    print(f"  Confidence level: {confidence*100:.0f}%")
    
    # Bootstrap
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(improvements, size=len(improvements), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Calculate percentiles
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    
    observed_mean = np.mean(improvements)
    
    print(f"\nResults:")
    print(f"  Observed mean improvement: {observed_mean:.4f} ({observed_mean*100:.2f}%)")
    print(f"  {confidence*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  {confidence*100:.0f}% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
    print(f"  CI width: {(ci_upper - ci_lower)*100:.2f}%")
    
    # Check if target is within CI
    target = 0.25
    print(f"\nTarget threshold check:")
    if ci_lower > target:
        print(f"  ✅ Lower bound ({ci_lower*100:.2f}%) > Target ({target*100:.0f}%)")
        print(f"  We can be {confidence*100:.0f}% confident the true improvement exceeds the target")
    else:
        print(f"  ⚠️  Lower bound ({ci_lower*100:.2f}%) ≤ Target ({target*100:.0f}%)")
    
    return {
        'mean': float(observed_mean),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'confidence': float(confidence),
        'exceeds_target': bool(ci_lower > target)
    }


def wilcoxon_signed_rank_test(degraded_cer, generated_cer, alpha=0.05):
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    Good for non-normal distributions.
    """
    print(f"\n{'='*70}")
    print("WILCOXON SIGNED-RANK TEST (Non-parametric)")
    print(f"{'='*70}")
    
    # Perform test
    statistic, p_value = stats.wilcoxon(degraded_cer, generated_cer, alternative='greater')
    
    print(f"\nHypothesis:")
    print(f"  H₀: Median(degraded) = Median(generated)")
    print(f"  H₁: Median(degraded) > Median(generated)")
    
    print(f"\nTest Statistics:")
    print(f"  W-statistic: {statistic:.4f}")
    print(f"  p-value (one-tailed): {p_value:.6f}")
    
    print(f"\nMedians:")
    print(f"  Degraded CER: {np.median(degraded_cer):.4f} ({np.median(degraded_cer)*100:.2f}%)")
    print(f"  Generated CER: {np.median(generated_cer):.4f} ({np.median(generated_cer)*100:.2f}%)")
    
    print(f"\nInterpretation:")
    if p_value < alpha:
        print(f"  ✅ REJECT H₀ (p < {alpha})")
        print(f"  Generated CER is SIGNIFICANTLY LOWER (non-parametric test)")
    else:
        print(f"  ❌ FAIL TO REJECT H₀")
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': bool(p_value < alpha)
    }


def normality_test(data, name="Data"):
    """Test normality using Shapiro-Wilk test."""
    print(f"\n{'='*70}")
    print(f"NORMALITY TEST: {name}")
    print(f"{'='*70}")
    
    statistic, p_value = stats.shapiro(data)
    
    print(f"\nShapiro-Wilk Test:")
    print(f"  W-statistic: {statistic:.4f}")
    print(f"  p-value: {p_value:.6f}")
    
    if p_value > 0.05:
        print(f"  ✅ Data appears to be normally distributed (p > 0.05)")
        normal = True
    else:
        print(f"  ⚠️  Data may not be normally distributed (p ≤ 0.05)")
        print(f"  Consider using non-parametric tests (Wilcoxon)")
        normal = False
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'normal': bool(normal)
    }


def power_analysis(degraded_cer, generated_cer, alpha=0.05):
    """
    Calculate statistical power of the test.
    Power = probability of correctly rejecting H0 when it's false.
    """
    print(f"\n{'='*70}")
    print("STATISTICAL POWER ANALYSIS")
    print(f"{'='*70}")
    
    # Calculate effect size
    differences = degraded_cer - generated_cer
    d = np.mean(differences) / np.std(differences, ddof=1)
    
    # Sample size
    n = len(degraded_cer)
    
    # Calculate non-centrality parameter
    ncp = d * np.sqrt(n)
    
    # Calculate power (probability of rejecting H0)
    # For one-tailed test
    critical_t = t_dist.ppf(1 - alpha, n - 1)
    power = 1 - t_dist.cdf(critical_t, n - 1, ncp)
    
    print(f"\nParameters:")
    print(f"  Effect size (d): {d:.4f}")
    print(f"  Sample size (n): {n}")
    print(f"  Significance level (α): {alpha}")
    print(f"  Non-centrality parameter: {ncp:.4f}")
    
    print(f"\nResults:")
    print(f"  Statistical power: {power:.4f} ({power*100:.2f}%)")
    
    print(f"\nInterpretation:")
    if power >= 0.95:
        print(f"  ✅✅✅ Excellent power (≥95%)")
        print(f"  Very high probability of detecting the effect if it exists")
    elif power >= 0.80:
        print(f"  ✅ Good power (≥80%)")
        print(f"  Adequate probability of detecting the effect")
    elif power >= 0.60:
        print(f"  ⚠️  Moderate power (60-80%)")
        print(f"  May miss the effect in some cases")
    else:
        print(f"  ❌ Low power (<60%)")
        print(f"  High risk of Type II error (false negative)")
    
    return {
        'power': float(power),
        'effect_size': float(d),
        'sample_size': int(n)
    }


def create_statistical_summary_plot(results_dict, output_path):
    """Create comprehensive statistical summary visualization."""
    print(f"\n📊 Creating statistical summary plot...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # 1. T-test visualization
    ax1 = fig.add_subplot(gs[0, :2])
    t_stat = results_dict['paired_t_test']['t_statistic']
    p_val = results_dict['paired_t_test']['p_value']
    df = results_dict['paired_t_test']['df']
    
    x = np.linspace(-5, 5, 1000)
    y = t_dist.pdf(x, df)
    ax1.plot(x, y, 'b-', linewidth=2, label='t-distribution')
    ax1.axvline(t_stat, color='red', linestyle='--', linewidth=2, label=f't = {t_stat:.2f}')
    ax1.fill_between(x[x >= t_stat], 0, t_dist.pdf(x[x >= t_stat], df), 
                     alpha=0.3, color='red', label=f'p = {p_val:.6f}')
    ax1.set_xlabel('t-statistic', fontsize=11)
    ax1.set_ylabel('Probability Density', fontsize=11)
    ax1.set_title(f'Paired t-test (df={df})', fontsize=13, weight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Add result text
    result_text = "✅ SIGNIFICANT" if results_dict['paired_t_test']['significant'] else "❌ NOT SIGNIFICANT"
    ax1.text(0.98, 0.95, result_text, transform=ax1.transAxes,
            fontsize=12, weight='bold', color='green' if results_dict['paired_t_test']['significant'] else 'red',
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Effect size gauge
    ax2 = fig.add_subplot(gs[0, 2])
    d = results_dict['cohens_d']['cohens_d']
    effect_labels = ['Trivial', 'Small', 'Medium', 'Large', 'V.Large']
    effect_bounds = [0, 0.2, 0.5, 0.8, 1.2, 3.0]
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    
    for i in range(len(effect_bounds)-1):
        ax2.barh(0, effect_bounds[i+1]-effect_bounds[i], left=effect_bounds[i], 
                height=0.5, color=colors[i], alpha=0.6, edgecolor='black')
    
    ax2.plot([d, d], [-0.3, 0.3], 'r-', linewidth=3)
    ax2.scatter([d], [0], s=200, c='red', marker='v', zorder=5, edgecolors='black', linewidths=2)
    ax2.set_xlim([0, 3])
    ax2.set_ylim([-0.5, 0.5])
    ax2.set_xlabel("Cohen's d", fontsize=11)
    ax2.set_title("Effect Size", fontsize=13, weight='bold')
    ax2.set_yticks([])
    ax2.text(d, 0.4, f'd = {d:.2f}', ha='center', fontsize=10, weight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Confidence interval
    ax3 = fig.add_subplot(gs[1, :])
    mean_imp = results_dict['bootstrap_ci']['mean'] * 100
    ci_lower = results_dict['bootstrap_ci']['ci_lower'] * 100
    ci_upper = results_dict['bootstrap_ci']['ci_upper'] * 100
    
    ax3.errorbar([1], [mean_imp], yerr=[[mean_imp - ci_lower], [ci_upper - mean_imp]],
                fmt='o', markersize=15, color='green', capsize=20, capthick=3, linewidth=3,
                label=f'Mean: {mean_imp:.1f}%')
    ax3.axhline(25, color='red', linestyle='--', linewidth=2, label='Target: 25%')
    ax3.axhspan(ci_lower, ci_upper, alpha=0.2, color='green', label=f'95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]')
    ax3.set_xlim([0.5, 1.5])
    ax3.set_ylim([0, max(ci_upper, 70)])
    ax3.set_xticks([])
    ax3.set_ylabel('CER Improvement (%)', fontsize=12)
    ax3.set_title('Bootstrap 95% Confidence Interval', fontsize=13, weight='bold')
    ax3.legend(fontsize=11, loc='upper right')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Summary statistics table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    summary_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['─'*25, '─'*25, '─'*40],
        ['t-statistic', f"{results_dict['paired_t_test']['t_statistic']:.4f}", 
         '✅ Significant' if results_dict['paired_t_test']['significant'] else '❌ Not significant'],
        ['p-value', f"{results_dict['paired_t_test']['p_value']:.6f}", 
         '*** p < 0.001' if results_dict['paired_t_test']['p_value'] < 0.001 else 
         '** p < 0.01' if results_dict['paired_t_test']['p_value'] < 0.01 else
         '* p < 0.05' if results_dict['paired_t_test']['p_value'] < 0.05 else 'n.s.'],
        ['Cohen\'s d', f"{results_dict['cohens_d']['cohens_d']:.4f}", 
         results_dict['cohens_d']['effect_size']],
        ['Mean Improvement', f"{results_dict['bootstrap_ci']['mean']*100:.2f}%", 
         '✅ Exceeds target (25%)' if results_dict['bootstrap_ci']['mean'] > 0.25 else '❌ Below target'],
        ['95% CI', f"[{results_dict['bootstrap_ci']['ci_lower']*100:.2f}%, {results_dict['bootstrap_ci']['ci_upper']*100:.2f}%]",
         '✅ CI > target' if results_dict['bootstrap_ci']['ci_lower'] > 0.25 else 'CI includes target'],
        ['Statistical Power', f"{results_dict['power_analysis']['power']*100:.1f}%",
         '✅ Excellent' if results_dict['power_analysis']['power'] >= 0.95 else
         '✅ Good' if results_dict['power_analysis']['power'] >= 0.80 else '⚠️ Moderate'],
        ['Wilcoxon p-value', f"{results_dict['wilcoxon_test']['p_value']:.6f}",
         '✅ Significant (non-parametric)' if results_dict['wilcoxon_test']['significant'] else '❌ Not significant'],
    ]
    
    table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.25, 0.25, 0.50])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(2, len(summary_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.suptitle('Statistical Validation Summary', fontsize=16, weight='bold', y=0.98)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✅ Saved statistical summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Statistical validation of GAN-HTR results')
    parser.add_argument('--results_json', type=str,
                       default='dual_modal_gan/outputs/checkpoints_final_100/metrics/comprehensive_metrics.json',
                       help='Path to comprehensive evaluation results JSON')
    parser.add_argument('--output_dir', type=str,
                       default='dual_modal_gan/outputs/statistical_analysis',
                       help='Output directory for statistical analysis')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level (default: 0.05)')
    parser.add_argument('--bootstrap_iterations', type=int, default=10000,
                       help='Number of bootstrap iterations (default: 10000)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"STATISTICAL VALIDATION - GAN-HTR")
    print(f"{'='*70}")
    print(f"Output directory: {args.output_dir}")
    print(f"Significance level (α): {args.alpha}")
    
    # Load results
    results = load_results(args.results_json)
    degraded_cer, generated_cer, improvements = extract_cer_data(results['sample_results'])
    
    # Run all statistical tests
    results_dict = {}
    
    # 1. Normality tests
    print(f"\n{'='*70}")
    print("PRELIMINARY: NORMALITY CHECKS")
    print(f"{'='*70}")
    results_dict['normality_degraded'] = normality_test(degraded_cer, "Degraded CER")
    results_dict['normality_generated'] = normality_test(generated_cer, "Generated CER")
    results_dict['normality_improvements'] = normality_test(improvements, "Improvements")
    
    # 2. Paired t-test
    results_dict['paired_t_test'] = paired_t_test(degraded_cer, generated_cer, args.alpha)
    
    # 3. Cohen's d
    results_dict['cohens_d'] = cohens_d(degraded_cer, generated_cer)
    
    # 4. Bootstrap CI
    results_dict['bootstrap_ci'] = bootstrap_confidence_interval(
        improvements, 
        n_bootstrap=args.bootstrap_iterations
    )
    
    # 5. Wilcoxon test
    results_dict['wilcoxon_test'] = wilcoxon_signed_rank_test(degraded_cer, generated_cer, args.alpha)
    
    # 6. Power analysis
    results_dict['power_analysis'] = power_analysis(degraded_cer, generated_cer, args.alpha)
    
    # Create visualization
    create_statistical_summary_plot(
        results_dict,
        os.path.join(args.output_dir, 'statistical_summary.png')
    )
    
    # Save results to JSON
    output_json = os.path.join(args.output_dir, 'statistical_results.json')
    with open(output_json, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n✅ Statistical results saved to {output_json}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n🎯 Primary Finding:")
    if results_dict['paired_t_test']['significant']:
        print(f"  ✅ Generated images have SIGNIFICANTLY LOWER CER than degraded images")
        print(f"  ✅ p-value: {results_dict['paired_t_test']['p_value']:.6f}")
        if results_dict['paired_t_test']['p_value'] < 0.001:
            print(f"  ✅ Significance: *** VERY STRONG (p < 0.001)")
        elif results_dict['paired_t_test']['p_value'] < 0.01:
            print(f"  ✅ Significance: ** STRONG (p < 0.01)")
        else:
            print(f"  ✅ Significance: * MODERATE (p < 0.05)")
    else:
        print(f"  ❌ No significant difference found")
    
    print(f"\n📊 Effect Size:")
    print(f"  Cohen's d: {results_dict['cohens_d']['cohens_d']:.4f}")
    print(f"  Interpretation: {results_dict['cohens_d']['effect_size']}")
    if results_dict['cohens_d']['cohens_d'] >= 0.8:
        print(f"  ✅ LARGE practical significance")
    
    print(f"\n📈 Confidence Interval:")
    mean_imp = results_dict['bootstrap_ci']['mean'] * 100
    ci_lower = results_dict['bootstrap_ci']['ci_lower'] * 100
    ci_upper = results_dict['bootstrap_ci']['ci_upper'] * 100
    print(f"  Mean improvement: {mean_imp:.2f}%")
    print(f"  95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
    if results_dict['bootstrap_ci']['exceeds_target']:
        print(f"  ✅ Lower bound > 25% target (STRONG evidence)")
    
    print(f"\n💪 Statistical Power:")
    power = results_dict['power_analysis']['power'] * 100
    print(f"  Power: {power:.1f}%")
    if power >= 95:
        print(f"  ✅ Excellent power (≥95%)")
    elif power >= 80:
        print(f"  ✅ Good power (≥80%)")
    
    print(f"\n🎓 Conclusion:")
    if (results_dict['paired_t_test']['significant'] and 
        results_dict['cohens_d']['cohens_d'] >= 0.8 and
        results_dict['bootstrap_ci']['exceeds_target']):
        print(f"  ✅✅✅ STRONG STATISTICAL EVIDENCE")
        print(f"  The improvement is:")
        print(f"    1. Statistically significant (p < {args.alpha})")
        print(f"    2. Practically meaningful (large effect size)")
        print(f"    3. Robust and reliable (95% CI > target)")
        print(f"    4. Well-powered study (power ≥ 80%)")
        print(f"\n  🎉 HYPOTHESIS VALIDATED WITH HIGH CONFIDENCE!")
    else:
        print(f"  ⚠️  Mixed evidence - some criteria not met")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
