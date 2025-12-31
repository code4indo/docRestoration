#!/usr/bin/env python3
"""
Recalculate Failure Pattern Distribution - Based on Character-Level Analysis
============================================================================

Script ini menghitung ulang distribusi pola kegagalan berdasarkan:
1. Data character-level errors dari analisis sebelumnya
2. Per-sample CER dan karakteristik error

Output: Distribusi pola kegagalan yang terverifikasi

Author: ML Engineer
Date: 2025-12-10
"""

import json
import os
from collections import defaultdict, Counter
from pathlib import Path

# Paths
CHARACTER_ERRORS_PATH = "dual_modal_gan/analysis/character_errors_restored.json"
TEST_PREDICTIONS_PATH = "dual_modal_gan/analysis/test_predictions_restored.json"
DETAILED_EVAL_PATH = "results/test_set_detailed_evaluation.json"

def load_character_errors():
    """Load character-level error analysis"""
    if os.path.exists(CHARACTER_ERRORS_PATH):
        with open(CHARACTER_ERRORS_PATH, 'r') as f:
            return json.load(f)
    return None

def load_test_predictions():
    """Load test predictions with GT and predicted text"""
    if os.path.exists(TEST_PREDICTIONS_PATH):
        with open(TEST_PREDICTIONS_PATH, 'r') as f:
            data = json.load(f)
        # Structure has 'predictions' key, not 'samples'
        predictions = data.get('predictions', [])
        # Calculate CER for each sample
        for pred in predictions:
            gt = pred.get('gt_text', '')
            pred_text = pred.get('pred_text_restored', '')
            if len(gt) > 0:
                import editdistance
                cer = editdistance.eval(gt, pred_text) / len(gt)
            else:
                cer = 0 if len(pred_text) == 0 else 1.0
            pred['cer'] = cer
        return {'samples': predictions, 'metadata': data.get('metadata', {})}
    return None

def load_detailed_evaluation():
    """Load detailed evaluation from results (gitignore-blocked, try anyway)"""
    if os.path.exists(DETAILED_EVAL_PATH):
        with open(DETAILED_EVAL_PATH, 'r') as f:
            return json.load(f)
    return None

def analyze_ligature_patterns(char_errors):
    """Analyze ligature-related errors from character error data"""
    if not char_errors:
        return None
    
    errors = char_errors.get('errors', [])
    total_errors = len(errors)
    
    # Count ligature errors
    ligature_errors = sum(1 for e in errors if e.get('is_ligature', False))
    
    # Count by sample
    samples_with_ligature_errors = set()
    for e in errors:
        if e.get('is_ligature', False):
            samples_with_ligature_errors.add(e.get('sample_id', -1))
    
    return {
        'total_errors': total_errors,
        'ligature_errors': ligature_errors,
        'ligature_error_percentage': ligature_errors / total_errors * 100 if total_errors > 0 else 0,
        'samples_with_ligature_errors': len(samples_with_ligature_errors),
        'sample_ids_with_ligature': sorted(list(samples_with_ligature_errors))
    }

def categorize_failure_by_cer(predictions):
    """Categorize samples by CER thresholds"""
    if not predictions:
        return None
    
    samples = predictions.get('samples', [])
    total_samples = len(samples)
    
    categories = {
        'excellent': [],      # CER < 15%
        'moderate': [],       # 15% <= CER < 50%
        'high_error': [],     # 50% <= CER < 90%
        'extreme': []         # CER >= 90%
    }
    
    for i, sample in enumerate(samples):
        cer = sample.get('cer', 0)
        sample_info = {
            'id': i,
            'cer': cer,
            'gt': sample.get('gt', ''),
            'pred': sample.get('pred', '')
        }
        
        if cer < 0.15:
            categories['excellent'].append(sample_info)
        elif cer < 0.50:
            categories['moderate'].append(sample_info)
        elif cer < 0.90:
            categories['high_error'].append(sample_info)
        else:
            categories['extreme'].append(sample_info)
    
    return {
        'total_samples': total_samples,
        'excellent_count': len(categories['excellent']),
        'excellent_pct': len(categories['excellent']) / total_samples * 100,
        'moderate_count': len(categories['moderate']),
        'moderate_pct': len(categories['moderate']) / total_samples * 100,
        'high_error_count': len(categories['high_error']),
        'high_error_pct': len(categories['high_error']) / total_samples * 100,
        'extreme_count': len(categories['extreme']),
        'extreme_pct': len(categories['extreme']) / total_samples * 100,
        'categories': categories
    }

def detect_content_type(gt_text):
    """Detect content type from ground truth text"""
    if not gt_text:
        return 'empty'
    
    # Count character types
    alpha_count = sum(1 for c in gt_text if c.isalpha())
    digit_count = sum(1 for c in gt_text if c.isdigit())
    symbol_count = sum(1 for c in gt_text if not c.isalnum() and not c.isspace())
    space_count = sum(1 for c in gt_text if c.isspace())
    total = len(gt_text)
    
    # Check for special symbols
    has_currency = 'ƒ' in gt_text or '€' in gt_text or '$' in gt_text
    has_section = '§' in gt_text
    has_dots_pattern = '. . .' in gt_text or '...' in gt_text
    
    # Categorize
    if has_currency or has_section:
        return 'numeric_symbol'
    if digit_count / max(total, 1) > 0.3:
        return 'numeric_heavy'
    if symbol_count / max(total, 1) > 0.2:
        return 'symbol_heavy'
    if has_dots_pattern:
        return 'dots_pattern'
    
    return 'text'

def recalculate_distribution():
    """Main function to recalculate failure distribution"""
    print("=" * 80)
    print("RECALCULATING FAILURE PATTERN DISTRIBUTION")
    print("=" * 80)
    
    # Load data
    print("\n[1/4] Loading character errors...")
    char_errors = load_character_errors()
    if char_errors:
        print(f"      Loaded {char_errors.get('metadata', {}).get('total_errors', 0)} character errors")
    else:
        print("      ❌ Character errors file not found")
    
    print("\n[2/4] Loading test predictions...")
    predictions = load_test_predictions()
    if predictions:
        print(f"      Loaded {len(predictions.get('samples', []))} samples")
    else:
        print("      ❌ Test predictions file not found")
    
    # Analyze ligature patterns
    print("\n[3/4] Analyzing ligature patterns from character errors...")
    ligature_analysis = analyze_ligature_patterns(char_errors)
    if ligature_analysis:
        print(f"      Total errors: {ligature_analysis['total_errors']}")
        print(f"      Ligature-context errors: {ligature_analysis['ligature_errors']} ({ligature_analysis['ligature_error_percentage']:.1f}%)")
        print(f"      Samples with ligature errors: {ligature_analysis['samples_with_ligature_errors']}")
    
    # CER-based categorization
    print("\n[4/4] Categorizing by CER thresholds...")
    cer_categories = categorize_failure_by_cer(predictions)
    
    if cer_categories:
        print(f"\n      CER Distribution (n={cer_categories['total_samples']}):")
        print(f"      - Excellent (CER < 15%): {cer_categories['excellent_count']} ({cer_categories['excellent_pct']:.1f}%)")
        print(f"      - Moderate (15-50%):    {cer_categories['moderate_count']} ({cer_categories['moderate_pct']:.1f}%)")
        print(f"      - High Error (50-90%):  {cer_categories['high_error_count']} ({cer_categories['high_error_pct']:.1f}%)")
        print(f"      - Extreme (>= 90%):     {cer_categories['extreme_count']} ({cer_categories['extreme_pct']:.1f}%)")
        
        # Analyze content types for high-error samples
        if cer_categories['categories']['high_error'] or cer_categories['categories']['extreme']:
            print("\n      Content Type Analysis for High/Extreme Error Samples:")
            high_extreme = cer_categories['categories']['high_error'] + cer_categories['categories']['extreme']
            content_types = Counter(detect_content_type(s['gt']) for s in high_extreme)
            for ct, count in content_types.most_common():
                print(f"        - {ct}: {count} ({count/len(high_extreme)*100:.1f}%)")
    
    # Calculate corrected ligature percentage
    print("\n" + "=" * 80)
    print("CORRECTED FINDINGS")
    print("=" * 80)
    
    if ligature_analysis and cer_categories:
        total_samples = cer_categories['total_samples']
        samples_with_lig_errors = ligature_analysis['samples_with_ligature_errors']
        lig_pct = samples_with_lig_errors / total_samples * 100
        
        print(f"\n1. LIGATURE-RELATED ERRORS:")
        print(f"   - Character errors in ligature context: {ligature_analysis['ligature_error_percentage']:.1f}%")
        print(f"   - Samples with at least one ligature error: {samples_with_lig_errors}/{total_samples} ({lig_pct:.1f}%)")
        
        # This is the key finding
        print(f"\n2. CER-BASED FAILURE DISTRIBUTION:")
        non_excellent = total_samples - cer_categories['excellent_count']
        print(f"   - Samples with CER >= 15%: {non_excellent}/{total_samples} ({non_excellent/total_samples*100:.1f}%)")
        
        print(f"\n3. RECOMMENDED CORRECTION FOR THESIS:")
        print(f"   Old claim: '62.9% of failures due to paleographic ligatures'")
        print(f"   Corrected: 'Samples with CER >= 15%: {non_excellent/total_samples*100:.1f}% ({non_excellent}/{total_samples})'")
        print(f"              'Character errors in ligature context: {ligature_analysis['ligature_error_percentage']:.1f}%'")
    
    # Save results
    results = {
        'ligature_analysis': ligature_analysis,
        'cer_categories': {
            'total_samples': cer_categories['total_samples'] if cer_categories else 0,
            'excellent_count': cer_categories['excellent_count'] if cer_categories else 0,
            'excellent_pct': cer_categories['excellent_pct'] if cer_categories else 0,
            'moderate_count': cer_categories['moderate_count'] if cer_categories else 0,
            'moderate_pct': cer_categories['moderate_pct'] if cer_categories else 0,
            'high_error_count': cer_categories['high_error_count'] if cer_categories else 0,
            'high_error_pct': cer_categories['high_error_pct'] if cer_categories else 0,
            'extreme_count': cer_categories['extreme_count'] if cer_categories else 0,
            'extreme_pct': cer_categories['extreme_pct'] if cer_categories else 0
        }
    }
    
    output_path = "dual_modal_gan/analysis/failure_distribution_recalculated.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {output_path}")
    
    return results

if __name__ == '__main__':
    recalculate_distribution()
