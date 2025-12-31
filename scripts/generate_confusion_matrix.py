#!/usr/bin/env python3
"""
Phase 2: Generate Character Confusion Matrix

Input: character_errors.json from Phase 1
Output:
- Confusion matrix heatmap (PNG/PDF)
- Top-K confused pairs table (CSV/JSON)
- Summary statistics

"""

import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_character_errors(json_path):
    """Load character errors from JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['errors'], data.get('metadata', {})

def build_confusion_matrix(errors, top_k=40):
    """
    Build confusion matrix from character errors
    
    Returns:
        confusion_df: DataFrame with GT chars as index, pred chars as columns
        char_list: Sorted list of characters
    """
    # Count confusions
    confusion_counts = defaultdict(lambda: defaultdict(int))
    
    for error in errors:
        if error['error_type'] != 'substitution':
            continue  # Only count substitutions for confusion matrix
        
        gt_char = error['gt_char']
        pred_char = error['pred_char']
        
        confusion_counts[gt_char][pred_char] += 1
    
    # Get most frequent characters
    all_gt_chars = Counter()
    for error in errors:
        if error['gt_char']:
            all_gt_chars[error['gt_char']] += 1
    
    # Select top-K most frequent chars
    top_chars = [char for char, _ in all_gt_chars.most_common(top_k)]
    
    # Build matrix
    matrix = np.zeros((len(top_chars), len(top_chars)))
    
    for i, gt_char in enumerate(top_chars):
        for j, pred_char in enumerate(top_chars):
            matrix[i, j] = confusion_counts[gt_char][pred_char]
    
    # Convert to DataFrame
    confusion_df = pd.DataFrame(matrix, index=top_chars, columns=top_chars)
    
    return confusion_df, top_chars

def plot_confusion_matrix(confusion_df, output_path='confusion_matrix.png', title='Character Confusion Matrix'):
    """
    Generate heatmap visualization of confusion matrix
    """
    fig, ax = plt.subplots(figsize=(18, 16))
    
    # Create heatmap
    sns.heatmap(confusion_df, 
                annot=False,  # Too many to annotate
                fmt='.0f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Confusion Count'},
                ax=ax,
                square=True)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Character', fontsize=14)
    ax.set_ylabel('Ground Truth Character', fontsize=14)
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved confusion matrix: {output_path}")

def extract_top_confusions(errors, top_k=20):
    """
    Extract top-K most confused character pairs
    
    Returns:
        List of dicts with confusion info
    """
    confusion_pairs = Counter()
    confusion_details = defaultdict(lambda: {
        'count': 0,
        'examples': []
    })
    
    for error in errors:
        if error['error_type'] != 'substitution':
            continue
        
        gt_char = error['gt_char']
        pred_char = error['pred_char']
        
        if gt_char == pred_char:
            continue  # Skip correct
        
        pair = (gt_char, pred_char)
        confusion_pairs[pair] += 1
        
        # Store example
        if len(confusion_details[pair]['examples']) < 5:
            confusion_details[pair]['examples'].append({
                'word': error['full_word'],
                'context': error['context_before'] + f"[{gt_char}→{pred_char}]" + error['context_after']
            })
        confusion_details[pair]['count'] = confusion_pairs[pair]
    
    # Get top-K
    top_confusions = []
    for (gt_char, pred_char), count in confusion_pairs.most_common(top_k):
        details = confusion_details[(gt_char, pred_char)]
        
        top_confusions.append({
            'rank': len(top_confusions) + 1,
            'gt_char': gt_char,
            'pred_char': pred_char,
            'count': count,
            'percentage': f"{count / len(errors) * 100:.2f}%",
            'example_words': [ex['word'] for ex in details['examples'][:3]],
            'example_contexts': [ex['context'] for ex in details['examples'][:3]]
        })
    
    return top_confusions

def generate_top_confusions_table(top_confusions, output_path='top_confusions.csv'):
    """
    Generate table of top confused pairs
    """
    # Create DataFrame
    df = pd.DataFrame(top_confusions)
    
    # Format for display
    display_df = df[['rank', 'gt_char', 'pred_char', 'count', 'percentage', 'example_words']].copy()
    display_df.columns = ['Rank', 'GT Char', 'Pred Char', 'Count', '% of Errors', 'Example Words']
    
    # Save to CSV
    display_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Saved top confusions table: {output_path}")
    
    # Also save detailed JSON
    json_path = output_path.replace('.csv', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(top_confusions, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved detailed confusions: {json_path}")
    
    return display_df

def analyze_confusion_patterns(errors):
    """
    Analyze patterns in confusions
    """
    stats = {
        'total_errors': len(errors),
        'by_error_type': Counter(e['error_type'] for e in errors),
        'by_position': Counter(e['word_position'] for e in errors),
        'ligature_errors': sum(1 for e in errors if e['is_ligature']),
        'capital_errors': sum(1 for e in errors if e['is_capital']),
        'punctuation_errors': sum(1 for e in errors if e['is_punctuation'])
    }
    
    return stats

def main(errors_json_path, output_dir='dual_modal_gan/analysis'):
    """
    Main analysis pipeline
    """
    print("="*80)
    print("PHASE 2: CONFUSION MATRIX ANALYSIS")
    print("="*80)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load errors
    print(f"\nLoading errors from: {errors_json_path}")
    errors, metadata = load_character_errors(errors_json_path)
    print(f"Total errors loaded: {len(errors)}")
    
    # Build confusion matrix
    print("\nBuilding confusion matrix...")
    confusion_df, char_list = build_confusion_matrix(errors, top_k=40)
    print(f"Matrix size: {confusion_df.shape[0]}×{confusion_df.shape[1]}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix heatmap...")
    plot_confusion_matrix(
        confusion_df,
        output_path=f"{output_dir}/confusion_matrix.png",
        title='Character Confusion Matrix (Top-40 Characters)'
    )
    
    # Extract top confusions
    print("\nExtracting top-20 confused pairs...")
    top_confusions = extract_top_confusions(errors, top_k=20)
    
    # Generate table
    print("\nGenerating top confusions table...")
    table_df = generate_top_confusions_table(
        top_confusions,
        output_path=f"{output_dir}/top_confusions.csv"
    )
    
    # Print top-10
    print(f"\n{'='*80}")
    print("TOP-10 MOST CONFUSED CHARACTER PAIRS:")
    print(f"{'='*80}")
    print(table_df.head(10).to_string(index=False))
    
    # Analyze patterns
    print(f"\n{'='*80}")
    print("CONFUSION PATTERN ANALYSIS:")
    print(f"{'='*80}")
    
    stats = analyze_confusion_patterns(errors)
    
    print(f"\nTotal errors: {stats['total_errors']:,}")
    print(f"\nError types:")
    for err_type, count in stats['by_error_type'].most_common():
        print(f"  {err_type:15s}: {count:6,} ({count/stats['total_errors']*100:5.1f}%)")
    
    print(f"\nWord position distribution:")
    for pos, count in stats['by_position'].most_common():
        print(f"  {pos:10s}: {count:6,} ({count/stats['total_errors']*100:5.1f}%)")
    
    print(f"\nCharacter properties:")
    print(f"  Ligature errors   : {stats['ligature_errors']:6,} ({stats['ligature_errors']/stats['total_errors']*100:5.1f}%)")
    print(f"  Capital errors    : {stats['capital_errors']:6,} ({stats['capital_errors']/stats['total_errors']*100:5.1f}%)")
    print(f"  Punctuation errors: {stats['punctuation_errors']:6,} ({stats['punctuation_errors']/stats['total_errors']*100:5.1f}%)")
    
    # Save summary stats
    stats_path = f"{output_dir}/confusion_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        # Convert Counter to dict for JSON serialization
        stats_json = {
            'total_errors': stats['total_errors'],
            'by_error_type': dict(stats['by_error_type']),
            'by_position': dict(stats['by_position']),
            'ligature_errors': stats['ligature_errors'],
            'capital_errors': stats['capital_errors'],
            'punctuation_errors': stats['punctuation_errors']
        }
        json.dump(stats_json, f, indent=2)
    print(f"\n✅ Saved stats: {stats_path}")
    
    print(f"\n{'='*80}")
    print("✅ PHASE 2 COMPLETE!")
    print(f"{'='*80}")
    print(f"Output files in: {output_dir}/")
    print(f"  - confusion_matrix.png (heatmap)")
    print(f"  - confusion_matrix.pdf (vector)")
    print(f"  - top_confusions.csv (table)")
    print(f"  - top_confusions.json (detailed)")
    print(f"  - confusion_stats.json (summary)")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        errors_json = sys.argv[1]
    else:
        errors_json = 'dual_modal_gan/analysis/character_errors.json'
    
    main(errors_json)
