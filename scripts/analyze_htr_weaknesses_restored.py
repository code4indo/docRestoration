#!/usr/bin/env python3
"""
Analyze HTR Weaknesses on RESTORED Images

Identify:
1. Which characters are most confused (after restoration)
2. Which character pairs are problematic
3. Common error patterns
4. Word patterns that fail
5. Context-dependent failures

This answers: "What are HTR's weaknesses on RESTORED images?"
"""

import json
from collections import Counter, defaultdict
import pandas as pd

def load_restored_errors():
    """Load character-level errors from restored images"""
    with open('dual_modal_gan/analysis/character_errors_restored.json', 'r') as f:
        data = json.load(f)
    return data['errors']

def analyze_character_confusions(errors):
    """Find most confused character pairs"""
    substitutions = [e for e in errors if e['error_type'] == 'substitution']
    
    confusion_pairs = Counter()
    for error in substitutions:
        gt = error['gt_char']
        pred = error['pred_char']
        if gt and pred and gt != pred:
            confusion_pairs[(gt, pred)] += 1
    
    return confusion_pairs

def analyze_deletion_patterns(errors):
    """Find which characters are still deleted (not recognized)"""
    deletions = [e for e in errors if e['error_type'] == 'deletion']
    
    deleted_chars = Counter(e['gt_char'] for e in deletions if e['gt_char'])
    
    return deleted_chars

def analyze_word_patterns(errors):
    """Find problematic words"""
    word_errors = defaultdict(list)
    
    for error in errors:
        word = error.get('full_word', '')
        if word:
            word_errors[word].append(error)
    
    # Sort by error count
    word_error_counts = {word: len(errs) for word, errs in word_errors.items()}
    most_problematic = sorted(word_error_counts.items(), key=lambda x: x[1], reverse=True)
    
    return most_problematic, word_errors

def analyze_context_patterns(errors):
    """Find problematic bigram/trigram contexts"""
    substitutions = [e for e in errors if e['error_type'] == 'substitution']
    
    bigram_errors = Counter()
    for error in substitutions:
        context_before = error.get('context_before', '')
        gt_char = error['gt_char']
        pred_char = error['pred_char']
        
        if len(context_before) >= 1:
            char_before = context_before[-1]
            bigram_errors[(char_before, gt_char, pred_char)] += 1
    
    return bigram_errors

def main():
    print("="*80)
    print("ANALISIS KELEMAHAN HTR PADA CITRA RESTORED")
    print("="*80)
    
    # Load errors
    print("\nMemuat error data dari citra restored...")
    errors = load_restored_errors()
    print(f"Total errors pada restored: {len(errors):,}")
    
    # Overall distribution
    error_types = Counter(e['error_type'] for e in errors)
    print(f"\nDistribusi error type:")
    for err_type, count in error_types.most_common():
        print(f"  {err_type:15s}: {count:6,} ({count/len(errors)*100:5.1f}%)")
    
    # 1. Character Confusions
    print("\n" + "="*80)
    print("1. KARAKTER YANG PALING SERING TERTUKAR (Substitution)")
    print("="*80)
    
    confusion_pairs = analyze_character_confusions(errors)
    
    print("\nTop-20 Pasangan Konfusi:")
    print(f"{'Rank':<6} {'GT → Pred':<15} {'Count':<8} {'% of Subst':<12} {'Deskripsi':<30}")
    print("-"*80)
    
    substitution_count = error_types['substitution']
    
    for i, ((gt, pred), count) in enumerate(confusion_pairs.most_common(20), 1):
        pct = count / substitution_count * 100
        
        # Add description
        desc = ""
        if gt == 'n' and pred == 'e':
            desc = "Cursive similarity"
        elif gt == 'e' and pred == 'c':
            desc = "Bleed-through effect"
        elif gt == 'o' and pred == 'e':
            desc = "Similar round shapes"
        elif gt == 'a' and pred == 'e':
            desc = "Vowel confusion"
        elif 'ſ' in [gt, pred]:
            desc = "Historical long 's'"
        
        print(f"{i:<6} {gt} → {pred:<12} {count:<8} {pct:5.1f}%      {desc:<30}")
    
    # 2. Still Deleted Characters
    print("\n" + "="*80)
    print("2. KARAKTER YANG MASIH SERING HILANG (Deletion)")
    print("="*80)
    
    deleted_chars = analyze_deletion_patterns(errors)
    
    print("\nTop-15 Karakter yang Masih Ter-delete:")
    print(f"{'Rank':<6} {'Char':<8} {'Count':<10} {'Deskripsi':<40}")
    print("-"*80)
    
    for i, (char, count) in enumerate(deleted_chars.most_common(15), 1):
        desc = ""
        if char == ' ':
            desc = "Space (word segmentation issue)"
        elif char in '.,;:':
            desc = f"Punctuation '{char}' (fine detail)"
        elif char in '0123456789':
            desc = f"Digit '{char}'"
        elif char.isupper():
            desc = f"Capital letter (variant issue)"
        
        char_display = repr(char) if char in [' ', '\t', '\n'] else char
        print(f"{i:<6} {char_display:<8} {count:<10} {desc:<40}")
    
    # 3. Problematic Words
    print("\n" + "="*80)
    print("3. KATA-KATA YANG PALING BERMASALAH")
    print("="*80)
    
    most_problematic, word_errors = analyze_word_patterns(errors)
    
    print("\nTop-20 Kata dengan Error Terbanyak:")
    print(f"{'Rank':<6} {'Word':<25} {'Errors':<10} {'Error Types':<30}")
    print("-"*80)
    
    for i, (word, error_count) in enumerate(most_problematic[:20], 1):
        errs = word_errors[word]
        types = Counter(e['error_type'] for e in errs)
        types_str = ", ".join(f"{t}:{c}" for t, c in types.most_common(2))
        print(f"{i:<6} {word:<25} {error_count:<10} {types_str:<30}")
    
    # 4. Context Patterns
    print("\n" + "="*80)
    print("4. POLA KONTEKS YANG BERMASALAH (Bigram)")
    print("="*80)
    
    bigram_errors = analyze_context_patterns(errors)
    
    print("\nTop-15 Kesalahan Berbasis Konteks:")
    print(f"{'Rank':<6} {'Pattern':<20} {'Count':<8} {'Interpretation':<35}")
    print("-"*80)
    
    for i, ((char_before, gt_char, pred_char), count) in enumerate(bigram_errors.most_common(15), 1):
        pattern = f"{char_before}{gt_char} → {char_before}{pred_char}"
        
        interp = ""
        if 'e' in [gt_char, pred_char] and 'n' in [gt_char, pred_char]:
            interp = "'en' digraph confusion"
        elif char_before == ' ':
            interp = "Word-initial error"
        elif gt_char == ' ':
            interp = "Space detection failure"
        
        print(f"{i:<6} {pattern:<20} {count:<8} {interp:<35}")
    
    # 5. Character Properties Analysis
    print("\n" + "="*80)
    print("5. ANALISIS BERDASARKAN PROPERTI KARAKTER")
    print("="*80)
    
    ligature_errors = sum(1 for e in errors if e.get('is_ligature', False))
    capital_errors = sum(1 for e in errors if e.get('is_capital', False))
    punctuation_errors = sum(1 for e in errors if e.get('is_punctuation', False))
    
    print(f"\nError pada karakter khusus:")
    print(f"  Ligature errors    : {ligature_errors:6,} ({ligature_errors/len(errors)*100:5.1f}%)")
    print(f"  Capital errors     : {capital_errors:6,} ({capital_errors/len(errors)*100:5.1f}%)")
    print(f"  Punctuation errors : {punctuation_errors:6,} ({punctuation_errors/len(errors)*100:5.1f}%)")
    
    # 6. Summary & Recommendations
    print("\n" + "="*80)
    print("6. RINGKASAN & REKOMENDASI")
    print("="*80)
    
    # Get top confusions
    top_3_confusions = confusion_pairs.most_common(3)
    top_3_deletions = deleted_chars.most_common(3)
    
    print("\nKELEMAHAN UTAMA HTR (Setelah Restorasi):")
    print("\n1. Confusion Terbanyak:")
    for (gt, pred), count in top_3_confusions:
        print(f"   - '{gt}' → '{pred}': {count} kali")
    
    print("\n2. Deletion Terbanyak:")
    for char, count in top_3_deletions:
        char_display = repr(char) if char in [' ', '\t', '\n'] else char
        print(f"   - Karakter '{char_display}': {count} kali hilang")
    
    print("\n3. Properti Rentan:")
    props = [
        ('Ligature', ligature_errors),
        ('Capital', capital_errors),
        ('Punctuation', punctuation_errors)
    ]
    props.sort(key=lambda x: x[1], reverse=True)
    for prop, count in props:
        print(f"   - {prop}: {count:,} errors")
    
    print("\nREKOMENDASI PERBAIKAN:")
    print("\n✓ Data Augmentation:")
    print(f"  - Tambah training data untuk pasangan: {', '.join(f'{gt}↔{pred}' for (gt, pred), _ in top_3_confusions[:3])}")
    print(f"  - Focus pada ligature training (9.0% dari total errors)")
    
    print("\n✓ Post-Processing:")
    print(f"  - Language model untuk koreksi substitusi ({error_types['substitution']:,} errors)")
    print(f"  - Context-based correction untuk 'en' digraph")
    
    print("\n✓ Model Architecture:")
    print(f"  - Attention mechanism untuk word-initial characters")
    print(f"  - Fine-tune pada capital letter variants")
    
    # Save detailed analysis
    results = {
        'total_errors': len(errors),
        'error_distribution': dict(error_types),
        'top_confusions': [
            {
                'gt_char': gt,
                'pred_char': pred,
                'count': count,
                'percentage': count / substitution_count * 100
            }
            for (gt, pred), count in confusion_pairs.most_common(20)
        ],
        'top_deletions': [
            {
                'char': char,
                'count': count,
                'percentage': count / error_types['deletion'] * 100
            }
            for char, count in deleted_chars.most_common(15)
        ],
        'problematic_words': [
            {
                'word': word,
                'error_count': count
            }
            for word, count in most_problematic[:20]
        ],
        'character_properties': {
            'ligature_errors': ligature_errors,
            'capital_errors': capital_errors,
            'punctuation_errors': punctuation_errors
        }
    }
    
    with open('dual_modal_gan/analysis/htr_weaknesses_restored.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Analisis lengkap disimpan di: htr_weaknesses_restored.json")
    
    print("\n" + "="*80)
    print("✅ ANALISIS KELEMAHAN HTR SELESAI")
    print("="*80)

if __name__ == '__main__':
    main()
