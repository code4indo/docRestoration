#!/usr/bin/env python3
"""
Phase 1 (Simplified): Character-Level Error Extraction from Existing Predictions

Extract character-level errors using existing evaluation results.
Instead of re-running inference, we parse saved predictions.

This is faster and uses existing validation outputs.
"""

import os
import json
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import List, Tuple
import editdistance

@dataclass
class CharacterError:
    """Single character-level error instance"""
    sample_id: int
    position_in_word: int
    position_in_sequence: int
    word_position: str  # 'start', 'middle', 'end'
    
    gt_char: str
    pred_char: str
    
    context_before: str
    context_after: str
    full_word: str
    full_gt_text: str
    full_pred_text: str
    
    error_type: str  # 'substitution', 'insertion', 'deletion'
    
    is_ligature: bool
    is_capital: bool
    is_punctuation: bool

def align_strings(gt: str, pred: str) -> List[Tuple[str, str, str]]:
    """
    Align two strings using edit distance
    
    Returns:
        List of (gt_char, pred_char, error_type)
    """
    n, m = len(gt), len(pred)
    
    # DP matrix for edit distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt[i-1] == pred[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # deletion
                    dp[i][j-1],      # insertion
                    dp[i-1][j-1]     # substitution
                )
    
    # Backtrack to get alignment
    alignments = []
    i, j = n, m
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and gt[i-1] == pred[j-1]:
            alignments.append((gt[i-1], pred[j-1], 'correct'))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignments.append((gt[i-1], pred[j-1], 'substitution'))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            alignments.append(('', pred[j-1], 'insertion'))
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            alignments.append((gt[i-1], '', 'deletion'))
            i -= 1
    
    alignments.reverse()
    return alignments

def detect_word_position(char_idx: int, text: str) -> str:
    """Detect if character is at start/middle/end of word"""
    if not text or char_idx >= len(text):
        return 'unknown'
    
    if char_idx == 0 or (char_idx > 0 and text[char_idx-1] == ' '):
        return 'start'
    
    if char_idx == len(text) - 1 or (char_idx + 1 < len(text) and text[char_idx+1] == ' '):
        return 'end'
    
    return 'middle'

def is_ligature(context: str) -> bool:
    """Detect if part of common Dutch paleographic ligature"""
    ligatures = ['st', 'ct', 'ck', 'ae', 'oe', 'ij', 'ff', 'll', 'ss']
    context_lower = context.lower()
    return any(lig in context_lower for lig in ligatures)

def extract_word(text: str, char_idx: int) -> Tuple[str, int]:
    """
    Extract word containing character at char_idx
    
    Returns:
        (word, position_in_word)
    """
    if not text or char_idx >= len(text):
        return '', 0
    
    # Find word boundaries
    start = char_idx
    while start > 0 and text[start-1] != ' ':
        start -= 1
    
    end = char_idx
    while end < len(text) and text[end] != ' ':
        end += 1
    
    word = text[start:end]
    pos_in_word = char_idx - start
    
    return word, pos_in_word

def extract_character_errors_from_text_pairs(
    gt_texts: List[str],
    pred_texts: List[str],
    output_path: str = 'character_errors_simple.json'
):
    """
    Extract character-level errors from paired GT and prediction texts
    
    Args:
        gt_texts: List of ground truth texts
        pred_texts: List of predicted texts
        output_path: Where to save results
    """
    
    print("="*80)
    print("CHARACTER-LEVEL ERROR EXTRACTION (Simplified)")
    print("="*80)
    
    all_errors = []
    
    for sample_id, (gt_text, pred_text) in enumerate(zip(gt_texts, pred_texts)):
        # Align sequences
        alignments = align_strings(gt_text, pred_text)
        
        # Track position in GT text
        gt_char_idx = 0
        
        for align_item in alignments:
            gt_char, pred_char, error_type = align_item
            
            # Skip correct matches (no diagnostic value)
            if error_type == 'correct':
                gt_char_idx += 1
                continue
            
            # For insertions, no GT character
            if error_type == 'insertion':
                # We could track these separately, but skip for now
                continue
            
            # Extract context and word info
            word, pos_in_word = extract_word(gt_text, gt_char_idx)
            word_pos = detect_word_position(gt_char_idx, gt_text)
            
            context_before = gt_text[max(0, gt_char_idx-2):gt_char_idx]
            context_after = gt_text[gt_char_idx+1:min(len(gt_text), gt_char_idx+3)]
            
            context_window = context_before + gt_char + context_after
            
            # Create error instance
            error = CharacterError(
                sample_id=sample_id,
                position_in_word=pos_in_word,
                position_in_sequence=gt_char_idx,
                word_position=word_pos,
                
                gt_char=gt_char,
                pred_char=pred_char,
                
                context_before=context_before,
                context_after=context_after,
                full_word=word,
                full_gt_text=gt_text,
                full_pred_text=pred_text,
                
                error_type=error_type,
                
                is_ligature=is_ligature(context_window),
                is_capital=gt_char.isupper() if gt_char else False,
                is_punctuation=not gt_char.isalnum() if gt_char else False
            )
            
            all_errors.append(asdict(error))
            
            # Move to next character in GT
            if error_type != 'insertion':
                gt_char_idx += 1
    
    # Save results
    print(f"\nExtraction complete!")
    print(f"Total samples: {len(gt_texts)}")
    print(f"Total character-level errors: {len(all_errors)}")
    print(f"Saving to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'total_samples': len(gt_texts),
                'total_errors': len(all_errors),
                'error_rate_per_sample': len(all_errors) / len(gt_texts) if gt_texts else 0
            },
            'errors': all_errors
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved: {output_path}")
    
    # Print quick stats
    print(f"\n{'='*80}")
    print("QUICK STATISTICS:")
    print(f"{'='*80}")
    
    error_types = Counter(e['error_type'] for e in all_errors)
    print(f"\nError Types:")
    for err_type, count in error_types.most_common():
        print(f"  {err_type:15s}: {count:6d} ({count/len(all_errors)*100:5.1f}%)")
    
    ligature_errors = sum(1 for e in all_errors if e['is_ligature'])
    capital_errors = sum(1 for e in all_errors if e['is_capital'])
    
    print(f"\nCharacter Properties:")
    print(f"  Ligature errors : {ligature_errors:6d} ({ligature_errors/len(all_errors)*100:5.1f}%)")
    print(f"  Capital errors  : {capital_errors:6d} ({capital_errors/len(all_errors)*100:5.1f}%)")
    
    position_errors = Counter(e['word_position'] for e in all_errors)
    print(f"\nWord Position Distribution:")
    for pos, count in position_errors.most_common():
        print(f"  {pos:10s}: {count:6d} ({count/len(all_errors)*100:5.1f}%)")
    
    return all_errors

def demo_with_sample_data():
    """Demo with sample GT and prediction pairs"""
    
    # Sample data (replace with real evaluation results)
    gt_sample = [
        "den twaelfden maii",
        "van Amsterdam naer Batavia",
        "waer ſtaet geschreven"
    ]
    
    pred_sample = [
        "den twelfden maii",  # ſ → s
        "van Amſterdam naer Batauia",  # s → ſ, v → u
        "waer staat geſchreuen"  # ſt ligature, v → u
    ]
    
    errors = extract_character_errors_from_text_pairs(
        gt_sample,
        pred_sample,
        output_path='demo_character_errors.json'
    )
    
    return errors

if __name__ == '__main__':
    print("Running demo with sample data...\n")
    demo_with_sample_data()
