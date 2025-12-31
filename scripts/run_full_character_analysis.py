#!/usr/bin/env python3
"""
Extract Character-Level Errors from Test Predictions

Input: test_predictions.json (GT + pred pairs for 712 samples)
Output: character_errors.json (detailed character-level errors)
"""

import json
import sys

sys.path.append('/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/scripts')
from extract_character_errors_simple import extract_character_errors_from_text_pairs

# Load predictions
print("Loading test predictions...")
with open('dual_modal_gan/analysis/test_predictions.json', 'r') as f:
    data = json.load(f)

predictions = data['predictions']
print(f"Loaded {len(predictions)} samples")

# Extract GT and pred texts
gt_texts = [p['gt_text'] for p in predictions]
pred_texts = [p['pred_text_degraded'] for p in predictions]

print(f"\nExtracting character-level errors...")
print(f"Total GT characters: {sum(len(gt) for gt in gt_texts):,}")
print(f"Total pred characters: {sum(len(pred) for pred in pred_texts):,}")

# Extract errors
errors = extract_character_errors_from_text_pairs(
    gt_texts,
    pred_texts,
    output_path='dual_modal_gan/analysis/character_errors_full.json'
)

print(f"\n✅ Character-level error extraction complete!")
print(f"   Total errors: {len(errors):,}")
print(f"   Saved to: dual_modal_gan/analysis/character_errors_full.json")
print(f"\n🚀 Now running Phase 2: Confusion Matrix Generation...")

# Run Phase 2
import subprocess
result = subprocess.run([
    'poetry', 'run', 'python',
    'scripts/generate_confusion_matrix.py',
    'dual_modal_gan/analysis/character_errors_full.json'
], cwd='/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration', capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
    sys.exit(1)

print("\n" + "="*80)
print("✅ FULL ANALYSIS COMPLETE!")
print("="*80)
print("\nFinal outputs:")
print("  - dual_modal_gan/analysis/character_errors_full.json")
print("  - dual_modal_gan/analysis/confusion_matrix.png")
print("  - dual_modal_gan/analysis/confusion_matrix.pdf")
print("  - dual_modal_gan/analysis/top_confusions.csv")
print("  - dual_modal_gan/analysis/top_confusions.json")
