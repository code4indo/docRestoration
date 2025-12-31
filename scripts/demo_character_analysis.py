#!/usr/bin/env python3
"""
Quick Character-Level Analysis Demo

While waiting for full extraction, demonstrate Phase 2 functionality
with sample data to show what the output will look like.
"""

import json
import sys

# Add our simplified extraction
sys.path.append('/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/scripts')
from extract_character_errors_simple import extract_character_errors_from_text_pairs

# Generate sample data with realistic Dutch paleography patterns
print("Generating sample data with realistic HTR errors...\n")

gt_texts = [
    "den twaelfden maii",  # 'ae' ligature
    "van Amsterdam naer Batavia",  # Will have ſ → s confusion
    "waer ſtaet geschreven",  # Long s
    "het contract met de compagnie",  # Standard text
    "bij deſe gelegenheid",  # ſ and ligatures
    "ingekomen brieven uijt Oostindien",  # ij, ui patterns
    "vermeld in het register",  # Common words
    "onderteeckent ende geſegh",  # ck ligature, ſ
    "ten overstaan van notaris",  # Common legal phrase
    "volgens specificatie daervan",  # ae, van
]

# Simulate realistic HTR errors from degraded images
pred_texts = [
    "den twelfden maii",  # ae→e (ligature split)
    "van Amſterdam naer Bataνia",  # s→ſ (inverse), v→ν (Greek nu confusion)
    "waer staat geſchreuen",  # ſ→s, ſt ligature, v→u
    "het contract mct de compagnic",  # e→c (bleed-through effect)
    "bij dese gelegenheid",  # ſ→s
    "ingckomen brieven uyt Oostindien",  # e→c, ij→y common
    "vcrmeld in het register",  # v→c (degradation)
    "onderteckent cnde gesegel",  # ck→ck, e→c, ſ→s, gh→g
    "tcn overstaan van notaris",  # t→tc (insertion)
    "volgens specificatic daeruan",  # e→c, v→u
]

# Add more samples to reach realistic n
for i in range(10):
    # Add variations
    gt_texts.append(f"doende verſlag nummer {i+1}")
    pred_texts.append(f"doende verslag nummer {i+1}")  # ſ→s

# Extract errors
print("="*80)
errors = extract_character_errors_from_text_pairs(
    gt_texts,
    pred_texts,
    output_path='dual_modal_gan/analysis/demo_character_errors.json'
)

print(f"\n✅ Demo data created!")
print(f"   Total samples: {len(gt_texts)}")
print(f"   Total errors: {len(errors)}")
print(f"\n   Now running Phase 2: Confusion Matrix Generation...\n")

# Run Phase 2 on demo data
import subprocess
result = subprocess.run([
    'poetry', 'run', 'python', 
    'scripts/generate_confusion_matrix.py',
    'dual_modal_gan/analysis/demo_character_errors.json'
], cwd='/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration', capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

print("\n"+"="*80)
print("✅ DEMO PHASE 1+2 COMPLETE!")
print("="*80)
print("\nOutput files:")
print("  - dual_modal_gan/analysis/demo_character_errors.json")
print("  - dual_modal_gan/analysis/confusion_matrix.png")  
print("  - dual_modal_gan/analysis/top_confusions.csv")
print("\nThis demonstrates what the full analysis will produce.")
print("Once full predictions are extracted (712 samples), we'll run on real data.")
