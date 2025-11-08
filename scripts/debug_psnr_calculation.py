#!/usr/bin/env python3
"""
Debug PSNR Calculation Bug

Test scenario:
    clean_images from TFRecord = [0, 1]
    
    WRONG CODE (current):
        clean_images_tanh = clean_images * 2.0 - 1.0  # [0,1] → [-1,1]
        clean_images_normalized = (clean_images_tanh + 1.0) / 2.0  # [-1,1] → [0,1]
        # Result: Equivalent to original [0,1] ✓
    
    But wait, this should be correct then?
    
    Let's trace actual values:
    - clean_images original: [0, 1]
    - clean_images * 2.0 - 1.0 = [-1, 1]
    - ([-1, 1] + 1.0) / 2.0 = [0, 2] / 2.0 = [0, 1] ✓
    
    So denormalization is correct mathematically!
    
    Then why is PSNR different?
    
    HYPOTHESIS: The bug might be in how validation dataset samples 
    are different from ALL validation data!
"""

import numpy as np

# Simulate the transformation
print("Testing normalization pipeline:")
print("="*80)

# Original clean image from TFRecord [0, 1]
clean_original = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
print(f"clean_images (TFRecord):          {clean_original}")

# Training code transformation
clean_tanh = clean_original * 2.0 - 1.0
print(f"clean_images_tanh (×2-1):         {clean_tanh}")

clean_normalized = (clean_tanh + 1.0) / 2.0
print(f"clean_images_normalized (+1)/2:   {clean_normalized}")

# Check if they match
print(f"\nDo they match? {np.allclose(clean_original, clean_normalized)}")
print(f"Max difference: {np.abs(clean_original - clean_normalized).max()}")

# HYPOTHESIS 2: Different samples between validation loop and sample generation?
print("\n" + "="*80)
print("HYPOTHESIS: Validation dataset iteration might be inconsistent")
print("="*80)
print("""
Potential issues:
1. Validation dataset might repeat/shuffle between iterations
2. Sample generation uses .take(5) which gets DIFFERENT samples
3. PSNR reported is AVERAGE over ALL validation data
4. Visual samples are only FIRST 5 from dataset

If validation data repeats or samples differently each epoch,
reported PSNR won't match visual samples!
""")
