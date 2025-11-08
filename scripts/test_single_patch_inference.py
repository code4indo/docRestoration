#!/usr/bin/env python3
"""Quick test: Time for single patch inference"""

import sys
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

# Add dual_modal_gan to path
project_root = Path(__file__).parent.parent
dual_modal_dir = project_root / "dual_modal_gan"
sys.path.insert(0, str(dual_modal_dir))

from src.models.generator_enhanced import unet_enhanced

# Load model
print("Loading model...")
start = time.time()
generator = unet_enhanced(input_size=(None, None, 1))
checkpoint_path = project_root / "dual_modal_gan/checkpoints/thin_stroke_preservation_v1_academic/best_model/ckpt-99"
checkpoint = tf.train.Checkpoint(generator=generator)
checkpoint.restore(str(checkpoint_path)).expect_partial()
print(f"Model loaded in {time.time() - start:.2f}s")

# Load test patch
patch_path = project_root / "outputs/anri_pseudo_patches/train/degraded/ID-ANRI_K66a_2482_0122_patch_0000.jpg"
with Image.open(patch_path) as img:
    if img.mode != 'L':
        img = img.convert('L')
    degraded = np.array(img, dtype=np.float32)

# Normalize
degraded_normalized = (degraded / 127.5) - 1.0
input_tensor = tf.convert_to_tensor(
    degraded_normalized[np.newaxis, :, :, np.newaxis],
    dtype=tf.float32
)

# First inference (with compilation)
print("\nFirst inference (with graph compilation)...")
start = time.time()
restored_tensor = generator(input_tensor, training=False)
first_time = time.time() - start
print(f"First inference: {first_time:.2f}s")

# Second inference (compiled)
print("\nSecond inference (compiled)...")
start = time.time()
restored_tensor = generator(input_tensor, training=False)
second_time = time.time() - start
print(f"Second inference: {second_time:.2f}s")

print(f"\nSpeed improvement: {first_time/second_time:.1f}x faster after compilation")
print(f"Estimated time for 8,882 patches:")
print(f"  If all patches take {first_time:.2f}s: {(8882 * first_time) / 3600:.1f} hours")
print(f"  If compiled ({second_time:.2f}s/patch): {(8882 * second_time) / 3600:.1f} hours")
