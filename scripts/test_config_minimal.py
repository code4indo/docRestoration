#!/usr/bin/env python3
"""
Minimal Test: Load config and build models only
No training, just verify config parsing and model building works
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
import json
import argparse

print("="*80)
print("MINIMAL CONFIG TEST - NO TRAINING")
print("="*80)

# Parse config
config_path = "configs/exp_proof_gt_v2_balanced.json"
print(f"\n1. Loading config: {config_path}")

with open(config_path) as f:
    config = json.load(f)

print(f"✅ Config loaded successfully")
print(f"   experiment_name: {config.get('experiment_name')}")
print(f"   discriminator_mode: {config.get('discriminator_mode')}")
print(f"   epochs: {config.get('epochs')}")
print(f"   batch_size: {config.get('batch_size')}")

# Check data files
print(f"\n2. Checking data files:")
tfrecord_path = config.get('tfrecord_path')
charset_path = config.get('charset_path')
recognizer_weights = config.get('recognizer_weights')

print(f"   TFRecord: {tfrecord_path}")
if Path(tfrecord_path).exists():
    print(f"      ✅ EXISTS ({Path(tfrecord_path).stat().st_size / 1024 / 1024:.1f} MB)")
else:
    print(f"      ❌ NOT FOUND")

print(f"   Charset: {charset_path}")
if Path(charset_path).exists():
    with open(charset_path) as f:
        charset = [line.strip() for line in f if line.strip()]
    print(f"      ✅ EXISTS ({len(charset)} characters)")
else:
    print(f"      ❌ NOT FOUND")
    
print(f"   Recognizer weights: {recognizer_weights}")
if Path(recognizer_weights).exists():
    print(f"      ✅ EXISTS")
else:
    print(f"      ❌ NOT FOUND")

# Build discriminator
print(f"\n3. Building discriminator with balanced config:")
from dual_modal_gan.src.models.discriminator_enhanced_v2_fixed import build_dual_modal_discriminator_enhanced_v2_fixed

disc_config = config.get('discriminator_config', {})
print(f"   Config: {disc_config}")

try:
    discriminator = build_dual_modal_discriminator_enhanced_v2_fixed(
        img_shape=(128, 1024, 1),
        vocab_size=len(charset) if Path(charset_path).exists() else 100,
        max_text_len=128,
        text_embed_dim=128,
        lstm_units=256,  # Will auto-adjust
        config=disc_config
    )
    print(f"\n✅ Discriminator built successfully!")
    print(f"   Total params: {discriminator.count_params():,}")
except Exception as e:
    print(f"\n❌ Discriminator build FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("MINIMAL TEST COMPLETED")
print("="*80)
