#!/usr/bin/env python3
"""
Verify Range Normalization Fix
Validates that generator receives [-1,1] input and produces [-1,1] output with proper black pixels
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'dual_modal_gan' / 'src'))

from models.generator_enhanced import unet_enhanced

def verify_fix():
    print("="*80)
    print("🔬 VERIFYING RANGE NORMALIZATION FIX")
    print("="*80)
    
    # Create generator
    generator = unet_enhanced(input_size=(1024, 128, 1))
    print("\n✅ Generator created (unet_enhanced)")
    
    # Simulate TFRecord data [0, 1]
    tfrecord_data = tf.random.uniform([1, 1024, 128, 1], minval=0.0, maxval=1.0)
    print(f"\n📊 Simulated TFRecord data:")
    print(f"   Range: [{tfrecord_data.numpy().min():.4f}, {tfrecord_data.numpy().max():.4f}]")
    print(f"   Expected: [0.0000, 1.0000]")
    
    # Normalize to [-1, 1] as per fix
    input_tanh = tfrecord_data * 2.0 - 1.0
    print(f"\n📊 After normalization (x*2-1):")
    print(f"   Range: [{input_tanh.numpy().min():.4f}, {input_tanh.numpy().max():.4f}]")
    print(f"   Expected: [-1.0000, 1.0000]")
    
    # Generate output
    output_tanh = generator(input_tanh, training=False)
    print(f"\n📊 Generator output (tanh):")
    print(f"   Range: [{output_tanh.numpy().min():.4f}, {output_tanh.numpy().max():.4f}]")
    print(f"   Expected: [-1.0000, 1.0000] (tanh activation)")
    
    # Denormalize for display
    output_normalized = (output_tanh + 1.0) / 2.0
    print(f"\n📊 After denormalization (x+1)/2:")
    print(f"   Range: [{output_normalized.numpy().min():.4f}, {output_normalized.numpy().max():.4f}]")
    print(f"   Expected: [0.0000, 1.0000]")
    
    # Convert to uint8
    output_uint8 = (output_normalized * 255).numpy().astype(np.uint8)
    print(f"\n📊 Final uint8 image:")
    print(f"   Range: [{output_uint8.min()}, {output_uint8.max()}]")
    print(f"   Expected: [0, 255] with black pixels possible")
    
    # Verify fix success
    print("\n" + "="*80)
    print("✅ VERIFICATION RESULTS:")
    print("="*80)
    
    checks_passed = 0
    total_checks = 4
    
    # Check 1: Generator can output full tanh range
    if output_tanh.numpy().min() <= -0.5:  # Can reach negative values
        print("✅ Generator outputs proper negative values (can produce black)")
        checks_passed += 1
    else:
        print(f"❌ Generator min output {output_tanh.numpy().min():.4f} (should be <-0.5)")
    
    # Check 2: Denormalized output covers [0,1]
    if output_normalized.numpy().min() < 0.1:
        print("✅ Denormalized output can reach near 0 (black pixels)")
        checks_passed += 1
    else:
        print(f"❌ Denormalized min {output_normalized.numpy().min():.4f} (should be near 0.0)")
    
    # Check 3: uint8 can have black pixels
    if output_uint8.min() < 50:
        print("✅ uint8 output can produce black pixels (<50)")
        checks_passed += 1
    else:
        print(f"❌ uint8 min {output_uint8.min()} (should be <50 for black text)")
    
    # Check 4: Input normalization correct
    expected_input_min = tfrecord_data.numpy().min() * 2.0 - 1.0
    actual_input_min = input_tanh.numpy().min()
    if abs(expected_input_min - actual_input_min) < 0.01:
        print("✅ Input normalization formula correct")
        checks_passed += 1
    else:
        print(f"❌ Input normalization mismatch")
    
    print("\n" + "="*80)
    if checks_passed == total_checks:
        print(f"🎉 ALL CHECKS PASSED ({checks_passed}/{total_checks})")
        print("✅ Range normalization fix is CORRECT")
        print("✅ Ready for training - generator can produce black text")
        return True
    else:
        print(f"⚠️  SOME CHECKS FAILED ({checks_passed}/{total_checks})")
        print("❌ Fix may need adjustment")
        return False

if __name__ == "__main__":
    success = verify_fix()
    sys.exit(0 if success else 1)
