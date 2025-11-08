#!/usr/bin/env python3
"""
AUDIT SCRIPT: Verifikasi Single-Modal Discriminator Architecture

Tujuan: Memastikan single-modal discriminator BENAR-BENAR single-modal (CNN-only)
        tanpa komponen text/HTR/LSTM.

Test yang dilakukan:
1. Architecture inspection (input/output layers)
2. Parameter count comparison
3. Forward pass test (image-only input)
4. Layer analysis (no LSTM/Embedding/Text layers)
5. Comparison dengan dual-modal architecture

Author: AI/ML Engineer
Date: 2025-11-06
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np

# Import discriminators
from dual_modal_gan.src.models.discriminator_single_modal import build_single_modal_discriminator_enhanced
from dual_modal_gan.src.models.discriminator_enhanced_v2_fixed import build_dual_modal_discriminator_enhanced_v2_fixed

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def test_single_modal_architecture():
    """Test 1: Architecture Inspection"""
    print_section("TEST 1: SINGLE-MODAL ARCHITECTURE INSPECTION")
    
    config = {
        'spatial_attention_kernel': 3,
        'batchnorm_momentum': 0.9,
        'dropout_rate': 0.1,
        'use_residual_blocks': True,
        'use_spatial_attention': True
    }
    
    discriminator = build_single_modal_discriminator_enhanced(
        img_shape=(1024, 128, 1),
        config=config
    )
    
    print("\n📊 INPUT ANALYSIS:")
    print(f"   Number of inputs: {len(discriminator.inputs)}")
    for i, inp in enumerate(discriminator.inputs):
        print(f"   Input {i}: name='{inp.name}', shape={inp.shape}, dtype={inp.dtype}")
    
    print("\n📊 OUTPUT ANALYSIS:")
    print(f"   Output: name='{discriminator.output.name}', shape={discriminator.output.shape}")
    
    print("\n📊 LAYER ANALYSIS:")
    text_layers = []
    lstm_layers = []
    embedding_layers = []
    
    for layer in discriminator.layers:
        layer_type = type(layer).__name__
        if 'LSTM' in layer_type or 'GRU' in layer_type or 'RNN' in layer_type:
            lstm_layers.append(layer.name)
        if 'Embedding' in layer_type:
            embedding_layers.append(layer.name)
        if 'text' in layer.name.lower() or 'sequential' in layer.name.lower():
            text_layers.append(layer.name)
    
    print(f"   Total layers: {len(discriminator.layers)}")
    print(f"   LSTM/RNN layers: {len(lstm_layers)} → {lstm_layers if lstm_layers else '✅ NONE (correct)'}")
    print(f"   Embedding layers: {len(embedding_layers)} → {embedding_layers if embedding_layers else '✅ NONE (correct)'}")
    print(f"   Text-related layers: {len(text_layers)} → {text_layers if text_layers else '✅ NONE (correct)'}")
    
    # Verification
    is_single_modal = (
        len(discriminator.inputs) == 1 and
        len(lstm_layers) == 0 and
        len(embedding_layers) == 0
    )
    
    print("\n" + "="*80)
    if is_single_modal:
        print("✅ PASS: Architecture is TRULY SINGLE-MODAL (CNN-only)")
    else:
        print("❌ FAIL: Architecture contains text/LSTM components!")
    print("="*80)
    
    return discriminator, is_single_modal

def test_forward_pass():
    """Test 2: Forward Pass dengan Image-Only Input"""
    print_section("TEST 2: FORWARD PASS TEST (Image-Only)")
    
    config = {
        'spatial_attention_kernel': 3,
        'batchnorm_momentum': 0.9,
        'dropout_rate': 0.1
    }
    
    discriminator = build_single_modal_discriminator_enhanced(
        img_shape=(1024, 128, 1),
        config=config
    )
    
    # Test dengan dummy image
    batch_size = 2
    test_images = np.random.randn(batch_size, 1024, 128, 1).astype(np.float32)
    
    print(f"\n📊 INPUT TEST:")
    print(f"   Batch size: {batch_size}")
    print(f"   Image shape: {test_images.shape}")
    print(f"   Image range: [{test_images.min():.3f}, {test_images.max():.3f}]")
    
    try:
        output = discriminator(test_images, training=False)
        
        print(f"\n📊 OUTPUT:")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.numpy().min():.4f}, {output.numpy().max():.4f}]")
        print(f"   Expected range: [0.0, 1.0] (sigmoid activation)")
        print(f"   Values: {output.numpy().flatten()}")
        
        # Verification
        is_valid = (
            output.shape == (batch_size, 1) and
            output.numpy().min() >= 0.0 and
            output.numpy().max() <= 1.0
        )
        
        print("\n" + "="*80)
        if is_valid:
            print("✅ PASS: Forward pass successful with image-only input")
        else:
            print("❌ FAIL: Output shape or range incorrect!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAIL: Forward pass failed with error:")
        print(f"   {str(e)}")
        print("="*80)
        return False

def test_dual_input_should_fail():
    """Test 3: Dual Input Should FAIL (memastikan tidak menerima text input)"""
    print_section("TEST 3: DUAL INPUT REJECTION TEST")
    
    config = {
        'spatial_attention_kernel': 3,
        'batchnorm_momentum': 0.9,
        'dropout_rate': 0.1
    }
    
    discriminator = build_single_modal_discriminator_enhanced(
        img_shape=(1024, 128, 1),
        config=config
    )
    
    # Test dengan image + text (should FAIL)
    batch_size = 2
    test_images = np.random.randn(batch_size, 1024, 128, 1).astype(np.float32)
    test_text = np.random.randint(0, 100, (batch_size, 128), dtype=np.int32)
    
    print(f"\n📊 INPUT TEST (Should FAIL):")
    print(f"   Image shape: {test_images.shape}")
    print(f"   Text shape: {test_text.shape}")
    print(f"   Trying to pass [image, text] to single-modal discriminator...")
    
    try:
        # Try to pass dual input (should fail)
        output = discriminator([test_images, test_text], training=False)
        
        print(f"\n❌ FAIL: Discriminator accepted dual input (image + text)!")
        print(f"   This indicates it's NOT truly single-modal!")
        print("="*80)
        return False
        
    except (ValueError, TypeError) as e:
        print(f"\n✅ PASS: Discriminator correctly REJECTED dual input")
        print(f"   Error message: {str(e)[:100]}...")
        print(f"   This confirms it's truly single-modal (image-only)")
        print("="*80)
        return True

def compare_with_dual_modal():
    """Test 4: Comparison dengan Dual-Modal Architecture"""
    print_section("TEST 4: COMPARISON WITH DUAL-MODAL ARCHITECTURE")
    
    config = {
        'spatial_attention_kernel': 3,
        'cross_modal_common_dim': 128,
        'batchnorm_momentum': 0.9,
        'dropout_rate': 0.1,
        'use_residual_blocks': True,
        'use_spatial_attention': True,
        'use_cross_modal_attention': True,
        'attention_type': 'scaled_dot_product'
    }
    
    # Build both discriminators
    print("\n📊 Building Single-Modal Discriminator...")
    single_modal = build_single_modal_discriminator_enhanced(
        img_shape=(1024, 128, 1),
        config=config
    )
    
    print("\n📊 Building Dual-Modal Discriminator...")
    dual_modal = build_dual_modal_discriminator_enhanced_v2_fixed(
        img_shape=(1024, 128, 1),
        vocab_size=108,
        max_text_len=128,
        config=config
    )
    
    # Compare architectures
    print("\n" + "="*80)
    print(" ARCHITECTURE COMPARISON")
    print("="*80)
    
    print(f"\n{'Aspect':<30} | {'Single-Modal':<25} | {'Dual-Modal':<25}")
    print("-" * 85)
    print(f"{'Number of inputs':<30} | {len(single_modal.inputs):<25} | {len(dual_modal.inputs):<25}")
    print(f"{'Input types':<30} | {'Image only':<25} | {'Image + Text':<25}")
    print(f"{'Total layers':<30} | {len(single_modal.layers):<25} | {len(dual_modal.layers):<25}")
    print(f"{'Total parameters':<30} | {single_modal.count_params():<25,} | {dual_modal.count_params():<25,}")
    print(f"{'Trainable parameters':<30} | {sum([tf.keras.backend.count_params(w) for w in single_modal.trainable_weights]):<25,} | {sum([tf.keras.backend.count_params(w) for w in dual_modal.trainable_weights]):<25,}")
    
    # Check for LSTM/Embedding layers
    single_lstm = sum(1 for l in single_modal.layers if 'LSTM' in type(l).__name__)
    dual_lstm = sum(1 for l in dual_modal.layers if 'LSTM' in type(l).__name__)
    single_embed = sum(1 for l in single_modal.layers if 'Embedding' in type(l).__name__)
    dual_embed = sum(1 for l in dual_modal.layers if 'Embedding' in type(l).__name__)
    
    print(f"{'LSTM layers':<30} | {single_lstm:<25} | {dual_lstm:<25}")
    print(f"{'Embedding layers':<30} | {single_embed:<25} | {dual_embed:<25}")
    
    print("\n" + "="*80)
    print(" VERIFICATION RESULTS")
    print("="*80)
    
    checks = {
        "Single-modal has 1 input": len(single_modal.inputs) == 1,
        "Dual-modal has 2 inputs": len(dual_modal.inputs) == 2,
        "Single-modal has NO LSTM": single_lstm == 0,
        "Dual-modal HAS LSTM": dual_lstm > 0,
        "Single-modal has NO Embedding": single_embed == 0,
        "Dual-modal HAS Embedding": dual_embed > 0,
        "Parameter count comparable": abs(single_modal.count_params() - dual_modal.count_params()) < 2_000_000
    }
    
    for check, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check}")
    
    all_passed = all(checks.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL CHECKS PASSED: Architectures are correctly differentiated")
    else:
        print("❌ SOME CHECKS FAILED: Review architecture implementation")
    print("="*80)
    
    return all_passed

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print(" SINGLE-MODAL DISCRIMINATOR ARCHITECTURE AUDIT")
    print(" Date: 2025-11-06")
    print(" Purpose: Verify single-modal is TRULY single-modal (CNN-only)")
    print("="*80)
    
    results = {
        "Architecture Inspection": False,
        "Forward Pass Test": False,
        "Dual Input Rejection": False,
        "Dual-Modal Comparison": False
    }
    
    # Test 1: Architecture inspection
    try:
        _, results["Architecture Inspection"] = test_single_modal_architecture()
    except Exception as e:
        print(f"\n❌ Test 1 FAILED with exception: {e}")
    
    # Test 2: Forward pass
    try:
        results["Forward Pass Test"] = test_forward_pass()
    except Exception as e:
        print(f"\n❌ Test 2 FAILED with exception: {e}")
    
    # Test 3: Dual input rejection
    try:
        results["Dual Input Rejection"] = test_dual_input_should_fail()
    except Exception as e:
        print(f"\n❌ Test 3 FAILED with exception: {e}")
    
    # Test 4: Comparison with dual-modal
    try:
        results["Dual-Modal Comparison"] = compare_with_dual_modal()
    except Exception as e:
        print(f"\n❌ Test 4 FAILED with exception: {e}")
    
    # Final summary
    print("\n" + "="*80)
    print(" FINAL AUDIT SUMMARY")
    print("="*80)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ AUDIT RESULT: PASS")
        print("\nConclusion:")
        print("  ✓ Single-modal discriminator is CORRECTLY IMPLEMENTED")
        print("  ✓ Architecture is TRULY single-modal (CNN-only, no text/LSTM)")
        print("  ✓ Safe to use for ablation study comparison")
        print("  ✓ Fair comparison dengan dual-modal discriminator")
    else:
        print("❌ AUDIT RESULT: FAIL")
        print("\nConclusion:")
        print("  ✗ Single-modal discriminator has implementation issues")
        print("  ✗ May contain text/LSTM components (not truly single-modal)")
        print("  ✗ NOT suitable for ablation study without fixes")
    print("="*80)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
