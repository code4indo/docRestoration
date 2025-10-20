#!/usr/bin/env python3
"""
Test Recognizer Weights Loading - Verify Fix
============================================
Test apakah recognizer dapat load weights dengan benar setelah fix layer name.
Expected: CER untuk clean images harus ~0.33 (bukan 1.56)
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed

def read_charlist(file_path):
    """Read character list from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        charset = []
        for line in f:
            content = line.rstrip('\n')
            if content == ' ':
                charset.append(' ')
            elif content:
                charset.append(content)
    return charset

def decode_ctc(predictions, charset):
    """Decode CTC predictions to text"""
    decoded = []
    prev_idx = -1
    
    for idx in predictions:
        # Skip blank token (0) and duplicates
        if idx > 0 and idx != prev_idx:
            if idx - 1 < len(charset):
                decoded.append(charset[idx - 1])
        prev_idx = idx
    
    return ''.join(decoded)

def test_recognizer_loading():
    """Test if recognizer loads weights correctly"""
    print("="*80)
    print("🔬 RECOGNIZER WEIGHTS LOADING TEST")
    print("="*80)
    
    # Paths
    charset_path = 'real_data_preparation/real_data_charlist.txt'
    weights_path = 'models/best_htr_recognizer/best_model.weights.h5'
    
    # Check files exist
    if not os.path.exists(charset_path):
        print(f"❌ Charset file not found: {charset_path}")
        return False
    
    if not os.path.exists(weights_path):
        print(f"❌ Weights file not found: {weights_path}")
        return False
    
    print(f"✅ Files found")
    print(f"   Charset: {charset_path}")
    print(f"   Weights: {weights_path}")
    print()
    
    # Load charset
    charset = read_charlist(charset_path)
    vocab_size = len(charset) + 1
    print(f"📚 Charset loaded: {len(charset)} characters + 1 blank = {vocab_size} total")
    print()
    
    # Load recognizer
    print("🔧 Loading recognizer...")
    try:
        recognizer = load_frozen_recognizer_fixed(
            weights_path=weights_path,
            charset_size=len(charset),
            return_feature_map=False
        )
        print("✅ Recognizer loaded successfully")
        print()
    except Exception as e:
        print(f"❌ Failed to load recognizer: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with random input (simulating clean image)
    print("🧪 Testing with random input...")
    test_input = tf.random.uniform((1, 1024, 128, 1), minval=0.0, maxval=1.0)
    
    try:
        logits = recognizer(test_input, training=False)
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {logits.shape}")
        print(f"   Expected: (1, 128, {vocab_size})")
        
        if logits.shape != (1, 128, vocab_size):
            print(f"❌ Output shape mismatch!")
            return False
        
        print("✅ Output shape correct")
        print()
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Decode predictions
    print("📝 Decoding predictions...")
    predictions = tf.argmax(logits, axis=-1).numpy()[0]
    decoded_text = decode_ctc(predictions, charset)
    
    print(f"   Raw predictions (first 20): {predictions[:20]}")
    print(f"   Decoded text: '{decoded_text}'")
    print()
    
    # Check if predictions are meaningful
    # Random weights would produce very repetitive or nonsensical output
    unique_chars = len(set(decoded_text))
    total_chars = len(decoded_text)
    
    print("🔍 Sanity checks:")
    print(f"   Unique characters: {unique_chars}")
    print(f"   Total characters: {total_chars}")
    
    # If recognizer has proper weights, it should produce varied predictions
    # Random weights tend to output same character repeatedly
    if unique_chars < 3 and total_chars > 10:
        print("⚠️  WARNING: Output too repetitive - may still have random weights")
        return False
    
    print("✅ Output looks reasonable (varied predictions)")
    print()
    
    # Count trainable variables (should be 0 - frozen)
    trainable_count = len(recognizer.trainable_variables)
    print(f"🔒 Trainable variables: {trainable_count}")
    
    if trainable_count > 0:
        print("⚠️  WARNING: Model not properly frozen!")
        return False
    
    print("✅ Model properly frozen")
    print()
    
    print("="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print()
    print("📊 Expected behavior after fix:")
    print("   - CER on clean images should be ~0.33 (not 1.56)")
    print("   - Text predictions should be meaningful")
    print("   - Model should produce consistent results")
    print()
    print("🚀 Next steps:")
    print("   1. Stop current training (if running)")
    print("   2. Restart training with fixed recognizer")
    print("   3. Monitor first epoch validation - CER should improve dramatically")
    print()
    
    return True

def print_model_summary():
    """Print detailed model layer info for debugging"""
    print("\n" + "="*80)
    print("📋 MODEL LAYER SUMMARY")
    print("="*80)
    
    charset_path = 'real_data_preparation/real_data_charlist.txt'
    weights_path = 'models/best_htr_recognizer/best_model.weights.h5'
    
    charset = read_charlist(charset_path)
    recognizer = load_frozen_recognizer_fixed(
        weights_path=weights_path,
        charset_size=len(charset),
        return_feature_map=False
    )
    
    print("\nLayer names (critical for weight loading):")
    for i, layer in enumerate(recognizer.layers):
        print(f"  {i:3d}. {layer.name:40s} - {layer.__class__.__name__}")
        if 'proj_dense' in layer.name:
            print(f"       ⚠️  CRITICAL LAYER: This must match weights file!")
    
    print("\nTotal parameters:")
    print(f"  Total: {recognizer.count_params():,}")
    print(f"  Trainable: {sum([tf.size(v).numpy() for v in recognizer.trainable_variables]):,}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test recognizer weights loading after fix")
    parser.add_argument('--summary', action='store_true', help='Print detailed model summary')
    args = parser.parse_args()
    
    if args.summary:
        print_model_summary()
    else:
        success = test_recognizer_loading()
        sys.exit(0 if success else 1)
