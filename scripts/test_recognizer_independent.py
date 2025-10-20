#!/usr/bin/env python3
"""
Independent Recognizer Test Script
Uji frozen recognizer untuk mengidentifikasi masalah teks random

Usage:
    poetry run python scripts/test_recognizer_independent.py
"""

import os
import sys
import tensorflow as tf
import numpy as np
import cv2

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dual_modal_gan.src.models.recognizer import load_frozen_recognizer

def read_charlist(path):
    """Read character list from file"""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

def decode_label(label_ids, charset):
    """Decode label IDs to text string"""
    decoded_chars = []
    prev_id = -1
    for label_id in label_ids:
        label_id = int(label_id)
        # Skip blank token (0) and CTC duplicates
        if label_id > 0 and label_id != prev_id:
            if label_id - 1 < len(charset):
                decoded_chars.append(charset[label_id - 1])
        prev_id = label_id
    return ''.join(decoded_chars)

def create_test_image():
    """Create a simple test image with text"""
    # Create blank image (128x1024)
    img = np.zeros((128, 1024, 1), dtype=np.float32)

    # Add some noise to simulate document
    noise = np.random.normal(0, 0.1, (128, 1024, 1))
    img = img + noise
    img = np.clip(img, 0, 1)

    return img

def test_recognizer():
    """Test frozen recognizer independently"""
    print("=" * 80)
    print("INDEPENDENT RECOGNIZER TEST")
    print("=" * 80)

    # Paths
    charset_path = "real_data_preparation/real_data_charlist.txt"
    weights_path = "models/best_htr_recognizer/best_model.weights.h5"

    # Check if files exist
    if not os.path.exists(charset_path):
        print(f"❌ Charset file not found: {charset_path}")
        return False

    if not os.path.exists(weights_path):
        print(f"❌ Weights file not found: {weights_path}")
        return False

    # Load charset
    charset = read_charlist(charset_path)
    print(f"✅ Charset loaded: {len(charset)} characters")
    print(f"   First 10 chars: {charset[:10]}")
    print(f"   Last 10 chars: {charset[-10:]}")

    # Model parameters
    charset_size = len(charset)  # 109
    vocab_size = charset_size + 1  # 110 (including blank)
    print(f"   Charset size: {charset_size}")
    print(f"   Vocab size (with blank): {vocab_size}")

    # Load recognizer
    print("\n🔄 Loading frozen recognizer...")
    try:
        recognizer = load_frozen_recognizer(
            weights_path=weights_path,
            charset_size=charset_size,
            return_feature_map=False
        )
        print("✅ Recognizer loaded successfully")
        print(f"   Model input shape: {recognizer.input_shape}")
        print(f"   Model output shape: {recognizer.output_shape}")
    except Exception as e:
        print(f"❌ Error loading recognizer: {e}")
        return False

    # Test with dummy input
    print("\n🧪 Testing with dummy input...")
    test_input = create_test_image()
    test_batch = np.expand_dims(test_input, axis=0)  # Add batch dimension

    print(f"   Input shape: {test_batch.shape}")
    print(f"   Input range: [{test_batch.min():.3f}, {test_batch.max():.3f}]")

    try:
        # Get predictions
        predictions = recognizer(test_input, training=False)
        print(f"✅ Prediction successful")
        print(f"   Output shape: {predictions.shape}")
        print(f"   Output range: [{tf.reduce_min(predictions):.3f}, {tf.reduce_max(predictions):.3f}]")

        # Decode predictions
        predicted_labels = tf.argmax(predictions, axis=-1, output_type=tf.int32)[0]
        predicted_text = decode_label(predicted_labels.numpy(), charset)

        print(f"   Predicted text: '{predicted_text}'")
        print(f"   Text length: {len(predicted_text)}")

        # Analyze prediction quality
        if len(predicted_text) == 0:
            print("   ⚠️  Warning: Empty prediction (all blank tokens)")
        elif len(set(predicted_text)) == 1 and len(predicted_text) > 10:
            print(f"   ⚠️  Warning: Repetitive character detected: '{predicted_text[0]}'")
        else:
            print(f"   ✅ Text looks reasonable")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return False

    # Test with multiple samples
    print("\n🧪 Testing with multiple random samples...")
    for i in range(3):
        test_input = create_test_image()
        test_batch = np.expand_dims(test_input, axis=0)

        try:
            predictions = recognizer(test_input, training=False)
            predicted_labels = tf.argmax(predictions, axis=-1, output_type=tf.int32)[0]
            predicted_text = decode_label(predicted_labels.numpy(), charset)
            print(f"   Sample {i+1}: '{predicted_text}' (length: {len(predicted_text)})")
        except Exception as e:
            print(f"   Sample {i+1}: Error - {e}")

    # Test model weights info
    print("\n📊 Model Information:")
    total_params = recognizer.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in recognizer.trainable_weights])
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen: {trainable_params == 0}")

    # Check final layer
    final_layer = recognizer.get_layer('logits')
    print(f"   Final layer units: {final_layer.units}")
    print(f"   Expected units: {vocab_size}")

    if final_layer.units != vocab_size:
        print(f"   ❌ MISMATCH: Final layer has {final_layer.units} units, expected {vocab_size}")
        return False
    else:
        print(f"   ✅ Final layer size matches expected vocab size")

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)
    return True

def check_model_compatibility():
    """Check if model weights are compatible with architecture"""
    print("\n🔍 CHECKING MODEL COMPATIBILITY...")

    weights_path = "models/best_htr_recognizer/best_model.weights.h5"

    try:
        # Load weights file to inspect
        with tf.io.gfile.GFile(weights_path, 'rb') as f:
            header = f.read(20)  # Read first 20 bytes

        print(f"   Weights file exists and readable")

        # Try to load weights into a temporary model
        charset = read_charlist("real_data_preparation/real_data_charlist.txt")
        charset_size = len(charset)

        temp_model = load_frozen_recognizer(
            weights_path=weights_path,
            charset_size=charset_size,
            return_feature_map=False
        )

        # Check if weights were actually loaded
        layer_count = len(temp_model.layers)
        print(f"   Model has {layer_count} layers")

        # Check some key layers
        key_layers = ['conv2d', 'dense', 'multi_head_attention']
        for layer_type in key_layers:
            matching_layers = [l for l in temp_model.layers if layer_type in l.name.lower()]
            if matching_layers:
                print(f"   Found {len(matching_layers)} {layer_type} layers")

        print("   ✅ Model compatibility check passed")

    except Exception as e:
        print(f"   ❌ Compatibility check failed: {e}")
        return False

    return True

if __name__ == "__main__":
    print("Starting independent recognizer test...")

    # Run basic recognizer test
    success = test_recognizer()

    if success:
        # Run additional compatibility check
        check_model_compatibility()

        print("\n🎯 NEXT STEPS:")
        print("1. If predictions are still random, the issue might be:")
        print("   - Model weights not loaded properly")
        print("   - Input preprocessing mismatch")
        print("   - Character encoding issue")
        print("2. Compare with training script predictions")
        print("3. Test with real data instead of random noise")

    else:
        print("\n❌ CRITICAL ISSUES FOUND:")
        print("1. Fix model loading issues first")
        print("2. Verify weight file integrity")
        print("3. Check charset consistency")