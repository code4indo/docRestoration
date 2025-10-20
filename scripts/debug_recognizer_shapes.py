#!/usr/bin/env python3
"""
Debug Recognizer Shapes Script
Identifikasi shape mismatch di CNN backbone

Usage:
    poetry run python scripts/debug_recognizer_shapes.py
"""

import os
import sys
import tensorflow as tf
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def read_charlist(path):
    """Read character list from file"""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

def test_cnn_shapes():
    """Test CNN backbone shapes step by step"""
    print("=" * 80)
    print("DEBUG CNN BACKBONE SHAPES")
    print("=" * 80)

    # Load charset
    charset = read_charlist("real_data_preparation/real_data_charlist.txt")
    charset_size = len(charset)
    print(f"Charset size: {charset_size}")

    # Import layers from recognizer
    from tensorflow.keras import layers

    # Constants from recognizer
    IMG_WIDTH = 1024
    IMG_HEIGHT = 128
    PROJ_DIM = 512
    DROPOUT_RATE = 0.20

    # Create test input
    inputs = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1), name='image_input')
    print(f"Input shape: {inputs.shape}")

    # Define conv_block from recognizer
    def conv_block(inp, filters, k=3, s=(1,1), name_prefix='cb', dropout=0.0):
        """Conv block with BatchNorm matching Stage 3"""
        y = layers.Conv2D(filters, k, strides=s, padding='same',
                         use_bias=False, name=f'{name_prefix}_conv')(inp)
        y = layers.BatchNormalization(name=f'{name_prefix}_bn')(y)
        y = layers.Activation('gelu', name=f'{name_prefix}_gelu')(y)
        if dropout > 0:
            y = layers.Dropout(dropout, name=f'{name_prefix}_drop')(y)
        return y

    x = inputs
    print(f"After input: {x.shape}")

    # Apply CNN layers step by step and print shapes
    x = conv_block(x, 64, k=7, s=(1,2), name_prefix='s1_1', dropout=DROPOUT_RATE*0.5)
    print(f"After s1_1 (conv7x7, stride=(1,2)): {x.shape}")

    x = conv_block(x, 64, k=3, s=(1,1), name_prefix='s1_2', dropout=DROPOUT_RATE*0.5)
    print(f"After s1_2 (conv3x3, stride=(1,1)): {x.shape}")

    x = layers.MaxPooling2D(pool_size=(2,2), name='pool1')(x)
    print(f"After pool1 (2x2): {x.shape}")

    x = conv_block(x, 128, k=3, s=(1,1), name_prefix='s2_1', dropout=DROPOUT_RATE*0.7)
    print(f"After s2_1 (conv3x3): {x.shape}")

    x = conv_block(x, 128, k=3, s=(1,1), name_prefix='s2_2', dropout=DROPOUT_RATE*0.7)
    print(f"After s2_2 (conv3x3): {x.shape}")

    x = layers.MaxPooling2D(pool_size=(2,2), name='pool2')(x)
    print(f"After pool2 (2x2): {x.shape}")

    x = conv_block(x, 256, k=3, s=(1,1), name_prefix='s3_1', dropout=DROPOUT_RATE)
    print(f"After s3_1 (conv3x3): {x.shape}")

    x = conv_block(x, 256, k=3, s=(1,1), name_prefix='s3_2', dropout=DROPOUT_RATE)
    print(f"After s3_2 (conv3x3): {x.shape}")

    x = layers.MaxPooling2D(pool_size=(2,1), name='pool3')(x)
    print(f"After pool3 (2x1): {x.shape}")

    x = conv_block(x, 512, k=3, s=(1,1), name_prefix='s4_1', dropout=DROPOUT_RATE)
    print(f"After s4_1 (conv3x3): {x.shape}")

    x = conv_block(x, 512, k=3, s=(1,1), name_prefix='s4_2', dropout=DROPOUT_RATE)
    print(f"After s4_2 (conv3x3): {x.shape}")

    # ========== SEQUENCE PROJECTION ==========
    print(f"\nBefore reshape: {x.shape}")

    # Check the flatten operation
    x_flat = layers.Lambda(
        lambda t: tf.reshape(t, (tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2]*tf.shape(t)[3])),
        name='flatten_height'
    )(x)

    print(f"After flatten_height: {x_flat.shape}")

    # Check projection
    x_proj = layers.Dense(PROJ_DIM, name='proj_dense')(x_flat)
    print(f"After proj_dense: {x_proj.shape}")
    print(f"Expected proj_dim: {PROJ_DIM}")

    # Create test model
    test_model = tf.keras.Model(inputs=inputs, outputs=x_proj, name='cnn_backbone_test')

    # Test with actual data
    print("\n🧪 Testing with actual data...")
    test_input = np.random.random((1, 1024, 128, 1)).astype(np.float32)
    print(f"Test input shape: {test_input.shape}")

    try:
        output = test_model(test_input, training=False)
        print(f"✅ CNN backbone test successful")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{tf.reduce_min(output):.3f}, {tf.reduce_max(output):.3f}]")
    except Exception as e:
        print(f"❌ CNN backbone test failed: {e}")
        return False

    # Compare with expected from training script
    print(f"\n📊 Shape Analysis:")
    print(f"   Final CNN output: H={x.shape[1]}, W={x.shape[2]}, C={x.shape[3]}")
    print(f"   Flattened features: {x.shape[2] * x.shape[3]}")
    print(f"   Expected proj_dim: {PROJ_DIM}")

    if x.shape[2] * x.shape[3] != PROJ_DIM:
        print(f"   ❌ MISMATCH: CNN features ({x.shape[2] * x.shape[3]}) != proj_dim ({PROJ_DIM})")
        print(f"   This explains the shape error!")
        return False
    else:
        print(f"   ✅ CNN features match projection dimension")

    return True

def check_training_script_shapes():
    """Check shapes from the original training script"""
    print("\n" + "=" * 80)
    print("CHECKING TRAINING SCRIPT SHAPES")
    print("=" * 80)

    # Try to import from training script
    try:
        # Add path to training script directory
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

        # Read the training script to extract architecture
        with open('scripts/train_transformer_improved_v2.py', 'r') as f:
            content = f.read()

        # Look for key architectural parameters
        print("📋 Training script architecture parameters:")

        if 'IMG_WIDTH = 1024' in content:
            print("   ✅ IMG_WIDTH = 1024")
        if 'IMG_HEIGHT = 128' in content:
            print("   ✅ IMG_HEIGHT = 128")
        if 'proj_dim = 512' in content:
            print("   ✅ proj_dim = 512")
        if 'NUM_HEADS = 8' in content:
            print("   ✅ NUM_HEADS = 8")
        if 'FF_DIM = 2048' in content:
            print("   ✅ FF_DIM = 2048")

        print("\n🔍 Looking for CNN architecture...")

        # Check conv blocks
        if 'conv_block(x, 64, k=7, s=(1,2)' in content:
            print("   ✅ First conv block: 64 filters, 7x7, stride=(1,2)")
        if 'MaxPooling2D(pool_size=(2,2)' in content:
            print("   ✅ Uses 2x2 max pooling")
        if 'conv_block(x, 512' in content:
            print("   ✅ Uses 512 filters in final conv blocks")

    except Exception as e:
        print(f"❌ Error reading training script: {e}")

if __name__ == "__main__":
    print("Debugging recognizer shapes...")

    success = test_cnn_shapes()
    check_training_script_shapes()

    if not success:
        print("\n❌ SHAPE MISMATCH DETECTED!")
        print("The CNN output features don't match the projection dimension.")
        print("This is why the recognizer produces random text - it can't process the input correctly!")

        print("\n🔧 POSSIBLE SOLUTIONS:")
        print("1. Adjust CNN architecture to match training script")
        print("2. Modify projection dimension to match CNN output")
        print("3. Check if the loaded weights are for a different architecture")
    else:
        print("\n✅ Shapes look correct - issue might be elsewhere")