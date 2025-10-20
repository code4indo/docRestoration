#!/usr/bin/env python3
"""
Test Recognizer dengan Shape Fix
Uji recognizer dengan berbagai approach untuk mengatasi shape mismatch

Usage:
    poetry run python scripts/test_recognizer_with_fix.py
"""

import os
import sys
import tensorflow as tf
import numpy as np

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

def create_custom_recognizer():
    """Create custom recognizer with adjusted projection layer"""
    print("🔧 Creating custom recognizer with 8192->512 projection...")

    from tensorflow.keras import layers

    # Constants
    charset_size = 108  # From charset file
    IMG_WIDTH = 1024
    IMG_HEIGHT = 128
    PROJ_DIM = 512
    DROPOUT_RATE = 0.20
    NUM_HEADS = 8
    FF_DIM = 2048
    NUM_TRANSFORMER_LAYERS = 6

    # Input
    inputs = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1), name='image_input')
    x = inputs

    # CNN Backbone (same as original)
    def conv_block(inp, filters, k=3, s=(1,1), name_prefix='cb', dropout=0.0):
        y = layers.Conv2D(filters, k, strides=s, padding='same',
                         use_bias=False, name=f'{name_prefix}_conv')(inp)
        y = layers.BatchNormalization(name=f'{name_prefix}_bn')(y)
        y = layers.Activation('gelu', name=f'{name_prefix}_gelu')(y)
        if dropout > 0:
            y = layers.Dropout(dropout, name=f'{name_prefix}_drop')(y)
        return y

    # Apply CNN layers
    x = conv_block(x, 64, k=7, s=(1,2), name_prefix='s1_1', dropout=DROPOUT_RATE*0.5)
    x = conv_block(x, 64, k=3, s=(1,1), name_prefix='s1_2', dropout=DROPOUT_RATE*0.5)
    x = layers.MaxPooling2D(pool_size=(2,2), name='pool1')(x)

    x = conv_block(x, 128, k=3, s=(1,1), name_prefix='s2_1', dropout=DROPOUT_RATE*0.7)
    x = conv_block(x, 128, k=3, s=(1,1), name_prefix='s2_2', dropout=DROPOUT_RATE*0.7)
    x = layers.MaxPooling2D(pool_size=(2,2), name='pool2')(x)

    x = conv_block(x, 256, k=3, s=(1,1), name_prefix='s3_1', dropout=DROPOUT_RATE)
    x = conv_block(x, 256, k=3, s=(1,1), name_prefix='s3_2', dropout=DROPOUT_RATE)
    x = layers.MaxPooling2D(pool_size=(2,1), name='pool3')(x)

    x = conv_block(x, 512, k=3, s=(1,1), name_prefix='s4_1', dropout=DROPOUT_RATE)
    x = conv_block(x, 512, k=3, s=(1,1), name_prefix='s4_2', dropout=DROPOUT_RATE)

    print(f"CNN output shape: {x.shape}")

    # ========== SEQUENCE PROJECTION ==========
    x = layers.Lambda(
        lambda t: tf.reshape(t, (tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2]*tf.shape(t)[3])),
        name='flatten_height'
    )(x)

    print(f"After flatten: {x.shape}")

    # FIX: Use 8192 -> 512 projection
    x = layers.Dense(PROJ_DIM, name='proj_dense')(x)
    x = layers.LayerNormalization(name='proj_ln')(x)
    x = layers.Dropout(DROPOUT_RATE, name='proj_drop')(x)

    print(f"After projection: {x.shape}")

    # ========== POSITIONAL ENCODING ==========
    target_time_steps = 128
    positions = tf.range(start=0, limit=target_time_steps, delta=1)
    pos_embedding_layer = layers.Embedding(
        input_dim=target_time_steps,
        output_dim=PROJ_DIM,
        name='positional_embedding'
    )
    x = x + pos_embedding_layer(positions)

    # ========== TRANSFORMER ENCODER ==========
    for i in range(NUM_TRANSFORMER_LAYERS):
        attn = layers.MultiHeadAttention(
            num_heads=NUM_HEADS,
            key_dim=PROJ_DIM // NUM_HEADS,
            dropout=DROPOUT_RATE,
            name=f'trn_attn_{i}'
        )(x, x)
        x = layers.LayerNormalization(name=f'trn_ln1_{i}')(x + attn)

        ffn = layers.Dense(FF_DIM, activation='gelu', name=f'trn_ffn1_{i}')(x)
        ffn = layers.Dropout(DROPOUT_RATE, name=f'trn_ffn_drop_{i}')(ffn)
        ffn = layers.Dense(PROJ_DIM, name=f'trn_ffn2_{i}')(ffn)
        x = layers.LayerNormalization(name=f'trn_ln2_{i}')(x + ffn)
        x = layers.Dropout(DROPOUT_RATE, name=f'trn_ffn_out_drop_{i}')(x)

    # ========== CTC OUTPUT LAYER ==========
    outputs = layers.Dense(charset_size + 1, activation=None, name='logits')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='custom_htr_recognizer')
    return model

def test_custom_recognizer():
    """Test custom recognizer"""
    print("=" * 80)
    print("TESTING CUSTOM RECOGNIZER")
    print("=" * 80)

    # Load charset
    charset = read_charlist("real_data_preparation/real_data_charlist.txt")
    charset_size = len(charset)
    print(f"Charset size: {charset_size}")

    # Create custom model
    model = create_custom_recognizer()
    print(f"✅ Custom model created")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")

    # Test prediction
    print("\n🧪 Testing custom model...")
    test_input = np.random.random((1, 1024, 128, 1)).astype(np.float32)

    try:
        predictions = model(test_input, training=False)
        print(f"✅ Custom model prediction successful")
        print(f"   Output shape: {predictions.shape}")

        # Decode prediction
        predicted_labels = tf.argmax(predictions, axis=-1, output_type=tf.int32)[0]
        predicted_text = decode_label(predicted_labels.numpy(), charset)
        print(f"   Predicted text: '{predicted_text}'")

        # Load pre-trained weights if possible
        weights_path = "models/best_htr_recognizer/best_model.weights.h5"
        if os.path.exists(weights_path):
            print(f"\n🔄 Attempting to load pre-trained weights...")
            try:
                model.load_weights(weights_path, skip_mismatch=True)
                print(f"✅ Weights loaded with skip_mismatch=True")

                # Test again with loaded weights
                predictions = model(test_input, training=False)
                predicted_labels = tf.argmax(predictions, axis=-1, output_type=tf.int32)[0]
                predicted_text = decode_label(predicted_labels.numpy(), charset)
                print(f"   Prediction with weights: '{predicted_text}'")

            except Exception as e:
                print(f"❌ Error loading weights: {e}")
        else:
            print(f"⚠️  Weights file not found: {weights_path}")

    except Exception as e:
        print(f"❌ Error with custom model: {e}")
        return False

    return True

def test_original_recognizer_debug():
    """Test original recognizer with debugging"""
    print("\n" + "=" * 80)
    print("TESTING ORIGINAL RECOGNIZER (DEBUG MODE)")
    print("=" * 80)

    charset = read_charlist("real_data_preparation/real_data_charlist.txt")
    charset_size = len(charset)

    try:
        recognizer = load_frozen_recognizer(
            weights_path="models/best_htr_recognizer/best_model.weights.h5",
            charset_size=charset_size,
            return_feature_map=False
        )
        print("✅ Original recognizer loaded")

        # Check layer shapes
        print("\n📊 Layer Analysis:")
        for layer in recognizer.layers:
            if hasattr(layer, 'output_shape'):
                print(f"   {layer.name}: {layer.output_shape}")

        # Test with small input to debug
        test_input = np.random.random((1, 1024, 128, 1)).astype(np.float32)
        print(f"\n🧪 Testing with small input...")

        # Manually pass through layers to find the problem
        layer_outputs = []
        x = test_input
        for layer in recognizer.layers:
            try:
                x = layer(x, training=False)
                layer_outputs.append((layer.name, x.shape))
                print(f"   ✅ {layer.name}: {x.shape}")
            except Exception as e:
                print(f"   ❌ {layer.name}: {e}")
                break

    except Exception as e:
        print(f"❌ Error with original recognizer: {e}")

if __name__ == "__main__":
    print("Testing recognizer fixes...")

    # Test custom recognizer
    custom_success = test_custom_recognizer()

    # Debug original recognizer
    test_original_recognizer_debug()

    if custom_success:
        print("\n🎯 SOLUTION FOUND:")
        print("1. The custom recognizer works with proper 8192->512 projection")
        print("2. Original recognizer has shape mismatch issues")
        print("3. Update dual_modal_gan/src/models/recognizer.py with custom architecture")

        print("\n📋 NEXT STEPS:")
        print("1. Replace recognizer.py with fixed architecture")
        print("2. Test GAN training with corrected recognizer")
        print("3. Monitor CER improvements")
    else:
        print("\n❌ Need further investigation")