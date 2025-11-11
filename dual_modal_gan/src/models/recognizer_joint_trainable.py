"""
HTR Recognizer - Joint Training Mode (TRAINABLE VERSION)

This is a modified version of recognizer_fixed.py that supports joint training
as proposed by Souibgui et al. for ablation study purposes.

⚠️ WARNING: This is for ABLATION STUDY ONLY
   Expected behavior: Catastrophic forgetting, gradient conflicts, instability

Key Differences from Frozen Version:
1. model.trainable = True (weights can be updated)
2. Exposed for optimizer gradient updates
3. Monitors recognizer drift from baseline CER (33.72%)

Reference: Souibgui et al. "Enhance to Read Better" (2021)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Bidirectional, GRU, Dense, 
    Reshape, Lambda, Dropout, BatchNormalization, LayerNormalization, 
    MultiHeadAttention, Add
)
import os


def gated_conv(x, filters, kernel_size, name, dropout_rate=0.2):
    """Gated convolution layer for feature extraction"""
    from tensorflow.keras.layers import Multiply
    
    # Linear convolution
    linear = Conv2D(filters, kernel_size, padding='same', name=f'{name}_linear')(x)
    # Gating convolution
    gate = Conv2D(filters, kernel_size, padding='same', activation='sigmoid', name=f'{name}_gate')(x)
    # Element-wise multiplication - use Keras Multiply layer instead of tf.multiply
    gated = Multiply(name=f'{name}_multiply')([linear, gate])
    # Batch normalization
    bn = BatchNormalization(name=f'{name}_bn')(gated)
    # Dropout
    if dropout_rate > 0:
        bn = Dropout(dropout_rate, name=f'{name}_dropout')(bn)
    return bn


def create_htr_model_joint_trainable(charset_size, img_height=128, img_width=1024):
    """
    Create HTR model for JOINT TRAINING (trainable=True)
    
    Architecture: CNN Encoder → Transformer Decoder → CTC
    Based on: Hybrid CNN-Transformer from htr_improved_v2
    
    ⚠️ This version is TRAINABLE for ablation study comparison
    
    Args:
        charset_size: Number of characters in charset
        img_height: Input image height (default: 128)
        img_width: Input image width (default: 1024)
    
    Returns:
        Model: Trainable HTR model (for joint training experiment)
    """
    
    inputs = Input(shape=(img_height, img_width, 1), name='input_image')
    
    # === CNN Encoder (Backbone) ===
    print(f"[Joint HTR] Building CNN encoder for joint training...")
    
    # Block 1: Initial feature extraction
    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_1')(inputs)
    x = BatchNormalization(name='bn1_1')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_2')(x)
    x = BatchNormalization(name='bn1_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)  # 64x512
    x = Dropout(0.2, name='dropout1')(x)
    
    # Block 2: Gated convolutions for text-aware features
    x = gated_conv(x, 64, (3, 3), name='gated_conv2_1', dropout_rate=0.2)
    x = gated_conv(x, 64, (3, 3), name='gated_conv2_2', dropout_rate=0.2)
    x = MaxPooling2D(pool_size=(2, 2), name='pool2')(x)  # 32x256
    
    # Block 3: Deep feature extraction
    x = gated_conv(x, 128, (3, 3), name='gated_conv3_1', dropout_rate=0.2)
    x = gated_conv(x, 128, (3, 3), name='gated_conv3_2', dropout_rate=0.2)
    x = MaxPooling2D(pool_size=(1, 2), name='pool3')(x)  # 32x128 (keep height for text)
    
    # Block 4: High-level features
    x = gated_conv(x, 256, (3, 3), name='gated_conv4_1', dropout_rate=0.3)
    x = gated_conv(x, 256, (3, 3), name='gated_conv4_2', dropout_rate=0.3)
    x = MaxPooling2D(pool_size=(1, 2), name='pool4')(x)  # 32x64
    
    # === Sequence Modeling (RNN/Transformer Hybrid) ===
    print(f"[Joint HTR] Building sequence decoder...")
    
    # Reshape for sequence processing: (batch, time_steps, features)
    # time_steps = width / (horizontal pooling factor) = 1024 / 16 = 64
    shape = tf.shape(x)
    x = Reshape((shape[2], shape[1] * shape[3]), name='reshape')(x)  # (batch, 64, 32*256)
    
    # Projection layer to reduce dimensionality
    proj_dim = 256
    x = Dense(proj_dim, activation='relu', name='proj_dense')(x)
    x = LayerNormalization(name='proj_ln')(x)
    x = Dropout(0.3, name='proj_dropout')(x)
    
    # Bidirectional GRU for temporal context
    x = Bidirectional(GRU(256, return_sequences=True, dropout=0.3), name='bi_gru1')(x)
    x = Bidirectional(GRU(256, return_sequences=True, dropout=0.3), name='bi_gru2')(x)
    
    # Transformer-style self-attention (optional enhancement)
    # Multi-head attention for long-range dependencies
    attn_output = MultiHeadAttention(
        num_heads=8, 
        key_dim=64,
        dropout=0.1,
        name='multi_head_attention'
    )(x, x)
    x = Add(name='attention_residual')([x, attn_output])
    x = LayerNormalization(name='attention_ln')(x)
    
    # === CTC Output Layer ===
    # Output dimension: charset_size + 1 (for CTC blank token)
    outputs = Dense(charset_size + 1, activation='softmax', name='ctc_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='htr_recognizer_joint_trainable')
    
    print(f"[Joint HTR] Model created for joint training:")
    print(f"   - Input shape: {inputs.shape}")
    print(f"   - Output shape: {outputs.shape}")
    print(f"   - Trainable: TRUE (joint training mode)")
    print(f"   - Total params: {model.count_params():,}")
    
    return model


def load_joint_trainable_recognizer(weights_path, charset_size,
                                     img_height=128, img_width=1024,
                                     return_feature_map=False):
    """
    Load HTR recognizer in JOINT TRAINING mode (trainable=True)
    
    ⚠️ CRITICAL: This is for ABLATION STUDY ONLY
       Expected behavior: Recognizer performance degradation due to:
       1. Catastrophic forgetting (pre-trained knowledge lost)
       2. Gradient conflicts with generator
       3. Training instability
    
    Args:
        weights_path: Path to pre-trained weights (.h5 or .weights.h5)
        charset_size: Number of characters
        img_height: Input height
        img_width: Input width
        return_feature_map: If True, return (logits, feature_map) for rec_feat loss
    
    Returns:
        Model: TRAINABLE recognizer (⚠️ will be updated during training)
    """
    
    print("\n" + "="*80)
    print("⚠️  LOADING RECOGNIZER IN JOINT TRAINING MODE (ABLATION STUDY)")
    print("="*80)
    print("This mode enables weight updates during GAN training.")
    print("Expected behavior: Catastrophic forgetting, instability, CER degradation")
    print("Baseline CER (frozen): 33.72%")
    print("Expected CER (joint): >40% (performance degradation)")
    print("="*80 + "\n")
    
    # Create model
    model = create_htr_model_joint_trainable(
        charset_size=charset_size,
        img_height=img_height,
        img_width=img_width
    )

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Recognizer weights not found at: {weights_path}")

    print(f"[Joint HTR] Loading pre-trained weights from: {weights_path}")
    # Load weights from frozen version
    try:
        model.load_weights(weights_path, skip_mismatch=True)
        print("[Joint HTR] ✅ Weights loaded successfully")
    except ValueError as e:
        print(f"[Joint HTR] ⚠️ Warning: Error loading weights: {e}")
        print("[Joint HTR] Attempting alternative loading method...")
        try:
            model.load_weights(weights_path, skip_mismatch=True, by_name=True)
            print("[Joint HTR] ✅ Weights loaded with by_name=True")
        except Exception as e2:
            print(f"[Joint HTR] ❌ Cannot load weights: {e2}")
            print("[Joint HTR] Continuing with random initialization (NOT RECOMMENDED)")

    # ⚠️ CRITICAL DIFFERENCE: KEEP TRAINABLE
    print("\n[Joint HTR] ⚠️  KEEPING MODEL TRAINABLE (joint training mode)")
    print("            This enables gradient updates from GAN loss")
    print("            Baseline CER 33.72% → Expected degradation to >40%\n")
    model.trainable = True  # ← KEY DIFFERENCE FROM FROZEN VERSION

    # If multi-output is requested, create a new model that outputs both logits and feature_map
    if return_feature_map:
        print("[Joint HTR] Creating multi-output model (logits + feature_map)...")
        # Find the feature layer (before transformer) - "proj_ln" is the projection layer
        feature_layer = model.get_layer('proj_ln').output

        # Create multi-output model
        multi_output_model = Model(
            inputs=model.input,
            outputs=[model.output, feature_layer],
            name='htr_recognizer_joint_trainable_multi_output'
        )
        multi_output_model.trainable = True  # ← KEEP TRAINABLE
        print("[Joint HTR] Multi-output model created: (logits, feature_map)")
        print(f"   - Logits shape: {model.output.shape}")
        print(f"   - Feature map shape: {feature_layer.shape}")
        print(f"   - Trainable: TRUE (joint training mode)")
        return multi_output_model

    print("[Joint HTR] ✅ Joint trainable HTR model ready")
    print("            Starting from pre-trained state (CER 33.72%)")
    print("            Will be updated during GAN training (expect degradation)\n")
    return model


# Convenience function for loading frozen version (for comparison)
def load_frozen_recognizer_for_comparison(weights_path, charset_size,
                                          img_height=128, img_width=1024,
                                          return_feature_map=False):
    """
    Load FROZEN recognizer for side-by-side comparison in ablation study
    This is the CONTROL condition (stable CER baseline)
    """
    model = create_htr_model_joint_trainable(
        charset_size=charset_size,
        img_height=img_height,
        img_width=img_width
    )
    
    if os.path.exists(weights_path):
        model.load_weights(weights_path, skip_mismatch=True)
    
    # FREEZE for control condition
    model.trainable = False
    print("[Frozen Control] Recognizer frozen (baseline CER 33.72% maintained)")
    
    if return_feature_map:
        feature_layer = model.get_layer('proj_ln').output
        multi_output_model = Model(
            inputs=model.input,
            outputs=[model.output, feature_layer],
            name='htr_recognizer_frozen_control'
        )
        multi_output_model.trainable = False
        return multi_output_model
    
    return model
