"""
Frozen, pre-trained Recognizer model for the Dual-Modal GAN.

Version 3: Replaces the model architecture with the correct ResNet-style
structure from `bck_train_transformer_simple_fixed_bfr_changeArsitektur.py`
(the true source file) to match the pre-trained weights. This also handles
the input shape transpose to align the GAN pipeline with the recognizer's expected input.
"""

import os
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Reshape,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    Embedding,
    Lambda,
    Activation
)
from tensorflow.keras.models import Model

# Constants from Stage 3 trained model (CER 33.72%)
IMG_WIDTH = 1024
IMG_HEIGHT = 128
NUM_HEADS = 8  # Stage 3 uses 8 heads
FF_DIM = 2048  # Stage 3 uses 2048 FFN dim
DROPOUT_RATE = 0.20  # Stage 3 uses 0.20 dropout
NUM_TRANSFORMER_LAYERS = 6  # Stage 3 uses 6 layers

def create_htr_model(charset_size, proj_dim=512, target_time_steps=128, 
                    num_transformer_layers=6, dropout_rate=0.20):
    """Creates the Transformer-based HTR model architecture.
    
    Architecture matches Stage 3 model from train_transformer_improved_v2.py
    (htr_improved_v2_20251001_221138, CER 33.72%)
    """
    from tensorflow.keras import layers
    
    # Input shape: (1024, 128, 1) - as trained
    inputs = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1), name='image_input')
    x = inputs
    
    # ========== CNN BACKBONE (Stage 3 architecture) ==========
    def conv_block(inp, filters, k=3, s=(1,1), name_prefix='cb', dropout=0.0):
        """Conv block with BatchNorm matching Stage 3"""
        y = layers.Conv2D(filters, k, strides=s, padding='same', 
                         use_bias=False, name=f'{name_prefix}_conv')(inp)
        y = layers.BatchNormalization(name=f'{name_prefix}_bn')(y)
        y = layers.Activation('gelu', name=f'{name_prefix}_gelu')(y)
        if dropout > 0:
            y = layers.Dropout(dropout, name=f'{name_prefix}_drop')(y)
        return y
    
    # Progressive feature extraction (Stage 3 config)
    x = conv_block(x, 64, k=7, s=(1,2), name_prefix='s1_1', dropout=dropout_rate*0.5)
    x = conv_block(x, 64, k=3, s=(1,1), name_prefix='s1_2', dropout=dropout_rate*0.5)
    x = layers.MaxPooling2D(pool_size=(2,2), name='pool1')(x)
    
    x = conv_block(x, 128, k=3, s=(1,1), name_prefix='s2_1', dropout=dropout_rate*0.7)
    x = conv_block(x, 128, k=3, s=(1,1), name_prefix='s2_2', dropout=dropout_rate*0.7)
    x = layers.MaxPooling2D(pool_size=(2,2), name='pool2')(x)
    
    x = conv_block(x, 256, k=3, s=(1,1), name_prefix='s3_1', dropout=dropout_rate)
    x = conv_block(x, 256, k=3, s=(1,1), name_prefix='s3_2', dropout=dropout_rate)
    x = layers.MaxPooling2D(pool_size=(2,1), name='pool3')(x)
    
    x = conv_block(x, 512, k=3, s=(1,1), name_prefix='s4_1', dropout=dropout_rate)
    x = conv_block(x, 512, k=3, s=(1,1), name_prefix='s4_2', dropout=dropout_rate)
    
    # ========== SEQUENCE PROJECTION ==========
    x = layers.Lambda(
        lambda t: tf.reshape(t, (tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2]*tf.shape(t)[3])), 
        name='flatten_height'
    )(x)
    
    x = layers.Dense(proj_dim, name='proj_dense')(x)
    x = layers.LayerNormalization(name='proj_ln')(x)
    x = layers.Dropout(dropout_rate, name='proj_drop')(x)
    
    # ========== POSITIONAL ENCODING ==========
    seq_len = target_time_steps
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_embedding_layer = layers.Embedding(
        input_dim=seq_len, 
        output_dim=proj_dim, 
        name='positional_embedding'
    )
    x = x + pos_embedding_layer(positions)
    
    # ========== TRANSFORMER ENCODER (6 layers for Stage 3) ==========
    for i in range(num_transformer_layers):
        # Multi-head attention
        attn = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, 
            key_dim=proj_dim // NUM_HEADS, 
            dropout=dropout_rate,
            name=f'trn_attn_{i}'
        )(x, x)
        x = layers.LayerNormalization(name=f'trn_ln1_{i}')(x + attn)
        
        # Feed-forward network
        ffn = layers.Dense(FF_DIM, activation='gelu', name=f'trn_ffn1_{i}')(x)
        ffn = layers.Dropout(dropout_rate, name=f'trn_ffn_drop_{i}')(ffn)
        ffn = layers.Dense(proj_dim, name=f'trn_ffn2_{i}')(ffn)
        x = layers.LayerNormalization(name=f'trn_ln2_{i}')(x + ffn)
        x = layers.Dropout(dropout_rate, name=f'trn_ffn_out_drop_{i}')(x)

    # ========== CTC OUTPUT LAYER ==========
    outputs = layers.Dense(charset_size + 1, activation=None, name='logits')(x)
    model = Model(inputs=inputs, outputs=outputs, name='htr_transformer_recognizer')
    return model

def load_frozen_recognizer(weights_path, charset_size, 
                          num_transformer_layers=NUM_TRANSFORMER_LAYERS,
                          dropout_rate=DROPOUT_RATE):
    """Loads the HTR model, applies pre-trained weights, and freezes it.
    
    Args:
        weights_path: Path to .weights.h5 file (Stage 3 model)
        charset_size: Number of characters (108 for our charset)
        num_transformer_layers: Number of transformer layers (6 for Stage 3)
        dropout_rate: Dropout rate (0.20 for Stage 3)
    """
    print(f"[Recognizer] Creating HTR model for {charset_size} characters...")
    print(f"[Recognizer] Architecture: {num_transformer_layers} layers, {NUM_HEADS} heads, "
          f"FFN dim {FF_DIM}, dropout {dropout_rate}")
    
    model = create_htr_model(
        charset_size=charset_size,
        num_transformer_layers=num_transformer_layers,
        dropout_rate=dropout_rate
    )

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Recognizer weights not found at: {weights_path}")

    print(f"[Recognizer] Loading weights from: {weights_path}")
    # Load weights (works for both .h5 and .weights.h5 formats)
    model.load_weights(weights_path)

    print("[Recognizer] Freezing model (setting trainable=False)...")
    model.trainable = False

    print("[Recognizer] Frozen HTR model ready (Stage 3, CER 33.72%).")
    return model
