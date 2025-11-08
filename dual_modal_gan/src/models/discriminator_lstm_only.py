"""
LSTM-Only Discriminator - Ablation Study Component.

Architecture untuk melengkapi ablation study discriminator:
- CNN-only (visual) ✓
- LSTM-only (text sequential) ← THIS FILE
- Dual-Modal (CNN + LSTM) ✓

Design Philosophy:
- Pure text sequence discriminator without visual features
- Relies solely on HTR recognizer predictions (text modal)
- Uses Bidirectional LSTM + Self-Attention for sequence modeling
- Parameter count matched to ~19M for fair comparison

Expected Challenge:
- Text input from generator predictions (CER ~27%)
- No visual guidance to detect image quality issues
- Expected to perform WORSE than CNN-only due to noisy text input

Scientific Value:
- Completes ablation study trifecta
- Demonstrates importance of visual modal
- Validates dual-modal architecture by showing both modals needed

Author: AI Research Assistant
Date: November 2, 2025
Purpose: H2A Ablation Study - LSTM-only baseline
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Bidirectional,
    LSTM,
    Dropout,
    BatchNormalization,
    LayerNormalization,
    Add,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Concatenate,
    Layer
)
from tensorflow.keras.models import Model


class SelfAttentionText(Layer):
    """
    Self-attention mechanism for text sequence.
    
    Allows model to weight character importance based on context.
    Identical to dual-modal version for consistency.
    """

    def __init__(self, **kwargs):
        super(SelfAttentionText, self).__init__(**kwargs)

    def build(self, input_shape):
        features = input_shape[-1]

        # Query, Key, Value projection layers
        self.query_dense = Dense(features, name='query')
        self.key_dense = Dense(features, name='key')
        self.value_dense = Dense(features, name='value')

        self.scale = tf.sqrt(tf.cast(features, tf.float32))
        super(SelfAttentionText, self).build(input_shape)

    def call(self, x):
        """
        Args:
            x: Input sequence (batch, seq_len, features)

        Returns:
            Attention-weighted features
        """
        # Query, Key, Value projections
        query = self.query_dense(x)
        key = self.key_dense(x)
        value = self.value_dense(x)

        # Scaled dot-product attention
        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / self.scale
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Apply attention to values
        attended = tf.matmul(attention_weights, value)

        return attended


def build_lstm_only_discriminator(
    text_shape=(32, 256),  # (max_text_len, vocab_size)
    config=None
):
    """
    Build LSTM-Only Discriminator untuk ablation study.
    
    Pure sequential text discriminator tanpa visual features.
    
    Architecture:
    - Input: Text sequence dari recognizer predictions (noisy, CER ~27%)
    - Processing: Bidirectional LSTM (2 layers) + Self-Attention
    - Output: Real/Fake validity score
    
    Challenge:
    - Text input quality terbatas oleh generator predictions
    - No visual guidance untuk detect image artifacts
    - Expected lower performance vs CNN-only
    
    Args:
        text_shape: (max_text_len, vocab_size) - Text sequence dimensions
        config: Configuration dictionary
        
    Returns:
        tf.keras.Model: LSTM-only discriminator
    """
    
    # Parse configuration
    if config is None:
        config = {}
    
    lstm_units = config.get('lstm_units', 512)  # Increased to match ~19M parameters
    dropout_rate = config.get('dropout_rate', 0.3)
    use_layer_norm = config.get('use_layer_norm', True)
    use_self_attention = config.get('use_self_attention', True)
    
    print("\n" + "="*80)
    print("BUILDING LSTM-ONLY DISCRIMINATOR")
    print("ABLATION STUDY: Text-Only Modal (No Visual Features)")
    print("="*80)
    print(f"Configuration:")
    print(f"  ✓ Text input shape: {text_shape}")
    print(f"  ✓ LSTM units: {lstm_units}")
    print(f"  ✓ Dropout rate: {dropout_rate}")
    print(f"  ✓ Layer normalization: {use_layer_norm}")
    print(f"  ✓ Self-attention: {use_self_attention}")
    
    # ========================================================================
    # TEXT BRANCH: Bidirectional LSTM + Self-Attention
    # ========================================================================
    print("\n[1/2] Building Text Sequential Processing Branch...")
    
    text_input = Input(shape=text_shape, name='text_input')
    
    # First Bidirectional LSTM layer
    text = Bidirectional(
        LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        name='bi_lstm_1'
    )(text_input)
    
    if use_layer_norm:
        text = LayerNormalization()(text)
    else:
        text = BatchNormalization()(text)
    
    text = Dropout(dropout_rate)(text)
    
    # Second Bidirectional LSTM layer (deeper processing)
    text = Bidirectional(
        LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        name='bi_lstm_2'
    )(text)
    
    if use_layer_norm:
        text = LayerNormalization()(text)
    else:
        text = BatchNormalization()(text)
    
    text = Dropout(dropout_rate)(text)
    
    # Self-attention mechanism (optional)
    if use_self_attention:
        print("   Applying self-attention to text sequence...")
        text_attended = SelfAttentionText()(text)
        
        # Residual connection
        text = Add()([text, text_attended])
        
        if use_layer_norm:
            text = LayerNormalization()(text)
    
    # Global pooling to get fixed-size representation
    # Use both average and max pooling for richer representation
    text_avg = GlobalAveragePooling1D()(text)
    text_max = GlobalMaxPooling1D()(text)
    
    text_features = Concatenate()([text_avg, text_max])  # Shape: (lstm_units*4,)
    
    print(f"   Text features shape: {lstm_units * 4} (from {lstm_units * 2} avg + {lstm_units * 2} max)")
    
    # ========================================================================
    # CLASSIFICATION HEAD: Dense layers for real/fake prediction
    # ========================================================================
    print("\n[2/2] Building Classification Head...")
    
    # Match parameter count dengan CNN-only (~19M)
    # Need larger dense layers since we don't have image branch
    
    x = Dense(2048, kernel_initializer='he_normal')(text_features)
    x = LayerNormalization()(x) if use_layer_norm else BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(1024, kernel_initializer='he_normal')(x)
    x = LayerNormalization()(x) if use_layer_norm else BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(512, kernel_initializer='he_normal')(x)
    x = LayerNormalization()(x) if use_layer_norm else BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = Dropout(dropout_rate)(x)
    
    # Output: real/fake score
    validity = Dense(1, activation='sigmoid', name='validity_score')(x)
    
    # ========================================================================
    # Create Model
    # ========================================================================
    model = Model(
        inputs=text_input,
        outputs=validity,
        name='lstm_only_discriminator'
    )
    
    print("\n" + "="*80)
    print("LSTM-ONLY DISCRIMINATOR SUMMARY")
    print("="*80)
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    
    print("\n" + "="*80)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Target (CNN-only): ~19,700,000")
    print(f"Parameter Difference: {abs(19_700_000 - total_params):,}")
    print(f"Fair Comparison: {'✅ YES' if abs(19_700_000 - total_params) < 2_000_000 else '❌ NO (adjust architecture)'}")
    print("="*80)
    
    print("\n✅ LSTM-Only Discriminator built successfully!")
    print("\nEXPERIMENT DESIGN:")
    print("  🎯 CNN-only: Visual features only")
    print("  🎯 LSTM-only (THIS): Text sequence only")
    print("  🎯 Dual-Modal: CNN + LSTM (combined)")
    print("  📊 Expected: LSTM-only < CNN-only < Dual-Modal")
    print("  📊 Reason: Text from generator is noisy (CER ~27%)")
    print("  📊 Validates: Visual modal is critical for quality")
    print()
    
    return model


if __name__ == "__main__":
    # Test build
    print("Testing LSTM-Only Discriminator build...")
    
    test_config = {
        'lstm_units': 512,  # Increased to match ~19M parameters
        'dropout_rate': 0.3,
        'use_layer_norm': True,
        'use_self_attention': True
    }
    
    discriminator = build_lstm_only_discriminator(
        text_shape=(32, 256),
        config=test_config
    )
    
    print("\n✅ Build test successful!")
    
    # Test forward pass
    import numpy as np
    batch_size = 2
    max_text_len = 32
    vocab_size = 256
    
    test_text = np.random.randn(batch_size, max_text_len, vocab_size).astype(np.float32)
    
    print("\nTesting forward pass...")
    output = discriminator(test_text, training=False)
    print(f"✅ Forward pass successful! Output shape: {output.shape}")
    print(f"   Output range: [{output.numpy().min():.4f}, {output.numpy().max():.4f}]")
    print(f"   Expected range: [0.0, 1.0] (sigmoid activation)")
