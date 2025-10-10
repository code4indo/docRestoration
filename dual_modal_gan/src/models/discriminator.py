"""
Dual-Modal Discriminator for the GAN-HTR project.

This discriminator is designed to assess the coherence between a generated
image and the text recognized from it.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    LeakyReLU,
    BatchNormalization,
    Dense,
    Flatten,
    Concatenate,
    Embedding,
    LSTM,
    GlobalAveragePooling1D
)
from tensorflow.keras.models import Model

def build_dual_modal_discriminator(img_shape=(128, 1024, 1), vocab_size=100, max_text_len=128, text_embed_dim=64, lstm_units=128):
    """Builds a discriminator that processes both image and text inputs.

    Args:
        img_shape (tuple): Shape of the input image.
        vocab_size (int): Size of the character vocabulary for text input.
        max_text_len (int): Maximum length of the input text sequence.
        text_embed_dim (int): Dimension for the text embedding layer.
        lstm_units (int): Number of units in the LSTM layer for text processing.

    Returns:
        tf.keras.Model: The dual-modal discriminator model.
    """
    # --- Image Input Branch ---
    image_input = Input(shape=img_shape, name='image_input')

    def d_layer(layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    df = 64
    img_features = d_layer(image_input, df, bn=False)
    img_features = d_layer(img_features, df * 2)
    img_features = d_layer(img_features, df * 4)
    img_features = d_layer(img_features, df * 8)
    img_features = Flatten()(img_features)

    # --- Text Input Branch ---
    text_input = Input(shape=(max_text_len,), name='text_input')
    # Embedding layer for integer-encoded text
    text_features = Embedding(input_dim=vocab_size + 1, output_dim=text_embed_dim)(text_input)
    # LSTM to process the sequence
    text_features = LSTM(lstm_units, return_sequences=False)(text_features)

    # --- Fusion and Classification ---
    # Concatenate the features from both modalities
    combined_features = Concatenate()([img_features, text_features])

    # Dense layers for final classification
    x = Dense(512)(combined_features)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Output a single validity score
    validity = Dense(1, activation='sigmoid', name='validity_score')(x)

    # Create and return the model
    model = Model(inputs=[image_input, text_input], outputs=validity, name='dual_modal_discriminator')
    
    print("[Discriminator] Dual-Modal Discriminator built.")
    model.summary()
    
    return model
