"""
Enhanced Generator model (U-Net with Residual Blocks and Attention Gates) 
for the Dual-Modal GAN.

This model incorporates architectural improvements to enhance its learning capacity
and focus on relevant features for better image restoration.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    LeakyReLU,
    BatchNormalization,
    Activation,
    Add
)
from tensorflow.keras.models import Model

def residual_conv_block(x, filters, kernel_size=3):
    """
    A residual block with two convolutional layers.
    
    Args:
        x (tf.Tensor): Input tensor.
        filters (int): Number of filters for the convolutional layers.
        kernel_size (int, optional): Kernel size. Defaults to 3.

    Returns:
        tf.Tensor: Output tensor of the residual block.
    """
    shortcut = x
    
    # First convolution
    conv = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU()(conv)
    
    # Second convolution
    conv = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    
    # Shortcut connection
    # If the number of filters changes, project the shortcut to the same dimension
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, padding='same', kernel_initializer='he_normal')(shortcut)
        shortcut = BatchNormalization()(shortcut)
        
    res_conn = Add()([shortcut, conv])
    output = LeakyReLU()(res_conn)
    
    return output

def attention_gate(encoder_features, decoder_features, inter_channels):
    """
    Attention Gate to focus on relevant features from the skip connection.

    Args:
        encoder_features (tf.Tensor): Features from the encoder path (skip connection).
        decoder_features (tf.Tensor): Features from the decoder path (upsampled).
        inter_channels (int): Number of intermediate channels.

    Returns:
        tf.Tensor: Attended features from the encoder path.
    """
    # Gating signal from decoder
    g = Conv2D(inter_channels, 1, padding='same', kernel_initializer='he_normal')(decoder_features)
    g = BatchNormalization()(g)

    # Features from encoder
    x = Conv2D(inter_channels, 1, padding='same', kernel_initializer='he_normal')(encoder_features)
    x = BatchNormalization()(x)

    # Combine and process
    psi = Add()([g, x])
    psi = LeakyReLU()(psi)
    psi = Conv2D(1, 1, padding='same', kernel_initializer='he_normal')(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)

    # Apply attention to encoder features
    return encoder_features * psi

def unet_enhanced(input_size=(1024, 128, 1)):
    """
    Enhanced U-Net with Residual Blocks and Attention Gates.

    Args:
        input_size (tuple, optional): The input shape. Defaults to (1024, 128, 1).

    Returns:
        tf.keras.Model: The enhanced U-Net generator model.
    """
    inputs = Input(input_size)

    # Encoder Path
    res1 = residual_conv_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(res1)

    res2 = residual_conv_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(res2)

    res3 = residual_conv_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(res3)

    res4 = residual_conv_block(pool3, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(res4)

    # Bottleneck (FIXED: 1024 -> 512 to reduce memory and align with Solution 3)
    res5 = residual_conv_block(pool4, 512)

    # Decoder Path (with Attention Gates)
    up6 = UpSampling2D(size=(2, 2))(res5)
    up6 = Conv2D(512, 2, padding='same', kernel_initializer='he_normal')(up6)
    up6 = BatchNormalization()(up6)
    up6 = LeakyReLU()(up6)
    
    att6 = attention_gate(res4, up6, 256)
    merge6 = Concatenate()([att6, up6])
    res6 = residual_conv_block(merge6, 512)

    up7 = UpSampling2D(size=(2, 2))(res6)
    up7 = Conv2D(256, 2, padding='same', kernel_initializer='he_normal')(up7)
    up7 = BatchNormalization()(up7)
    up7 = LeakyReLU()(up7)

    att7 = attention_gate(res3, up7, 128)
    merge7 = Concatenate()([att7, up7])
    res7 = residual_conv_block(merge7, 256)

    up8 = UpSampling2D(size=(2, 2))(res7)
    up8 = Conv2D(128, 2, padding='same', kernel_initializer='he_normal')(up8)
    up8 = BatchNormalization()(up8)
    up8 = LeakyReLU()(up8)

    att8 = attention_gate(res2, up8, 64)
    merge8 = Concatenate()([att8, up8])
    res8 = residual_conv_block(merge8, 128)

    up9 = UpSampling2D(size=(2, 2))(res8)
    up9 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(up9)
    up9 = BatchNormalization()(up9)
    up9 = LeakyReLU()(up9)

    att9 = attention_gate(res1, up9, 32)
    merge9 = Concatenate()([att9, up9])
    res9 = residual_conv_block(merge9, 64)
    
    # Output Layer (FIXED: sigmoid -> tanh to match [-1, 1] data range)
    conv10 = Conv2D(1, 1, activation='tanh')(res9)

    model = Model(inputs=inputs, outputs=conv10, name='generator_enhanced')

    return model
