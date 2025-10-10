"""
Generator model (U-Net) for the Dual-Modal GAN.

This code is adapted from the unet function in the original GAN_AHTR.py.
Updated: Added Batch Normalization and Dropout as per Souibgui et al. paper.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    LeakyReLU,
    BatchNormalization
    # Dropout removed to fix XLA layout optimization error
)
from tensorflow.keras.models import Model

def unet(input_size=(1024, 128, 1)):
    """U-Net implementation with Batch Normalization (Dropout removed for stability).
    
    Reference: Souibgui et al. "Enhance to read better" (Pattern Recognition 2021)
    - Added Batch Normalization after each Conv2D layer
    - Dropout REMOVED to fix TensorFlow XLA layout optimization error

    Args:
        input_size (tuple, optional): The input shape of the images. 
                                      Defaults to (1024, 128, 1) to match HTR recognizer.

    Returns:
        tf.keras.Model: The U-Net generator model.
    """
    inputs = Input(input_size)

    # Encoder Path
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU()(conv1)
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU()(conv2)
    conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU()(conv3)
    conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU()(conv4)
    conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU()(conv5)
    conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU()(conv5)
    # conv5 = Dropout(0.2)(conv5)  # REMOVED: XLA layout error fix

    # Decoder Path (with Batch Norm, Dropout removed for stability)
    up6 = Conv2D(512, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
    up6 = BatchNormalization()(up6)
    up6 = LeakyReLU()(up6)
    merge6 = Concatenate()([conv4, up6])
    conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU()(conv6)
    conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU()(conv6)
    # conv6 = Dropout(0.2)(conv6)  # REMOVED: XLA layout error fix

    up7 = Conv2D(256, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    up7 = BatchNormalization()(up7)
    up7 = LeakyReLU()(up7)
    merge7 = Concatenate()([conv3, up7])
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU()(conv7)
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU()(conv7)
    # conv7 = Dropout(0.2)(conv7)  # REMOVED: XLA layout error fix

    up8 = Conv2D(128, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    up8 = BatchNormalization()(up8)
    up8 = LeakyReLU()(up8)
    merge8 = Concatenate()([conv2, up8])
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU()(conv8)
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU()(conv8)
    # conv8 = Dropout(0.2)(conv8)  # REMOVED: XLA layout error fix

    up9 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    up9 = BatchNormalization()(up9)
    up9 = LeakyReLU()(up9)
    merge9 = Concatenate()([conv1, up9])
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = LeakyReLU()(conv9)
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = LeakyReLU()(conv9)
    # conv9 = Dropout(0.2)(conv9)  # REMOVED: XLA layout error fix
    
    # Output Layer
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10, name='generator')

    return model
