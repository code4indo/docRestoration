#!/usr/bin/env python3
"""
Modifikasi arsitektur CRNN yang ditingkatkan untuk performa yang lebih baik
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf

from contextlib import redirect_stdout
from tensorflow.keras import backend as K
from tensorflow.keras import Model

from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.constraints import MaxNorm

from network.layers import FullGatedConv2D, GatedConv2D, OctConv2D
from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.layers import Input, Add, Activation, Lambda, MaxPooling2D, Reshape
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Flatten
from tensorflow.keras.layers import RepeatVector, Permute, Multiply, Concatenate, UpSampling2D

def flor_enhanced(input_size, d_model):
    """
    Enhanced Flor Architecture dengan:
    - Deeper convolutional layers
    - Attention mechanism
    - Residual connections
    - Larger hidden units
    - Multi-scale features
    """

    input_data = Input(name="input", shape=input_size)

    # Initial feature extraction with larger filters
    cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer="he_uniform")(input_data)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)

    # Residual block 1
    cnn_res1 = cnn
    cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = FullGatedConv2D(filters=64, kernel_size=(3, 3), padding="same")(cnn)
    cnn = Conv2D(filters=32, kernel_size=(1, 1), padding="same")(cnn)  # Residual projection
    cnn = Add()([cnn, cnn_res1])  # Residual connection

    # Downsampling with larger filters
    cnn = Conv2D(filters=80, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = FullGatedConv2D(filters=80, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    # Residual block 2
    cnn_res2 = cnn
    cnn = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = FullGatedConv2D(filters=96, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = Conv2D(filters=80, kernel_size=(1, 1), padding="same")(cnn)  # Residual projection
    cnn = Add()([cnn, cnn_res2])  # Residual connection
    cnn = Dropout(rate=0.2)(cnn)

    # Further downsampling
    cnn = Conv2D(filters=112, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = FullGatedConv2D(filters=112, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    # Final convolutional layers with larger filters
    cnn = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)

    cnn = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)

    # Reshape for RNN
    shape = K.int_shape(cnn)
    bgru = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    # Enhanced RNN with larger units and attention
    bgru = Bidirectional(GRU(units=256, return_sequences=True, dropout=0.3))(bgru)
    bgru = Dense(units=512)(bgru)

    # Attention mechanism
    attention = Dense(1, activation='tanh')(bgru)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(512)(attention)
    attention = Permute([2, 1])(attention)
    bgru_att = Multiply()([bgru, attention])

    bgru = Bidirectional(GRU(units=256, return_sequences=True, dropout=0.3))(bgru_att)
    output_data = Dense(units=d_model)(bgru)

    return (input_data, output_data)

def flor_multi_scale(input_size, d_model):
    """
    Multi-scale Flor Architecture:
    - Multiple parallel convolutional streams
    - Feature fusion
    - Enhanced feature extraction
    """

    input_data = Input(name="input", shape=input_size)

    # Multi-scale feature extraction
    # Scale 1: Fine details
    scale1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(input_data)
    scale1 = PReLU(shared_axes=[1, 2])(scale1)
    scale1 = BatchNormalization()(scale1)
    scale1 = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(scale1)
    scale1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(scale1)

    # Scale 2: Medium features
    scale2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding="same", kernel_initializer="he_uniform")(input_data)
    scale2 = PReLU(shared_axes=[1, 2])(scale2)
    scale2 = BatchNormalization()(scale2)
    scale2 = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(scale2)

    # Scale 3: Coarse features
    scale3 = Conv2D(filters=48, kernel_size=(7, 7), strides=(4, 4), padding="same", kernel_initializer="he_uniform")(input_data)
    scale3 = PReLU(shared_axes=[1, 2])(scale3)
    scale3 = BatchNormalization()(scale3)
    scale3 = FullGatedConv2D(filters=48, kernel_size=(3, 3), padding="same")(scale3)

    # Fuse multi-scale features
    # Resize scale1 to match scale2
    scale1_resized = Conv2D(filters=32, kernel_size=(1, 1), padding="same")(scale1)
    scale1_resized = UpSampling2D(size=(2, 2))(scale1_resized)

    # Resize scale3 to match scale2
    scale3_resized = Conv2D(filters=32, kernel_size=(1, 1), padding="same")(scale3)
    scale3_resized = UpSampling2D(size=(2, 2))(scale3_resized)

    # Match spatial dimensions by using same stride/padding
    scale1_resized = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same")(scale1_resized)
    scale3_resized = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same")(scale3_resized)

    # Concatenate features
    fused = Concatenate()([scale1_resized, scale2, scale3_resized])

    # Further processing
    cnn = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(fused)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = FullGatedConv2D(filters=96, kernel_size=(3, 3), padding="same")(cnn)

    cnn = Conv2D(filters=112, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = FullGatedConv2D(filters=112, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)

    cnn = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)

    # RNN layers
    shape = K.int_shape(cnn)
    bgru = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    bgru = Bidirectional(GRU(units=256, return_sequences=True, dropout=0.3))(bgru)
    bgru = Dense(units=512)(bgru)

    bgru = Bidirectional(GRU(units=256, return_sequences=True, dropout=0.3))(bgru)
    output_data = Dense(units=d_model)(bgru)

    return (input_data, output_data)

def flor_transformer(input_size, d_model):
    """
    Flor with Transformer-inspired architecture:
    - Self-attention mechanism
    - Multi-head attention
    - Feed-forward networks
    """

    input_data = Input(name="input", shape=input_size)

    # Enhanced CNN feature extraction
    cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer="he_uniform")(input_data)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)

    cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = FullGatedConv2D(filters=64, kernel_size=(3, 3), padding="same")(cnn)

    cnn = Conv2D(filters=96, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = FullGatedConv2D(filters=96, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization()(cnn)

    cnn = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)

    # Reshape for transformer-like processing
    shape = K.int_shape(cnn)
    seq = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    # Position encoding - menggunakan Keras layer
    positions = tf.range(start=0, limit=shape[1], delta=1)
    pos_embedding = tf.keras.layers.Embedding(input_dim=shape[1], output_dim=shape[2] * shape[3])
    pos_encoding = pos_embedding(positions)

    # Expand to match batch size using Lambda layer
    def expand_pos_encoding(x):
        pos_enc, seq_tensor = x
        batch_size = tf.shape(seq_tensor)[0]
        expanded_pos = tf.expand_dims(pos_enc, axis=0)
        tiled_pos = tf.tile(expanded_pos, [batch_size, 1, 1])
        return tiled_pos

    pos_encoding_expanded = Lambda(expand_pos_encoding)([pos_encoding, seq])
    seq = Add()([seq, pos_encoding_expanded])

    # Multi-head self-attention
    attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(seq, seq)
    attention_output = Dropout(0.1)(attention_output)
    seq = Add()([seq, attention_output])  # Residual
    seq = LayerNormalization()(seq)

    # Feed-forward network
    ffn = Dense(units=512, activation='relu')(seq)
    ffn = Dense(units=shape[2] * shape[3])(ffn)
    ffn = Dropout(0.1)(ffn)
    seq = Add()([seq, ffn])  # Residual
    seq = LayerNormalization()(seq)

    # Another attention block
    attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(seq, seq)
    attention_output = Dropout(0.1)(attention_output)
    seq = Add()([seq, attention_output])  # Residual
    seq = LayerNormalization()(seq)

    # Final RNN for sequence modeling
    bgru = Bidirectional(GRU(units=256, return_sequences=True, dropout=0.3))(seq)
    output_data = Dense(units=d_model)(bgru)

    return (input_data, output_data)

def create_enhanced_crnn_model(architecture_name="flor_enhanced"):
    """
    Factory function untuk membuat model CRNN yang ditingkatkan
    """
    from network.model import HTRModel

    # Model configuration
    input_size = (128, 1024, 1)  # Sesuai dengan data training
    vocab_size = 80  # Sesuai dengan charset

    # Pilih arsitektur
    if architecture_name == "flor_enhanced":
        architecture = flor_enhanced
    elif architecture_name == "flor_multi_scale":
        architecture = flor_multi_scale
    elif architecture_name == "flor_transformer":
        architecture = flor_transformer
    else:
        raise ValueError(f"Arsitektur {architecture_name} tidak ditemukan")

    # Buat model
    model = HTRModel(
        architecture=architecture_name,
        input_size=input_size,
        vocab_size=vocab_size,
        greedy=False,
        beam_width=10,
        top_paths=1
    )

    # Compile model
    model.compile(learning_rate=1e-4)

    return model

# Simpan fungsi ke globals agar bisa diakses oleh HTRModel
import sys
current_module = sys.modules[__name__]
setattr(current_module, 'flor_enhanced', flor_enhanced)
setattr(current_module, 'flor_multi_scale', flor_multi_scale)
setattr(current_module, 'flor_transformer', flor_transformer)