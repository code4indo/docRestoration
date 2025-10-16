"""
Enhanced Generator V2 - SOTA Architecture for Document Restoration
Target: PSNR > 20 dB pada 1 epoch training

Improvements over V1:
1. CBAM (Channel + Spatial Attention) - Focus on relevant features
2. Residual Dense Blocks - Better gradient flow + feature reuse
3. Multi-Scale Feature Pyramid - Capture details at multiple scales
4. Deeper bottleneck - More representational power
5. Progressive feature refinement

References:
- Woo et al. (2018) - CBAM: Convolutional Block Attention Module
- Zhang et al. (2018) - Residual Dense Network for Image Super-Resolution
- Souibgui et al. (2022) - Enhance to Read Better (baseline: PSNR 40+ dB)
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
    Add,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Reshape,
    Dense,
    Multiply,
    Lambda
)
from tensorflow.keras.models import Model


def channel_attention(x, ratio=8, name_prefix='ca'):
    """
    Channel Attention Module from CBAM.
    Focuses on 'what' is meaningful in the feature maps.
    
    Args:
        x: Input tensor
        ratio: Reduction ratio for bottleneck
        name_prefix: Prefix for layer names
    
    Returns:
        Channel-attended features
    """
    channels = x.shape[-1]
    
    # Shared MLP
    shared_dense_1 = Dense(channels // ratio, activation='relu', name=f'{name_prefix}_fc1')
    shared_dense_2 = Dense(channels, name=f'{name_prefix}_fc2')
    
    # Global Average Pooling path
    avg_pool = GlobalAveragePooling2D()(x)
    avg_pool = Reshape((1, 1, channels))(avg_pool)
    avg_pool = shared_dense_1(avg_pool)
    avg_pool = shared_dense_2(avg_pool)
    
    # Global Max Pooling path
    max_pool = GlobalMaxPooling2D()(x)
    max_pool = Reshape((1, 1, channels))(max_pool)
    max_pool = shared_dense_1(max_pool)
    max_pool = shared_dense_2(max_pool)
    
    # Combine and apply sigmoid
    attention = Add()([avg_pool, max_pool])
    attention = Activation('sigmoid')(attention)
    
    return Multiply()([x, attention])


def spatial_attention(x, kernel_size=7, name_prefix='sa'):
    """
    Spatial Attention Module from CBAM.
    Focuses on 'where' is meaningful in the feature maps.
    
    Args:
        x: Input tensor
        kernel_size: Kernel size for convolution
        name_prefix: Prefix for layer names
    
    Returns:
        Spatially-attended features
    """
    # Average pooling along channels - wrapped in Lambda
    avg_pool = Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
                     name=f'{name_prefix}_avg_pool')(x)
    
    # Max pooling along channels - wrapped in Lambda
    max_pool = Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
                     name=f'{name_prefix}_max_pool')(x)
    
    # Concatenate and apply conv
    concat = Concatenate(axis=-1, name=f'{name_prefix}_concat')([avg_pool, max_pool])
    attention = Conv2D(1, kernel_size, padding='same', activation='sigmoid',
                      kernel_initializer='he_normal', name=f'{name_prefix}_conv')(concat)
    
    return Multiply(name=f'{name_prefix}_multiply')([x, attention])


def cbam_block(x, ratio=8, kernel_size=7, name_prefix='cbam'):
    """
    CBAM: Convolutional Block Attention Module.
    Sequential channel-then-spatial attention.
    
    Args:
        x: Input tensor
        ratio: Channel reduction ratio
        kernel_size: Spatial attention kernel size
        name_prefix: Prefix for layer names
    
    Returns:
        Attended features
    """
    x = channel_attention(x, ratio, name_prefix=f'{name_prefix}_channel')
    x = spatial_attention(x, kernel_size, name_prefix=f'{name_prefix}_spatial')
    return x


def residual_dense_block(x, filters, growth_rate=32, num_layers=3, name_prefix='rdb'):
    """
    Residual Dense Block for better feature reuse and gradient flow.
    Inspired by RDN (Residual Dense Network).
    
    Args:
        x: Input tensor
        filters: Number of output filters
        growth_rate: Growth rate for dense connections
        num_layers: Number of dense layers
        name_prefix: Prefix for layer names
    
    Returns:
        Dense feature-fused output
    """
    # Project input to target filters if needed
    if x.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, padding='same', kernel_initializer='he_normal',
                         name=f'{name_prefix}_project')(x)
        shortcut = BatchNormalization(name=f'{name_prefix}_project_bn')(shortcut)
    else:
        shortcut = x
    
    concat_features = [shortcut]
    
    for i in range(num_layers):
        # Concatenate all previous features
        concat = Concatenate(name=f'{name_prefix}_concat{i}')(concat_features) if len(concat_features) > 1 else shortcut
        
        # Conv layer
        conv = Conv2D(growth_rate, 3, padding='same', kernel_initializer='he_normal',
                     name=f'{name_prefix}_conv{i+1}')(concat)
        conv = BatchNormalization(name=f'{name_prefix}_bn{i+1}')(conv)
        conv = LeakyReLU(name=f'{name_prefix}_relu{i+1}')(conv)
        
        concat_features.append(conv)
    
    # Local Feature Fusion
    lff = Concatenate(name=f'{name_prefix}_concat_all')(concat_features)
    lff = Conv2D(filters, 1, padding='same', kernel_initializer='he_normal',
                name=f'{name_prefix}_lff')(lff)
    
    # Local Residual Learning - now shapes match
    output = Add(name=f'{name_prefix}_residual')([shortcut, lff])
    
    return output


def multi_scale_feature_pyramid(x, filters, scales=[1, 2, 4], name_prefix='msfp'):
    """
    Multi-Scale Feature Pyramid for capturing details at different scales.
    
    Args:
        x: Input tensor
        filters: Number of filters per scale
        scales: Dilation rates for different scales
        name_prefix: Prefix for layer names
    
    Returns:
        Multi-scale fused features
    """
    features = []
    
    for i, scale in enumerate(scales):
        # Dilated convolution for multi-scale receptive fields
        conv = Conv2D(filters, 3, padding='same', dilation_rate=scale,
                     kernel_initializer='he_normal',
                     name=f'{name_prefix}_scale{scale}_conv')(x)
        conv = BatchNormalization(name=f'{name_prefix}_scale{scale}_bn')(conv)
        conv = LeakyReLU(name=f'{name_prefix}_scale{scale}_relu')(conv)
        features.append(conv)
    
    # Fuse multi-scale features
    fused = Concatenate(name=f'{name_prefix}_concat')(features)
    fused = Conv2D(filters, 1, padding='same', kernel_initializer='he_normal',
                  name=f'{name_prefix}_fuse')(fused)
    fused = BatchNormalization(name=f'{name_prefix}_fuse_bn')(fused)
    fused = LeakyReLU(name=f'{name_prefix}_fuse_relu')(fused)
    
    return fused


def encoder_block(x, filters, name_prefix='enc', use_rdb=True, use_cbam=True):
    """
    Enhanced encoder block with RDB + CBAM.
    
    Args:
        x: Input tensor
        filters: Number of filters
        name_prefix: Prefix for layer names
        use_rdb: Whether to use Residual Dense Block
        use_cbam: Whether to use CBAM attention
    
    Returns:
        Encoded features
    """
    # Project to target filters if needed
    if x.shape[-1] != filters:
        x = Conv2D(filters, 1, padding='same', kernel_initializer='he_normal',
                  name=f'{name_prefix}_project')(x)
        x = BatchNormalization(name=f'{name_prefix}_project_bn')(x)
    
    # Residual Dense Block for feature extraction
    if use_rdb:
        x = residual_dense_block(x, filters, name_prefix=f'{name_prefix}_rdb')
    else:
        # Fallback to simple residual block
        conv = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal',
                     name=f'{name_prefix}_conv1')(x)
        conv = BatchNormalization(name=f'{name_prefix}_bn1')(conv)
        conv = LeakyReLU(name=f'{name_prefix}_relu1')(conv)
        
        conv = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal',
                     name=f'{name_prefix}_conv2')(conv)
        conv = BatchNormalization(name=f'{name_prefix}_bn2')(conv)
        
        x = Add(name=f'{name_prefix}_residual')([x, conv])
        x = LeakyReLU(name=f'{name_prefix}_relu2')(x)
    
    # CBAM attention
    if use_cbam:
        x = cbam_block(x, name_prefix=f'{name_prefix}_cbam')
    
    return x


def decoder_block(x, skip, filters, name_prefix='dec', use_rdb=True, use_cbam=True):
    """
    Enhanced decoder block with attention gating and RDB.
    
    Args:
        x: Upsampled decoder features
        skip: Skip connection from encoder
        filters: Number of filters
        name_prefix: Prefix for layer names
        use_rdb: Whether to use Residual Dense Block
        use_cbam: Whether to use CBAM attention
    
    Returns:
        Decoded features
    """
    # Upsample
    up = UpSampling2D(size=(2, 2), name=f'{name_prefix}_upsample')(x)
    up = Conv2D(filters, 2, padding='same', kernel_initializer='he_normal',
               name=f'{name_prefix}_conv_up')(up)
    up = BatchNormalization(name=f'{name_prefix}_bn_up')(up)
    up = LeakyReLU(name=f'{name_prefix}_relu_up')(up)
    
    # Attention gate for skip connection
    # Simple attention: use decoder features to gate encoder features
    gate = Conv2D(filters, 1, padding='same', kernel_initializer='he_normal',
                 name=f'{name_prefix}_gate_g')(up)
    gate = BatchNormalization(name=f'{name_prefix}_gate_g_bn')(gate)
    
    skip_conv = Conv2D(filters, 1, padding='same', kernel_initializer='he_normal',
                      name=f'{name_prefix}_gate_x')(skip)
    skip_conv = BatchNormalization(name=f'{name_prefix}_gate_x_bn')(skip_conv)
    
    psi = Add(name=f'{name_prefix}_gate_add')([gate, skip_conv])
    psi = LeakyReLU(name=f'{name_prefix}_gate_relu')(psi)
    psi = Conv2D(1, 1, padding='same', kernel_initializer='he_normal',
                name=f'{name_prefix}_gate_psi')(psi)
    psi = BatchNormalization(name=f'{name_prefix}_gate_psi_bn')(psi)
    psi = Activation('sigmoid', name=f'{name_prefix}_gate_sigmoid')(psi)
    
    skip_attended = Multiply(name=f'{name_prefix}_gate_multiply')([skip, psi])
    
    # Merge with upsampled features
    merge = Concatenate(name=f'{name_prefix}_merge')([skip_attended, up])
    
    # Residual Dense Block
    if use_rdb:
        merge = residual_dense_block(merge, filters, name_prefix=f'{name_prefix}_rdb')
    else:
        # Simple residual block
        conv = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal',
                     name=f'{name_prefix}_conv1')(merge)
        conv = BatchNormalization(name=f'{name_prefix}_bn1')(conv)
        conv = LeakyReLU(name=f'{name_prefix}_relu1')(conv)
        
        conv = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal',
                     name=f'{name_prefix}_conv2')(conv)
        conv = BatchNormalization(name=f'{name_prefix}_bn2')(conv)
        merge = LeakyReLU(name=f'{name_prefix}_relu2')(conv)
    
    # CBAM attention
    if use_cbam:
        merge = cbam_block(merge, name_prefix=f'{name_prefix}_cbam')
    
    return merge


def unet_enhanced_v2(input_size=(1024, 128, 1)):
    """
    Enhanced U-Net V2 with SOTA techniques for document restoration.
    
    Target: PSNR > 20 dB on 1 epoch
    
    Key Features:
    1. Residual Dense Blocks (RDB) for better gradient flow
    2. CBAM attention (Channel + Spatial) at each level
    3. Multi-Scale Feature Pyramid in bottleneck
    4. Attention gates in decoder
    5. Deeper architecture (5 scales vs 4)
    
    Args:
        input_size: Input shape (H, W, C)
    
    Returns:
        Keras Model
    """
    inputs = Input(input_size, name='input')
    
    # ========== ENCODER ==========
    # Level 1: 1024x128 -> 512x64
    e1 = encoder_block(inputs, 64, name_prefix='enc1', use_rdb=True, use_cbam=True)
    p1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(e1)
    
    # Level 2: 512x64 -> 256x32
    e2 = encoder_block(p1, 128, name_prefix='enc2', use_rdb=True, use_cbam=True)
    p2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(e2)
    
    # Level 3: 256x32 -> 128x16
    e3 = encoder_block(p2, 256, name_prefix='enc3', use_rdb=True, use_cbam=True)
    p3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(e3)
    
    # Level 4: 128x16 -> 64x8
    e4 = encoder_block(p3, 512, name_prefix='enc4', use_rdb=True, use_cbam=True)
    p4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(e4)
    
    # Level 5: 64x8 -> 32x4 (DEEPER!)
    e5 = encoder_block(p4, 512, name_prefix='enc5', use_rdb=True, use_cbam=True)
    p5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(e5)
    
    # ========== BOTTLENECK ==========
    # Multi-scale feature extraction at bottleneck (32x4)
    bottleneck = multi_scale_feature_pyramid(p5, 512, scales=[1, 2, 4], 
                                             name_prefix='bottleneck_msfp')
    bottleneck = residual_dense_block(bottleneck, 512, name_prefix='bottleneck_rdb')
    bottleneck = cbam_block(bottleneck, name_prefix='bottleneck_cbam')
    
    # ========== DECODER ==========
    # Level 5: 32x4 -> 64x8
    d5 = decoder_block(bottleneck, e5, 512, name_prefix='dec5', use_rdb=True, use_cbam=True)
    
    # Level 4: 64x8 -> 128x16
    d4 = decoder_block(d5, e4, 512, name_prefix='dec4', use_rdb=True, use_cbam=True)
    
    # Level 3: 128x16 -> 256x32
    d3 = decoder_block(d4, e3, 256, name_prefix='dec3', use_rdb=True, use_cbam=True)
    
    # Level 2: 256x32 -> 512x64
    d2 = decoder_block(d3, e2, 128, name_prefix='dec2', use_rdb=True, use_cbam=True)
    
    # Level 1: 512x64 -> 1024x128
    d1 = decoder_block(d2, e1, 64, name_prefix='dec1', use_rdb=True, use_cbam=True)
    
    # ========== OUTPUT ==========
    # Final refinement
    output = Conv2D(32, 3, padding='same', kernel_initializer='he_normal',
                   name='output_refine_conv')(d1)
    output = BatchNormalization(name='output_refine_bn')(output)
    output = LeakyReLU(name='output_refine_relu')(output)
    
    # Output layer (tanh for [-1, 1] range)
    output = Conv2D(1, 1, activation='tanh', name='output')(output)
    
    model = Model(inputs=inputs, outputs=output, name='generator_enhanced_v2')
    
    return model


# For testing
if __name__ == '__main__':
    print("Building Enhanced Generator V2...")
    model = unet_enhanced_v2()
    model.summary()
    
    print(f"\n✅ Total parameters: {model.count_params():,}")
    print(f"✅ Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
