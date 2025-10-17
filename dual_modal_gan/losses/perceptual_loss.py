"""
VGG Perceptual Loss for Enhanced Training
Implements perceptual similarity based on VGG19 features
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

class VGGPerceptualLoss(Layer):
    """
    Perceptual loss based on VGG19 feature extraction.
    Computes L1 distance between features extracted at multiple layers.
    Implemented as Keras Layer for compatibility with @tf.function.
    """
    
    def __init__(self, layers=None, weights='imagenet', **kwargs):
        """
        Initialize VGG perceptual loss.
        
        Args:
            layers: List of layer names to extract features from.
                   Default: ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4']
            weights: Pre-trained weights ('imagenet' or None)
        """
        super(VGGPerceptualLoss, self).__init__(**kwargs)
        
        if layers is None:
            # Standard configuration from research literature
            layers = [
                'block1_conv2',  # Early features (64 channels)
                'block2_conv2',  # Mid features (128 channels)
                'block3_conv4',  # Mid-high features (256 channels)
                'block4_conv4',  # High features (512 channels)
                'block5_conv4'   # Highest features (512 channels)
            ]
        
        self.layer_names = layers
        
        # Load pre-trained VGG19
        vgg = VGG19(include_top=False, weights=weights)
        vgg.trainable = False  # Freeze VGG weights
        
        # Extract intermediate layer outputs
        outputs = [vgg.get_layer(name).output for name in layers]
        self.model = Model(inputs=vgg.input, outputs=outputs)
        self.model.trainable = False  # Ensure no trainable params
        
        self.num_layers = len(layers)
        
        # ImageNet mean for preprocessing (BGR order)
        self.vgg_mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        
        print(f"✅ VGG Perceptual Loss initialized with {len(layers)} layers:")
        for i, name in enumerate(layers):
            shape = outputs[i].shape
            print(f"   - {name}: {shape}")
    
    def call(self, y_true, y_pred):
        """
        Compute perceptual loss between true and predicted images.
        
        Args:
            y_true: Ground truth images, shape (batch, height, width, channels)
            y_pred: Predicted images, shape (batch, height, width, channels)
        
        Returns:
            Perceptual loss (scalar)
        """
        # Preprocess both images
        y_true_preprocessed = self.preprocess(y_true)
        y_pred_preprocessed = self.preprocess(y_pred)
        
        # Extract features from both images
        features_true = self.model(y_true_preprocessed, training=False)
        features_pred = self.model(y_pred_preprocessed, training=False)
        
        # Ensure features are lists
        if not isinstance(features_true, list):
            features_true = [features_true]
            features_pred = [features_pred]
        
        # Compute L1 loss at each layer
        loss = 0.0
        for ft, fp in zip(features_true, features_pred):
            # Mean absolute error at each feature layer
            loss += tf.reduce_mean(tf.abs(ft - fp))
        
        # Average across all layers
        loss = loss / tf.cast(self.num_layers, tf.float32)
        
        return loss
    
    def preprocess(self, images):
        """
        Preprocess images for VGG19.
        Converts grayscale to RGB and normalizes to ImageNet range.
        
        Args:
            images: Tensor of shape (batch, height, width, channels)
                   Values in range [0, 1]
        
        Returns:
            Preprocessed tensor for VGG19
        """
        # Convert grayscale to RGB if needed
        if images.shape[-1] == 1:
            images = tf.image.grayscale_to_rgb(images)
        
        # Scale to [0, 255]
        images = images * 255.0
        
        # VGG19 preprocessing (mean subtraction)
        images = images - self.vgg_mean
        
        return images

def create_perceptual_loss(layers=None, weights='imagenet'):
    """
    Factory function to create VGG perceptual loss.
    
    Args:
        layers: List of VGG layer names (default: standard 5-layer config)
        weights: VGG weights ('imagenet' or None)
    
    Returns:
        VGGPerceptualLoss instance
    """
    return VGGPerceptualLoss(layers=layers, weights=weights)


# Example usage:
if __name__ == "__main__":
    # Test perceptual loss
    import numpy as np
    
    print("Testing VGG Perceptual Loss...")
    
    # Create dummy images
    batch_size = 2
    height, width = 64, 256
    channels = 1  # Grayscale
    
    y_true = tf.random.uniform((batch_size, height, width, channels))
    y_pred = tf.random.uniform((batch_size, height, width, channels))
    
    # Initialize loss
    perceptual_loss = create_perceptual_loss()
    
    # Compute loss
    loss_value = perceptual_loss(y_true, y_pred)
    
    print(f"\n✅ Perceptual loss computed: {loss_value.numpy():.4f}")
    print(f"   Input shape: {y_true.shape}")
    print(f"   Output type: {type(loss_value)}")
    print("\nTest passed!")
