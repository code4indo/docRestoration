"""
Dual-Modal GAN Training with Keras fit() API - Native Multi-GPU Support

This version uses tf.keras.Model subclassing with custom train_step()
which is fully compatible with model.fit() and MirroredStrategy.

Key Benefits:
1. Native multi-GPU support via MirroredStrategy
2. Automatic gradient distribution and aggregation
3. Built-in callbacks (ModelCheckpoint, TensorBoard, etc.)
4. Proper batch handling for distributed training
5. No manual strategy.run() required

Usage:
    poetry run python dual_modal_gan/scripts/train_gan_keras_fit.py \
        --gpu_id "0,1" --batch_size 4 --epochs 50
"""

import os
import sys
import argparse
import json
from datetime import datetime
import tensorflow as tf
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dual_modal_gan.src.models.generator import unet
from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
from dual_modal_gan.src.models.generator_enhanced_v2 import unet_enhanced_v2
from dual_modal_gan.src.models.discriminator import build_dual_modal_discriminator
from dual_modal_gan.src.models.discriminator_enhanced_v2 import build_dual_modal_discriminator_enhanced_v2
from dual_modal_gan.src.models.discriminator_enhanced_v2_fixed import build_dual_modal_discriminator_enhanced_v2_fixed
from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed as load_frozen_recognizer
# from dual_modal_gan.src.utils import read_charlist  # NOT EXIST - define below
from dual_modal_gan.losses.perceptual_loss import create_perceptual_loss


# ============================================================================
# UTILITY FUNCTIONS (copied from train_enhanced.py)
# ============================================================================

def read_charlist(path):
    """Read charset from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]


# ============================================================================
# DATA LOADING FUNCTIONS (copied from train_enhanced.py)
# ============================================================================

def _parse_tfrecord_fn(example_proto):
    """Parse TFRecord with degraded image, clean image, and text label."""
    feature_description = {
        'degraded_image_raw': tf.io.FixedLenFeature([], tf.string),
        'degraded_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'degraded_image_dtype': tf.io.FixedLenFeature([], tf.string),
        'clean_image_raw': tf.io.FixedLenFeature([], tf.string),
        'clean_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'clean_image_dtype': tf.io.FixedLenFeature([], tf.string),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'label_shape': tf.io.FixedLenFeature([1], tf.int64),
        'label_dtype': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    # Deserialize degraded image
    degraded_image_shape = tf.cast(example['degraded_image_shape'], tf.int32)
    degraded_image = tf.io.decode_raw(example['degraded_image_raw'], tf.float32)
    degraded_image = tf.reshape(degraded_image, degraded_image_shape)
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])  # (H,W,C) → (W,H,C)
    degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])
    
    # Deserialize clean image
    clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
    clean_image = tf.reshape(clean_image, clean_image_shape)
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])  # (H,W,C) → (W,H,C)
    clean_image = tf.ensure_shape(clean_image, [1024, 128, 1])
    
    # Deserialize label
    label_shape = tf.cast(example['label_shape'], tf.int32)
    label = tf.io.decode_raw(example['label_raw'], tf.int64)
    label = tf.reshape(label, label_shape)
    label = tf.cast(label, tf.int32)
    
    # Pad label to static shape
    padding = [[0, 128 - tf.shape(label)[0]]]
    label = tf.pad(label, padding, "CONSTANT", constant_values=0)
    label.set_shape([128])
    
    return degraded_image, clean_image, label

def create_dataset(tfrecord_path, batch_size, train_split=0.7, val_split=0.15):
    """Create train/val/test split with academic protocol."""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # Get total size
    total_size = sum(1 for _ in dataset)
    
    # Calculate split sizes
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    # Parse and split
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = dataset.take(train_size)
    remaining = dataset.skip(train_size)
    val_dataset = remaining.take(val_size)
    test_dataset = remaining.skip(val_size)
    
    # Process datasets
    train_dataset = train_dataset.shuffle(buffer_size=1024).repeat().batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    print(f"\n📊 Dataset Split:")
    print(f"   Total: {total_size} | Train: {train_size} | Val: {val_size} | Test: {test_size}")
    
    return train_dataset, val_dataset, test_dataset, train_size, val_size, test_size


class DualModalGAN(tf.keras.Model):
    """
    Dual-Modal GAN-HTR model using Keras Model API.
    
    This enables multi-GPU training via model.fit() with MirroredStrategy.
    """
    
    def __init__(
        self,
        generator,
        discriminator,
        recognizer,
        perceptual_loss_layer=None,
        pixel_loss_weight=100.0,
        adv_loss_weight=1.0,
        rec_feat_loss_weight=8.0,
        ctc_loss_weight=0.15,
        perceptual_loss_weight=2.0,
        ctc_loss_clip_max=300.0,
        discriminator_mode='predicted',
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.generator = generator
        self.discriminator = discriminator
        self.recognizer = recognizer
        self.perceptual_loss_layer = perceptual_loss_layer
        
        # Loss weights
        self.pixel_loss_weight = pixel_loss_weight
        self.adv_loss_weight = adv_loss_weight
        self.rec_feat_loss_weight = rec_feat_loss_weight
        self.ctc_loss_weight = ctc_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.ctc_loss_clip_max = ctc_loss_clip_max
        self.discriminator_mode = discriminator_mode
        
        # Loss functions
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.mae_loss = tf.keras.losses.MeanAbsoluteError()
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        
        # Metrics
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.adv_loss_tracker = tf.keras.metrics.Mean(name="adv_loss")
        self.pixel_loss_tracker = tf.keras.metrics.Mean(name="pixel_loss")
        self.rec_feat_loss_tracker = tf.keras.metrics.Mean(name="rec_feat_loss")
        self.percep_loss_tracker = tf.keras.metrics.Mean(name="percep_loss")
        self.ctc_loss_tracker = tf.keras.metrics.Mean(name="ctc_loss")
    
    @property
    def metrics(self):
        """Return list of metrics tracked by the model."""
        return [
            self.gen_loss_tracker,
            self.disc_loss_tracker,
            self.adv_loss_tracker,
            self.pixel_loss_tracker,
            self.rec_feat_loss_tracker,
            self.percep_loss_tracker,
            self.ctc_loss_tracker,
        ]
    
    def compile(self, g_optimizer, d_optimizer):
        """
        Compile the model with optimizers.
        
        Note: We override compile() to store optimizers for train_step().
        """
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
    
    def build(self, input_shape):
        """
        Build method to mark model as built.
        
        This is required for model.fit() but we don't need to do anything here
        since our generator, discriminator, and recognizer are already built.
        """
        super().build(input_shape)
    
    def call(self, inputs, training=False):
        """
        Forward pass (required for Keras Model).
        
        For GAN training with custom train_step(), this method is not actually used,
        but it needs to exist for Keras Model API compatibility.
        
        Args:
            inputs: Input tensor or tuple
            training: Whether in training mode
            
        Returns:
            Generator output
        """
        # Handle both single tensor and tuple inputs
        if isinstance(inputs, (list, tuple)):
            degraded_images = inputs[0]
        else:
            degraded_images = inputs
        
        return self.generator(degraded_images, training=training)
    
    def train_step(self, data):
        """
        Custom training step for GAN.
        
        This method is called by model.fit() and handles:
        - Multi-GPU distribution (automatic via MirroredStrategy)
        - Gradient computation and application
        - Metric updates
        
        Args:
            data: Tuple of (degraded_images, (clean_images, ground_truth_text))
        
        Returns:
            Dictionary of metrics
        """
        # Unpack data
        degraded_images, (clean_images, ground_truth_text) = data
        
        # Get actual batch size (important for multi-GPU!)
        batch_size = tf.shape(degraded_images)[0]
        
        # Labels for discriminator (one-sided label smoothing)
        real_labels = tf.ones([batch_size, 1]) * 0.9
        fake_labels = tf.zeros([batch_size, 1])
        
        # ============= DISCRIMINATOR TRAINING =============
        with tf.GradientTape() as disc_tape:
            # Normalize inputs to [-1, 1] for tanh generator
            clean_images_tanh = clean_images * 2.0 - 1.0
            degraded_images_tanh = degraded_images * 2.0 - 1.0
            
            # Generate fake images
            generated_images = self.generator(degraded_images_tanh, training=True)
            
            # Denormalize for recognizer (expects [0, 1])
            clean_images_01 = (clean_images_tanh + 1.0) / 2.0
            generated_images_01 = (generated_images + 1.0) / 2.0
            
            # Get text predictions from recognizer
            recognizer_output_clean = self.recognizer(clean_images_01, training=False)
            recognizer_output_generated = self.recognizer(generated_images_01, training=False)
            
            # Extract logits
            if isinstance(recognizer_output_clean, (list, tuple)):
                clean_logits = recognizer_output_clean[0]
                generated_logits = recognizer_output_generated[0]
            else:
                clean_logits = recognizer_output_clean
                generated_logits = recognizer_output_generated
            
            clean_text_pred = tf.argmax(clean_logits, axis=-1, output_type=tf.int32)
            generated_text_pred = tf.argmax(generated_logits, axis=-1, output_type=tf.int32)
            
            # Discriminator forward pass
            if self.discriminator_mode == 'ground_truth':
                real_output = self.discriminator([clean_images_tanh, ground_truth_text], training=True)
            else:
                real_output = self.discriminator([clean_images_tanh, clean_text_pred], training=True)
            
            fake_output = self.discriminator([generated_images, generated_text_pred], training=True)
            
            # Discriminator losses
            disc_loss_real = self.bce_loss(real_labels, real_output)
            disc_loss_fake = self.bce_loss(fake_labels, fake_output)
            total_disc_loss = disc_loss_real + disc_loss_fake
        
        # Apply discriminator gradients
        disc_gradients = disc_tape.gradient(total_disc_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        # ============= GENERATOR TRAINING =============
        with tf.GradientTape() as gen_tape:
            # Generate images (reuse from above would be more efficient but needs refactor)
            generated_images = self.generator(degraded_images_tanh, training=True)
            generated_images_01 = (generated_images + 1.0) / 2.0
            
            # Get recognizer outputs
            recognizer_output_clean = self.recognizer(clean_images_01, training=False)
            recognizer_output_generated = self.recognizer(generated_images_01, training=False)
            
            # Extract logits and features
            if isinstance(recognizer_output_clean, (list, tuple)):
                clean_logits = recognizer_output_clean[0]
                clean_feature_map = recognizer_output_clean[1]
                generated_logits = recognizer_output_generated[0]
                generated_feature_map = recognizer_output_generated[1]
            else:
                clean_logits = recognizer_output_clean
                generated_logits = recognizer_output_generated
                # Dummy features if not available
                clean_feature_map = tf.zeros([batch_size, 1], dtype=tf.float32)
                generated_feature_map = tf.zeros([batch_size, 1], dtype=tf.float32)
            
            generated_text_pred = tf.argmax(generated_logits, axis=-1, output_type=tf.int32)
            
            # Get fake output from discriminator
            fake_output = self.discriminator([generated_images, generated_text_pred], training=True)
            
            # Generator losses
            adversarial_loss = self.bce_loss(real_labels, fake_output)
            pixel_loss = self.mae_loss(clean_images_tanh, generated_images)
            rec_feat_loss = self.mse_loss(clean_feature_map, generated_feature_map)
            
            # Perceptual loss
            if self.perceptual_loss_layer is not None:
                perceptual_loss = self.perceptual_loss_layer(clean_images_tanh, generated_images)
            else:
                perceptual_loss = 0.0
            
            # CTC loss
            label_len = tf.math.count_nonzero(ground_truth_text, axis=1, dtype=tf.int32)
            logit_len = tf.fill([batch_size], generated_logits.shape[1])
            
            ctc_loss_raw = tf.reduce_mean(
                tf.nn.ctc_loss(
                    labels=tf.cast(ground_truth_text, tf.int32),
                    logits=generated_logits,
                    label_length=label_len,
                    logit_length=logit_len,
                    logits_time_major=False,
                    blank_index=0
                )
            )
            ctc_loss = tf.clip_by_value(ctc_loss_raw, 0.0, self.ctc_loss_clip_max)
            
            # Total generator loss
            total_gen_loss = (
                (self.adv_loss_weight * adversarial_loss) +
                (self.pixel_loss_weight * pixel_loss) +
                (self.rec_feat_loss_weight * rec_feat_loss) +
                (self.perceptual_loss_weight * perceptual_loss) +
                (self.ctc_loss_weight * ctc_loss)
            )
        
        # Apply generator gradients
        gen_gradients = gen_tape.gradient(total_gen_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        
        # Update metrics
        self.gen_loss_tracker.update_state(total_gen_loss)
        self.disc_loss_tracker.update_state(total_disc_loss)
        self.adv_loss_tracker.update_state(adversarial_loss)
        self.pixel_loss_tracker.update_state(pixel_loss)
        self.rec_feat_loss_tracker.update_state(rec_feat_loss)
        self.percep_loss_tracker.update_state(perceptual_loss)
        self.ctc_loss_tracker.update_state(ctc_loss)
        
        return {m.name: m.result() for m in self.metrics}


def main(args):
    """Main training function."""
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_id.split(',')]
    num_gpus = len(gpu_ids)
    
    print(f"\n{'='*70}")
    print(f"  Dual-Modal GAN-HTR Training (Keras fit() API)")
    print(f"{'='*70}")
    print(f"  Multi-GPU: {num_gpus}x GPU (IDs: {gpu_ids})")
    print(f"  Strategy: MirroredStrategy (native multi-GPU)")
    print(f"  Precision: Pure FP32")
    print(f"{'='*70}\n")
    
    # Create distribution strategy
    if num_gpus > 1:
        # Multi-GPU training
        strategy = tf.distribute.MirroredStrategy()
        print(f"✅ MirroredStrategy initialized with {strategy.num_replicas_in_sync} GPUs")
    else:
        # Single GPU training
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        strategy = tf.distribute.get_strategy()
        print(f"✅ Single GPU mode: GPU {args.gpu_id}")
    
    # Setup directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    # Load dataset and charset
    print("\n[Phase 1/3] Loading Dataset...")
    charset = read_charlist(args.charset_path)
    vocab_size = len(charset) + 1
    print(f"  Charset: {vocab_size} characters")
    
    # Calculate global batch size for multi-GPU
    # model.fit() handles per-replica batching automatically
    global_batch_size = args.batch_size * num_gpus if num_gpus > 1 else args.batch_size
    
    train_dataset, val_dataset, test_dataset, train_count, val_count, test_count = create_dataset(
        args.tfrecord_path,
        global_batch_size
    )
    
    print(f"  Training samples: {train_count}")
    print(f"  Validation samples: {val_count}")
    print(f"  Test samples: {test_count} (LOCKED for final eval)")
    print(f"  Global batch size: {global_batch_size} (per-GPU: {args.batch_size})")
    
    # Build models within strategy scope
    with strategy.scope():
        print("\n[Phase 2/3] Building Models...")
        
        # Generator
        if args.generator_version == 'enhanced':
            generator = unet_enhanced(input_size=(1024, 128, 1))
            print("  Generator: Enhanced U-Net")
        elif args.generator_version == 'enhanced_v2':
            generator = unet_enhanced_v2(input_size=(1024, 128, 1))
            print("  Generator: Enhanced V2 U-Net")
        else:
            generator = unet(input_size=(1024, 128, 1))
            print("  Generator: Base U-Net")
        
        # Discriminator
        if args.discriminator_version == 'enhanced_v2_fixed':
            discriminator = build_dual_modal_discriminator_enhanced_v2_fixed(
                img_shape=(1024, 128, 1),
                vocab_size=vocab_size,
                max_text_len=128
            )
            print("  Discriminator: Enhanced V2 Fixed")
        elif args.discriminator_version == 'enhanced_v2':
            discriminator = build_dual_modal_discriminator_enhanced_v2(
                img_shape=(1024, 128, 1),
                vocab_size=vocab_size,
                max_text_len=128
            )
            print("  Discriminator: Enhanced V2")
        else:
            discriminator = build_dual_modal_discriminator(
                img_shape=(1024, 128, 1),
                vocab_size=vocab_size,
                max_text_len=128
            )
            print("  Discriminator: Base")
        
        # Recognizer
        use_rec_feat_loss = args.rec_feat_loss_weight > 0.0
        recognizer = load_frozen_recognizer(
            weights_path=args.recognizer_weights,
            charset_size=vocab_size - 1,
            return_feature_map=use_rec_feat_loss
        )
        print("  Recognizer: Frozen HTR model")
        
        # Perceptual loss
        if args.perceptual_loss_weight > 0:
            perceptual_loss_layer = create_perceptual_loss()
            print("  Perceptual Loss: VGG-based")
        else:
            perceptual_loss_layer = None
            print("  Perceptual Loss: Disabled")
        
        # Optimizers
        g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.lr_g,
            beta_1=0.5,
            clipnorm=args.gradient_clip_norm
        )
        d_optimizer = tf.keras.optimizers.SGD(
            learning_rate=args.lr_d,
            momentum=0.9,
            clipnorm=args.gradient_clip_norm
        )
        
        print("  Optimizers: Adam (G), SGD (D)")
        
        # Create GAN model
        gan_model = DualModalGAN(
            generator=generator,
            discriminator=discriminator,
            recognizer=recognizer,
            perceptual_loss_layer=perceptual_loss_layer,
            pixel_loss_weight=args.pixel_loss_weight,
            adv_loss_weight=args.adv_loss_weight,
            rec_feat_loss_weight=args.rec_feat_loss_weight,
            ctc_loss_weight=args.ctc_loss_weight,
            perceptual_loss_weight=args.perceptual_loss_weight,
            ctc_loss_clip_max=args.ctc_loss_clip_max,
            discriminator_mode=args.discriminator_mode
        )
        
        # Compile model
        gan_model.compile(g_optimizer=g_optimizer, d_optimizer=d_optimizer)
        print("  GAN Model: Compiled")
        
        # Build model explicitly
        # Input shape: (batch_size, 1024, 128, 1)
        gan_model.build((global_batch_size, 1024, 128, 1))
        print("  GAN Model: Built")
    
    # Prepare dataset for fit()
    # Format: (degraded, (clean, text))
    def format_dataset(dataset):
        """Format dataset for model.fit() input signature."""
        def format_fn(degraded, clean, text):
            return degraded, (clean, text)
        return dataset.map(format_fn)
    
    train_dataset_formatted = format_dataset(train_dataset)
    val_dataset_formatted = format_dataset(val_dataset)
    
    # Callbacks
    print("\n[Phase 3/3] Setting up Callbacks...")
    callbacks = []
    
    # Model checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, "ckpt-{epoch:03d}.weights.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_freq='epoch',
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # TensorBoard
    log_dir = os.path.join(args.checkpoint_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=False,
        update_freq='epoch'
    )
    callbacks.append(tensorboard_callback)
    
    # CSV Logger
    csv_path = os.path.join(args.checkpoint_dir, "training_log.csv")
    csv_callback = tf.keras.callbacks.CSVLogger(csv_path, append=True)
    callbacks.append(csv_callback)
    
    print(f"  Checkpoints: {args.checkpoint_dir}")
    print(f"  TensorBoard: {log_dir}")
    print(f"  CSV Log: {csv_path}")
    
    # Calculate steps
    steps_per_epoch = args.steps_per_epoch or (train_count // global_batch_size)
    validation_steps = val_count // global_batch_size
    
    print(f"\n{'='*70}")
    print(f"  Starting Training...")
    print(f"{'='*70}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Validation steps: {validation_steps}")
    print(f"{'='*70}\n")
    
    # Train with model.fit()
    history = gan_model.fit(
        train_dataset_formatted,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset_formatted,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "final_model")
    gan_model.save_weights(final_path)
    print(f"\n✅ Training complete! Final model saved to: {final_path}")
    
    # Save training history
    history_path = os.path.join(args.checkpoint_dir, "history.json")
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    print(f"✅ Training history saved to: {history_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Dual-Modal GAN with Keras fit() API')
    
    # Model selection
    parser.add_argument('--generator_version', type=str, default='enhanced',
                       choices=['base', 'enhanced', 'enhanced_v2'])
    parser.add_argument('--discriminator_version', type=str, default='enhanced_v2_fixed',
                       choices=['base', 'enhanced_v2', 'enhanced_v2_fixed'])
    
    # Paths
    parser.add_argument('--tfrecord_path', type=str,
                       default='dual_modal_gan/data/dataset_gan.tfrecord')
    parser.add_argument('--charset_path', type=str,
                       default='real_data_preparation/real_data_charlist.txt')
    parser.add_argument('--recognizer_weights', type=str,
                       default='models/best_htr_recognizer/best_model.weights.h5')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='dual_modal_gan/checkpoints/keras_fit_training')
    parser.add_argument('--sample_dir', type=str,
                       default='dual_modal_gan/outputs/keras_fit_samples')
    
    # Training config
    parser.add_argument('--gpu_id', type=str, default='0,1',
                       help='GPU IDs (comma-separated for multi-GPU, e.g., "0,1")')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Per-GPU batch size')
    parser.add_argument('--steps_per_epoch', type=int, default=None)
    
    # Optimizers
    parser.add_argument('--lr_g', type=float, default=0.0002)
    parser.add_argument('--lr_d', type=float, default=0.0002)
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0)
    
    # Loss weights
    parser.add_argument('--pixel_loss_weight', type=float, default=100.0)
    parser.add_argument('--adv_loss_weight', type=float, default=1.0)
    parser.add_argument('--rec_feat_loss_weight', type=float, default=8.0)
    parser.add_argument('--ctc_loss_weight', type=float, default=0.15)
    parser.add_argument('--perceptual_loss_weight', type=float, default=2.0)
    parser.add_argument('--ctc_loss_clip_max', type=float, default=300.0)
    
    # Discriminator mode
    parser.add_argument('--discriminator_mode', type=str, default='predicted',
                       choices=['predicted', 'ground_truth'])
    
    args = parser.parse_args()
    main(args)
