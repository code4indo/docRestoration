#!/usr/bin/env python3
"""
Baseline 1: Plain U-Net (No GAN, No HTR-aware loss)
Train standard U-Net untuk image enhancement tanpa adversarial training
Hanya menggunakan L1 + SSIM loss

Usage:
    poetry run python dual_modal_gan/scripts/train_baseline_unet.py \
        --epochs 50 --batch_size 4 --gpu_id 1
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime
import tensorflow as tf
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dual_modal_gan.src.models.generator import unet


def _parse_tfrecord_fn(example_proto):
    """Parse TFRecord example."""
    feature_description = {
        'degraded_image_raw': tf.io.FixedLenFeature([], tf.string),
        'degraded_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'clean_image_raw': tf.io.FixedLenFeature([], tf.string),
        'clean_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'label_shape': tf.io.FixedLenFeature([1], tf.int64),
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
    
    return degraded_image, clean_image


def load_dataset(tfrecord_path, batch_size, shuffle=False, repeat=False):
    """Load dataset from TFRecord."""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024)
    if repeat:
        dataset = dataset.repeat()
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images."""
    return tf.image.ssim(
        tf.constant(img1[np.newaxis, ...]),
        tf.constant(img2[np.newaxis, ...]),
        max_val=1.0
    ).numpy()[0]


class BaselineUNetTrainer:
    def __init__(self, args):
        self.args = args
        self.setup_gpu()
        self.setup_directories()
        
        # Build model
        print("\n🏗️  Building U-Net model...")
        self.generator = unet()
        print(f"  ✅ Generator built: {self.generator.count_params():,} parameters")
        
        # Optimizer
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate,
            beta_1=0.9,
            beta_2=0.999
        )
        
        # Metrics tracking
        self.history = {
            'epoch': [],
            'g_loss': [],
            'l1_loss': [],
            'ssim_loss': [],
            'psnr': [],
            'ssim': [],
            'time': []
        }
        
        self.best_psnr = 0.0
        self.best_ssim = 0.0
    
    def setup_gpu(self):
        """Setup GPU configuration."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Use specified GPU
                tf.config.set_visible_devices(gpus[self.args.gpu_id], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[self.args.gpu_id], True)
                print(f"✅ Using GPU: {gpus[self.args.gpu_id].name}")
            except Exception as e:
                print(f"⚠️  GPU setup failed: {e}")
    
    def setup_directories(self):
        """Setup output directories."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"dual_modal_gan/outputs/baseline_unet_{timestamp}"
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.metrics_dir = os.path.join(self.output_dir, "metrics")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir}")
    
    @tf.function
    def train_step(self, degraded_batch, clean_batch):
        """Single training step."""
        with tf.GradientTape() as tape:
            # Generate enhanced images
            generated = self.generator(degraded_batch, training=True)
            
            # L1 loss
            l1_loss = tf.reduce_mean(tf.abs(clean_batch - generated))
            
            # SSIM loss
            ssim_loss = 1.0 - tf.reduce_mean(
                tf.image.ssim(clean_batch, generated, max_val=1.0)
            )
            
            # Total generator loss (no adversarial component)
            g_loss = l1_loss + 0.5 * ssim_loss
        
        # Update generator
        gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        
        return g_loss, l1_loss, ssim_loss, generated
    
    def evaluate_batch(self, degraded_batch, clean_batch):
        """Evaluate on a batch."""
        generated = self.generator(degraded_batch, training=False)
        
        # Calculate metrics
        psnr_values = []
        ssim_values = []
        
        for i in range(generated.shape[0]):
            psnr = calculate_psnr(clean_batch[i].numpy(), generated[i].numpy())
            ssim = calculate_ssim(clean_batch[i].numpy(), generated[i].numpy())
            psnr_values.append(psnr)
            ssim_values.append(ssim)
        
        return np.mean(psnr_values), np.mean(ssim_values)
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*70}")
        print(f"BASELINE 1: PLAIN U-NET TRAINING")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  Epochs: {self.args.epochs}")
        print(f"  Batch size: {self.args.batch_size}")
        print(f"  Learning rate: {self.args.learning_rate}")
        print(f"  Dataset: {self.args.dataset_path}")
        print(f"  Output: {self.output_dir}")
        
        # Load datasets
        print(f"\n📁 Loading datasets...")
        train_dataset = load_dataset(
            self.args.dataset_path,
            batch_size=self.args.batch_size,
            shuffle=True,
            repeat=True
        )
        
        # Use subset of training data for validation (since val dataset doesn't exist)
        val_dataset = load_dataset(
            self.args.dataset_path,
            batch_size=self.args.batch_size,
            shuffle=False,
            repeat=False
        ).take(20)  # Use 20 batches for validation
        print(f"  ✅ Datasets loaded (using training subset for validation)")
        
        # Calculate steps
        steps_per_epoch = self.args.steps_per_epoch
        
        print(f"\n🚀 Starting training...")
        start_time = time.time()
        
        for epoch in range(self.args.epochs):
            epoch_start = time.time()
            
            # Training phase
            g_losses = []
            l1_losses = []
            ssim_losses = []
            
            train_iter = iter(train_dataset)
            
            for step in range(steps_per_epoch):
                try:
                    degraded_batch, clean_batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataset)
                    degraded_batch, clean_batch = next(train_iter)
                
                g_loss, l1_loss, ssim_loss, _ = self.train_step(degraded_batch, clean_batch)
                
                g_losses.append(float(g_loss))
                l1_losses.append(float(l1_loss))
                ssim_losses.append(float(ssim_loss))
                
                # Progress
                if (step + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{self.args.epochs} - Step {step+1}/{steps_per_epoch} - "
                          f"G_loss: {np.mean(g_losses[-20:]):.4f}, "
                          f"L1: {np.mean(l1_losses[-20:]):.4f}, "
                          f"SSIM: {np.mean(ssim_losses[-20:]):.4f}")
            
            # Validation phase
            psnr_values = []
            ssim_values = []
            
            for degraded_batch, clean_batch in val_dataset:
                psnr, ssim = self.evaluate_batch(degraded_batch, clean_batch)
                psnr_values.append(psnr)
                ssim_values.append(ssim)
            
            avg_psnr = np.mean(psnr_values)
            avg_ssim = np.mean(ssim_values)
            
            epoch_time = time.time() - epoch_start
            
            # Update history
            self.history['epoch'].append(epoch + 1)
            self.history['g_loss'].append(float(np.mean(g_losses)))
            self.history['l1_loss'].append(float(np.mean(l1_losses)))
            self.history['ssim_loss'].append(float(np.mean(ssim_losses)))
            self.history['psnr'].append(float(avg_psnr))
            self.history['ssim'].append(float(avg_ssim))
            self.history['time'].append(float(epoch_time))
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1}/{self.args.epochs} Summary:")
            print(f"  G_loss: {np.mean(g_losses):.4f}")
            print(f"  L1_loss: {np.mean(l1_losses):.4f}")
            print(f"  SSIM_loss: {np.mean(ssim_losses):.4f}")
            print(f"  Val PSNR: {avg_psnr:.2f} dB")
            print(f"  Val SSIM: {avg_ssim:.4f}")
            print(f"  Time: {epoch_time:.1f}s")
            
            # Save best model
            if avg_psnr > self.best_psnr:
                self.best_psnr = avg_psnr
                self.best_ssim = avg_ssim
                print(f"  ✅ New best PSNR! Saving model...")
                self.save_checkpoint('best_model')
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')
            
            # Save history
            self.save_history()
            
            print(f"{'─'*70}\n")
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETED!")
        print(f"{'='*70}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best PSNR: {self.best_psnr:.2f} dB")
        print(f"Best SSIM: {self.best_ssim:.4f}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\n")
    
    def save_checkpoint(self, name):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, name)
        os.makedirs(checkpoint_path, exist_ok=True)
        self.generator.save_weights(os.path.join(checkpoint_path, 'generator.weights.h5'))
    
    def save_history(self):
        """Save training history."""
        history_path = os.path.join(self.metrics_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train Baseline 1: Plain U-Net')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--steps_per_epoch', type=int, default=100, help='Steps per epoch')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID to use')
    parser.add_argument('--dataset_path', type=str,
                       default='dual_modal_gan/data/dataset_gan.tfrecord',
                       help='Path to training dataset')
    parser.add_argument('--val_dataset_path', type=str,
                       default='dual_modal_gan/data/dataset_gan_val.tfrecord',
                       help='Path to validation dataset')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = BaselineUNetTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
