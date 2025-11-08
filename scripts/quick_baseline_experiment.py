#!/usr/bin/env python3
"""
Quick Baseline Experiment untuk Paper Validation
Train U-Net baseline (no GAN, no HTR loss) untuk validasi claim di Table VII
Target: 5 epochs, evaluate SSIM dan CER untuk comparison dengan frozen recognizer

Usage:
    poetry run python scripts/quick_baseline_experiment.py
"""

import os
import sys
import time
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dual_modal_gan.src.models.generator import unet
from dual_modal_gan.src.models.recognizer import create_htr_model


def decode_ctc_predictions(logits, charset):
    """Decode CTC predictions to text."""
    results = []
    predictions = tf.argmax(logits, axis=-1)
    
    for pred in predictions:
        pred_np = pred.numpy()
        
        # CTC decode: remove consecutive duplicates and blank (0)
        result = []
        prev = -1
        for p in pred_np:
            if p != prev and p != 0:
                result.append(p - 1)  # charset index (0 is blank, 1 is first char)
            prev = p
        
        decoded = ''.join([charset[i] if 0 <= i < len(charset) else '<?>' for i in result])
        results.append(decoded)
    
    return results


def decode_label(label_ids, charset):
    """Decode label IDs to text string."""
    decoded_chars = []
    for label_id in label_ids:
        if label_id > 0 and label_id <= len(charset):
            decoded_chars.append(charset[label_id - 1])
    return ''.join(decoded_chars)


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
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])
    degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])
    
    # Deserialize clean image
    clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
    clean_image = tf.reshape(clean_image, clean_image_shape)
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])
    clean_image = tf.ensure_shape(clean_image, [1024, 128, 1])
    
    # Deserialize label
    label_shape = tf.cast(example['label_shape'], tf.int32)
    label = tf.io.decode_raw(example['label_raw'], tf.int32)
    label = tf.reshape(label, label_shape)
    
    return degraded_image, clean_image, label


def load_dataset(tfrecord_path, batch_size, shuffle=False):
    """Load dataset from TFRecord."""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=512)
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images."""
    return tf.image.ssim(img1, img2, max_val=1.0)


def calculate_cer(recognizer, restored_images, true_labels, charset):
    """Calculate CER using recognizer."""
    # Get predictions
    predictions = recognizer(restored_images, training=False)
    pred_texts = decode_ctc_predictions(predictions, charset)
    
    # Decode true labels
    true_texts = []
    for label in true_labels:
        label_np = label.numpy()
        true_text = decode_label(label_np, charset)
        true_texts.append(true_text)
    
    # Calculate CER
    total_chars = 0
    total_errors = 0
    
    for pred, truth in zip(pred_texts, true_texts):
        # Levenshtein distance
        errors = levenshtein_distance(pred, truth)
        total_errors += errors
        total_chars += len(truth)
    
    cer = (total_errors / total_chars * 100) if total_chars > 0 else 0.0
    return cer


def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


class QuickBaselineExperiment:
    def __init__(self):
        self.setup_gpu()
        
        # Paths
        self.dataset_path = 'dual_modal_gan/data/dataset_gan.tfrecord'
        self.charset_path = 'real_data_preparation/real_data_charlist.txt'
        self.recognizer_weights = '/home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5'
        
        # Hyperparameters
        self.epochs = 5
        self.batch_size = 4
        self.steps_per_epoch = 50  # Quick experiment
        self.eval_samples = 150  # 150 samples untuk validation
        
        print("\n" + "="*60)
        print("🧪 QUICK BASELINE EXPERIMENT - Paper Validation")
        print("="*60)
        print(f"Purpose: Validate baseline comparison claim in Table VII")
        print(f"Training: {self.epochs} epochs, {self.steps_per_epoch} steps/epoch")
        print(f"Evaluation: {self.eval_samples} samples")
        print("="*60 + "\n")
        
    def setup_gpu(self):
        """Setup GPU."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Use GPU 0
                tf.config.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print(f"✅ Using GPU: {gpus[0]}")
            except RuntimeError as e:
                print(f"❌ GPU setup error: {e}")
        else:
            print("⚠️  No GPU found, using CPU")
    
    def build_models(self):
        """Build U-Net generator and recognizer."""
        print("\n🏗️  Building models...")
        
        # Build U-Net generator
        self.generator = unet()
        print(f"  ✅ U-Net Generator: {self.generator.count_params():,} parameters")
        
        # Build recognizer for evaluation
        with open(self.charset_path, 'r', encoding='utf-8') as f:
            self.charset = [line.strip() for line in f]
        
        self.recognizer = create_htr_model(
            charset_size=len(self.charset) + 1,  # +1 for CTC blank
            proj_dim=512,
            target_time_steps=128
        )
        
        # Load pre-trained recognizer weights
        print(f"  📥 Loading recognizer weights: {self.recognizer_weights}")
        self.recognizer.load_weights(self.recognizer_weights)
        print(f"  ✅ Recognizer loaded: {self.recognizer.count_params():,} parameters")
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
    
    @tf.function
    def train_step(self, degraded_images, clean_images):
        """Single training step - L1 + SSIM loss only."""
        with tf.GradientTape() as tape:
            # Generate restored images
            restored_images = self.generator(degraded_images, training=True)
            
            # L1 loss
            l1_loss = tf.reduce_mean(tf.abs(restored_images - clean_images))
            
            # SSIM loss
            ssim_val = tf.reduce_mean(tf.image.ssim(restored_images, clean_images, max_val=1.0))
            ssim_loss = 1.0 - ssim_val
            
            # Total loss
            total_loss = l1_loss + ssim_loss
        
        # Update generator
        gradients = tape.gradient(total_loss, self.generator.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        
        return total_loss, l1_loss, ssim_loss
    
    def train(self):
        """Train baseline U-Net."""
        print("\n🚀 Starting baseline training...")
        
        # Load datasets
        train_dataset = load_dataset(self.dataset_path, self.batch_size, shuffle=True)
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Training phase
            losses = []
            l1_losses = []
            ssim_losses = []
            
            train_iter = iter(train_dataset)
            
            for step in range(self.steps_per_epoch):
                try:
                    degraded_batch, clean_batch, _ = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataset)
                    degraded_batch, clean_batch, _ = next(train_iter)
                
                loss, l1_loss, ssim_loss = self.train_step(degraded_batch, clean_batch)
                
                losses.append(float(loss))
                l1_losses.append(float(l1_loss))
                ssim_losses.append(float(ssim_loss))
                
                if (step + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{self.epochs} - Step {step+1}/{self.steps_per_epoch} - "
                          f"Loss: {np.mean(losses[-10:]):.4f}, "
                          f"L1: {np.mean(l1_losses[-10:]):.4f}, "
                          f"SSIM_loss: {np.mean(ssim_losses[-10:]):.4f}")
            
            epoch_time = time.time() - epoch_start
            print(f"  ✅ Epoch {epoch+1} completed in {epoch_time:.1f}s - "
                  f"Avg Loss: {np.mean(losses):.4f}\n")
        
        total_time = time.time() - start_time
        print(f"✅ Training completed in {total_time/60:.1f} minutes\n")
    
    def evaluate(self):
        """Evaluate on validation set."""
        print("\n📊 Evaluating baseline performance...")
        
        # Load validation data
        val_dataset = load_dataset(self.dataset_path, self.batch_size, shuffle=False)
        
        ssim_values = []
        cer_values = []
        num_evaluated = 0
        
        for degraded_batch, clean_batch, label_batch in val_dataset:
            if num_evaluated >= self.eval_samples:
                break
            
            # Generate restored images
            restored_batch = self.generator(degraded_batch, training=False)
            
            # Calculate SSIM
            ssim_batch = calculate_ssim(restored_batch, clean_batch)
            ssim_values.extend(ssim_batch.numpy())
            
            # Calculate CER
            cer = calculate_cer(self.recognizer, restored_batch, label_batch, self.charset)
            cer_values.append(cer)
            
            num_evaluated += self.batch_size
            
            if num_evaluated % 40 == 0:
                print(f"  Evaluated {num_evaluated}/{self.eval_samples} samples...")
        
        # Calculate statistics
        mean_ssim = np.mean(ssim_values)
        std_ssim = np.std(ssim_values)
        mean_cer = np.mean(cer_values)
        std_cer = np.std(cer_values)
        
        print("\n" + "="*60)
        print("📈 BASELINE RESULTS (No GAN, No HTR Loss)")
        print("="*60)
        print(f"Samples evaluated: {num_evaluated}")
        print(f"\nSSIM: {mean_ssim:.4f} ± {std_ssim:.4f}")
        print(f"CER:  {mean_cer:.2f}% ± {std_cer:.2f}%")
        print("="*60 + "\n")
        
        # Save results
        results = {
            'experiment': 'baseline_unet_quick',
            'timestamp': datetime.now().isoformat(),
            'epochs': self.epochs,
            'steps_per_epoch': self.steps_per_epoch,
            'eval_samples': num_evaluated,
            'metrics': {
                'ssim_mean': float(mean_ssim),
                'ssim_std': float(std_ssim),
                'cer_mean': float(mean_cer),
                'cer_std': float(std_cer)
            }
        }
        
        result_path = 'results/baseline_quick_experiment.json'
        os.makedirs('results', exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to: {result_path}\n")
        
        return results


def main():
    experiment = QuickBaselineExperiment()
    experiment.build_models()
    experiment.train()
    results = experiment.evaluate()
    
    print("\n🎯 COMPARISON WITH FROZEN RECOGNIZER APPROACH:")
    print("="*60)
    print("Frozen Recognizer (Production):  SSIM 0.987, CER 34.9%")
    print(f"Baseline U-Net (This exp):       SSIM {results['metrics']['ssim_mean']:.3f}, CER {results['metrics']['cer_mean']:.1f}%")
    print("="*60)
    print("\n✅ Experiment completed! Data can be used for Table VII validation.\n")


if __name__ == '__main__':
    main()
