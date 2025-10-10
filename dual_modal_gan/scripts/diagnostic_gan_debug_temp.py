#!/usr/bin/env python3
"""
Skrip Diagnostik Komprehensif untuk GAN Training Issues
Tujuan: Mengidentifikasi akar penyebab Generator stuck pada output seragam
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from dual_modal_gan.src.models.generator import unet
from dual_modal_gan.src.models.discriminator import build_dual_modal_discriminator
from dual_modal_gan.src.models.recognizer import load_frozen_recognizer

# ============================================================================
# TEST 1: VERIFIKASI DATA PIPELINE
# ============================================================================
def test_data_pipeline():
    """Test apakah data dari TFRecord benar-benar terbaca dengan baik"""
    print("\n" + "="*80)
    print("TEST 1: VERIFIKASI DATA PIPELINE")
    print("="*80)
    
    tfrecord_path = 'dual_modal_gan/data/dataset_gan.tfrecord'
    
    def _parse_tfrecord_fn(example_proto):
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
        
        degraded_image_shape = tf.cast(example['degraded_image_shape'], tf.int32)
        degraded_image = tf.io.decode_raw(example['degraded_image_raw'], tf.float32)
        degraded_image = tf.reshape(degraded_image, degraded_image_shape)

        clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
        clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
        clean_image = tf.reshape(clean_image, clean_image_shape)

        label_shape = tf.cast(example['label_shape'], tf.int32)
        label = tf.io.decode_raw(example['label_raw'], tf.int64)
        label = tf.reshape(label, label_shape)
        label = tf.cast(label, tf.int32)
        
        return degraded_image, clean_image, label
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_tfrecord_fn)
    dataset = dataset.batch(4)
    
    output_dir = Path('dual_modal_gan/outputs/debug_diagnostic')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    degraded_batch, clean_batch, label_batch = next(iter(dataset))
    
    print(f"✅ Degraded batch shape: {degraded_batch.shape}")
    print(f"✅ Clean batch shape: {clean_batch.shape}")
    print(f"✅ Label batch shape: {label_batch.shape}")
    
    # Statistik gambar
    print(f"\n📊 Degraded image stats:")
    print(f"   Min: {tf.reduce_min(degraded_batch):.4f}, Max: {tf.reduce_max(degraded_batch):.4f}")
    print(f"   Mean: {tf.reduce_mean(degraded_batch):.4f}, Std: {tf.math.reduce_std(degraded_batch):.4f}")
    
    print(f"\n📊 Clean image stats:")
    print(f"   Min: {tf.reduce_min(clean_batch):.4f}, Max: {tf.reduce_max(clean_batch):.4f}")
    print(f"   Mean: {tf.reduce_mean(clean_batch):.4f}, Std: {tf.math.reduce_std(clean_batch):.4f}")
    
    # Simpan sample
    for i in range(2):
        cv2.imwrite(str(output_dir / f'test1_degraded_{i}.png'), 
                    (degraded_batch[i].numpy()[:,:,0] * 255).astype(np.uint8))
        cv2.imwrite(str(output_dir / f'test1_clean_{i}.png'), 
                    (clean_batch[i].numpy()[:,:,0] * 255).astype(np.uint8))
    
    print(f"\n✅ Sampel gambar disimpan di: {output_dir}")
    return degraded_batch, clean_batch, label_batch

# ============================================================================
# TEST 2: VERIFIKASI ARSITEKTUR GENERATOR
# ============================================================================
def test_generator_architecture():
    """Test apakah Generator memiliki arsitektur yang benar"""
    print("\n" + "="*80)
    print("TEST 2: VERIFIKASI ARSITEKTUR GENERATOR")
    print("="*80)
    
    generator = unet(input_size=(128, 1024, 1))
    generator.summary()
    
    # Cek aktivasi layer terakhir
    last_layer = generator.layers[-1]
    print(f"\n🔍 Layer terakhir: {last_layer.name}")
    print(f"🔍 Tipe: {type(last_layer)}")
    print(f"🔍 Config: {last_layer.get_config()}")
    
    activation = last_layer.get_config().get('activation', 'none')
    print(f"\n✅ Aktivasi layer output: {activation}")
    
    if activation != 'sigmoid':
        print("⚠️ WARNING: Aktivasi bukan sigmoid! Ini bisa menyebabkan output tidak bounded!")
    
    # Test forward pass dengan input random
    test_input = tf.random.uniform((1, 128, 1024, 1), minval=0, maxval=1)
    test_output = generator(test_input, training=False)
    
    print(f"\n📊 Test forward pass:")
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {test_output.shape}")
    print(f"   Output min: {tf.reduce_min(test_output):.4f}")
    print(f"   Output max: {tf.reduce_max(test_output):.4f}")
    print(f"   Output mean: {tf.reduce_mean(test_output):.4f}")
    print(f"   Output std: {tf.math.reduce_std(test_output):.4f}")
    
    # Test dengan input hitam dan putih
    black_input = tf.zeros((1, 128, 1024, 1))
    white_input = tf.ones((1, 128, 1024, 1))
    
    black_output = generator(black_input, training=False)
    white_output = generator(white_input, training=False)
    
    print(f"\n📊 Test dengan input ekstrem:")
    print(f"   Black input -> Output mean: {tf.reduce_mean(black_output):.4f}, std: {tf.math.reduce_std(black_output):.4f}")
    print(f"   White input -> Output mean: {tf.reduce_mean(white_output):.4f}, std: {tf.math.reduce_std(white_output):.4f}")
    
    # Cek apakah ada gradient flow
    print(f"\n🔍 Cek trainable parameters:")
    total_params = sum([tf.size(var).numpy() for var in generator.trainable_variables])
    print(f"   Total trainable params: {total_params:,}")
    
    return generator

# ============================================================================
# TEST 3: VERIFIKASI GRADIENT FLOW
# ============================================================================
def test_gradient_flow(generator, degraded_batch, clean_batch):
    """Test apakah gradient mengalir dengan baik"""
    print("\n" + "="*80)
    print("TEST 3: VERIFIKASI GRADIENT FLOW")
    print("="*80)
    
    optimizer = tf.keras.optimizers.Adam(2e-4)
    mae_loss_fn = tf.keras.losses.MeanAbsoluteError()
    
    # Ambil 1 sample
    degraded_input = degraded_batch[:1]
    clean_target = clean_batch[:1]
    
    print(f"Input shape: {degraded_input.shape}")
    print(f"Target shape: {clean_target.shape}")
    
    # Forward + backward pass
    with tf.GradientTape() as tape:
        generated = generator(degraded_input, training=True)
        loss = mae_loss_fn(clean_target, generated)
    
    gradients = tape.gradient(loss, generator.trainable_variables)
    
    print(f"\n📊 Loss value: {loss.numpy():.6f}")
    print(f"\n🔍 Gradient statistics:")
    
    grad_norms = []
    zero_grads = 0
    nan_grads = 0
    
    for i, (grad, var) in enumerate(zip(gradients, generator.trainable_variables)):
        if grad is None:
            print(f"   Layer {i} ({var.name}): NONE gradient!")
            continue
        
        grad_norm = tf.norm(grad).numpy()
        grad_norms.append(grad_norm)
        
        if grad_norm == 0:
            zero_grads += 1
        if np.isnan(grad_norm):
            nan_grads += 1
        
        if i < 5 or i >= len(gradients) - 5:  # Print first and last 5
            print(f"   Layer {i} ({var.name[:40]}): norm={grad_norm:.6e}")
    
    print(f"\n📊 Gradient summary:")
    print(f"   Total layers: {len(gradients)}")
    print(f"   Layers with zero gradients: {zero_grads}")
    print(f"   Layers with NaN gradients: {nan_grads}")
    print(f"   Min gradient norm: {min(grad_norms):.6e}")
    print(f"   Max gradient norm: {max(grad_norms):.6e}")
    print(f"   Mean gradient norm: {np.mean(grad_norms):.6e}")
    
    if zero_grads > len(gradients) * 0.5:
        print("⚠️ WARNING: Lebih dari 50% layer memiliki gradient = 0!")
    
    if nan_grads > 0:
        print("⚠️ WARNING: Ada gradient NaN!")
    
    # Apply gradients
    optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    
    # Test output setelah 1 step
    generated_after = generator(degraded_input, training=False)
    loss_after = mae_loss_fn(clean_target, generated_after)
    
    print(f"\n📊 After 1 gradient step:")
    print(f"   Loss before: {loss.numpy():.6f}")
    print(f"   Loss after: {loss_after.numpy():.6f}")
    print(f"   Loss change: {(loss_after - loss).numpy():.6f}")

# ============================================================================
# TEST 4: TRAINING LOOP MINI (10 STEPS)
# ============================================================================
def test_mini_training(generator, degraded_batch, clean_batch):
    """Test training loop mini untuk melihat apakah loss turun"""
    print("\n" + "="*80)
    print("TEST 4: MINI TRAINING LOOP (10 steps)")
    print("="*80)
    
    optimizer = tf.keras.optimizers.Adam(2e-4)
    mae_loss_fn = tf.keras.losses.MeanAbsoluteError()
    
    output_dir = Path('dual_modal_gan/outputs/debug_diagnostic')
    
    losses = []
    
    print("\nMenjalankan 10 langkah training dengan L1 loss saja...")
    
    for step in range(10):
        with tf.GradientTape() as tape:
            generated = generator(degraded_batch, training=True)
            loss = mae_loss_fn(clean_batch, generated)
        
        gradients = tape.gradient(loss, generator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
        
        losses.append(loss.numpy())
        
        print(f"Step {step+1}/10: Loss = {loss.numpy():.6f}")
        
        # Simpan output setiap 5 step
        if (step + 1) % 5 == 0:
            gen_img = (generated[0].numpy()[:,:,0] * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / f'test4_generated_step{step+1}.png'), gen_img)
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o')
    plt.xlabel('Step')
    plt.ylabel('L1 Loss')
    plt.title('Mini Training Loss Curve')
    plt.grid(True)
    plt.savefig(output_dir / 'test4_loss_curve.png')
    plt.close()
    
    print(f"\n📊 Loss statistics:")
    print(f"   Initial loss: {losses[0]:.6f}")
    print(f"   Final loss: {losses[-1]:.6f}")
    print(f"   Loss reduction: {(losses[0] - losses[-1]):.6f}")
    print(f"   Loss reduction %: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")
    
    if losses[-1] >= losses[0] * 0.95:
        print("⚠️ WARNING: Loss tidak turun signifikan! Ada masalah learning!")
    else:
        print("✅ Loss turun dengan baik!")
    
    # Cek output diversity
    gen_output = generator(degraded_batch, training=False).numpy()
    output_stds = [np.std(gen_output[i]) for i in range(gen_output.shape[0])]
    
    print(f"\n📊 Output diversity (std per gambar):")
    for i, std in enumerate(output_stds):
        print(f"   Sample {i}: std = {std:.6f}")
    
    if max(output_stds) < 0.01:
        print("⚠️ WARNING: Output sangat seragam (low std)! Generator collapse!")

# ============================================================================
# TEST 5: WEIGHT INITIALIZATION CHECK
# ============================================================================
def test_weight_initialization():
    """Test apakah weight initialization bagus"""
    print("\n" + "="*80)
    print("TEST 5: WEIGHT INITIALIZATION CHECK")
    print("="*80)
    
    generator = unet(input_size=(128, 1024, 1))
    
    print("\nMemeriksa distribusi weight pada beberapa layer...")
    
    for i, layer in enumerate(generator.layers):
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
        
        kernel = weights[0]
        
        # Hanya print layer Conv2D
        if 'conv' in layer.name.lower():
            print(f"\n🔍 Layer: {layer.name}")
            print(f"   Shape: {kernel.shape}")
            print(f"   Mean: {np.mean(kernel):.6e}")
            print(f"   Std: {np.std(kernel):.6e}")
            print(f"   Min: {np.min(kernel):.6e}")
            print(f"   Max: {np.max(kernel):.6e}")
            
            if np.std(kernel) < 1e-6:
                print("   ⚠️ WARNING: Std sangat kecil!")
            if np.abs(np.mean(kernel)) > 0.1:
                print("   ⚠️ WARNING: Mean tidak mendekati 0!")

# ============================================================================
# TEST 6: DISCRIMINATOR BEHAVIOR CHECK
# ============================================================================
def test_discriminator_behavior():
    """Test apakah discriminator berfungsi dengan baik"""
    print("\n" + "="*80)
    print("TEST 6: DISCRIMINATOR BEHAVIOR CHECK")
    print("="*80)
    
    discriminator = build_dual_modal_discriminator(
        img_shape=(128, 1024, 1),
        vocab_size=100,
        max_text_len=128
    )
    
    # Test dengan input dummy
    real_img = tf.random.uniform((2, 128, 1024, 1))
    fake_img = tf.random.uniform((2, 128, 1024, 1))
    text_pred = tf.random.uniform((2, 128), minval=0, maxval=100, dtype=tf.int32)
    
    real_score = discriminator([real_img, text_pred], training=False)
    fake_score = discriminator([fake_img, text_pred], training=False)
    
    print(f"\n📊 Discriminator scores (before training):")
    print(f"   Real image scores: {real_score.numpy().flatten()}")
    print(f"   Fake image scores: {fake_score.numpy().flatten()}")
    print(f"   Mean real: {tf.reduce_mean(real_score):.4f}")
    print(f"   Mean fake: {tf.reduce_mean(fake_score):.4f}")
    
    # Test apakah discriminator bisa belajar
    optimizer = tf.keras.optimizers.Adam(2e-4)
    bce_loss = tf.keras.losses.BinaryCrossentropy()
    
    print("\nTraining discriminator 10 steps...")
    for step in range(10):
        with tf.GradientTape() as tape:
            real_out = discriminator([real_img, text_pred], training=True)
            fake_out = discriminator([fake_img, text_pred], training=True)
            
            loss_real = bce_loss(tf.ones_like(real_out) * 0.9, real_out)
            loss_fake = bce_loss(tf.zeros_like(fake_out), fake_out)
            total_loss = loss_real + loss_fake
        
        grads = tape.gradient(total_loss, discriminator.trainable_variables)
        optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
        
        if (step + 1) % 5 == 0:
            print(f"   Step {step+1}: Loss={total_loss.numpy():.4f}")
    
    real_score_after = discriminator([real_img, text_pred], training=False)
    fake_score_after = discriminator([fake_img, text_pred], training=False)
    
    print(f"\n📊 After training:")
    print(f"   Mean real: {tf.reduce_mean(real_score_after):.4f}")
    print(f"   Mean fake: {tf.reduce_mean(fake_score_after):.4f}")
    
    if tf.reduce_mean(real_score_after) > tf.reduce_mean(fake_score_after):
        print("✅ Discriminator belajar membedakan real vs fake!")
    else:
        print("⚠️ WARNING: Discriminator tidak belajar dengan baik!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("\n" + "="*80)
    print("DIAGNOSTIC GAN DEBUG - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("\nTujuan: Mengidentifikasi root cause mengapa Generator stuck")
    print("pada output seragam (hitam/abu-abu/putih)")
    print("="*80)
    
    # Run all tests
    try:
        degraded_batch, clean_batch, label_batch = test_data_pipeline()
        generator = test_generator_architecture()
        test_gradient_flow(generator, degraded_batch, clean_batch)
        test_mini_training(generator, degraded_batch, clean_batch)
        test_weight_initialization()
        test_discriminator_behavior()
        
        print("\n" + "="*80)
        print("SEMUA TEST SELESAI")
        print("="*80)
        print("\nCek hasil di: dual_modal_gan/outputs/debug_diagnostic/")
        print("\nAnalisis hasil test untuk menentukan langkah selanjutnya.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
