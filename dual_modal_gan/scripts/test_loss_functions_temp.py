#!/usr/bin/env python3
"""
Test Loss Functions secara Terpisah
Tujuan: Memverifikasi bahwa setiap komponen loss bekerja dengan benar
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from dual_modal_gan.src.models.generator import unet
from dual_modal_gan.src.models.recognizer import load_frozen_recognizer
from dual_modal_gan.src.models.discriminator import build_dual_modal_discriminator

def read_charlist(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

# ============================================================================
# TEST L1/MAE LOSS
# ============================================================================
def test_l1_loss():
    """Test L1/MAE loss secara terpisah"""
    print("\n" + "="*80)
    print("TEST: L1/MAE PIXEL LOSS")
    print("="*80)
    
    mae_loss_fn = tf.keras.losses.MeanAbsoluteError()
    
    # Test case 1: Identical images
    img1 = tf.random.uniform((4, 128, 1024, 1))
    img2 = tf.identity(img1)
    
    loss = mae_loss_fn(img1, img2)
    print(f"\n✅ Test 1 - Identical images:")
    print(f"   Loss: {loss.numpy():.8f} (should be ~0)")
    
    # Test case 2: Completely different
    img3 = tf.zeros((4, 128, 1024, 1))
    img4 = tf.ones((4, 128, 1024, 1))
    
    loss = mae_loss_fn(img3, img4)
    print(f"\n✅ Test 2 - Black vs White:")
    print(f"   Loss: {loss.numpy():.8f} (should be ~1.0)")
    
    # Test case 3: Half different
    img5 = tf.random.uniform((4, 128, 1024, 1))
    img6 = img5 + tf.random.normal((4, 128, 1024, 1), mean=0, stddev=0.1)
    img6 = tf.clip_by_value(img6, 0, 1)
    
    loss = mae_loss_fn(img5, img6)
    print(f"\n✅ Test 3 - Small noise added:")
    print(f"   Loss: {loss.numpy():.8f} (should be small)")
    
    # Test gradient flow
    generator = unet(input_size=(128, 1024, 1))
    optimizer = tf.keras.optimizers.Adam(2e-4)
    
    input_img = tf.random.uniform((1, 128, 1024, 1))
    target_img = tf.random.uniform((1, 128, 1024, 1))
    
    with tf.GradientTape() as tape:
        output = generator(input_img, training=True)
        loss = mae_loss_fn(target_img, output)
    
    gradients = tape.gradient(loss, generator.trainable_variables)
    grad_norms = [tf.norm(g).numpy() for g in gradients if g is not None]
    
    print(f"\n✅ Gradient flow test:")
    print(f"   Loss value: {loss.numpy():.6f}")
    print(f"   Num gradients: {len(grad_norms)}")
    print(f"   Mean gradient norm: {np.mean(grad_norms):.6e}")
    print(f"   Max gradient norm: {np.max(grad_norms):.6e}")
    print(f"   Min gradient norm: {np.min(grad_norms):.6e}")

# ============================================================================
# TEST ADVERSARIAL LOSS
# ============================================================================
def test_adversarial_loss():
    """Test adversarial loss (BCE) secara terpisah"""
    print("\n" + "="*80)
    print("TEST: ADVERSARIAL LOSS (BCE)")
    print("="*80)
    
    bce_loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    # Test case 1: Perfect prediction
    true_labels = tf.ones((4, 1))
    pred_probs = tf.ones((4, 1)) * 0.99
    
    loss = bce_loss_fn(true_labels, pred_probs)
    print(f"\n✅ Test 1 - Perfect prediction (1 vs 0.99):")
    print(f"   Loss: {loss.numpy():.8f} (should be very small)")
    
    # Test case 2: Wrong prediction
    pred_probs2 = tf.ones((4, 1)) * 0.01
    loss = bce_loss_fn(true_labels, pred_probs2)
    print(f"\n✅ Test 2 - Wrong prediction (1 vs 0.01):")
    print(f"   Loss: {loss.numpy():.8f} (should be large)")
    
    # Test case 3: Random prediction
    pred_probs3 = tf.random.uniform((4, 1))
    loss = bce_loss_fn(true_labels, pred_probs3)
    print(f"\n✅ Test 3 - Random prediction:")
    print(f"   Loss: {loss.numpy():.8f}")
    
    # Test discriminator loss
    discriminator = build_dual_modal_discriminator(
        img_shape=(128, 1024, 1),
        vocab_size=100,
        max_text_len=128
    )
    
    real_img = tf.random.uniform((2, 128, 1024, 1))
    fake_img = tf.random.uniform((2, 128, 1024, 1))
    text_pred = tf.random.uniform((2, 128), minval=0, maxval=100, dtype=tf.int32)
    
    real_output = discriminator([real_img, text_pred], training=False)
    fake_output = discriminator([fake_img, text_pred], training=False)
    
    real_labels = tf.ones((2, 1)) * 0.9
    fake_labels = tf.zeros((2, 1))
    
    disc_loss_real = bce_loss_fn(real_labels, real_output)
    disc_loss_fake = bce_loss_fn(fake_labels, fake_output)
    
    print(f"\n✅ Discriminator loss test:")
    print(f"   Real output: {real_output.numpy().flatten()}")
    print(f"   Fake output: {fake_output.numpy().flatten()}")
    print(f"   Loss (real): {disc_loss_real.numpy():.6f}")
    print(f"   Loss (fake): {disc_loss_fake.numpy():.6f}")
    print(f"   Total D loss: {(disc_loss_real + disc_loss_fake).numpy():.6f}")

# ============================================================================
# TEST CTC LOSS
# ============================================================================
def test_ctc_loss():
    """Test CTC loss untuk recognizer secara terpisah"""
    print("\n" + "="*80)
    print("TEST: CTC LOSS (HTR)")
    print("="*80)
    
    charset_path = 'real_data_preparation/real_data_charlist.txt'
    charset = read_charlist(charset_path)
    vocab_size = len(charset) + 1
    
    print(f"Vocab size: {vocab_size}")
    
    recognizer = load_frozen_recognizer(
        weights_path='transformer_improved_results_fixed_20250930_061317_best/best_model_fixed.weights.h5',
        charset_size=vocab_size - 1
    )
    
    print("✅ Recognizer loaded")
    
    # Test forward pass
    test_img = tf.random.uniform((2, 128, 1024, 1))
    logits = recognizer(test_img, training=False)
    
    print(f"\n📊 Recognizer output:")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Logits min: {tf.reduce_min(logits):.4f}")
    print(f"   Logits max: {tf.reduce_max(logits):.4f}")
    print(f"   Logits mean: {tf.reduce_mean(logits):.4f}")
    
    # Test CTC loss dengan label dummy
    batch_size = 2
    ground_truth_text = tf.constant([[5, 10, 15, 20, 0, 0], [3, 7, 11, 0, 0, 0]], dtype=tf.int32)
    
    label_len = tf.math.count_nonzero(ground_truth_text, axis=1, keepdims=False, dtype=tf.int32)
    logit_len = tf.fill([batch_size], logits.shape[1])
    
    print(f"\n📊 CTC Loss inputs:")
    print(f"   Label shape: {ground_truth_text.shape}")
    print(f"   Label lengths: {label_len.numpy()}")
    print(f"   Logit lengths: {logit_len.numpy()}")
    
    try:
        ctc_loss = tf.nn.ctc_loss(
            labels=tf.cast(ground_truth_text, tf.int32),
            logits=logits,
            label_length=label_len,
            logit_length=logit_len,
            logits_time_major=False,
            blank_index=0
        )
        
        print(f"\n✅ CTC loss computed:")
        print(f"   Per-sample loss: {ctc_loss.numpy()}")
        print(f"   Mean loss: {tf.reduce_mean(ctc_loss).numpy():.6f}")
        
        # Check if loss is reasonable
        mean_loss = tf.reduce_mean(ctc_loss).numpy()
        if np.isnan(mean_loss):
            print("⚠️ WARNING: CTC loss is NaN!")
        elif np.isinf(mean_loss):
            print("⚠️ WARNING: CTC loss is Inf!")
        elif mean_loss > 100:
            print("⚠️ WARNING: CTC loss sangat besar (>100)!")
        else:
            print("✅ CTC loss dalam range normal")
            
    except Exception as e:
        print(f"❌ ERROR in CTC loss: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# TEST COMBINED LOSS
# ============================================================================
def test_combined_loss():
    """Test kombinasi loss seperti di training"""
    print("\n" + "="*80)
    print("TEST: COMBINED LOSS (Full Training Loss)")
    print("="*80)
    
    charset_path = 'real_data_preparation/real_data_charlist.txt'
    charset = read_charlist(charset_path)
    vocab_size = len(charset) + 1
    
    # Build models
    generator = unet(input_size=(128, 1024, 1))
    recognizer = load_frozen_recognizer(
        weights_path='transformer_improved_results_fixed_20250930_061317_best/best_model_fixed.weights.h5',
        charset_size=vocab_size - 1
    )
    discriminator = build_dual_modal_discriminator(
        img_shape=(128, 1024, 1),
        vocab_size=vocab_size,
        max_text_len=128
    )
    
    # Loss functions
    bce_loss_fn = tf.keras.losses.BinaryCrossentropy()
    mae_loss_fn = tf.keras.losses.MeanAbsoluteError()
    
    # Test dengan berbagai kombinasi weight
    test_configs = [
        {'pixel': 100.0, 'adv': 0.0, 'ctc': 0.0, 'name': 'L1 only'},
        {'pixel': 10.0, 'adv': 1.0, 'ctc': 0.0, 'name': 'L1 + Adv'},
        {'pixel': 10.0, 'adv': 0.0, 'ctc': 10.0, 'name': 'L1 + CTC'},
        {'pixel': 10.0, 'adv': 1.0, 'ctc': 10.0, 'name': 'Full (L1+Adv+CTC)'},
    ]
    
    batch_size = 2
    degraded_images = tf.random.uniform((batch_size, 128, 1024, 1))
    clean_images = tf.random.uniform((batch_size, 128, 1024, 1))
    ground_truth_text = tf.constant([[5, 10, 15, 20, 0, 0, 0, 0], 
                                     [3, 7, 11, 0, 0, 0, 0, 0]], dtype=tf.int32)
    
    print("\nMenguji berbagai kombinasi loss weight:\n")
    
    for config in test_configs:
        print(f"{'='*60}")
        print(f"Config: {config['name']}")
        print(f"  pixel_weight={config['pixel']}, adv_weight={config['adv']}, ctc_weight={config['ctc']}")
        print(f"{'='*60}")
        
        with tf.GradientTape() as gen_tape:
            # Forward pass
            generated_images = generator(degraded_images, training=True)
            
            # Pixel loss
            pixel_loss = mae_loss_fn(clean_images, generated_images)
            
            # Adversarial loss
            generated_logits = recognizer(generated_images, training=False)
            generated_text_pred = tf.argmax(generated_logits, axis=-1, output_type=tf.int32)
            fake_output = discriminator([generated_images, generated_text_pred], training=False)
            real_labels = tf.ones([batch_size, 1]) * 0.9
            adversarial_loss = bce_loss_fn(real_labels, fake_output)
            
            # CTC loss
            label_len = tf.math.count_nonzero(ground_truth_text, axis=1, keepdims=False, dtype=tf.int32)
            logit_len = tf.fill([batch_size], generated_logits.shape[1])
            ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(
                labels=tf.cast(ground_truth_text, tf.int32),
                logits=generated_logits,
                label_length=label_len,
                logit_length=logit_len,
                logits_time_major=False,
                blank_index=0
            ))
            
            # Total loss
            total_gen_loss = (config['adv'] * adversarial_loss) + \
                           (config['pixel'] * pixel_loss) + \
                           (config['ctc'] * ctc_loss)
        
        # Check gradients
        gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
        grad_norms = [tf.norm(g).numpy() for g in gradients if g is not None]
        
        print(f"  Pixel loss: {pixel_loss.numpy():.6f} (weighted: {(config['pixel'] * pixel_loss).numpy():.6f})")
        print(f"  Adv loss: {adversarial_loss.numpy():.6f} (weighted: {(config['adv'] * adversarial_loss).numpy():.6f})")
        print(f"  CTC loss: {ctc_loss.numpy():.6f} (weighted: {(config['ctc'] * ctc_loss).numpy():.6f})")
        print(f"  TOTAL LOSS: {total_gen_loss.numpy():.6f}")
        print(f"  Mean gradient norm: {np.mean(grad_norms):.6e}")
        print(f"  Max gradient norm: {np.max(grad_norms):.6e}")
        
        # Check for issues
        if np.isnan(total_gen_loss.numpy()):
            print("  ⚠️ WARNING: Total loss is NaN!")
        elif np.isinf(total_gen_loss.numpy()):
            print("  ⚠️ WARNING: Total loss is Inf!")
        elif np.mean(grad_norms) < 1e-10:
            print("  ⚠️ WARNING: Gradients sangat kecil (vanishing)!")
        elif np.max(grad_norms) > 100:
            print("  ⚠️ WARNING: Gradients sangat besar (exploding)!")
        else:
            print("  ✅ Loss dan gradients dalam range normal")
        
        print()

# ============================================================================
# TEST WEIGHT BALANCE
# ============================================================================
def test_weight_balance():
    """Test apakah weight balance antara loss components reasonable"""
    print("\n" + "="*80)
    print("TEST: LOSS WEIGHT BALANCE ANALYSIS")
    print("="*80)
    
    print("\nAnalisis magnitude relatif dari setiap loss component:\n")
    print("Tujuan: Memastikan tidak ada loss yang terlalu dominan atau terlalu kecil")
    print("sehingga gradient dari loss lain hilang.\n")
    
    # Estimasi typical loss values
    print("📊 Estimasi typical loss values (berdasarkan teori & observasi):")
    print("   - L1/MAE loss: ~0.1 - 0.5 (untuk gambar normalized [0,1])")
    print("   - Adversarial loss (BCE): ~0.3 - 0.7 (random init)")
    print("   - CTC loss: ~10 - 100 (tergantung panjang sequence)")
    
    print("\n📊 Dengan weight default:")
    print("   pixel_weight=100.0, adv_weight=1.0, ctc_weight=10.0")
    print()
    print("   Weighted losses:")
    print("   - L1 weighted: 0.3 * 100.0 = 30.0")
    print("   - Adv weighted: 0.5 * 1.0 = 0.5")
    print("   - CTC weighted: 50.0 * 10.0 = 500.0")
    print()
    print("   ⚠️ MASALAH: CTC loss terlalu dominan (500 vs 30)!")
    print("   Generator akan fokus menurunkan CTC loss, bukan memperbaiki visual quality")
    
    print("\n💡 Rekomendasi weight yang lebih balanced:")
    print("   Opsi 1 - L1 only (pre-training):")
    print("     pixel=100.0, adv=0.0, ctc=0.0")
    print()
    print("   Opsi 2 - L1 + Adv (GAN training):")
    print("     pixel=100.0, adv=1.0, ctc=0.0")
    print()
    print("   Opsi 3 - Balanced (full training):")
    print("     pixel=100.0, adv=2.0, ctc=1.0")
    print("     (CTC weight dikurangi drastis)")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*80)
    print("TEST LOSS FUNCTIONS - COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    try:
        test_l1_loss()
        test_adversarial_loss()
        test_ctc_loss()
        test_combined_loss()
        test_weight_balance()
        
        print("\n" + "="*80)
        print("SEMUA TEST LOSS SELESAI")
        print("="*80)
        print("\n💡 Kesimpulan akan membantu menentukan kombinasi loss weight optimal")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
