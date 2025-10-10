#!/usr/bin/env python3
"""
Test & Inspect Frozen Recognizer Model
Tujuan: Memverifikasi bahwa recognizer benar-benar frozen dan berfungsi dengan baik
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

from dual_modal_gan.src.models.recognizer import load_frozen_recognizer

def read_charlist(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

# ============================================================================
# TEST 1: Model Loading & Architecture
# ============================================================================
def test_model_loading():
    """Test apakah model recognizer ter-load dengan benar"""
    print("\n" + "="*80)
    print("TEST 1: RECOGNIZER MODEL LOADING")
    print("="*80)
    
    charset_path = 'real_data_preparation/real_data_charlist.txt'
    weights_path = 'transformer_improved_results_fixed_20250930_061317_best/best_model_fixed.weights.h5'
    
    charset = read_charlist(charset_path)
    vocab_size = len(charset) + 1
    
    print(f"Charset size: {len(charset)}")
    print(f"Vocab size (with blank): {vocab_size}")
    print(f"Weights path: {weights_path}")
    
    try:
        recognizer = load_frozen_recognizer(
            weights_path=weights_path,
            charset_size=vocab_size - 1
        )
        print("✅ Model loaded successfully!")
        
        # Print architecture
        print(f"\n📊 Model architecture:")
        recognizer.summary()
        
        return recognizer, vocab_size
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ============================================================================
# TEST 2: Check if Model is Frozen
# ============================================================================
def test_frozen_status(recognizer):
    """Verifikasi apakah model benar-benar frozen"""
    print("\n" + "="*80)
    print("TEST 2: VERIFY FROZEN STATUS")
    print("="*80)
    
    if recognizer is None:
        print("❌ No model to test")
        return
    
    print(f"Model trainable: {recognizer.trainable}")
    
    trainable_vars = recognizer.trainable_variables
    non_trainable_vars = recognizer.non_trainable_variables
    
    print(f"\n📊 Variable counts:")
    print(f"   Trainable variables: {len(trainable_vars)}")
    print(f"   Non-trainable variables: {len(non_trainable_vars)}")
    
    if len(trainable_vars) == 0:
        print("✅ Model is properly frozen (no trainable variables)")
    else:
        print(f"⚠️ WARNING: Model has {len(trainable_vars)} trainable variables!")
        print("   Ini bisa menyebabkan recognizer berubah selama GAN training!")
        
        # List trainable variables
        print("\n   Trainable variables:")
        for var in trainable_vars[:5]:  # Show first 5
            print(f"     - {var.name}: {var.shape}")
    
    # Test if gradients can flow through the model
    test_input = tf.random.uniform((1, 128, 1024, 1))
    
    with tf.GradientTape() as tape:
        tape.watch(test_input)
        output = recognizer(test_input, training=False)
        loss = tf.reduce_mean(output)
    
    gradients_wrt_input = tape.gradient(loss, test_input)
    
    print(f"\n📊 Gradient flow test:")
    if gradients_wrt_input is not None:
        grad_norm = tf.norm(gradients_wrt_input).numpy()
        print(f"   Gradient w.r.t. input exists: norm = {grad_norm:.6e}")
        print("   ✅ Backpropagation through recognizer works")
    else:
        print("   ❌ No gradient w.r.t. input!")

# ============================================================================
# TEST 3: Forward Pass with Random Input
# ============================================================================
def test_forward_pass(recognizer):
    """Test forward pass dengan berbagai input"""
    print("\n" + "="*80)
    print("TEST 3: FORWARD PASS WITH VARIOUS INPUTS")
    print("="*80)
    
    if recognizer is None:
        print("❌ No model to test")
        return
    
    # Test 1: Random input
    print("\n📊 Test 1: Random input")
    random_input = tf.random.uniform((4, 128, 1024, 1))
    logits = recognizer(random_input, training=False)
    
    print(f"   Input shape: {random_input.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Output dtype: {logits.dtype}")
    print(f"   Output min: {tf.reduce_min(logits):.4f}")
    print(f"   Output max: {tf.reduce_max(logits):.4f}")
    print(f"   Output mean: {tf.reduce_mean(logits):.4f}")
    print(f"   Output std: {tf.math.reduce_std(logits):.4f}")
    
    # Test 2: Black input
    print("\n📊 Test 2: Black input (all zeros)")
    black_input = tf.zeros((2, 128, 1024, 1))
    logits_black = recognizer(black_input, training=False)
    
    print(f"   Output mean: {tf.reduce_mean(logits_black):.4f}")
    print(f"   Output std: {tf.math.reduce_std(logits_black):.4f}")
    
    # Test 3: White input
    print("\n📊 Test 3: White input (all ones)")
    white_input = tf.ones((2, 128, 1024, 1))
    logits_white = recognizer(white_input, training=False)
    
    print(f"   Output mean: {tf.reduce_mean(logits_white):.4f}")
    print(f"   Output std: {tf.math.reduce_std(logits_white):.4f}")
    
    # Check if outputs are different
    diff_black_white = tf.reduce_mean(tf.abs(logits_black - logits_white))
    print(f"\n📊 Difference between black and white outputs:")
    print(f"   Mean absolute difference: {diff_black_white:.4f}")
    
    if diff_black_white < 0.01:
        print("   ⚠️ WARNING: Outputs are very similar! Model might not be sensitive to input")
    else:
        print("   ✅ Outputs are different - model is responsive to input")
    
    # Test output distribution
    print("\n📊 Analyzing output distribution (random input):")
    predictions = tf.argmax(logits, axis=-1)
    
    print(f"   Prediction shape: {predictions.shape}")
    print(f"   Unique predicted classes: {len(tf.unique(tf.reshape(predictions, [-1]))[0])}")
    print(f"   Sample predictions (first 20 timesteps of first sample):")
    print(f"   {predictions[0, :20].numpy()}")
    
    # Check if all predictions are the same (mode collapse)
    flat_preds = tf.reshape(predictions, [-1])
    unique_preds = tf.unique(flat_preds)[0]
    
    if len(unique_preds) == 1:
        print("   ⚠️ WARNING: All predictions are the same! Model collapsed!")
    else:
        print(f"   ✅ Model predicts {len(unique_preds)} different classes")

# ============================================================================
# TEST 4: Output Stability
# ============================================================================
def test_output_stability(recognizer):
    """Test apakah output konsisten untuk input yang sama"""
    print("\n" + "="*80)
    print("TEST 4: OUTPUT STABILITY TEST")
    print("="*80)
    
    if recognizer is None:
        print("❌ No model to test")
        return
    
    test_input = tf.random.uniform((2, 128, 1024, 1), seed=42)
    
    # Run 3 times with training=False
    outputs = []
    for i in range(3):
        out = recognizer(test_input, training=False)
        outputs.append(out)
    
    # Compare outputs
    diff_01 = tf.reduce_mean(tf.abs(outputs[0] - outputs[1])).numpy()
    diff_02 = tf.reduce_mean(tf.abs(outputs[0] - outputs[2])).numpy()
    diff_12 = tf.reduce_mean(tf.abs(outputs[1] - outputs[2])).numpy()
    
    print(f"📊 Output differences (should be ~0 for frozen model):")
    print(f"   Run 0 vs Run 1: {diff_01:.10f}")
    print(f"   Run 0 vs Run 2: {diff_02:.10f}")
    print(f"   Run 1 vs Run 2: {diff_12:.10f}")
    
    max_diff = max(diff_01, diff_02, diff_12)
    
    if max_diff < 1e-6:
        print("   ✅ Outputs are perfectly stable (deterministic)")
    elif max_diff < 1e-3:
        print("   ✅ Outputs are very stable (minor numerical differences)")
    else:
        print(f"   ⚠️ WARNING: Outputs are NOT stable! Max diff: {max_diff}")
        print("   Model might have dropout or other non-deterministic layers active")

# ============================================================================
# TEST 5: Real Image Test
# ============================================================================
def test_real_image(recognizer, vocab_size):
    """Test dengan gambar asli dari dataset"""
    print("\n" + "="*80)
    print("TEST 5: RECOGNITION ON REAL IMAGES")
    print("="*80)
    
    if recognizer is None:
        print("❌ No model to test")
        return
    
    # Load charlist for decoding
    charset_path = 'real_data_preparation/real_data_charlist.txt'
    charset = read_charlist(charset_path)
    
    # Try to load a real image
    image_dir = Path('real_data_preparation/images_fixed')
    label_file = Path('real_data_preparation/labels_fixed.txt')
    
    if not label_file.exists():
        print("⚠️ Label file tidak ditemukan")
        return
    
    # Read first few entries
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = [line.strip().split(' ', 1) for line in f if line.strip()][:3]
    
    output_dir = Path('dual_modal_gan/outputs/debug_recognizer')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, ground_truth in lines:
        image_path = image_dir / filename
        
        if not image_path.exists():
            print(f"⚠️ Image not found: {filename}")
            continue
        
        print(f"\n📄 Testing on: {filename}")
        print(f"   Ground truth: {ground_truth}")
        
        # Load and preprocess image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img.shape != (128, 1024):
            img = cv2.resize(img, (1024, 128))
        
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)  # Add channel dim
        img = np.expand_dims(img, axis=0)   # Add batch dim
        img_tensor = tf.convert_to_tensor(img)
        
        # Forward pass
        logits = recognizer(img_tensor, training=False)
        predictions = tf.argmax(logits, axis=-1)[0]  # Shape: (timesteps,)
        
        # Decode prediction
        pred_chars = []
        prev_idx = -1
        for idx in predictions.numpy():
            if idx != 0 and idx != prev_idx:  # Skip blank and repeats
                if idx <= len(charset):
                    pred_chars.append(charset[idx - 1])
            prev_idx = idx
        
        predicted_text = ''.join(pred_chars)
        
        print(f"   Predicted: {predicted_text}")
        
        # Calculate simple accuracy
        if predicted_text == ground_truth:
            print("   ✅ Perfect match!")
        else:
            # Simple character-level comparison
            correct_chars = sum(1 for a, b in zip(predicted_text, ground_truth) if a == b)
            total_chars = max(len(predicted_text), len(ground_truth))
            acc = correct_chars / total_chars * 100 if total_chars > 0 else 0
            print(f"   Accuracy: {acc:.1f}%")
        
        # Save image for reference
        cv2.imwrite(str(output_dir / f'test_{filename}'), (img[0, :, :, 0] * 255).astype(np.uint8))
    
    print(f"\n✅ Test images saved to: {output_dir}")

# ============================================================================
# TEST 6: Gradient Flow in GAN Context
# ============================================================================
def test_gradient_in_gan_context(recognizer):
    """Test apakah gradient dari CTC loss bisa flow ke generator"""
    print("\n" + "="*80)
    print("TEST 6: GRADIENT FLOW IN GAN CONTEXT")
    print("="*80)
    
    if recognizer is None:
        print("❌ No model to test")
        return
    
    from dual_modal_gan.src.models.generator import unet
    
    generator = unet(input_size=(128, 1024, 1))
    
    batch_size = 2
    degraded_input = tf.random.uniform((batch_size, 128, 1024, 1))
    ground_truth_labels = tf.constant([[5, 10, 15, 20, 0, 0], 
                                       [3, 7, 11, 0, 0, 0]], dtype=tf.int32)
    
    print("📊 Testing gradient flow: Generator -> Recognizer -> CTC Loss")
    
    with tf.GradientTape() as tape:
        # Generator forward
        generated_images = generator(degraded_input, training=True)
        
        # Recognizer forward (frozen)
        logits = recognizer(generated_images, training=False)
        
        # CTC loss
        label_len = tf.math.count_nonzero(ground_truth_labels, axis=1, dtype=tf.int32)
        logit_len = tf.fill([batch_size], logits.shape[1])
        
        ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(
            labels=tf.cast(ground_truth_labels, tf.int32),
            logits=logits,
            label_length=label_len,
            logit_length=logit_len,
            logits_time_major=False,
            blank_index=0
        ))
    
    # Get gradients w.r.t. generator
    gen_gradients = tape.gradient(ctc_loss, generator.trainable_variables)
    
    # Get gradients w.r.t. recognizer (should be None)
    rec_gradients = tape.gradient(ctc_loss, recognizer.trainable_variables)
    
    print(f"\n   CTC Loss: {ctc_loss.numpy():.6f}")
    
    gen_grad_norms = [tf.norm(g).numpy() for g in gen_gradients if g is not None]
    print(f"\n   Generator gradients:")
    print(f"     Count: {len(gen_grad_norms)}")
    print(f"     Mean norm: {np.mean(gen_grad_norms):.6e}")
    print(f"     Max norm: {np.max(gen_grad_norms):.6e}")
    
    if len(rec_gradients) == 0:
        print(f"\n   Recognizer gradients: None (as expected - model is frozen)")
        print("   ✅ Gradient hanya mengalir ke Generator, tidak ke Recognizer")
    else:
        rec_grad_norms = [tf.norm(g).numpy() for g in rec_gradients if g is not None]
        if len(rec_grad_norms) > 0:
            print(f"\n   ⚠️ WARNING: Recognizer has {len(rec_grad_norms)} gradients!")
            print("   Model mungkin tidak fully frozen!")
        else:
            print(f"\n   Recognizer gradients: All None")
            print("   ✅ Recognizer properly frozen")
    
    if np.mean(gen_grad_norms) < 1e-10:
        print("\n   ⚠️ WARNING: Generator gradients sangat kecil!")
        print("   CTC loss mungkin tidak memberikan learning signal yang cukup")
    elif np.max(gen_grad_norms) > 100:
        print("\n   ⚠️ WARNING: Generator gradients sangat besar!")
        print("   Mungkin perlu gradient clipping")
    else:
        print("\n   ✅ Generator gradients dalam range normal")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*80)
    print("FROZEN RECOGNIZER TEST SUITE")
    print("="*80)
    print("\nMemverifikasi bahwa recognizer model:")
    print("  1. Ter-load dengan benar")
    print("  2. Benar-benar frozen (tidak trainable)")
    print("  3. Berfungsi dengan baik untuk forward pass")
    print("  4. Gradient bisa flow untuk CTC loss dalam GAN training")
    print("="*80)
    
    try:
        recognizer, vocab_size = test_model_loading()
        test_frozen_status(recognizer)
        test_forward_pass(recognizer)
        test_output_stability(recognizer)
        test_real_image(recognizer, vocab_size)
        test_gradient_in_gan_context(recognizer)
        
        print("\n" + "="*80)
        print("SEMUA TEST RECOGNIZER SELESAI")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
