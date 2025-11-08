#!/usr/bin/env python3
"""
DIAGNOSTIC: Debug checkpoint loading and inference issues
Compares training-time samples with post-hoc inference results
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add project root
import sys
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dual_modal_gan.src.models.generator_enhanced import unet_enhanced

def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR between two images"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_val / np.sqrt(mse))

def load_sample_images(sample_dir, epoch=10, sample_idx=0):
    """Load degraded, clean, and generated images from saved samples"""
    sample_path = sample_dir / f"comparison_epoch_{epoch:04d}_sample_{sample_idx}.png"
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample not found: {sample_path}")
    
    # Load comparison image (3 images side-by-side)
    comparison = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
    
    # Split into 3 images
    height, width = comparison.shape
    img_width = width // 3
    
    degraded = comparison[:, :img_width]
    clean = comparison[:, img_width:2*img_width]
    generated_training = comparison[:, 2*img_width:]
    
    # Normalize to [0, 1]
    degraded = degraded.astype(np.float32) / 255.0
    clean = clean.astype(np.float32) / 255.0
    generated_training = generated_training.astype(np.float32) / 255.0
    
    return degraded, clean, generated_training

def test_checkpoint_inference(checkpoint_dir, sample_dir, epoch=10):
    """Test if checkpoint produces same output as training-time samples"""
    print("\n" + "="*80)
    print("🔬 DIAGNOSTIC: Checkpoint Loading & Inference Test")
    print("="*80)
    
    # Load generator
    print(f"\n[1/4] Loading Generator...")
    generator = unet_enhanced(input_size=(1024, 128, 1))
    print(f"  Parameters: {generator.count_params():,}")
    
    # Load checkpoint
    print(f"\n[2/4] Loading Checkpoint...")
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint_path = Path(checkpoint_dir)
    
    if (checkpoint_path / 'checkpoint').exists():
        status = checkpoint.restore(tf.train.latest_checkpoint(str(checkpoint_path)))
        status.expect_partial()
        print(f"  ✅ Checkpoint loaded from: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
    
    # Load training-time sample
    print(f"\n[3/4] Loading Training-Time Sample...")
    degraded, clean, generated_training = load_sample_images(
        Path(sample_dir), epoch=epoch, sample_idx=0
    )
    print(f"  Image shape: {degraded.shape}")
    print(f"  Degraded range: [{degraded.min():.3f}, {degraded.max():.3f}]")
    print(f"  Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
    print(f"  Generated (training) range: [{generated_training.min():.3f}, {generated_training.max():.3f}]")
    
    # Calculate PSNR from training-time generation
    psnr_training = calculate_psnr(clean, generated_training, max_val=1.0)
    print(f"\n  📊 PSNR (training-time): {psnr_training:.4f} dB")
    
    # Run POST-HOC inference
    print(f"\n[4/4] Running Post-Hoc Inference...")
    
    # Prepare input (transpose to match recognizer format: H,W,C → W,H,C)
    degraded_input = np.transpose(degraded, (1, 0))  # (128, 1024) → (1024, 128)
    degraded_input = np.expand_dims(degraded_input, axis=-1)  # → (1024, 128, 1)
    degraded_input = np.expand_dims(degraded_input, axis=0)  # → (1, 1024, 128, 1)
    
    print(f"  Input shape: {degraded_input.shape}")
    print(f"  Input range: [{degraded_input.min():.3f}, {degraded_input.max():.3f}]")
    
    # Generate
    generated_posthoc = generator(degraded_input, training=False)
    generated_posthoc = generated_posthoc.numpy()[0]  # Remove batch dimension
    
    print(f"  Output shape: {generated_posthoc.shape}")
    print(f"  Output range: [{generated_posthoc.min():.3f}, {generated_posthoc.max():.3f}]")
    
    # Transpose back to H,W,C for comparison
    generated_posthoc = np.transpose(generated_posthoc.squeeze(), (1, 0))  # (W,H) → (H,W)
    
    # Calculate PSNR from post-hoc generation
    psnr_posthoc = calculate_psnr(clean, generated_posthoc, max_val=1.0)
    print(f"\n  📊 PSNR (post-hoc): {psnr_posthoc:.4f} dB")
    
    # Compare outputs
    diff = np.abs(generated_training - generated_posthoc)
    print(f"\n  🔍 Output Comparison:")
    print(f"     Mean absolute difference: {diff.mean():.6f}")
    print(f"     Max absolute difference: {diff.max():.6f}")
    print(f"     PSNR difference: {psnr_training - psnr_posthoc:+.4f} dB")
    
    # Verdict
    print(f"\n" + "="*80)
    print("📊 DIAGNOSTIC RESULTS")
    print("="*80)
    
    if abs(psnr_training - psnr_posthoc) < 0.5:
        print("✅ MATCH: Post-hoc inference produces SAME output as training")
        print(f"   PSNR training: {psnr_training:.2f} dB")
        print(f"   PSNR post-hoc: {psnr_posthoc:.2f} dB")
        print(f"   Difference: {abs(psnr_training - psnr_posthoc):.2f} dB (acceptable)")
    else:
        print("❌ MISMATCH: Post-hoc inference produces DIFFERENT output!")
        print(f"   PSNR training: {psnr_training:.2f} dB")
        print(f"   PSNR post-hoc: {psnr_posthoc:.2f} dB")
        print(f"   Difference: {abs(psnr_training - psnr_posthoc):.2f} dB (suspicious!)")
        print(f"\n   Possible causes:")
        print(f"   1. Checkpoint mismatch (loaded wrong checkpoint)")
        print(f"   2. Input preprocessing difference (normalization, transposition)")
        print(f"   3. Training used different generator architecture")
        print(f"   4. Generator weights not properly saved/restored")
    
    print("="*80 + "\n")
    
    return {
        'psnr_training': psnr_training,
        'psnr_posthoc': psnr_posthoc,
        'diff_mean': diff.mean(),
        'diff_max': diff.max()
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--sample_dir', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=10)
    
    args = parser.parse_args()
    
    test_checkpoint_inference(args.checkpoint_dir, args.sample_dir, args.epoch)
