"""
Generate sample images from joint training checkpoints (post-hoc)

Purpose: Retroactively create visualization samples from completed joint training
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import cv2

# Use GPU 1 (GPU 0 is busy with frozen training)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Pure FP32
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')

from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
from dual_modal_gan.scripts.train_enhanced import create_dataset

def main():
    # Config
    config_path = "configs/ablation_joint_training_vs_frozen.json"
    checkpoint_dir = "dual_modal_gan/checkpoints/ablation_joint_training"
    sample_dir = "dual_modal_gan/outputs/samples_ablation_joint_training"
    
    print("="*80)
    print("GENERATE JOINT TRAINING SAMPLES (POST-HOC)")
    print("="*80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"\n[1/4] Loading dataset...")
    train_ds, val_dataset, test_ds, train_size, val_size, test_size = create_dataset(
        tfrecord_path=config['tfrecord_path'],
        batch_size=2,
        train_split=config.get('train_split', 0.7),
        val_split=config.get('val_split', 0.15)
    )
    print(f"  ✅ Dataset loaded: {val_size} validation samples")
    
    print(f"\n[2/4] Building generator...")
    generator = unet_enhanced(input_size=(1024, 128, 1))
    
    print(f"\n[3/4] Loading checkpoints...")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Generate samples for each checkpoint
    epochs_to_generate = [5, 10, 15, 20]
    num_samples = 5
    
    for epoch in epochs_to_generate:
        ckpt_path = os.path.join(checkpoint_dir, f'ckpt-{epoch}_gen.weights.h5')
        
        if not os.path.exists(ckpt_path):
            print(f"  ⚠️  Checkpoint not found: {ckpt_path}")
            continue
        
        print(f"\n[4/{len(epochs_to_generate)}] Generating samples for epoch {epoch}...")
        generator.load_weights(ckpt_path)
        print(f"  ✅ Loaded: {ckpt_path}")
        
        # Collect samples
        num_samples = 5
        collected_degraded = []
        collected_clean = []
        
        val_iter = iter(val_dataset)
        samples_collected = 0
        
        while samples_collected < num_samples:
            try:
                batch = next(val_iter)
                degraded_batch, clean_batch, _ = batch
                batch_size_actual = degraded_batch.shape[0]
                samples_needed = min(batch_size_actual, num_samples - samples_collected)
                
                collected_degraded.append(degraded_batch[:samples_needed])
                collected_clean.append(clean_batch[:samples_needed])
                
                samples_collected += samples_needed
            except StopIteration:
                break
        
        if not collected_degraded:
            print("  ⚠️  No samples collected")
            continue
        
        # Concatenate samples
        degraded_samples = tf.concat(collected_degraded, axis=0)[:num_samples]
        clean_samples = tf.concat(collected_clean, axis=0)[:num_samples]
        
        # Generate restored images ONE BY ONE to avoid OOM
        generated_list = []
        for i in range(degraded_samples.shape[0]):
            single_input = tf.expand_dims(degraded_samples[i], axis=0)  # Add batch dim
            single_output = generator(single_input, training=False)
            generated_list.append(single_output[0])  # Remove batch dim
        
        generated_samples = tf.stack(generated_list, axis=0)
        
        # Save samples
        for i in range(min(num_samples, degraded_samples.shape[0])):
            # Convert to uint8
            generated_img = (generated_samples[i].numpy() * 255).astype(np.uint8)
            degraded_img = (degraded_samples[i].numpy() * 255).astype(np.uint8)
            clean_img = (clean_samples[i].numpy() * 255).astype(np.uint8)
            
            # Remove channel dimension if grayscale
            if generated_img.shape[-1] == 1:
                generated_img = np.squeeze(generated_img, axis=-1)
                degraded_img = np.squeeze(degraded_img, axis=-1)
                clean_img = np.squeeze(clean_img, axis=-1)
            
            # Transpose if needed (W,H) -> (H,W)
            if generated_img.shape[0] > generated_img.shape[1]:
                generated_img = np.transpose(generated_img)
                degraded_img = np.transpose(degraded_img)
                clean_img = np.transpose(clean_img)
            
            # Save individual restored image
            sample_path = os.path.join(sample_dir, f'epoch_{epoch:04d}_sample_{i}.png')
            cv2.imwrite(sample_path, generated_img)
            
            # Create vertical comparison: [Degraded | Ground Truth | Restored]
            comparison = np.vstack([degraded_img, clean_img, generated_img])
            comparison_path = os.path.join(sample_dir, f'comparison_epoch_{epoch:04d}_sample_{i}.png')
            cv2.imwrite(comparison_path, comparison)
        
        print(f"  ✅ Saved {min(num_samples, degraded_samples.shape[0])} samples for epoch {epoch}")
    
    print("\n" + "="*80)
    print("✅ SAMPLE GENERATION COMPLETED!")
    print("="*80)
    print(f"Samples saved to: {sample_dir}")
    print(f"Total images: {len(os.listdir(sample_dir))} files")

if __name__ == '__main__':
    main()
