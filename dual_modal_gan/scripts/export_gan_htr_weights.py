#!/usr/bin/env python3
"""
Export GAN-HTR Generator dari TF Checkpoint ke Keras .weights.h5

Converts TensorFlow checkpoint format to Keras weights format
for easier loading in visualization scripts.
"""

import os
import sys
import argparse
from pathlib import Path
import tensorflow as tf

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dual_modal_gan.src.models.generator import unet


def export_generator_weights(checkpoint_path, output_path):
    """Export generator weights from TF checkpoint to Keras format"""
    
    print("\n" + "="*80)
    print("EXPORT GAN-HTR GENERATOR WEIGHTS")
    print("="*80)
    
    print(f"\n📁 Checkpoint path: {checkpoint_path}")
    print(f"📁 Output path: {output_path}")
    
    # Create generator
    print("\n🎨 Creating generator model...")
    generator = unet()
    print("  ✅ Generator created")
    
    # Load from TF checkpoint
    print(f"\n📥 Loading weights from TensorFlow checkpoint...")
    
    # Find latest checkpoint
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint is None:
        print(f"❌ No checkpoint found in {checkpoint_dir}")
        return False
    
    print(f"  Found: {latest_checkpoint}")
    
    # Create checkpoint object
    checkpoint = tf.train.Checkpoint(generator=generator)
    
    # Restore
    status = checkpoint.restore(latest_checkpoint)
    
    try:
        status.assert_existing_objects_matched()
        print("  ✅ Checkpoint loaded successfully")
    except Exception as e:
        print(f"  ⚠️ Warning: {e}")
        print("  Attempting to continue anyway...")
    
    # Save in Keras format
    print(f"\n💾 Saving weights in Keras format...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generator.save_weights(output_path)
    print(f"  ✅ Saved: {output_path}")
    
    # Verify file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"  File size: {file_size:.1f} MB")
    
    print("\n" + "="*80)
    print("✅ EXPORT COMPLETE!")
    print("="*80)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Export GAN-HTR generator weights')
    parser.add_argument('--checkpoint_path', type=str,
                       default='dual_modal_gan/outputs/checkpoints_final_100/best_model-10',
                       help='Path to TF checkpoint (without extension)')
    parser.add_argument('--output_path', type=str,
                       default='dual_modal_gan/outputs/checkpoints_final_100/generator.weights.h5',
                       help='Output path for Keras weights')
    
    args = parser.parse_args()
    
    success = export_generator_weights(args.checkpoint_path, args.output_path)
    
    if not success:
        print("\n❌ Export failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
