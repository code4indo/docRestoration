#!/usr/bin/env python3
"""
Comprehensive PSNR Evaluation Script
For objective PSNR calculation on entire validation dataset
"""

import os
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
import json
from datetime import datetime

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def parse_tfrecord(example_proto):
    """Parse TFRecord - sama seperti training script"""
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

    # Parse degraded image
    degraded_image_shape = tf.cast(example['degraded_image_shape'], tf.int32)
    degraded_image = tf.io.decode_raw(example['degraded_image_raw'], tf.float32)
    degraded_image = tf.reshape(degraded_image, degraded_image_shape)
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])
    degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])

    # Parse clean image
    clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
    clean_image = tf.reshape(clean_image, clean_image_shape)
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])
    clean_image = tf.ensure_shape(clean_image, [1024, 128, 1])

    return degraded_image, clean_image

def create_validation_dataset(tfrecord_path, batch_size=1):
    """Create validation dataset - TANPA shuffle!"""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    # Split validation set (10% dari total)
    total_size = sum(1 for _ in dataset)
    train_size = int(total_size * 0.9)
    val_size = total_size - train_size

    # Ambil validation set TANPA shuffle untuk konsistensi
    dataset = dataset.skip(train_size)
    val_dataset = dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    return val_dataset, val_size

def calculate_comprehensive_psnr(generator, val_dataset, val_size, output_dir):
    """Hitung PSNR untuk SELURUH validation dataset"""

    print(f"📊 Evaluating PSNR on {val_size} validation samples...")

    psnr_values = []
    ssim_values = []
    processed_samples = 0

    # Progress bar
    pbar = tqdm(val_dataset, desc="Processing validation samples")

    for batch_idx, (degraded_images, clean_images) in enumerate(pbar):
        try:
            # Generate enhanced images
            generated_images = generator(degraded_images, training=False)

            # Denormalize dari [-1,1] ke [0,1]
            generated_norm = (generated_images + 1.0) / 2.0
            clean_norm = (clean_images + 1.0) / 2.0

            # Calculate PSNR untuk setiap gambar di batch
            for i in range(generated_norm.shape[0]):
                psnr = tf.image.psnr(clean_norm[i:i+1], generated_norm[i:i+1], max_val=1.0)
                ssim = tf.image.ssim(clean_norm[i:i+1], generated_norm[i:i+1], max_val=1.0)

                psnr_values.append(float(psnr.numpy()))
                ssim_values.append(float(ssim.numpy()))
                processed_samples += 1

                # Save sample images untuk visual inspection
                if i < 5:  # Save first 5 samples per batch
                    save_comparison_images(
                        degraded_images[i], clean_images[i], generated_norm[i],
                        batch_idx, i, output_dir
                    )

            # Update progress bar dengan statistik running
            current_psnr_mean = np.mean(psnr_values)
            current_psnr_std = np.std(psnr_values)
            pbar.set_postfix({
                'PSNR': f'{current_psnr_mean:.2f}±{current_psnr_std:.2f}',
                'Samples': processed_samples
            })

        except Exception as e:
            print(f"⚠️  Error processing batch {batch_idx}: {e}")
            continue

    return psnr_values, ssim_values, processed_samples

def save_comparison_images(degraded, clean, generated, batch_idx, img_idx, output_dir):
    """Save comparison images for visual inspection"""

    # Denormalize
    degraded_norm = (degraded + 1.0) / 2.0
    clean_norm = (clean + 1.0) / 2.0

    # Convert to uint8
    deg_img = (degraded_norm * 255).numpy().astype(np.uint8)
    clean_img = (clean_norm * 255).numpy().astype(np.uint8)
    gen_img = (generated * 255).numpy().astype(np.uint8)

    # Remove channel dimension and transpose if needed
    deg_img = np.squeeze(deg_img, axis=-1)
    clean_img = np.squeeze(clean_img, axis=-1)
    gen_img = np.squeeze(gen_img, axis=-1)

    # Transpose untuk horizontal text
    if deg_img.shape[0] > deg_img.shape[1]:
        deg_img = np.transpose(deg_img)
        clean_img = np.transpose(clean_img)
        gen_img = np.transpose(gen_img)

    # Create comparison
    comparison = np.vstack([deg_img, clean_img, gen_img])

    # Save image
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, f'comparison_batch{batch_idx:03d}_img{img_idx}.png')
    cv2.imwrite(img_path, comparison)

def main():
    """Main evaluation function"""
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive PSNR Evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to generator model (.h5) or checkpoint directory')
    parser.add_argument('--tfrecord_path', type=str,
                       default='dual_modal_gan/data/dataset_gan.tfrecord',
                       help='Path to TFRecord dataset')
    parser.add_argument('--output_dir', type=str, default='psnr_evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--is_checkpoint', action='store_true',
                       help='Treat model_path as checkpoint directory')
    parser.add_argument('--checkpoint_prefix', type=str, default='ckpt',
                       help='Checkpoint file prefix')

    args = parser.parse_args()

    print("🔬 COMPREHENSIVE PSNR EVALUATION")
    print("=" * 50)

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'evaluation_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"\n📂 Loading model from: {args.model_path}")

    if args.is_checkpoint:
        # Load from checkpoint - try different generator architectures
        # Try enhanced first (most likely)
        try:
            from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
            generator = unet_enhanced(input_size=(1024, 128, 1))
            generator_type = "enhanced"
            print("🔧 Trying enhanced generator architecture...")
        except ImportError:
            try:
                from dual_modal_gan.src.models.generator_enhanced_v2 import unet_enhanced_v2
                generator = unet_enhanced_v2(input_size=(1024, 128, 1))
                generator_type = "enhanced_v2"
                print("🔧 Trying enhanced_v2 generator architecture...")
            except ImportError:
                from dual_modal_gan.src.models.generator import unet
                generator = unet(input_size=(1024, 128, 1))
                generator_type = "base"
                print("🔧 Using base generator architecture...")

        checkpoint = tf.train.Checkpoint(generator=generator)
        checkpoint_path = tf.train.latest_checkpoint(args.model_path)

        if checkpoint_path:
            try:
                checkpoint.restore(checkpoint_path).expect_partial()
                print(f"✅ Checkpoint restored: {checkpoint_path}")
                print(f"   Generator type: {generator_type}")
            except Exception as e:
                print(f"❌ Failed to restore checkpoint with {generator_type}: {e}")
                # Try other architectures if first attempt fails
                if generator_type != "base":
                    print("🔄 Trying base generator as fallback...")
                    from dual_modal_gan.src.models.generator import unet
                    generator = unet(input_size=(1024, 128, 1))
                    checkpoint = tf.train.Checkpoint(generator=generator)
                    checkpoint.restore(checkpoint_path).expect_partial()
                    print(f"✅ Base generator checkpoint restored: {checkpoint_path}")
        else:
            print(f"❌ No checkpoint found in: {args.model_path}")
            return
    else:
        # Load .h5 model
        generator = tf.keras.models.load_model(args.model_path)
        print(f"✅ Model loaded: {args.model_path}")

    # Create validation dataset
    print(f"\n📊 Creating validation dataset from: {args.tfrecord_path}")
    val_dataset, val_size = create_validation_dataset(args.tfrecord_path, args.batch_size)
    print(f"   Validation samples: {val_size}")

    # Calculate comprehensive PSNR
    print(f"\n🧮 Calculating comprehensive PSNR metrics...")
    psnr_values, ssim_values, processed_samples = calculate_comprehensive_psnr(
        generator, val_dataset, val_size, output_dir
    )

    # Calculate statistics
    if psnr_values:
        psnr_mean = np.mean(psnr_values)
        psnr_std = np.std(psnr_values)
        psnr_median = np.median(psnr_values)
        psnr_min = np.min(psnr_values)
        psnr_max = np.max(psnr_values)

        # Confidence interval (95%)
        confidence_interval = 1.96 * (psnr_std / np.sqrt(len(psnr_values)))

        ssim_mean = np.mean(ssim_values)
        ssim_std = np.std(ssim_values)

        print(f"\n📈 COMPREHENSIVE RESULTS ({processed_samples} samples):")
        print(f"   PSNR: {psnr_mean:.2f} ± {psnr_std:.2f} dB")
        print(f"   PSNR Median: {psnr_median:.2f} dB")
        print(f"   PSNR Range: [{psnr_min:.2f}, {psnr_max:.2f}] dB")
        print(f"   95% CI: [{psnr_mean-confidence_interval:.2f}, {psnr_mean+confidence_interval:.2f}] dB")
        print(f"   SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")

        # Save results
        results = {
            'evaluation_timestamp': timestamp,
            'model_path': args.model_path,
            'dataset_path': args.tfrecord_path,
            'total_validation_samples': val_size,
            'processed_samples': processed_samples,
            'psnr_statistics': {
                'mean': float(psnr_mean),
                'std': float(psnr_std),
                'median': float(psnr_median),
                'min': float(psnr_min),
                'max': float(psnr_max),
                'confidence_interval_95': {
                    'lower': float(psnr_mean - confidence_interval),
                    'upper': float(psnr_mean + confidence_interval)
                }
            },
            'ssim_statistics': {
                'mean': float(ssim_mean),
                'std': float(ssim_std)
            },
            'sample_psnr_values': [float(x) for x in psnr_values[:100]]  # First 100 samples
        }

        results_file = os.path.join(output_dir, 'psnr_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {results_file}")
        print(f"🖼️  Sample images saved to: {output_dir}")

        # Academic format output
        print(f"\n📚 ACADEMIC REPORTING FORMAT:")
        print(f"   \"The proposed method achieved PSNR of {psnr_mean:.2f} ± {psnr_std:.2f} dB")
        print(f"    (95% CI: [{psnr_mean-confidence_interval:.2f}, {psnr_mean+confidence_interval:.2f}] dB)")
        print(f"    on {processed_samples} validation documents, ranging from {psnr_min:.1f} to {psnr_max:.1f} dB.\"")

    else:
        print(f"\n❌ No PSNR values calculated. Check model and dataset.")

if __name__ == '__main__':
    main()