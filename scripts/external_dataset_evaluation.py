#!/usr/bin/env python3
"""
External Dataset Evaluation
Test model pada dataset berbeda untuk generalisasi
"""

import os
import numpy as np
import tensorflow as tf
import json
from datetime import datetime
import cv2
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_synthetic_test_dataset(n_samples=100, noise_types=['gaussian', 'poisson', 'speckle']):
    """
    Create synthetic test dataset dengan berbagai jenis degradasi
    """
    print(f"🧪 Creating synthetic test dataset ({n_samples} samples)")

    test_images = []
    ground_truth_images = []

    for i in tqdm(range(n_samples), desc="Generating synthetic documents"):
        # Generate clean document (synthetic text)
        clean_img = generate_synthetic_document(1024, 128)

        # Apply random degradation
        noise_type = np.random.choice(noise_types)
        degraded_img = apply_degradation(clean_img, noise_type)

        test_images.append(degraded_img)
        ground_truth_images.append(clean_img)

    return np.array(test_images), np.array(ground_truth_images)

def generate_synthetic_document(width, height):
    """
    Generate synthetic document with text-like patterns
    """
    # Create base image
    img = np.ones((height, width), dtype=np.float32) * 0.9  # Light background

    # Add text-like horizontal lines
    n_lines = np.random.randint(15, 25)
    for _ in range(n_lines):
        y = np.random.randint(10, height - 10)
        thickness = np.random.randint(2, 4)

        # Create text-like pattern
        x_positions = np.random.randint(50, width - 50, np.random.randint(5, 15))
        for x in x_positions:
            char_width = np.random.randint(10, 30)
            if x + char_width < width:
                img[y-thickness//2:y+thickness//2+1, x:x+char_width] = 0.1

    # Add some noise for realism
    noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 1)

    # Add channel dimension
    img = np.expand_dims(img, axis=-1)
    return img

def apply_degradation(clean_img, noise_type='gaussian'):
    """
    Apply various types of degradation to simulate real document damage
    """
    degraded = clean_img.copy()

    if noise_type == 'gaussian':
        # Gaussian noise
        noise = np.random.normal(0, 0.1, degraded.shape).astype(np.float32)
        degraded = np.clip(degraded + noise, 0, 1)

    elif noise_type == 'poisson':
        # Poisson noise (common in scanned documents)
        # Scale to [0, 255] for Poisson, then back to [0, 1]
        img_scaled = (degraded * 255).astype(np.uint8)
        noisy = np.random.poisson(img_scaled * 0.5) / (255 * 0.5)
        degraded = np.clip(noisy, 0, 1).astype(np.float32)

    elif noise_type == 'speckle':
        # Speckle noise (common in document scanning)
        noise = np.random.normal(0, 0.2, degraded.shape)
        degraded = np.clip(degraded + degraded * noise, 0, 1)

    # Add blur for realistic scanning effect
    if np.random.random() > 0.5:
        kernel_size = np.random.choice([1, 3])
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            for i in range(degraded.shape[2]):
                degraded[:, :, i] = cv2.filter2D(degraded[:, :, i], -1, kernel)

    # Add contrast variations
    if np.random.random() > 0.7:
        contrast_factor = np.random.uniform(0.8, 1.2)
        degraded = np.clip((degraded - 0.5) * contrast_factor + 0.5, 0, 1)

    return degraded

def evaluate_on_external_data(model, test_images, ground_truth_images, dataset_name="External Test"):
    """
    Evaluate model on external/synthetic test data
    """
    print(f"\n🔍 Evaluating on {dataset_name}")
    print(f"   Test samples: {len(test_images)}")

    psnr_values = []
    ssim_values = []

    # Convert to tensors for model
    test_images_tensor = tf.constant(test_images, dtype=tf.float32)
    ground_truth_tensor = tf.constant(ground_truth_images, dtype=tf.float32)

    # Process in batches
    batch_size = 4
    n_batches = (len(test_images) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_images))

        batch_degraded = test_images_tensor[start_idx:end_idx]
        batch_clean = ground_truth_tensor[start_idx:end_idx]

        # Generate enhanced images
        generated = model(batch_degraded, training=False)

        # Denormalize from [-1,1] to [0,1]
        generated_norm = (generated + 1.0) / 2.0
        clean_norm = (batch_clean + 1.0) / 2.0

        # Calculate metrics for each sample in batch
        for i in range(generated_norm.shape[0]):
            psnr = tf.image.psnr(clean_norm[i:i+1], generated_norm[i:i+1], max_val=1.0)
            ssim = tf.image.ssim(clean_norm[i:i+1], generated_norm[i:i+1], max_val=1.0)

            psnr_values.append(float(psnr.numpy()))
            ssim_values.append(float(ssim.numpy()))

    return np.array(psnr_values), np.array(ssim_values)

def load_model_for_evaluation(model_path):
    """
    Load model dari checkpoint untuk evaluasi
    """
    print(f"📂 Loading model from: {model_path}")

    try:
        from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
        generator = unet_enhanced(input_size=(1024, 128, 1))

        checkpoint = tf.train.Checkpoint(generator=generator)
        checkpoint_path = tf.train.latest_checkpoint(model_path)

        if checkpoint_path:
            checkpoint.restore(checkpoint_path).expect_partial()
            print(f"✅ Model loaded: {checkpoint_path}")
            return generator
        else:
            print(f"❌ No checkpoint found in: {model_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def main():
    """Main function untuk external dataset evaluation"""
    import argparse
    parser = argparse.ArgumentParser(description='External Dataset Evaluation')
    parser.add_argument('--model_path', type=str,
                       default='dual_modal_gan/checkpoints/full_training_production_v1/best_model',
                       help='Path to model checkpoint directory')
    parser.add_argument('--output_dir', type=str, default='external_evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--n_synthetic_samples', type=int, default=100,
                       help='Number of synthetic test samples to generate')
    parser.add_argument('--use_real_data', action='store_true',
                       help='Use real test data if available')
    parser.add_argument('--real_data_path', type=str,
                       default='dual_modal_gan/data/test_dataset.tfrecord',
                       help='Path to real test dataset')

    args = parser.parse_args()

    print("🔬 EXTERNAL DATASET EVALUATION")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Synthetic samples: {args.n_synthetic_samples}")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'external_eval_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = load_model_for_evaluation(args.model_path)
    if model is None:
        return

    all_results = {}

    # 1. Evaluate on synthetic test data
    print(f"\n{'='*60}")
    print("1. SYNTHETIC DATASET EVALUATION")
    print(f"{'='*60}")

    synthetic_test, synthetic_gt = create_synthetic_test_dataset(args.n_synthetic_samples)

    synthetic_psnr, synthetic_ssim = evaluate_on_external_data(
        model, synthetic_test, synthetic_gt, "Synthetic Test Dataset"
    )

    all_results['synthetic'] = {
        'n_samples': len(synthetic_test),
        'psnr_mean': float(np.mean(synthetic_psnr)),
        'psnr_std': float(np.std(synthetic_psnr)),
        'psnr_min': float(np.min(synthetic_psnr)),
        'psnr_max': float(np.max(synthetic_psnr)),
        'ssim_mean': float(np.mean(synthetic_ssim)),
        'ssim_std': float(np.std(synthetic_ssim))
    }

    print(f"Synthetic Results:")
    print(f"   PSNR: {all_results['synthetic']['psnr_mean']:.2f} ± {all_results['synthetic']['psnr_std']:.2f} dB")
    print(f"   SSIM: {all_results['synthetic']['ssim_mean']:.4f} ± {all_results['synthetic']['ssim_std']:.4f}")

    # 2. Evaluate on real test data if available
    if args.use_real_data and os.path.exists(args.real_data_path):
        print(f"\n{'='*60}")
        print("2. REAL TEST DATASET EVALUATION")
        print(f"{'='*60}")

        # Load real test data (implement if available)
        # This would require loading from the real test TFRecord
        print(f"Real test data evaluation would go here...")
        # For now, skip real data evaluation

    # 3. Compare with validation results (previous evaluation)
    print(f"\n{'='*60}")
    print("3. COMPARISON WITH VALIDATION RESULTS")
    print(f"{'='*60}")

    # Load previous validation results
    val_results_file = 'psnr_evaluation_production_v1/evaluation_20251021_054254/psnr_results.json'
    if os.path.exists(val_results_file):
        with open(val_results_file, 'r') as f:
            val_results = json.load(f)

        all_results['validation'] = {
            'n_samples': val_results['processed_samples'],
            'psnr_mean': val_results['psnr_statistics']['mean'],
            'psnr_std': val_results['psnr_statistics']['std'],
            'ssim_mean': val_results['ssim_statistics']['mean'],
            'ssim_std': val_results['ssim_statistics']['std']
        }

        # Calculate performance difference
        val_psnr = all_results['validation']['psnr_mean']
        synth_psnr = all_results['synthetic']['psnr_mean']
        psnr_diff = synth_psnr - val_psnr

        print(f"Validation vs Synthetic Performance:")
        print(f"   Validation PSNR: {val_psnr:.2f} ± {all_results['validation']['psnr_std']:.2f} dB")
        print(f"   Synthetic PSNR: {synth_psnr:.2f} ± {all_results['synthetic']['psnr_std']:.2f} dB")
        print(f"   Difference: {psnr_diff:+.2f} dB")

        if abs(psnr_diff) < 2.0:
            print(f"   ✅ Good generalization (|ΔPSNR| < 2.0 dB)")
        elif abs(psnr_diff) < 5.0:
            print(f"   ⚠️  Moderate generalization gap (2.0 < |ΔPSNR| < 5.0 dB)")
        else:
            print(f"   ❌ Poor generalization (|ΔPSNR| > 5.0 dB)")

    # Save results
    results = {
        'evaluation_timestamp': timestamp,
        'model_path': args.model_path,
        'evaluation_types': ['synthetic', 'validation_comparison'],
        'all_results': all_results,
        'generalization_assessment': {
            'method': 'synthetic_dataset_testing',
            'advantage': 'Tests model on data distributions not seen during training',
            'interpretation': 'Small performance gap indicates good generalization'
        }
    }

    if 'validation' in all_results and 'synthetic' in all_results:
        results['performance_gap'] = {
            'psnr_difference': all_results['synthetic']['psnr_mean'] - all_results['validation']['psnr_mean'],
            'ssim_difference': all_results['synthetic']['ssim_mean'] - all_results['validation']['ssim_mean']
        }

    results_file = os.path.join(output_dir, 'external_evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Academic reporting format
    print(f"\n📚 ACADEMIC REPORTING FORMAT:")
    if 'validation' in all_results and 'synthetic' in all_results:
        val_psnr = all_results['validation']['psnr_mean']
        synth_psnr = all_results['synthetic']['psnr_mean']

        print(f"   \"To assess generalization capability, we evaluated the model on {all_results['synthetic']['n_samples']} synthetic document samples")
        print(f"    with varying degradation types. The model achieved PSNR of {synth_psnr:.2f} ± {all_results['synthetic']['psnr_std']:.2f} dB on synthetic data,")
        print(f"    showing a {abs(synth_psnr - val_psnr):.2f} dB difference compared to validation performance ({val_psnr:.2f} dB),")
        print(f"    indicating {'good' if abs(synth_psnr - val_psnr) < 2.0 else 'moderate' if abs(synth_psnr - val_psnr) < 5.0 else 'limited'} generalization capability.\"")

    print(f"\n✅ External evaluation completed!")

if __name__ == '__main__':
    main()