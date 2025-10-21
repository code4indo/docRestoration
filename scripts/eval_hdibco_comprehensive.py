#!/usr/bin/env python3
"""
Comprehensive H-DIBCO Evaluation with Patch-Based Inference
Evaluasi lengkap pada dataset H-DIBCO dengan patch-based processing
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.patch_based_inference import PatchExtractor, ImageReconstructor

def simulate_gan_enhancement(patch):
    """
    Simulasi enhancement GAN-HTR untuk evaluasi
    Menggunakan berbagai teknik preprocessing umum untuk document enhancement
    """
    # Convert to uint8 untuk processing
    if patch.dtype == np.float32:
        patch_uint8 = (patch * 255).astype(np.uint8)
    else:
        patch_uint8 = patch

    # 1. Contrast Enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(patch_uint8)

    # 2. Noise Reduction
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # 3. Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    # 4. Morphological operations untuk text enhancement
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

    # 5. Convert kembali ke float [0,1]
    enhanced_float = enhanced.astype(np.float32) / 255.0

    return enhanced_float

def evaluate_single_image(image_path, patch_config):
    """
    Evaluasi single image dengan patch-based processing
    """
    try:
        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None

        # Normalize ke [0,1]
        image = image.astype(np.float32) / 255.0

        # Patch extraction
        extractor = PatchExtractor(
            patch_size=patch_config['size'],
            overlap=patch_config['overlap'],
            strategy=patch_config['strategy']
        )

        patches, positions = extractor.extract_patches(image)

        # Handle case dimana tidak ada patch terextract
        if len(patches) == 0:
            # Fallback: resize image ke minimum patch size
            min_h, min_w = patch_config['size']
            scale_h = min_h / image.shape[0]
            scale_w = min_w / image.shape[1]
            scale = max(scale_h, scale_w)

            new_h = int(image.shape[0] * scale)
            new_w = int(image.shape[1] * scale)
            image = cv2.resize(image, (new_w, new_h))

            # Retry extraction
            patches, positions = extractor.extract_patches(image)

        # Process patches
        processed_patches = []
        for patch in patches:
            enhanced_patch = simulate_gan_enhancement(patch)
            processed_patches.append(enhanced_patch)

        # Reconstruction
        reconstructor = ImageReconstructor(
            original_size=image.shape,
            patch_size=patch_config['size'],
            stride=extractor.effective_stride[0],
            blend_method=patch_config['blend_method']
        )

        enhanced_image = reconstructor.reconstruct_image(
            np.array(processed_patches), positions
        )

        # Calculate metrics
        metrics = reconstructor.calculate_reconstruction_quality(image, enhanced_image)

        return {
            'image_path': str(image_path),
            'original_shape': image.shape,
            'enhanced_shape': enhanced_image.shape,
            'n_patches': len(patches),
            'patch_config': patch_config,
            'metrics': metrics,
            'enhanced_image': enhanced_image,
            'original_image': image
        }

    except Exception as e:
        print(f"   ❌ Error processing {image_path.name}: {e}")
        return None

def run_comprehensive_evaluation():
    """
    Jalankan evaluasi komprehensif pada H-DIBCO dataset
    """
    print("🔬 COMPREHENSIVE H-DIBCO EVALUATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load dataset
    dataset_dir = Path("dibco_datasets/DIPCO2016_dataset")
    if not dataset_dir.exists():
        print(f"❌ Dataset not found: {dataset_dir}")
        return None

    image_files = sorted(list(dataset_dir.glob("*.bmp")))
    print(f"📁 Found {len(image_files)} images in H-DIBCO 2016 dataset")

    # Test configurations
    configs = [
        {
            'name': 'Small Patches',
            'size': (256, 32),
            'overlap': 0.2,
            'strategy': 'sliding',
            'blend_method': 'linear'
        },
        {
            'name': 'Medium Patches',
            'size': (512, 64),
            'overlap': 0.25,
            'strategy': 'sliding',
            'blend_method': 'linear'
        },
        {
            'name': 'Large Patches',
            'size': (1024, 128),
            'overlap': 0.3,
            'strategy': 'sliding',
            'blend_method': 'gaussian'
        }
    ]

    # Output directory
    output_dir = Path("hdibco_evaluation_results")
    output_dir.mkdir(exist_ok=True, parents=True)

    all_results = {}

    for config in configs:
        print(f"\n🔧 Testing {config['name']} Configuration:")
        print(f"   Patch size: {config['size']}")
        print(f"   Overlap: {config['overlap']:.1%}")
        print(f"   Strategy: {config['strategy']}")
        print(f"   Blend method: {config['blend_method']}")

        config_results = []
        successful = 0

        for i, image_path in enumerate(image_files, 1):
            print(f"   🖼️ Image {i:2d}/10: {image_path.name:<8s}", end=" ")

            result = evaluate_single_image(image_path, config)

            if result:
                config_results.append(result)
                successful += 1

                # Save images for this configuration
                config_dir = output_dir / config['name'].lower().replace(' ', '_')
                config_dir.mkdir(exist_ok=True)

                # Save original and enhanced
                original_path = config_dir / f"{image_path.stem}_original.png"
                enhanced_path = config_dir / f"{image_path.stem}_enhanced.png"
                comparison_path = config_dir / f"{image_path.stem}_comparison.png"

                cv2.imwrite(str(original_path), (result['original_image'] * 255).astype(np.uint8))
                cv2.imwrite(str(enhanced_path), (result['enhanced_image'] * 255).astype(np.uint8))

                # Create comparison
                original_uint8 = (result['original_image'] * 255).astype(np.uint8)
                enhanced_uint8 = (result['enhanced_image'] * 255).astype(np.uint8)
                comparison = np.hstack([original_uint8, enhanced_uint8])
                cv2.imwrite(str(comparison_path), comparison)

                print(f"✅ Patches: {result['n_patches']:3d} PSNR: {result['metrics']['psnr']:5.2f} SSIM: {result['metrics']['ssim']:.3f}")
            else:
                print("❌ Failed")

        all_results[config['name']] = {
            'config': config,
            'results': config_results,
            'successful_count': successful,
            'success_rate': successful / len(image_files)
        }

        print(f"   📊 Success rate: {successful}/{len(image_files)} ({successful/len(image_files):.1%})")

        if config_results:
            # Calculate average metrics
            avg_psnr = np.mean([r['metrics']['psnr'] for r in config_results])
            avg_ssim = np.mean([r['metrics']['ssim'] for r in config_results if r['metrics']['ssim'] is not None])
            avg_mse = np.mean([r['metrics']['mse'] for r in config_results])
            avg_mae = np.mean([r['metrics']['mae'] for r in config_results])

            print(f"   📈 Average PSNR: {avg_psnr:.2f} dB")
            if not np.isnan(avg_ssim):
                print(f"   📈 Average SSIM: {avg_ssim:.4f}")
            print(f"   📈 Average MSE:  {avg_mse:.6f}")
            print(f"   📈 Average MAE:  {avg_mae:.6f}")

            # Save metrics for plotting
            config_dir = output_dir / config['name'].lower().replace(' ', '_')
            metrics_data = {
                'image_names': [Path(r['image_path']).stem for r in config_results],
                'psnr_values': [r['metrics']['psnr'] for r in config_results],
                'ssim_values': [r['metrics']['ssim'] for r in config_results],
                'n_patches': [r['n_patches'] for r in config_results]
            }

            with open(config_dir / "metrics.json", 'w') as f:
                json.dump(metrics_data, f, indent=2)

    # Generate comprehensive report
    print(f"\n📋 GENERATING COMPREHENSIVE REPORT")
    print("=" * 40)

    report = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'dataset': 'H-DIBCO 2016',
        'total_images': len(image_files),
        'configurations': {}
    }

    best_config = None
    best_psnr = -1

    for config_name, config_data in all_results.items():
        if config_data['results']:
            avg_psnr = np.mean([r['metrics']['psnr'] for r in config_data['results']])
            avg_ssim = np.mean([r['metrics']['ssim'] for r in config_data['results'] if r['metrics']['ssim'] is not None])

            config_summary = {
                'patch_size': config_data['config']['size'],
                'overlap': config_data['config']['overlap'],
                'strategy': config_data['config']['strategy'],
                'blend_method': config_data['config']['blend_method'],
                'success_rate': config_data['success_rate'],
                'images_processed': config_data['successful_count'],
                'avg_psnr': float(avg_psnr),
                'avg_ssim': float(avg_ssim) if not np.isnan(avg_ssim) else None,
                'avg_mse': float(np.mean([r['metrics']['mse'] for r in config_data['results']])),
                'avg_mae': float(np.mean([r['metrics']['mae'] for r in config_data['results']]))
            }

            report['configurations'][config_name] = config_summary

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_config = config_name

    # Save full report
    with open(output_dir / "comprehensive_evaluation_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    # Print final summary
    print(f"\n🏆 EVALUATION SUMMARY")
    print("=" * 40)
    print(f"Dataset: H-DIBCO 2016 ({len(image_files)} images)")
    print(f"Configurations tested: {len(all_results)}")
    print(f"Best configuration: {best_config}")
    print(f"Best average PSNR: {best_psnr:.2f} dB")
    print(f"Results saved to: {output_dir}")

    # Create performance comparison plot
    try:
        create_performance_plot(all_results, output_dir)
    except Exception as e:
        print(f"⚠️ Could not create performance plot: {e}")

    return report

def create_performance_plot(all_results, output_dir):
    """
    Buat plot perbandingan performa antar konfigurasi
    """
    config_names = []
    avg_psnrs = []
    avg_ssims = []
    success_rates = []

    for config_name, config_data in all_results.items():
        if config_data['results']:
            config_names.append(config_name.replace(' ', '\n'))
            avg_psnrs.append(np.mean([r['metrics']['psnr'] for r in config_data['results']]))

            ssim_values = [r['metrics']['ssim'] for r in config_data['results'] if r['metrics']['ssim'] is not None]
            avg_ssims.append(np.mean(ssim_values) if ssim_values else 0)

            success_rates.append(config_data['success_rate'] * 100)

    # Create subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('H-DIBCO 2016 Patch-Based Processing Performance', fontsize=16, fontweight='bold')

    # PSNR comparison
    bars1 = ax1.bar(config_names, avg_psnrs, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax1.set_title('Average PSNR (dB)', fontweight='bold')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_ylim(0, max(avg_psnrs) + 5)
    for bar, value in zip(bars1, avg_psnrs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.2f}', ha='center', va='bottom')
    ax1.grid(True, alpha=0.3)

    # SSIM comparison
    bars2 = ax2.bar(config_names, avg_ssims, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax2.set_title('Average SSIM', fontweight='bold')
    ax2.set_ylabel('SSIM')
    ax2.set_ylim(0, 1)
    for bar, value in zip(bars2, avg_ssims):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom')
    ax2.grid(True, alpha=0.3)

    # Success rate comparison
    bars3 = ax3.bar(config_names, success_rates, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax3.set_title('Success Rate (%)', fontweight='bold')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_ylim(0, 105)
    for bar, value in zip(bars3, success_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{value:.0f}%', ha='center', va='bottom')
    ax3.grid(True, alpha=0.3)

    # Combined metrics scatter plot
    ax4.scatter(avg_psnrs, avg_ssims, s=100, c=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.7)
    for i, config in enumerate(config_names):
        ax4.annotate(config.replace('\n', ' '), (avg_psnrs[i], avg_ssims[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax4.set_xlabel('Average PSNR (dB)')
    ax4.set_ylabel('Average SSIM')
    ax4.set_title('PSNR vs SSIM', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(min(avg_psnrs) - 1, max(avg_psnrs) + 1)
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Performance plot saved to: {output_dir}/performance_comparison.png")

if __name__ == '__main__':
    # Run comprehensive evaluation
    report = run_comprehensive_evaluation()

    if report:
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📁 Check results in: hdibco_evaluation_results/")
        print(f"📊 Performance plot and comprehensive report generated!")
    else:
        print(f"\n❌ Evaluation failed!")
        sys.exit(1)