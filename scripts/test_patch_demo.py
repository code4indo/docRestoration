#!/usr/bin/env python3
"""
Demo Patch-Based Inference System
Demonstrasi sistem inferensi universal tanpa perlu model
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.patch_based_inference import PatchExtractor, ImageReconstructor

def simulate_enhancement(patch):
    """Simulate enhancement process (placeholder for actual model)"""
    # Add some enhancement-like processing
    # 1. Contrast enhancement
    enhanced = cv2.convertScaleAbs(patch, alpha=1.2, beta=10)

    # 2. Noise reduction
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)

    # 3. Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    return enhanced

def process_with_simulation(image_path, output_dir, patch_size=(512, 64), overlap=0.25):
    """Process image using patch-based approach with simulated enhancement"""

    print(f"\n🎯 Processing: {image_path}")
    print("=" * 60)

    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None

    # Convert to float and normalize
    image = image.astype(np.float32) / 255.0
    print(f"Original shape: {image.shape}")

    # Extract patches
    extractor = PatchExtractor(
        patch_size=patch_size,
        overlap=overlap,
        strategy='sliding'
    )

    patches, positions = extractor.extract_patches(image)
    print(f"Extracted {len(patches)} patches")

    # Process patches (simulate enhancement)
    print("🔄 Processing patches (simulated enhancement)...")
    processed_patches = []

    for i, patch in enumerate(patches):
        # Convert to uint8 for processing
        patch_uint8 = (patch * 255).astype(np.uint8)

        # Simulate enhancement
        enhanced_patch = simulate_enhancement(patch_uint8)

        # Convert back to float
        processed_patch = enhanced_patch.astype(np.float32) / 255.0
        processed_patches.append(processed_patch)

        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(patches)} patches")

    # Reconstruct image
    reconstructor = ImageReconstructor(
        original_size=image.shape,
        patch_size=patch_size,
        stride=extractor.effective_stride[0],
        blend_method='linear'
    )

    enhanced_image = reconstructor.reconstruct_image(
        np.array(processed_patches),
        positions
    )

    # Calculate metrics
    metrics = reconstructor.calculate_reconstruction_quality(image, enhanced_image)
    print(f"Reconstruction metrics:")
    print(f"   PSNR: {metrics['psnr']:.2f} dB")
    if metrics['ssim'] is not None:
        print(f"   SSIM: {metrics['ssim']:.4f}")
    print(f"   MSE: {metrics['mse']:.6f}")
    print(f"   MAE: {metrics['mae']:.6f}")

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Convert back to uint8 for saving
    original_uint8 = (image * 255).astype(np.uint8)
    enhanced_uint8 = (enhanced_image * 255).astype(np.uint8)

    # Save individual images
    image_name = Path(image_path).stem
    original_path = output_dir / f"{image_name}_original.png"
    enhanced_path = output_dir / f"{image_name}_enhanced.png"
    comparison_path = output_dir / f"{image_name}_comparison.png"

    cv2.imwrite(str(original_path), original_uint8)
    cv2.imwrite(str(enhanced_path), enhanced_uint8)

    # Create comparison
    comparison = np.hstack([original_uint8, enhanced_uint8])
    cv2.imwrite(str(comparison_path), comparison)

    print(f"💾 Results saved:")
    print(f"   Original: {original_path}")
    print(f"   Enhanced: {enhanced_path}")
    print(f"   Comparison: {comparison_path}")

    return {
        'original_path': str(original_path),
        'enhanced_path': str(enhanced_path),
        'comparison_path': str(comparison_path),
        'original_shape': image.shape,
        'enhanced_shape': enhanced_image.shape,
        'n_patches': len(patches),
        'metrics': metrics
    }

def demo_with_hdibco():
    """Demo with H-DIBCO dataset"""
    print("🎨 DEMO: Patch-Based Document Enhancement")
    print("=" * 60)

    # Check dataset
    hdibco_dir = Path("dibco_datasets/DIPCO2016_dataset")
    if not hdibco_dir.exists():
        print(f"❌ H-DIBCO dataset not found at {hdibco_dir}")
        return False

    # Find images
    image_files = list(hdibco_dir.glob("*.bmp"))
    if not image_files:
        print(f"❌ No BMP images found in {hdibco_dir}")
        return False

    print(f"Found {len(image_files)} images in H-DIBCO dataset")

    # Process first few images
    output_dir = Path("demo_outputs/patch_enhancement_demo")
    output_dir.mkdir(exist_ok=True, parents=True)

    results = []
    max_images = min(3, len(image_files))  # Process up to 3 images

    for i, image_path in enumerate(image_files[:max_images]):
        print(f"\n🖼️ Image {i+1}/{max_images}: {image_path.name}")

        result = process_with_simulation(
            image_path,
            output_dir,
            patch_size=(512, 64),
            overlap=0.2
        )

        if result:
            results.append(result)

    # Save summary
    summary = {
        'demo_timestamp': datetime.now().isoformat(),
        'dataset': 'H-DIBCO 2016',
        'total_images_found': len(image_files),
        'images_processed': len(results),
        'processing_method': 'patch-based_with_simulated_enhancement',
        'configuration': {
            'patch_size': (512, 64),
            'overlap': 0.2,
            'strategy': 'sliding',
            'blend_method': 'linear'
        },
        'results': results,
        'output_directory': str(output_dir)
    }

    summary_path = output_dir / "demo_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n📊 Demo Summary:")
    print(f"   Dataset: H-DIBCO 2016")
    print(f"   Images processed: {len(results)}/{len(image_files)}")
    print(f"   Patch size: 512x64")
    print(f"   Overlap: 20%")
    print(f"   Summary saved: {summary_path}")
    print(f"   Output directory: {output_dir}")

    return True

def demo_different_patch_sizes():
    """Demo with different patch sizes"""
    print("\n🔧 DEMO: Different Patch Sizes")
    print("=" * 50)

    # Create test image
    test_image = np.random.rand(1500, 800) * 255
    test_image = test_image.astype(np.uint8)

    # Add document-like structure
    for i in range(15):
        y = int(i * 1500 / 15)
        test_image[y:y+8, :] = np.random.randint(0, 100, (8, 800))

    # Save test image
    test_path = Path("demo_outputs/test_document.png")
    test_path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(test_path), test_image)

    print(f"Generated test document: {test_path}")

    # Test different patch sizes
    configs = [
        {'patch_size': (256, 32), 'name': 'Small'},
        {'patch_size': (512, 64), 'name': 'Medium'},
        {'patch_size': (1024, 128), 'name': 'Large'},
    ]

    output_dir = Path("demo_outputs/patch_size_comparison")
    output_dir.mkdir(exist_ok=True, parents=True)

    for config in configs:
        print(f"\n📐 Testing {config['name']} patches: {config['patch_size']}")

        result = process_with_simulation(
            test_path,
            output_dir / f"patches_{config['name'].lower()}",
            patch_size=config['patch_size'],
            overlap=0.25
        )

        if result:
            print(f"   ✅ {config['name']} patches: {result['n_patches']} patches")
            print(f"   PSNR: {result['metrics']['psnr']:.2f} dB")
        else:
            print(f"   ❌ {config['name']} patches failed")

def main():
    """Run all demos"""
    print("🚀 PATCH-BASED INFERENCE DEMONSTRATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Working directory: {os.getcwd()}")
    print()

    # Run demos
    success = True

    # Demo 1: H-DIBCO dataset
    if not demo_with_hdibco():
        success = False

    # Demo 2: Different patch sizes
    try:
        demo_different_patch_sizes()
    except Exception as e:
        print(f"❌ Patch size demo failed: {e}")
        success = False

    # Final summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All demos completed successfully!")
        print("\n📋 What was demonstrated:")
        print("   ✅ Patch extraction from arbitrary-sized images")
        print("   ✅ Patch processing with simulated enhancement")
        print("   ✅ Image reconstruction with seamless blending")
        print("   ✅ Different patch sizes and overlap settings")
        print("   ✅ Processing real H-DIBCO document images")
        print("\n📁 Demo outputs saved to: demo_outputs/")
        print("\n🔧 Next steps:")
        print("   1. Replace simulated enhancement with actual GAN-HTR model")
        print("   2. Test with different document types")
        print("   3. Optimize patch sizes for different image dimensions")
        print("   4. Add quality assessment metrics")
    else:
        print("⚠️ Some demos failed. Check logs for details.")
    print("="*60)

if __name__ == '__main__':
    main()