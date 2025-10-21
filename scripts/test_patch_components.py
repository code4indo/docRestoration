#!/usr/bin/env python3
"""
Test Patch Components Only (No Model Loading)
Menguji komponen patch extraction dan reconstruction tanpa model
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.patch_based_inference import PatchExtractor, ImageReconstructor

def test_patch_extraction_fixed():
    """Test patch extraction functionality after bugfix"""
    print("🧪 Testing Patch Extraction (Fixed)")
    print("=" * 50)

    # Create test image with various sizes
    test_sizes = [
        (2000, 300),   # Large image
        (1000, 400),   # Medium image
        (500, 200),    # Small image
        (1024, 128),   # Exact patch size
    ]

    strategies = ['sliding', 'grid', 'adaptive']
    patch_size = (1024, 128)

    for test_size in test_sizes:
        print(f"\n📐 Testing image size: {test_size}")

        # Create test image
        test_image = np.random.rand(*test_size) * 255
        test_image = test_image.astype(np.uint8)

        for strategy in strategies:
            print(f"   🔍 Strategy: {strategy}")

            try:
                extractor = PatchExtractor(
                    patch_size=patch_size,
                    overlap=0.25,
                    strategy=strategy
                )

                # Get patch info
                info = extractor.get_patch_info(test_image.shape)
                print(f"      Expected patches: {info['n_patches']}")
                print(f"      Coverage: {info['coverage']}")

                # Extract patches
                patches, positions = extractor.extract_patches(test_image)
                print(f"      Actual patches: {len(patches)}")

                # Verify patches
                if len(patches) > 0:
                    patch_shape = patches[0].shape
                    print(f"      Patch shape: {patch_shape}")

                    # Check if patch dimensions are reasonable
                    max_h, max_w = patch_size
                    actual_h, actual_w = patch_shape[:2]

                    if actual_h <= max_h and actual_w <= max_w:
                        print(f"      ✅ Patch dimensions OK")
                    else:
                        print(f"      ⚠️ Unexpected patch dimensions")

                    # Test positions
                    print(f"      Sample positions: {positions[:3]}")
                else:
                    print(f"      ⚠️ No patches extracted")

            except Exception as e:
                print(f"      ❌ Error: {e}")

def test_image_reconstruction_comprehensive():
    """Test image reconstruction comprehensively"""
    print("\n🧪 Testing Image Reconstruction (Comprehensive)")
    print("=" * 50)

    # Create more realistic test image (document-like)
    h, w = 1500, 600
    original_image = np.random.rand(h, w) * 255
    original_image = original_image.astype(np.uint8)

    # Add some structure (lines of text simulation)
    for i in range(10):
        y = int(i * h / 10)
        original_image[y:y+5, :] = np.random.randint(0, 50, (5, w))  # Dark lines

    original_image = original_image.astype(np.float32) / 255.0

    print(f"Test image shape: {original_image.shape}")
    print(f"Image type: Document-like simulation")

    # Test configurations
    configs = [
        {'patch_size': (512, 64), 'overlap': 0.1, 'blend_method': 'linear'},
        {'patch_size': (512, 64), 'overlap': 0.25, 'blend_method': 'gaussian'},
        {'patch_size': (256, 32), 'overlap': 0.2, 'blend_method': 'linear'},
        {'patch_size': (1024, 128), 'overlap': 0.3, 'blend_method': 'gaussian'},
    ]

    for i, config in enumerate(configs, 1):
        print(f"\n🔧 Test {i}: {config}")

        try:
            # Extract patches
            extractor = PatchExtractor(
                patch_size=config['patch_size'],
                overlap=config['overlap'],
                strategy='sliding'
            )

            patches, positions = extractor.extract_patches(original_image)
            print(f"   Extracted {len(patches)} patches")

            # Test reconstruction
            reconstructor = ImageReconstructor(
                original_size=original_image.shape,
                patch_size=config['patch_size'],
                stride=extractor.effective_stride[0],
                blend_method=config['blend_method']
            )

            reconstructed = reconstructor.reconstruct_image(patches, positions)
            print(f"   Reconstructed shape: {reconstructed.shape}")

            # Calculate quality metrics
            metrics = reconstructor.calculate_reconstruction_quality(
                original_image, reconstructed
            )

            print(f"   PSNR: {metrics['psnr']:.2f} dB")
            if metrics['ssim'] is not None:
                print(f"   SSIM: {metrics['ssim']:.4f}")
            print(f"   MSE: {metrics['mse']:.6f}")
            print(f"   MAE: {metrics['mae']:.6f}")

            # Save test images
            test_dir = Path("test_outputs/reconstruction_test")
            test_dir.mkdir(exist_ok=True, parents=True)

            # Save original, reconstructed, and difference
            original_uint8 = (original_image * 255).astype(np.uint8)
            reconstructed_uint8 = (reconstructed * 255).astype(np.uint8)

            cv2.imwrite(str(test_dir / f"test_{i}_original.png"), original_uint8)
            cv2.imwrite(str(test_dir / f"test_{i}_reconstructed_{config['blend_method']}.png"), reconstructed_uint8)

            # Create difference image
            diff = np.abs(original_uint8.astype(np.float32) - reconstructed_uint8.astype(np.float32))
            diff_uint8 = diff.astype(np.uint8)
            cv2.imwrite(str(test_dir / f"test_{i}_difference.png"), diff_uint8)

            print(f"   📁 Test images saved to: {test_dir}")

            # Verify reconstruction quality
            if metrics['psnr'] > 25:  # Good reconstruction
                print(f"   ✅ Excellent reconstruction quality")
            elif metrics['psnr'] > 20:
                print(f"   ✅ Good reconstruction quality")
            elif metrics['psnr'] > 15:
                print(f"   ⚠️ Moderate reconstruction quality")
            else:
                print(f"   ❌ Poor reconstruction quality")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🧪 Testing Edge Cases")
    print("=" * 50)

    # Test 1: Image smaller than patch size
    print("\n📏 Test 1: Image smaller than patch size")
    try:
        small_image = np.random.rand(500, 100) * 255
        small_image = small_image.astype(np.uint8)

        extractor = PatchExtractor(patch_size=(1024, 128), strategy='sliding')
        patches, positions = extractor.extract_patches(small_image)
        print(f"   Small image result: {len(patches)} patches")

        if len(patches) == 0:
            print("   ✅ Correctly handled too-small image")
        else:
            print(f"   ⚠️ Unexpected patches from small image")

    except Exception as e:
        print(f"   ❌ Error with small image: {e}")

    # Test 2: Very large image
    print("\n📏 Test 2: Very large image")
    try:
        large_image = np.random.rand(3000, 2000) * 255
        large_image = large_image.astype(np.uint8)

        extractor = PatchExtractor(patch_size=(512, 64), overlap=0.3, strategy='sliding')
        info = extractor.get_patch_info(large_image.shape)
        print(f"   Large image patch estimate: {info['n_patches']}")

        if info['n_patches'] > 100:
            print("   ⚠️ Large number of patches - memory intensive")
        else:
            print("   ✅ Reasonable number of patches")

    except Exception as e:
        print(f"   ❌ Error with large image: {e}")

    # Test 3: Extreme overlap
    print("\n📏 Test 3: Extreme overlap values")
    try:
        test_image = np.random.rand(1000, 400) * 255
        test_image = test_image.astype(np.uint8)

        for overlap in [0.0, 0.5, 0.8]:
            print(f"   Testing overlap: {overlap}")
            extractor = PatchExtractor(patch_size=(256, 32), overlap=overlap, strategy='sliding')
            patches, positions = extractor.extract_patches(test_image)
            print(f"      Patches with {overlap:.0%} overlap: {len(patches)}")

    except Exception as e:
        print(f"   ❌ Error with extreme overlap: {e}")

def run_component_tests():
    """Run all component tests"""
    print("🚀 Running Patch-Based Component Tests")
    print("=" * 60)
    print(f"Working directory: {os.getcwd()}")
    print()

    # Create test outputs directory
    test_output_dir = Path("test_outputs")
    test_output_dir.mkdir(exist_ok=True)

    # Run tests
    test_patch_extraction_fixed()
    test_image_reconstruction_comprehensive()
    test_edge_cases()

    print(f"\n{'='*60}")
    print("✅ All component tests completed!")
    print(f"Test outputs saved to: {test_output_dir}")
    print("="*60)

if __name__ == '__main__':
    run_component_tests()