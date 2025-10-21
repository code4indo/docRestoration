#!/usr/bin/env python3
"""
Test Patch-Based GAN-HTR Inference System
Menguji sistem inferensi universal dengan dataset H-DIBCO
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

from scripts.patch_based_inference import UniversalGANHTRProcessor

def test_patch_extraction():
    """Test patch extraction functionality"""
    print("🧪 Testing Patch Extraction")
    print("=" * 50)

    # Import PatchExtractor
    from scripts.patch_based_inference import PatchExtractor

    # Create test image
    test_image = np.random.rand(2000, 300) * 255
    test_image = test_image.astype(np.uint8)

    print(f"Test image shape: {test_image.shape}")

    # Test different strategies
    strategies = ['sliding', 'grid', 'adaptive']
    patch_size = (1024, 128)

    for strategy in strategies:
        print(f"\n🔍 Testing {strategy} strategy:")

        try:
            extractor = PatchExtractor(
                patch_size=patch_size,
                overlap=0.25,
                strategy=strategy
            )

            # Get patch info
            info = extractor.get_patch_info(test_image.shape)
            print(f"   Patch info: {info}")

            # Extract patches
            patches, positions = extractor.extract_patches(test_image)
            print(f"   Extracted {len(patches)} patches")
            print(f"   Patch shape: {patches[0].shape if len(patches) > 0 else 'None'}")
            print(f"   Position sample: {positions[:3] if len(positions) > 3 else positions}")

            # Verify patch dimensions
            if len(patches) > 0:
                expected_patch_h, expected_patch_w = patch_size
                actual_patch_h, actual_patch_w = patches[0].shape[:2]
                print(f"   Patch dimensions: {actual_patch_h}x{actual_patch_w}")

                # Check if patches are within expected size range
                if actual_patch_h <= expected_patch_h and actual_patch_w <= expected_patch_w:
                    print(f"   ✅ Patch dimensions OK")
                else:
                    print(f"   ⚠️ Patch dimensions unexpected")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n✅ Patch extraction test completed\n")

def test_image_reconstruction():
    """Test image reconstruction functionality"""
    print("🧪 Testing Image Reconstruction")
    print("=" * 50)

    from scripts.patch_based_inference import PatchExtractor, ImageReconstructor

    # Create test image
    original_image = np.random.rand(1000, 400) * 255
    original_image = original_image.astype(np.uint8)
    original_image = original_image.astype(np.float32) / 255.0

    print(f"Original image shape: {original_image.shape}")

    # Extract patches
    extractor = PatchExtractor(
        patch_size=(512, 64),
        overlap=0.2,
        strategy='sliding'
    )

    patches, positions = extractor.extract_patches(original_image)
    print(f"Extracted {len(patches)} patches")

    # Test reconstruction methods
    blend_methods = ['linear', 'gaussian']

    for method in blend_methods:
        print(f"\n🔨 Testing {method} blending:")

        try:
            reconstructor = ImageReconstructor(
                original_size=original_image.shape,
                patch_size=(512, 64),
                stride=extractor.effective_stride[0],
                blend_method=method
            )

            # Reconstruct image
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

            # Verify reconstruction quality
            if metrics['psnr'] > 20:  # Reasonable PSNR for perfect reconstruction
                print(f"   ✅ Reconstruction quality OK")
            else:
                print(f"   ⚠️ Low reconstruction quality")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n✅ Image reconstruction test completed\n")

def test_with_hdibco():
    """Test with actual H-DIBCO dataset"""
    print("🧪 Testing with H-DIBCO Dataset")
    print("=" * 50)

    # Check if dataset exists
    hdibco_dir = Path("dibco_datasets/DIPCO2016_dataset")
    if not hdibco_dir.exists():
        print(f"❌ H-DIBCO dataset not found at {hdibco_dir}")
        print("   Please run download_dibco.py first")
        return False

    # Find images (BMP format for H-DIBCO)
    image_files = list(hdibco_dir.glob("*.bmp"))
    if not image_files:
        print(f"❌ No BMP images found in {hdibco_dir}")
        return False

    test_image = image_files[0]
    print(f"Testing with: {test_image}")

    try:
        # Initialize processor
        model_path = "dual_modal_gan/checkpoints/full_training_production_v1/best_model"
        if not os.path.exists(model_path):
            print(f"❌ Model checkpoint not found: {model_path}")
            print("   Available checkpoints:")
            checkpoint_base = "dual_modal_gan/checkpoints"
            if os.path.exists(checkpoint_base):
                for item in os.listdir(checkpoint_base):
                    if os.path.isdir(os.path.join(checkpoint_base, item)):
                        print(f"     - {item}")
            return False

        # Create processor with small patches for testing
        processor = UniversalGANHTRProcessor(
            model_path=model_path,
            patch_size=(512, 64),  # Smaller patches for testing
            overlap=0.2,
            strategy='sliding',
            blend_method='linear',
            batch_size=2  # Small batch for testing
        )

        # Process image
        output_dir = Path("test_outputs/patch_inference_test")
        output_dir.mkdir(exist_ok=True, parents=True)

        result = processor.process_image(
            input_path=str(test_image),
            output_path=str(output_dir / f"{test_image.stem}_enhanced.png")
        )

        print(f"\n✅ H-DIBCO test completed successfully!")
        print(f"   Input: {result['input_path']}")
        print(f"   Output: {result['output_path']}")
        print(f"   Comparison: {result['comparison_path']}")
        print(f"   Patches processed: {result['n_patches']}")
        print(f"   PSNR: {result['reconstruction_metrics']['psnr']:.2f} dB")

        return True

    except Exception as e:
        print(f"❌ Error during H-DIBCO test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_processor_configuration():
    """Test different processor configurations"""
    print("🧪 Testing Processor Configurations")
    print("=" * 50)

    model_path = "dual_modal_gan/checkpoints/full_training_production_v1/best_model"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    # Test configurations
    configs = [
        {
            'name': 'Small Patches',
            'patch_size': (256, 32),
            'overlap': 0.1,
            'strategy': 'sliding',
            'blend_method': 'linear'
        },
        {
            'name': 'Medium Patches',
            'patch_size': (512, 64),
            'overlap': 0.25,
            'strategy': 'grid',
            'blend_method': 'linear'
        },
        {
            'name': 'Large Patches',
            'patch_size': (1024, 128),
            'overlap': 0.3,
            'strategy': 'sliding',
            'blend_method': 'gaussian'
        }
    ]

    # Create test image
    test_image_path = Path("test_outputs/test_image.png")
    test_image_path.parent.mkdir(exist_ok=True, parents=True)

    # Generate test document-like image
    test_image = np.random.rand(1500, 600) * 255
    test_image = test_image.astype(np.uint8)
    cv2.imwrite(str(test_image_path), test_image)

    print(f"Generated test image: {test_image_path}")
    print(f"Test image shape: {test_image.shape}")

    for config in configs:
        print(f"\n🔧 Testing {config['name']}:")
        print(f"   Config: {config}")

        try:
            processor = UniversalGANHTRProcessor(
                model_path=model_path,
                patch_size=config['patch_size'],
                overlap=config['overlap'],
                strategy=config['strategy'],
                blend_method=config['blend_method'],
                batch_size=2
            )

            output_path = test_image_path.parent / f"test_{config['name'].lower().replace(' ', '_')}.png"

            result = processor.process_image(
                input_path=str(test_image_path),
                output_path=str(output_path)
            )

            print(f"   ✅ Configuration successful!")
            print(f"   Patches: {result['n_patches']}")
            print(f"   PSNR: {result['reconstruction_metrics']['psnr']:.2f} dB")
            print(f"   Output: {output_path}")

        except Exception as e:
            print(f"   ❌ Configuration failed: {e}")

def run_all_tests():
    """Run all tests"""
    print("🚀 Running Patch-Based Inference System Tests")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Working directory: {os.getcwd()}")
    print()

    # Create test outputs directory
    test_output_dir = Path("test_outputs")
    test_output_dir.mkdir(exist_ok=True)

    # Run individual tests
    tests = [
        ("Patch Extraction", test_patch_extraction),
        ("Image Reconstruction", test_image_reconstruction),
        ("Processor Configuration", test_processor_configuration),
        ("H-DIBCO Dataset", test_with_hdibco)
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")

        try:
            start_time = datetime.now()
            success = test_func()
            end_time = datetime.now()

            results[test_name] = {
                'success': success if 'success' in locals() else True,
                'duration': str(end_time - start_time),
                'timestamp': end_time.isoformat()
            }

        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results[test_name] = {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    # Save test results
    test_results = {
        'test_run_timestamp': datetime.now().isoformat(),
        'working_directory': os.getcwd(),
        'test_results': results,
        'summary': {
            'total_tests': len(tests),
            'passed': sum(1 for r in results.values() if r.get('success', False)),
            'failed': sum(1 for r in results.values() if not r.get('success', False))
        }
    }

    results_file = test_output_dir / "patch_inference_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {test_results['summary']['total_tests']}")
    print(f"Passed: {test_results['summary']['passed']}")
    print(f"Failed: {test_results['summary']['failed']}")
    print(f"Results saved to: {results_file}")

    if test_results['summary']['failed'] == 0:
        print("\n🎉 All tests passed! Patch-based inference system is ready.")
    else:
        print(f"\n⚠️ {test_results['summary']['failed']} test(s) failed. Check logs for details.")

    return test_results

if __name__ == '__main__':
    # Run tests
    results = run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if results['summary']['failed'] == 0 else 1)