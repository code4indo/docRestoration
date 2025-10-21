#!/usr/bin/env python3
"""
Test Real GAN-HTR Model on H-DIBCO Dataset
Testing actual model inference with patch-based processing
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

def test_real_model_on_single_image():
    """Test real GAN-HTR model on single H-DIBCO image"""
    print("🤖 TESTING REAL GAN-HTR MODEL")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Find available model checkpoints
    model_paths = [
        "dual_modal_gan/checkpoints/full_training_production_v1/best_model",
        "dual_modal_gan/checkpoints/stable_training_enhanced_v2_fixed"
    ]

    working_model = None
    for model_path in model_paths:
        if os.path.exists(model_path):
            working_model = model_path
            break

    if not working_model:
        print("❌ No working model found")
        return False

    print(f"📂 Using model: {working_model}")

    # Find H-DIBCO images
    dataset_dir = Path("dibco_datasets/DIPCO2016_dataset")
    if not dataset_dir.exists():
        print(f"❌ Dataset not found: {dataset_dir}")
        return False

    image_files = sorted(list(dataset_dir.glob("*.bmp")))
    if not image_files:
        print("❌ No images found")
        return False

    # Test with first image
    test_image = image_files[0]
    print(f"🖼️ Testing with: {test_image.name}")
    print(f"   Image size: {cv2.imread(str(test_image), cv2.IMREAD_GRAYSCALE).shape}")

    try:
        # Initialize processor with small patches for testing
        processor = UniversalGANHTRProcessor(
            model_path=working_model,
            patch_size=(256, 32),  # Small patches for faster testing
            overlap=0.2,
            strategy='sliding',
            blend_method='linear',
            batch_size=2  # Small batch size
        )

        # Process image
        output_dir = Path("real_model_test_results")
        output_dir.mkdir(exist_ok=True, parents=True)

        result = processor.process_image(
            input_path=str(test_image),
            output_path=str(output_dir / f"{test_image.stem}_enhanced_real.png")
        )

        if result:
            print(f"\n✅ Real model processing successful!")
            print(f"   Input: {result['input_path']}")
            print(f"   Output: {result['output_path']}")
            print(f"   Comparison: {result['comparison_path']}")
            print(f"   Patches processed: {result['n_patches']}")
            print(f"   PSNR: {result['reconstruction_metrics']['psnr']:.2f} dB")
            if result['reconstruction_metrics']['ssim'] is not None:
                print(f"   SSIM: {result['reconstruction_metrics']['ssim']:.4f}")

            # Save result info
            result_info = {
                'timestamp': datetime.now().isoformat(),
                'model_path': working_model,
                'image_path': str(test_image),
                'result': result
            }

            with open(output_dir / "real_model_test_result.json", 'w') as f:
                json.dump(result_info, f, indent=2)

            return True
        else:
            print("❌ Processing failed")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_real_vs_simulated():
    """Compare real model vs simulated enhancement"""
    print("\n🔄 COMPARING REAL MODEL vs SIMULATED ENHANCEMENT")
    print("=" * 60)

    # Find available model
    model_path = "dual_modal_gan/checkpoints/full_training_production_v1/best_model"
    if not os.path.exists(model_path):
        print("❌ Model not found")
        return False

    dataset_dir = Path("dibco_datasets/DIPCO2016_dataset")
    image_files = sorted(list(dataset_dir.glob("*.bmp")))

    if len(image_files) < 2:
        print("❌ Need at least 2 images for comparison")
        return False

    # Test first 2 images
    test_images = image_files[:2]
    output_dir = Path("comparison_results")
    output_dir.mkdir(exist_ok=True, parents=True)

    comparison_results = []

    for i, image_path in enumerate(test_images):
        print(f"\n📊 Image {i+1}: {image_path.name}")

        try:
            # Real model processing
            print("   🤖 Processing with real model...")
            processor = UniversalGANHTRProcessor(
                model_path=model_path,
                patch_size=(256, 32),
                overlap=0.2,
                strategy='sliding',
                blend_method='linear',
                batch_size=2
            )

            real_result = processor.process_image(
                input_path=str(image_path),
                output_path=str(output_dir / f"{image_path.stem}_real_enhanced.png")
            )

            if real_result:
                print(f"      ✅ Real model PSNR: {real_result['reconstruction_metrics']['psnr']:.2f} dB")

                # Simulated processing for comparison
                print("   🎭 Processing with simulated enhancement...")
                from scripts.eval_hdibco_comprehensive import evaluate_single_image

                config = {
                    'size': (256, 32),
                    'overlap': 0.2,
                    'strategy': 'sliding',
                    'blend_method': 'linear'
                }

                sim_result = evaluate_single_image(image_path, config)

                if sim_result:
                    print(f"      ✅ Simulated PSNR: {sim_result['metrics']['psnr']:.2f} dB")

                    # Save comparison
                    comparison = {
                        'image': image_path.name,
                        'real_model': {
                            'psnr': real_result['reconstruction_metrics']['psnr'],
                            'ssim': real_result['reconstruction_metrics']['ssim'],
                            'n_patches': real_result['n_patches']
                        },
                        'simulated': {
                            'psnr': sim_result['metrics']['psnr'],
                            'ssim': sim_result['metrics']['ssim'],
                            'n_patches': sim_result['n_patches']
                        },
                        'improvement': real_result['reconstruction_metrics']['psnr'] - sim_result['metrics']['psnr']
                    }

                    comparison_results.append(comparison)

                    # Create side-by-side comparison of real vs simulated
                    real_img = cv2.imread(real_result['enhanced_path'], cv2.IMREAD_GRAYSCALE)
                    sim_img = cv2.imread(output_dir / f"{image_path.stem}_simulated_enhanced.png", cv2.IMREAD_GRAYSCALE)

                    if real_img is not None and sim_img is not None:
                        # Resize to same dimensions for comparison
                        h1, w1 = real_img.shape
                        h2, w2 = sim_img.shape

                        if h1 != h2 or w1 != w2:
                            sim_img = cv2.resize(sim_img, (w1, h1))

                        # Create comparison: Original | Real Model | Simulated
                        original_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                        original_img = cv2.resize(original_img, (w1, h1))

                        comparison_img = np.hstack([original_img, real_img, sim_img])
                        comparison_path = output_dir / f"{image_path.stem}_comparison_real_vs_sim.png"
                        cv2.imwrite(str(comparison_path), comparison_img)

                        print(f"      📊 Comparison saved: {comparison_path}")

                else:
                    print("      ❌ Simulated processing failed")
            else:
                print("      ❌ Real model processing failed")

        except Exception as e:
            print(f"   ❌ Error processing {image_path.name}: {e}")

    # Save comparison summary
    if comparison_results:
        avg_improvement = np.mean([r['improvement'] for r in comparison_results])
        print(f"\n📈 COMPARISON SUMMARY:")
        print(f"   Images compared: {len(comparison_results)}")
        print(f"   Average PSNR improvement (Real vs Simulated): {avg_improvement:.2f} dB")

        comparison_summary = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'results': comparison_results,
            'average_improvement': float(avg_improvement)
        }

        with open(output_dir / "real_vs_simulated_comparison.json", 'w') as f:
            json.dump(comparison_summary, f, indent=2)

        return True
    else:
        print("❌ No successful comparisons")
        return False

if __name__ == '__main__':
    print("🧪 REAL GAN-HTR MODEL TESTING")
    print("=" * 50)

    success = False

    # Test 1: Single image with real model
    if test_real_model_on_single_image():
        success = True
        print("\n" + "="*50)

        # Test 2: Comparison with simulated enhancement
        if compare_real_vs_simulated():
            print("\n🎉 All real model tests completed successfully!")
            print("📁 Results saved in: real_model_test_results/ and comparison_results/")
        else:
            print("\n⚠️ Comparison failed, but real model test succeeded")
    else:
        print("\n❌ Real model testing failed")

    sys.exit(0 if success else 1)