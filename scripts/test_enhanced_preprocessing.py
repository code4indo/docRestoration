#!/usr/bin/env python3
"""
Test Enhanced Preprocessing with Real GAN-HTR Model
Validating preprocessing consistency and performance improvements
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import logging
import time
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.enhanced_preprocessing import EnhancedGANHTRPreprocessor, DocumentAwarePreprocessor
from scripts.enhanced_patch_inference import EnhancedUniversalGANHTRProcessor

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('enhanced_preprocessing_test.log')
        ]
    )
    return logging.getLogger(__name__)

def load_test_image(image_path: str, logger: logging.Logger) -> np.ndarray:
    """Load test image"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    logger.info(f"📁 Loaded test image: {Path(image_path).name} ({image.shape})")
    return image

def test_basic_preprocessing_consistency(logger: logging.Logger):
    """Test basic preprocessing consistency"""
    logger.info("🧪 Testing Basic Preprocessing Consistency...")

    # Create test image similar to training data (text line)
    test_image = np.random.randint(0, 256, (128, 1024), dtype=np.uint8)

    # Initialize preprocessor
    preprocessor = EnhancedGANHTRPreprocessor(target_size=(128, 1024), logger=logger)

    # Test preprocessing
    processed = preprocessor.preprocess_for_inference(test_image)

    # Validate shape and range
    assert processed.shape == (1024, 128, 1), f"Unexpected shape: {processed.shape}"
    assert processed.min() >= -1.0 and processed.max() <= 1.0, f"Unexpected range: [{processed.min()}, {processed.max()}]"

    # Test postprocessing
    postprocessed = preprocessor.postprocess_output(processed)
    assert postprocessed.shape == (128, 1024), f"Unexpected postprocessed shape: {postprocessed.shape}"
    assert postprocessed.dtype == np.uint8, f"Unexpected dtype: {postprocessed.dtype}"

    logger.info("✅ Basic preprocessing consistency test PASSED")
    return True

def test_document_aware_preprocessing(logger: logging.Logger):
    """Test document-aware preprocessing"""
    logger.info("🧪 Testing Document-Aware Preprocessing...")

    # Load H-DIBCO test image
    test_image_path = "dibco_datasets/DIPCO2016_dataset/1.bmp"
    if not os.path.exists(test_image_path):
        logger.warning(f"⚠️ Test image not found: {test_image_path}")
        return False

    test_image = load_test_image(test_image_path, logger)

    # Initialize document-aware preprocessor
    doc_preprocessor = DocumentAwarePreprocessor(target_size=(128, 1024), logger=logger)

    # Test document preprocessing
    processed_lines = doc_preprocessor.preprocess_document(test_image)

    assert len(processed_lines) > 0, "No text lines extracted"

    # Validate each processed line
    for i, line in enumerate(processed_lines):
        assert line.shape == (1024, 128, 1), f"Line {i} unexpected shape: {line.shape}"
        assert line.min() >= -1.0 and line.max() <= 1.0, f"Line {i} unexpected range"

    # Test reconstruction
    reconstructed = doc_preprocessor.reconstruct_document(processed_lines, test_image.shape)
    assert reconstructed.shape == test_image.shape, f"Reconstruction shape mismatch"

    logger.info(f"✅ Document-aware preprocessing test PASSED")
    logger.info(f"   Extracted {len(processed_lines)} text lines")
    logger.info(f"   Reconstruction shape: {reconstructed.shape}")

    return True

def test_with_real_model(logger: logging.Logger):
    """Test enhanced preprocessing with real GAN-HTR model"""
    logger.info("🧪 Testing with Real GAN-HTR Model...")

    # Check model availability
    model_paths = [
        "dual_modal_gan/outputs/checkpoints_fp32",
        "dual_modal_gan/outputs/checkpoints_full_training_production_v1",
        "dual_modal_gan/checkpoints/full_training_production_v1/best_model"
    ]

    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break

    if not model_path:
        logger.error("❌ No trained model found")
        return False

    logger.info(f"📦 Using model: {model_path}")

    # Check for test images
    test_images = []
    dibco_dir = Path("dibco_datasets/DIPCO2016_dataset")
    if dibco_dir.exists():
        test_images = sorted(list(dibco_dir.glob("*.bmp")))[:3]  # Test first 3 images

    if not test_images:
        logger.error("❌ No test images found")
        return False

    # Initialize enhanced processor
    processor = EnhancedUniversalGANHTRProcessor(
        model_path=model_path,
        domain_aware=True,  # Use document-aware processing
        logger=logger
    )

    results = []

    for i, image_path in enumerate(test_images):
        logger.info(f"🖼️ Testing image {i+1}/3: {image_path.name}")

        try:
            # Process image
            output_path = f"enhanced_preprocessing_test_output_{i+1}.png"
            start_time = time.time()

            result = processor.process_image(str(image_path), output_path)
            processing_time = time.time() - start_time

            results.append(result)

            logger.info(f"   ✅ Processed in {processing_time:.2f}s")
            logger.info(f"   📊 PSNR: {result['reconstruction_metrics']['psnr']:.2f} dB")
            if result['reconstruction_metrics']['ssim'] is not None:
                logger.info(f"   📊 SSIM: {result['reconstruction_metrics']['ssim']:.4f}")
            logger.info(f"   📝 Lines processed: {result['n_lines_processed']}")

        except Exception as e:
            logger.error(f"   ❌ Failed to process {image_path.name}: {e}")
            continue

    # Calculate summary statistics
    if results:
        avg_psnr = np.mean([r['reconstruction_metrics']['psnr'] for r in results])
        avg_ssim = np.mean([r['reconstruction_metrics']['ssim'] for r in results if r['reconstruction_metrics']['ssim'] is not None])
        avg_processing_time = np.mean([r['processing_time'] for r in results])

        logger.info(f"📈 Summary Statistics:")
        logger.info(f"   Average PSNR: {avg_psnr:.2f} dB")
        if avg_ssim is not None:
            logger.info(f"   Average SSIM: {avg_ssim:.4f}")
        logger.info(f"   Average processing time: {avg_processing_time:.2f}s")
        logger.info(f"   Images processed: {len(results)}")

        # Save results
        results_summary = {
            'test_timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'preprocessing_method': 'enhanced_document_aware',
            'results': [
                {
                    'image_name': Path(r['input_path']).name,
                    'psnr': r['reconstruction_metrics']['psnr'],
                    'ssim': r['reconstruction_metrics']['ssim'],
                    'n_lines': r['n_lines_processed'],
                    'processing_time': r['processing_time']
                } for r in results
            ],
            'summary': {
                'avg_psnr': float(avg_psnr),
                'avg_ssim': float(avg_ssim) if avg_ssim is not None else None,
                'avg_processing_time': float(avg_processing_time),
                'total_images': len(results)
            }
        }

        with open('enhanced_preprocessing_test_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)

        logger.info("💾 Results saved to: enhanced_preprocessing_test_results.json")

        # Compare with baseline (19 dB from previous tests)
        baseline_psnr = 19.0
        improvement = avg_psnr - baseline_psnr

        logger.info(f"🎯 Performance Analysis:")
        logger.info(f"   Baseline PSNR: {baseline_psnr:.2f} dB")
        logger.info(f"   Enhanced PSNR: {avg_psnr:.2f} dB")
        logger.info(f"   Improvement: {improvement:+.2f} dB ({improvement/baseline_psnr*100:+.1f}%)")

        if improvement > 5:
            logger.info("🎉 SIGNIFICANT IMPROVEMENT ACHIEVED!")
        elif improvement > 0:
            logger.info("✅ Improvement achieved")
        else:
            logger.warning("⚠️ No improvement or performance degradation")

        return True
    else:
        logger.error("❌ No successful processing results")
        return False

def compare_with_original_inference(logger: logging.Logger):
    """Compare enhanced preprocessing with original inference"""
    logger.info("🔄 Comparing with Original Inference...")

    # Check if original inference results exist
    original_results_file = "hdibco_test_results.json"
    if not os.path.exists(original_results_file):
        logger.warning(f"⚠️ Original results not found: {original_results_file}")
        return False

    try:
        with open(original_results_file, 'r') as f:
            original_results = json.load(f)

        # Load enhanced results
        enhanced_results_file = "enhanced_preprocessing_test_results.json"
        if not os.path.exists(enhanced_results_file):
            logger.warning(f"⚠️ Enhanced results not found: {enhanced_results_file}")
            return False

        with open(enhanced_results_file, 'r') as f:
            enhanced_results = json.load(f)

        # Compare metrics
        original_psnr = original_results.get('average_metrics', {}).get('avg_psnr', 0)
        enhanced_psnr = enhanced_results.get('summary', {}).get('avg_psnr', 0)

        improvement = enhanced_psnr - original_psnr

        logger.info(f"📊 Comparison Results:")
        logger.info(f"   Original PSNR: {original_psnr:.2f} dB")
        logger.info(f"   Enhanced PSNR: {enhanced_psnr:.2f} dB")
        logger.info(f"   Improvement: {improvement:+.2f} dB")

        if improvement > 0:
            logger.info("✅ Enhanced preprocessing OUTPERFORMS original inference")
        elif improvement == 0:
            logger.info("➖ Enhanced preprocessing MATCHES original inference")
        else:
            logger.warning("⚠️ Enhanced preprocessing UNDERPERFORMS original inference")

        return True

    except Exception as e:
        logger.error(f"❌ Error during comparison: {e}")
        return False

def main():
    """Main test function"""
    logger = setup_logging()

    logger.info("🚀 Enhanced Preprocessing Test Suite")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    test_results = {}

    # Test 1: Basic preprocessing consistency
    test_results['basic_consistency'] = test_basic_preprocessing_consistency(logger)

    # Test 2: Document-aware preprocessing
    test_results['document_aware'] = test_document_aware_preprocessing(logger)

    # Test 3: Real model testing
    test_results['real_model'] = test_with_real_model(logger)

    # Test 4: Comparison with original inference
    test_results['comparison'] = compare_with_original_inference(logger)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 TEST SUMMARY")
    logger.info("=" * 60)

    passed_tests = sum(test_results.values())
    total_tests = len(test_results)

    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"   {test_name.replace('_', ' ').title()}: {status}")

    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED - Enhanced preprocessing is working correctly!")
    elif passed_tests > 0:
        logger.info("⚠️ Some tests passed - Enhanced preprocessing partially working")
    else:
        logger.error("❌ ALL TESTS FAILED - Enhanced preprocessing needs fixes")

    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)