#!/usr/bin/env python3
"""
Quick Validation Test for Enhanced Preprocessing
Focus on testing model loading and single image processing
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import logging
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.enhanced_preprocessing import DocumentAwarePreprocessor
from scripts.enhanced_patch_inference import EnhancedUniversalGANHTRProcessor

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def quick_test_single_image(logger):
    """Quick test with single image to validate model loading and processing"""
    logger.info("🚀 Quick Validation Test")
    logger.info("=" * 50)

    # Check image availability
    test_image_path = "dibco_datasets/DIPCO2016_dataset/1.bmp"
    if not os.path.exists(test_image_path):
        logger.error(f"❌ Test image not found: {test_image_path}")
        return False

    logger.info(f"📁 Test image: {test_image_path}")

    # Check model availability
    model_path = "dual_modal_gan/outputs/checkpoints_fp32"
    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found: {model_path}")
        return False

    logger.info(f"📦 Model: {model_path}")

    try:
        # Initialize processor
        start_time = time.time()
        processor = EnhancedUniversalGANHTRProcessor(
            model_path=model_path,
            domain_aware=True,
            logger=logger
        )
        init_time = time.time() - start_time
        logger.info(f"✅ Processor initialized in {init_time:.2f}s")

        # Process single image
        output_path = "quick_validation_output.png"
        start_time = time.time()

        result = processor.process_image(test_image_path, output_path)

        processing_time = time.time() - start_time

        if result:
            logger.info(f"✅ SUCCESS! Processing completed in {processing_time:.2f}s")
            logger.info(f"📊 Results:")
            logger.info(f"   Original shape: {result['original_shape']}")
            logger.info(f"   Enhanced shape: {result['enhanced_shape']}")
            logger.info(f"   Lines processed: {result['n_lines_processed']}")
            logger.info(f"   Processing method: {result['processing_method']}")

            # Show metrics
            metrics = result['reconstruction_metrics']
            logger.info(f"   PSNR: {metrics['psnr']:.2f} dB")
            if metrics['ssim'] is not None:
                logger.info(f"   SSIM: {metrics['ssim']:.4f}")

            logger.info(f"   Output saved: {output_path}")

            # Determine success based on PSNR
            psnr_value = metrics['psnr']
            if psnr_value >= 25.0:
                logger.info(f"🎉 EXCELLENT! PSNR {psnr_value:.2f} dB exceeds target 25 dB")
                return True
            elif psnr_value >= 20.0:
                logger.info(f"✅ GOOD! PSNR {psnr_value:.2f} dB exceeds baseline 19 dB")
                return True
            elif psnr_value >= 15.0:
                logger.info(f"⚠️ ACCEPTABLE! PSNR {psnr_value:.2f} dB needs improvement")
                return True
            else:
                logger.warning(f"❌ POOR! PSNR {psnr_value:.2f} dB below acceptable range")
                return False
        else:
            logger.error(f"❌ Processing failed")
            return False

    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def compare_with_baseline(logger):
    """Compare results with known baseline"""
    logger.info("\n📈 Comparison with Baseline")
    logger.info("=" * 50)

    # Known results from previous tests
    baseline_psnr = 19.0  # Original inference results
    target_psnr = 25.0    # Target enhancement

    # Check if we have new results
    results_file = "enhanced_preprocessing_test_results.json"
    if os.path.exists(results_file):
        import json
        with open(results_file, 'r') as f:
            results = json.load(f)

        avg_psnr = results.get('summary', {}).get('avg_psnr', 0)

        logger.info(f"   Baseline PSNR: {baseline_psnr:.2f} dB")
        logger.info(f"   Enhanced PSNR: {avg_psnr:.2f} dB")
        logger.info(f"   Target PSNR: {target_psnr:.2f} dB")

        improvement = avg_psnr - baseline_psnr
        logger.info(f"   Improvement: {improvement:+.2f} dB ({improvement/baseline_psnr*100:+.1f}%)")

        if avg_psnr >= target_psnr:
            logger.info("🎉 TARGET ACHIEVED! Enhanced preprocessing meets performance goals")
            return True
        elif improvement > 0:
            logger.info("✅ IMPROVEMENT ACHIEVED! Enhanced preprocessing shows positive results")
            return True
        else:
            logger.warning("⚠️ NO IMPROVEMENT! Enhanced preprocessing needs further optimization")
            return False
    else:
        logger.info("ℹ️ No previous results found for comparison")
        return False

def main():
    """Main function"""
    logger = setup_logging()
    logger.info(f"🚀 Quick Validation Test for Enhanced Preprocessing")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run quick test
    success = quick_test_single_image(logger)

    # Compare with baseline if possible
    if success:
        compare_with_baseline(logger)

    # Final status
    logger.info("\n" + "=" * 50)
    if success:
        logger.info("🎉 QUICK VALIDATION PASSED!")
        logger.info("Enhanced preprocessing is working correctly")
    else:
        logger.info("❌ QUICK VALIDATION FAILED!")
        logger.info("Enhanced preprocessing needs optimization")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)