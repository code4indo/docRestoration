#!/usr/bin/env python3
"""
Test script to verify Gradio inference pipeline is working
"""
import sys
import os
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

# Setup paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "dual_modal_gan"))

from dual_modal_gan.scripts.inference_portrait_overlap_experiment import (
    load_model, 
    process_portrait_document,
    setup_logging
)

def test_inference():
    print("="*80)
    print("GRADIO INFERENCE TEST")
    print("="*80)
    
    # Create test image (synthetic degraded document)
    print("\n1. Creating synthetic test image...")
    test_img = np.ones((2000, 1500), dtype=np.uint8) * 240
    # Add some "degraded" text-like patterns
    for i in range(10, 1990, 100):
        cv2.rectangle(test_img, (100, i), (1400, i+20), 50, -1)
    print(f"   ✓ Test image created: {test_img.shape}")
    
    # Load model
    print("\n2. Loading model...")
    ckpt_dir = current_dir / "dual_modal_gan/checkpoints/production_full_coverage_vgg_v1"
    ckpt_name = "ckpt-94"
    
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   ✓ GPU detected: {gpus[0].name}")
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print("   ⚠ No GPU, using CPU")
    
    generator = load_model(str(ckpt_dir), ckpt_name, gpu_id=0)
    print(f"   ✓ Model loaded: {type(generator)}")
    
    # Test inference
    print("\n3. Running inference...")
    import logging
    logger = logging.getLogger("TEST")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    
    restored = process_portrait_document(
        test_img,
        generator,
        logger,
        alpha=0.0,
        post_processing=True,
        aggressive=False,
        thin_strokes=False,
        gamma=1.0
    )
    
    print(f"\n4. Inference complete!")
    print(f"   Input shape:  {test_img.shape}")
    print(f"   Output shape: {restored.shape}")
    print(f"   Output dtype: {restored.dtype}")
    print(f"   Output range: [{restored.min()}, {restored.max()}]")
    
    # Save result
    output_path = current_dir / "test_gradio_output.png"
    cv2.imwrite(str(output_path), restored)
    print(f"   ✓ Result saved: {output_path}")
    
    print("\n" + "="*80)
    print("TEST PASSED ✓")
    print("="*80)
    
    return True

if __name__ == "__main__":
    try:
        test_inference()
    except Exception as e:
        print(f"\n❌ TEST FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
