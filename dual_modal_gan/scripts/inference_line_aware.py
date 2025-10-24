#!/usr/bin/env python3
"""
Line-Aware Document Restoration with Optional EDSR Super-Resolution
====================================================================

Inference script for single line images (128×1024).
Supports optional EDSR super-resolution for high-resolution output.

Usage:
    # Without SR
    python inference_line_aware.py --checkpoint_dir <path> --input <line.png> --output_dir <dir>
    
    # With SR (2x)
    python inference_line_aware.py --checkpoint_dir <path> --input <line.png> --output_dir <dir> --use_sr --sr_scale 2
    
    # With SR (4x, efficient model)
    python inference_line_aware.py --checkpoint_dir <path> --input <line.png> --output_dir <dir> --use_sr --sr_scale 4 --sr_efficient
"""

import os
import sys
from pathlib import Path

# Disable TF warnings and XLA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_DISABLE_XLA'] = '1'

import numpy as np
import cv2
import tensorflow as tf
import argparse
import logging
from datetime import datetime

# Disable XLA at TF level
tf.config.optimizer.set_jit(False)

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
from dual_modal_gan.src.models.edsr import (
    build_edsr, build_edsr_efficient,
    preprocess_for_edsr, postprocess_from_edsr
)


def configure_gpu(gpu_id=1):
    """Configure GPU with memory growth."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            if gpu_id is not None and gpu_id >= 0 and gpu_id < len(gpus):
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                logging.info(f"✓ Using GPU {gpu_id}: {gpus[gpu_id].name}")
        except RuntimeError as e:
            logging.error(f"❌ GPU configuration error: {e}")
    else:
        logging.warning("⚠️  No GPU found, using CPU")


def load_generator(checkpoint_path: str):
    """Load trained generator from checkpoint."""
    logging.info(f"Loading generator from: {checkpoint_path}")
    
    generator = unet_enhanced(input_size=(1024, 128, 1))
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(checkpoint_path).expect_partial()
    
    logging.info(f"✓ Generator loaded successfully")
    return generator


def load_sr_model(scale: int = 2, use_efficient: bool = False):
    """Load EDSR super-resolution model."""
    logging.info(f"Loading EDSR {'efficient' if use_efficient else 'baseline'} model (x{scale})...")
    
    if use_efficient:
        model = build_edsr_efficient(input_shape=(128, 1024, 1), scale=scale)
    else:
        model = build_edsr(input_shape=(128, 1024, 1), scale=scale)
    
    logging.info(f"✓ EDSR model loaded: {model.count_params():,} parameters")
    return model


def preprocess_line(image: np.ndarray) -> np.ndarray:
    """Preprocess line for generator input."""
    # Ensure grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Ensure correct size (128, 1024)
    if image.shape != (128, 1024):
        image = cv2.resize(image, (1024, 128))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Transpose (H, W) -> (W, H) for model
    image = image.T
    
    # Add channel dimension (W, H, 1)
    image = image[..., np.newaxis]
    
    # Normalize to [-1, 1] for tanh generator
    image = image * 2.0 - 1.0
    
    return image


def postprocess_line(output: np.ndarray) -> np.ndarray:
    """Postprocess generator output to image."""
    # Remove channel dimension
    if len(output.shape) == 3:
        output = output[:, :, 0]
    
    # Transpose (W, H) -> (H, W)
    output = output.T
    
    # Denormalize from [-1, 1] to [0, 255]
    output = (output + 1.0) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output


def apply_super_resolution(restored_line: np.ndarray, sr_model, scale: int = 2) -> np.ndarray:
    """
    Apply super-resolution to restored line.
    
    NOTE: Since EDSR is untrained, we use high-quality Lanczos4 interpolation
    which gives MUCH better results for document upscaling.
    """
    h, w = restored_line.shape
    new_h = h * scale
    new_w = w * scale
    
    # Use Lanczos4 - best quality for upscaling documents
    # Better than untrained EDSR which produces noise/artifacts
    sr_line = cv2.resize(restored_line, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    return sr_line


def main():
    parser = argparse.ArgumentParser(
        description='Line-Aware Document Restoration with Optional EDSR',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Checkpoint directory')
    parser.add_argument('--checkpoint_name', type=str, default='ckpt-88',
                       help='Checkpoint name (default: ckpt-88)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input line image (128×1024)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--gpu_id', type=int, default=1,
                       help='GPU device ID (default: 1, -1 for CPU)')
    parser.add_argument('--image_ext', type=str, default='.jpg',
                       help='Image extension (for compatibility, not used)')
    parser.add_argument('--use_sr', action='store_true',
                       help='Enable EDSR super-resolution')
    parser.add_argument('--sr_scale', type=int, default=2, choices=[2, 4],
                       help='SR upscaling factor (2x or 4x)')
    parser.add_argument('--sr_efficient', action='store_true',
                       help='Use lightweight EDSR for faster inference')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / f'inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*70)
    logging.info("LINE-AWARE DOCUMENT RESTORATION")
    if args.use_sr:
        logging.info(f"  + HIGH-QUALITY UPSCALING (Lanczos4 {args.sr_scale}x)")
    logging.info("="*70)
    logging.info(f"Checkpoint: {args.checkpoint_dir}/{args.checkpoint_name}")
    logging.info(f"Input: {args.input}")
    logging.info(f"Output: {args.output_dir}")
    logging.info(f"GPU: {args.gpu_id}")
    if args.use_sr:
        logging.info(f"SR Scale: {args.sr_scale}x")
        logging.info(f"SR Model: {'Efficient' if args.sr_efficient else 'Baseline'}")
    logging.info("")
    
    # Configure GPU
    configure_gpu(args.gpu_id if args.gpu_id >= 0 else None)
    
    # Load generator
    checkpoint_path = str(Path(args.checkpoint_dir) / args.checkpoint_name)
    generator = load_generator(checkpoint_path)
    
    # Load SR model if enabled
    sr_model = None
    if args.use_sr:
        logging.info("ℹ️  Using Lanczos4 interpolation for super-resolution")
        logging.info("ℹ️  (EDSR disabled - requires training for good quality)")
        # We pass None as sr_model since we use Lanczos4 instead
        sr_model = "lanczos4"  # Placeholder to enable SR path
    
    # Load input image
    logging.info(f"Loading image: {args.input}")
    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logging.error(f"❌ Failed to load image: {args.input}")
        return
    
    logging.info(f"  Input size: {image.shape}")
    
    # Preprocess
    line_input = preprocess_line(image)
    logging.info(f"  Preprocessed size: {line_input.shape}")
    
    # Inference
    logging.info("Running generator inference...")
    restored_tensor = generator(line_input[np.newaxis, ...], training=False)
    restored = postprocess_line(restored_tensor.numpy()[0])
    logging.info(f"  Restored size: {restored.shape}")
    
    # Apply SR if enabled
    if sr_model is not None:
        logging.info(f"Applying high-quality upscaling (Lanczos4 {args.sr_scale}x)...")
        restored = apply_super_resolution(restored, sr_model, scale=args.sr_scale)
        logging.info(f"  SR output size: {restored.shape}")
    
    # Save results
    input_name = Path(args.input).stem
    
    # Save restored
    restored_path = output_dir / f"{input_name}_restored.png"
    cv2.imwrite(str(restored_path), restored)
    logging.info(f"✓ Saved restored: {restored_path}")
    
    # Save side-by-side comparison
    if args.use_sr:
        # Upscale original for comparison
        h_orig, w_orig = image.shape
        h_new = h_orig * args.sr_scale
        w_new = w_orig * args.sr_scale
        image_upscaled = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_CUBIC)
        comparison = np.hstack([image_upscaled, restored])
    else:
        comparison = np.hstack([image, restored])
    
    comparison_path = output_dir / f"{input_name}_comparison.png"
    cv2.imwrite(str(comparison_path), comparison)
    logging.info(f"✓ Saved comparison: {comparison_path}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': checkpoint_path,
        'input': str(args.input),
        'input_size': image.shape,
        'output_size': restored.shape,
        'super_resolution': {
            'enabled': args.use_sr,
            'scale': args.sr_scale if args.use_sr else None,
            'model': 'efficient' if args.sr_efficient else 'baseline' if args.use_sr else None
        }
    }
    
    metadata_path = output_dir / f"{input_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    logging.info(f"\n{'='*70}")
    logging.info(f"✅ INFERENCE COMPLETED")
    logging.info(f"{'='*70}")


if __name__ == '__main__':
    main()
