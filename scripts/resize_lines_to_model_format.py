#!/usr/bin/env python3
"""
Resize Handwritten Line Images to Model Format (1024×128)
=========================================================

Script untuk resize gambar baris tulisan tangan dengan berbagai ukuran
ke format yang sesuai dengan model: 1024×128 pixels (Width×Height).

Features:
- Support multiple image formats (jpg, jpeg, png, bmp, tif, tiff)
- Preserve aspect ratio dengan padding
- Batch processing untuk multiple files
- Optional: Maintain aspect ratio vs forced resize

Model Expectations:
- Input shape: (1024, 128, 1) dalam format (Width, Height, Channel)
- OpenCV convention: (Height, Width) = (128, 1024)
- Grayscale images

Usage:
    # Resize dengan maintain aspect ratio (recommended)
    python scripts/resize_lines_to_model_format.py \\
        --input_dir DokumenRusak/lines_original \\
        --output_dir DokumenRusak/lines_resized \\
        --mode preserve_aspect
    
    # Resize dengan forced stretch (not recommended, causes distortion)
    python scripts/resize_lines_to_model_format.py \\
        --input_dir DokumenRusak/lines_original \\
        --output_dir DokumenRusak/lines_resized \\
        --mode stretch

Author: AI Assistant
Date: 2025-10-22
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import numpy as np
import cv2
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

# Model expected dimensions
TARGET_WIDTH = 1024   # Model expects width first
TARGET_HEIGHT = 128   # Model expects height second

# Supported image formats
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# Padding color for preserve_aspect mode
PADDING_COLOR = 255  # White padding (typical document background)


# ============================================================================
# Resize Functions
# ============================================================================

def resize_preserve_aspect(image: np.ndarray, 
                          target_width: int = TARGET_WIDTH,
                          target_height: int = TARGET_HEIGHT,
                          padding_color: int = PADDING_COLOR) -> np.ndarray:
    """
    Resize image while preserving aspect ratio, with padding.
    
    This method is RECOMMENDED for handwritten text to avoid distortion.
    
    Args:
        image: Input grayscale image (H, W)
        target_width: Target width (default: 1024)
        target_height: Target height (default: 128)
        padding_color: Color for padding (default: 255 white)
    
    Returns:
        Resized image with shape (target_height, target_width)
    """
    height, width = image.shape[:2]
    
    # Calculate aspect ratios
    aspect_ratio = width / height
    target_aspect = target_width / target_height
    
    # Determine scaling to fit within target dimensions
    if aspect_ratio > target_aspect:
        # Image is wider than target - fit to width
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Image is taller than target - fit to height
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create canvas with padding
    canvas = np.ones((target_height, target_width), dtype=np.uint8) * padding_color
    
    # Calculate padding offsets (center the image)
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    # Place resized image on canvas
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return canvas


def resize_stretch(image: np.ndarray,
                  target_width: int = TARGET_WIDTH,
                  target_height: int = TARGET_HEIGHT) -> np.ndarray:
    """
    Resize image by stretching (may cause distortion).
    
    WARNING: This may distort handwriting and is NOT RECOMMENDED.
    Use only if aspect ratio distortion is acceptable.
    
    Args:
        image: Input grayscale image (H, W)
        target_width: Target width (default: 1024)
        target_height: Target height (default: 128)
    
    Returns:
        Resized image with shape (target_height, target_width)
    """
    # Direct resize without preserving aspect ratio
    resized = cv2.resize(image, (target_width, target_height), 
                        interpolation=cv2.INTER_AREA)
    return resized


def resize_with_border(image: np.ndarray,
                      target_width: int = TARGET_WIDTH,
                      target_height: int = TARGET_HEIGHT,
                      border_size: int = 10,
                      padding_color: int = PADDING_COLOR) -> np.ndarray:
    """
    Resize image with guaranteed border/margin.
    
    Useful for ensuring text doesn't touch edges.
    
    Args:
        image: Input grayscale image (H, W)
        target_width: Target width (default: 1024)
        target_height: Target height (default: 128)
        border_size: Minimum border size in pixels (default: 10)
        padding_color: Color for padding (default: 255 white)
    
    Returns:
        Resized image with shape (target_height, target_width)
    """
    # Calculate effective target size (accounting for borders)
    effective_width = target_width - (2 * border_size)
    effective_height = target_height - (2 * border_size)
    
    # Resize to fit within effective area
    height, width = image.shape[:2]
    aspect_ratio = width / height
    target_aspect = effective_width / effective_height
    
    if aspect_ratio > target_aspect:
        new_width = effective_width
        new_height = int(effective_width / aspect_ratio)
    else:
        new_height = effective_height
        new_width = int(effective_height * aspect_ratio)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create canvas with padding
    canvas = np.ones((target_height, target_width), dtype=np.uint8) * padding_color
    
    # Calculate offsets (center with guaranteed border)
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    # Place resized image on canvas
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return canvas


# ============================================================================
# Image Processing
# ============================================================================

def load_image(image_path: Path) -> Tuple[np.ndarray, bool]:
    """
    Load image and convert to grayscale if needed.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Tuple of (grayscale_image, was_converted)
    """
    # Read image
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert to grayscale if needed
    was_converted = False
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        was_converted = True
    
    return img, was_converted


def process_image(image_path: Path,
                 output_path: Path,
                 mode: str = 'preserve_aspect',
                 border_size: int = 0) -> dict:
    """
    Process single image: load, resize, save.
    
    Args:
        image_path: Input image path
        output_path: Output image path
        mode: Resize mode ('preserve_aspect', 'stretch', 'with_border')
        border_size: Border size for 'with_border' mode
    
    Returns:
        Processing statistics dict
    """
    try:
        # Load image
        original, was_converted = load_image(image_path)
        original_shape = original.shape
        
        # Resize based on mode
        if mode == 'preserve_aspect':
            resized = resize_preserve_aspect(original)
        elif mode == 'stretch':
            resized = resize_stretch(original)
        elif mode == 'with_border':
            resized = resize_with_border(original, border_size=border_size)
        else:
            raise ValueError(f"Unknown resize mode: {mode}")
        
        # Verify output shape
        assert resized.shape == (TARGET_HEIGHT, TARGET_WIDTH), \
            f"Output shape mismatch: {resized.shape} != ({TARGET_HEIGHT}, {TARGET_WIDTH})"
        
        # Save resized image
        cv2.imwrite(str(output_path), resized)
        
        return {
            'status': 'success',
            'original_shape': original_shape,
            'output_shape': resized.shape,
            'was_converted': was_converted
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


# ============================================================================
# Batch Processing
# ============================================================================

def find_images(input_dir: Path) -> List[Path]:
    """
    Find all images in input directory.
    
    Args:
        input_dir: Input directory path
    
    Returns:
        List of image file paths
    """
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(input_dir.glob(f'*{ext}'))
        images.extend(input_dir.glob(f'*{ext.upper()}'))
    
    return sorted(images)


def batch_process(input_dir: Path,
                 output_dir: Path,
                 mode: str = 'preserve_aspect',
                 border_size: int = 0,
                 preserve_names: bool = True) -> dict:
    """
    Batch process all images in directory.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        mode: Resize mode
        border_size: Border size for 'with_border' mode
        preserve_names: Keep original filenames
    
    Returns:
        Processing summary dict
    """
    # Find all images
    image_files = find_images(input_dir)
    
    if not image_files:
        logging.warning(f"No images found in {input_dir}")
        return {'total': 0, 'success': 0, 'errors': 0}
    
    logging.info(f"Found {len(image_files)} images to process")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    results = {
        'total': len(image_files),
        'success': 0,
        'errors': 0,
        'error_files': []
    }
    
    for img_path in tqdm(image_files, desc="Processing images"):
        # Determine output filename
        if preserve_names:
            output_name = img_path.stem + '.png'  # Always save as PNG
        else:
            output_name = f"{img_path.stem}_resized.png"
        
        output_path = output_dir / output_name
        
        # Process image
        result = process_image(img_path, output_path, mode, border_size)
        
        if result['status'] == 'success':
            results['success'] += 1
            logging.debug(f"✓ {img_path.name}: {result['original_shape']} → {result['output_shape']}")
        else:
            results['errors'] += 1
            results['error_files'].append({
                'file': img_path.name,
                'error': result['error']
            })
            logging.error(f"✗ {img_path.name}: {result['error']}")
    
    return results


# ============================================================================
# Visualization
# ============================================================================

def create_comparison_grid(input_dir: Path,
                          output_dir: Path,
                          num_samples: int = 5,
                          save_path: Path = None):
    """
    Create before/after comparison grid for visual inspection.
    
    Args:
        input_dir: Original images directory
        output_dir: Resized images directory
        num_samples: Number of samples to show
        save_path: Path to save comparison image
    """
    try:
        import matplotlib.pyplot as plt
        
        # Find matching pairs
        original_files = find_images(input_dir)[:num_samples]
        
        if not original_files:
            logging.warning("No images found for comparison")
            return
        
        # Create figure
        fig, axes = plt.subplots(num_samples, 2, figsize=(15, 3*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for idx, orig_path in enumerate(original_files):
            # Load original
            orig_img, _ = load_image(orig_path)
            
            # Load resized
            resized_path = output_dir / (orig_path.stem + '.png')
            if resized_path.exists():
                resized_img, _ = load_image(resized_path)
            else:
                logging.warning(f"Resized image not found: {resized_path}")
                continue
            
            # Plot original
            axes[idx, 0].imshow(orig_img, cmap='gray')
            axes[idx, 0].set_title(f'Original: {orig_img.shape}')
            axes[idx, 0].axis('off')
            
            # Plot resized
            axes[idx, 1].imshow(resized_img, cmap='gray')
            axes[idx, 1].set_title(f'Resized: {resized_img.shape}')
            axes[idx, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logging.info(f"Comparison grid saved to {save_path}")
        else:
            plt.savefig(output_dir / 'comparison_grid.png', dpi=150, bbox_inches='tight')
            logging.info(f"Comparison grid saved to {output_dir / 'comparison_grid.png'}")
        
        plt.close()
        
    except ImportError:
        logging.warning("matplotlib not available, skipping comparison grid")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Resize handwritten line images to model format (1024×128)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preserve aspect ratio (recommended)
  python scripts/resize_lines_to_model_format.py \\
      --input_dir DokumenRusak/lines_original \\
      --output_dir DokumenRusak/lines_resized \\
      --mode preserve_aspect
  
  # With guaranteed borders
  python scripts/resize_lines_to_model_format.py \\
      --input_dir DokumenRusak/lines_original \\
      --output_dir DokumenRusak/lines_resized \\
      --mode with_border \\
      --border_size 10
  
  # Stretch mode (not recommended)
  python scripts/resize_lines_to_model_format.py \\
      --input_dir DokumenRusak/lines_original \\
      --output_dir DokumenRusak/lines_resized \\
      --mode stretch
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing line images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for resized images')
    parser.add_argument('--mode', type=str, default='preserve_aspect',
                       choices=['preserve_aspect', 'stretch', 'with_border'],
                       help='Resize mode (default: preserve_aspect)')
    parser.add_argument('--border_size', type=int, default=10,
                       help='Border size for with_border mode (default: 10)')
    parser.add_argument('--create_comparison', action='store_true',
                       help='Create before/after comparison grid')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples in comparison grid (default: 5)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input directory
    if not input_dir.exists():
        logging.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Print configuration
    logging.info("="*70)
    logging.info("Resize Line Images to Model Format")
    logging.info("="*70)
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Target dimensions: {TARGET_WIDTH}×{TARGET_HEIGHT} (W×H)")
    logging.info(f"Resize mode: {args.mode}")
    if args.mode == 'with_border':
        logging.info(f"Border size: {args.border_size} pixels")
    logging.info("")
    
    # Process images
    start_time = datetime.now()
    results = batch_process(
        input_dir=input_dir,
        output_dir=output_dir,
        mode=args.mode,
        border_size=args.border_size
    )
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Print summary
    logging.info("")
    logging.info("="*70)
    logging.info("PROCESSING SUMMARY")
    logging.info("="*70)
    logging.info(f"Total images: {results['total']}")
    logging.info(f"Successfully processed: {results['success']}")
    logging.info(f"Errors: {results['errors']}")
    logging.info(f"Processing time: {elapsed:.2f} seconds")
    logging.info(f"Average time per image: {elapsed/results['total']:.3f} seconds")
    
    if results['errors'] > 0:
        logging.info("")
        logging.info("Error files:")
        for err in results['error_files']:
            logging.info(f"  - {err['file']}: {err['error']}")
    
    # Create comparison grid if requested
    if args.create_comparison:
        logging.info("")
        logging.info("Creating comparison grid...")
        create_comparison_grid(
            input_dir=input_dir,
            output_dir=output_dir,
            num_samples=args.num_samples
        )
    
    logging.info("")
    logging.info(f"✓ All images saved to: {output_dir}")
    logging.info("="*70)


if __name__ == '__main__':
    main()
