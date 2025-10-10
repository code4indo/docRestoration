#!/usr/bin/env python3
"""
Document Restoration using Sliding Window Approach

PURE RESTORATION - No line detection needed!

Strategy:
    1. Slide fixed window (1024×128) across document
    2. Process each window with GAN
    3. Blend overlapping regions with Gaussian weights
    4. Reconstruct full document

Benefits:
    - No dependency on line detection
    - Works on ANY document (text, non-text, mixed)
    - Perfect alignment (fixed grid)
    - Simple and robust
"""

import argparse
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import time
from typing import Tuple


class SlidingWindowRestoration:
    """
    Document restoration using sliding window approach.
    Pure restoration - no line detection required.
    """
    
    def __init__(self, gan_model_dir: str, gpu_id: int = 0):
        """
        Initialize restoration system.
        
        Args:
            gan_model_dir: Directory containing GAN model
            gpu_id: GPU device ID
        """
        self.gan_model_dir = gan_model_dir
        self.gpu_id = gpu_id
        
        # Window parameters (GAN input size)
        self.window_height = 128
        self.window_width = 1024
        
        # Stride (overlap)
        self.stride_y = 64   # 50% vertical overlap
        self.stride_x = 512  # 50% horizontal overlap
        
        # Load GAN
        print(f"🔧 Loading GAN model from: {gan_model_dir}")
        self._setup_gpu()
        self.generator = self._load_generator()
        print(f"✅ GAN model loaded")
    
    def _setup_gpu(self):
        """Configure GPU."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus and self.gpu_id < len(gpus):
            try:
                tf.config.set_visible_devices(gpus[self.gpu_id], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[self.gpu_id], True)
                print(f"✅ Using GPU {self.gpu_id}")
            except RuntimeError as e:
                print(f"⚠️  GPU setup warning: {e}")
    
    def _load_generator(self):
        """Load GAN generator."""
        # Try loading saved model
        model_path = Path(self.gan_model_dir) / "generator.keras"
        if not model_path.exists():
            # Try alternative paths
            alternatives = [
                Path(self.gan_model_dir) / "generator_best.keras",
                Path(self.gan_model_dir) / "generator.h5",
            ]
            for alt in alternatives:
                if alt.exists():
                    model_path = alt
                    break
        
        if not model_path.exists():
            raise FileNotFoundError(f"Generator model not found in: {self.gan_model_dir}")
        
        try:
            generator = tf.keras.models.load_model(str(model_path), compile=False)
            return generator
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def create_gaussian_weight(self, height: int, width: int) -> np.ndarray:
        """
        Create Gaussian weight map for smooth blending.
        
        Args:
            height: Window height
            width: Window width
        
        Returns:
            Weight map (H×W) with values in [0, 1]
        """
        # Create 1D Gaussian for each dimension
        y = np.linspace(-1, 1, height)
        x = np.linspace(-1, 1, width)
        
        # Gaussian function: exp(-x^2 / (2*sigma^2))
        sigma = 0.5
        gauss_y = np.exp(-y**2 / (2 * sigma**2))
        gauss_x = np.exp(-x**2 / (2 * sigma**2))
        
        # 2D Gaussian (outer product)
        weight = np.outer(gauss_y, gauss_x)
        
        return weight
    
    def extract_windows(self, image: np.ndarray) -> Tuple[list, list]:
        """
        Extract sliding windows from image.
        
        Args:
            image: Input image (H×W×C)
        
        Returns:
            (windows, positions) - list of windows and their (y, x) positions
        """
        h, w, c = image.shape
        
        windows = []
        positions = []
        
        # Slide vertically
        y = 0
        while y + self.window_height <= h:
            # Slide horizontally
            x = 0
            while x + self.window_width <= w:
                # Extract window
                window = image[y:y+self.window_height, x:x+self.window_width].copy()
                windows.append(window)
                positions.append((y, x))
                
                x += self.stride_x
                
                # Handle right edge
                if x + self.window_width > w and x < w:
                    x = w - self.window_width
            
            y += self.stride_y
            
            # Handle bottom edge
            if y + self.window_height > h and y < h:
                y = h - self.window_height
        
        print(f"✅ Extracted {len(windows)} windows")
        print(f"   Grid: ~{len(set([p[0] for p in positions]))} rows × "
              f"~{len(set([p[1] for p in positions]))} cols")
        
        return windows, positions
    
    def process_windows_batch(self, windows: list, batch_size: int = 8) -> list:
        """
        Process windows with GAN in batches.
        
        Args:
            windows: List of window images
            batch_size: Batch size
        
        Returns:
            List of restored windows
        """
        num_windows = len(windows)
        num_batches = (num_windows + batch_size - 1) // batch_size
        
        print(f"🎨 Processing {num_windows} windows in {num_batches} batches...")
        
        restored_windows = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_windows)
            
            batch = windows[start_idx:end_idx]
            
            # Normalize to [0, 1]
            batch_normalized = np.array([w.astype(np.float32) / 255.0 for w in batch])
            
            # GAN inference
            batch_restored = self.generator(batch_normalized, training=False).numpy()
            
            # Denormalize to [0, 255]
            batch_restored = np.clip(batch_restored * 255.0, 0, 255).astype(np.uint8)
            
            restored_windows.extend(batch_restored)
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f"  ✅ Batch {batch_idx + 1}/{num_batches} complete")
        
        return restored_windows
    
    def reconstruct_document(self, image_shape: Tuple[int, int, int],
                            restored_windows: list,
                            positions: list) -> np.ndarray:
        """
        Reconstruct full document from restored windows with weighted blending.
        
        Args:
            image_shape: Original image shape (H, W, C)
            restored_windows: List of restored window images
            positions: List of (y, x) positions
        
        Returns:
            Reconstructed document
        """
        h, w, c = image_shape
        
        # Create accumulation arrays
        canvas = np.zeros((h, w, c), dtype=np.float64)
        weight_map = np.zeros((h, w, c), dtype=np.float64)
        
        # Create Gaussian weight
        gaussian_weight = self.create_gaussian_weight(self.window_height, self.window_width)
        gaussian_weight_3d = gaussian_weight[:, :, np.newaxis]  # Broadcast to RGB
        
        print(f"🔧 Reconstructing document with weighted blending...")
        
        # Accumulate weighted windows
        for window, (y, x) in zip(restored_windows, positions):
            # Convert to float
            window_float = window.astype(np.float64)
            
            # Apply Gaussian weight
            weighted_window = window_float * gaussian_weight_3d
            
            # Accumulate
            canvas[y:y+self.window_height, x:x+self.window_width] += weighted_window
            weight_map[y:y+self.window_height, x:x+self.window_width] += gaussian_weight_3d
        
        # Normalize by weight
        # Avoid division by zero
        weight_map[weight_map == 0] = 1.0
        reconstructed = canvas / weight_map
        
        # Clip and convert to uint8
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        
        return reconstructed
    
    def process_document(self, image_path: str, output_dir: str) -> dict:
        """
        Process full document with sliding window restoration.
        
        Args:
            image_path: Path to input document
            output_dir: Output directory
        
        Returns:
            Processing metadata
        """
        start_time = time.time()
        
        print("=" * 80)
        print(f"📄 SLIDING WINDOW RESTORATION")
        print("=" * 80)
        print(f"Input: {image_path}")
        print(f"Output: {output_dir}")
        print("")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        h, w, c = image.shape
        print(f"📐 Image size: {w}×{h}×{c}")
        print(f"🔲 Window size: {self.window_width}×{self.window_height}")
        print(f"📏 Stride: {self.stride_x}×{self.stride_y} (overlap: 50%)")
        print("")
        
        # 1. Extract windows
        print("✂️  Extracting sliding windows...")
        windows, positions = self.extract_windows(image)
        print("")
        
        # 2. Process with GAN
        restored_windows = self.process_windows_batch(windows, batch_size=8)
        print("")
        
        # 3. Reconstruct document
        restored_document = self.reconstruct_document(
            image.shape, restored_windows, positions
        )
        print("✅ Reconstruction complete")
        print("")
        
        # 4. Save output
        output_path = Path(output_dir) / f"{Path(image_path).stem}_restored.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_path), restored_document)
        print(f"💾 Saved: {output_path}")
        
        # 5. Save metadata
        processing_time = time.time() - start_time
        
        metadata = {
            "success": True,
            "document": Path(image_path).stem,
            "method": "sliding_window",
            "input_size": {"width": w, "height": h},
            "window_size": {"width": self.window_width, "height": self.window_height},
            "stride": {"x": self.stride_x, "y": self.stride_y},
            "num_windows": len(windows),
            "processing_time_seconds": round(processing_time, 2),
            "output_path": str(output_path)
        }
        
        metadata_path = output_path.parent / f"{Path(image_path).stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📊 Metadata saved: {metadata_path}")
        print(f"⏱️  Processing time: {processing_time:.2f}s")
        print("=" * 80)
        print()
        
        return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Document restoration using sliding window (no line detection)"
    )
    parser.add_argument('--input', required=True, help='Input document image')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--gan-model-dir', required=True, help='GAN model directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    
    args = parser.parse_args()
    
    # Initialize restoration system
    restoration = SlidingWindowRestoration(
        gan_model_dir=args.gan_model_dir,
        gpu_id=args.gpu
    )
    
    # Process document
    metadata = restoration.process_document(
        image_path=args.input,
        output_dir=args.output_dir
    )
    
    if metadata['success']:
        print("✅ Document restoration complete!")
    else:
        print("❌ Restoration failed!")


if __name__ == "__main__":
    main()
