#!/usr/bin/env python3
"""
Document Restoration using Projection Profile Line Detection

Pure OpenCV approach - No external ML dependencies for line detection.
Uses horizontal projection profile to detect text lines naturally.

Architecture:
    1. Projection Profile → Detect line regions
    2. Extract lines → Full-width strips
    3. GAN Processing → Restore each line
    4. Reconstruction → Place back at exact coordinates

Benefits:
    - Perfect vertical alignment (natural boundaries)
    - No baseline zigzag issues
    - Simple and fast
    - Robust to degradation
"""

import argparse
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import json
import time
from typing import List, Tuple, Optional


class ProjectionProfileRestoration:
    """
    Document-level restoration using projection profile line detection.
    """
    
    def __init__(self, gan_model_path: str, gpu_id: int = 0):
        """
        Initialize restoration system.
        
        Args:
            gan_model_path: Path to GAN generator model
            gpu_id: GPU device ID
        """
        self.gan_model_path = gan_model_path
        self.gpu_id = gpu_id
        
        # Load GAN model
        print(f"🔧 Loading GAN model from: {gan_model_path}")
        self._setup_gpu()
        self.generator = self._load_generator()
        print(f"✅ GAN model loaded successfully")
        
        # Parameters untuk projection profile
        self.projection_sigma = 5.0  # Gaussian smoothing
        self.min_line_height = 30    # Minimum line height (pixels)
        self.min_distance = 50       # Minimum distance between peaks
        self.prominence_ratio = 0.15 # Peak prominence threshold
        
    def _setup_gpu(self):
        """Configure GPU."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus and self.gpu_id < len(gpus):
            try:
                tf.config.set_visible_devices(gpus[self.gpu_id], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[self.gpu_id], True)
                print(f"✅ Using GPU {self.gpu_id}: {gpus[self.gpu_id]}")
            except RuntimeError as e:
                print(f"⚠️  GPU setup warning: {e}")
        else:
            print("⚠️  No GPU available, using CPU")
    
    def _load_generator(self):
        """Load GAN generator model."""
        try:
            # Try loading as full model (.keras)
            if self.gan_model_path.endswith('.keras'):
                generator = tf.keras.models.load_model(
                    self.gan_model_path,
                    compile=False
                )
                return generator
            
            # Load weights (.weights.h5)
            elif self.gan_model_path.endswith('.weights.h5') or self.gan_model_path.endswith('.h5'):
                print("📦 Loading model architecture + weights...")
                
                # Import generator architecture
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
                from models.generator import unet
                
                # Build architecture (NOTE: unet uses (1024, 128, 1) but we use RGB)
                generator = unet(input_size=(1024, 128, 3))
                
                # Load weights
                generator.load_weights(self.gan_model_path)
                print("✅ Weights loaded successfully")
                
                return generator
            else:
                raise ValueError(f"Unsupported model format: {self.gan_model_path}")
                
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def compute_projection_profile(self, image: np.ndarray) -> np.ndarray:
        """
        Compute horizontal projection profile.
        
        Args:
            image: Grayscale image
        
        Returns:
            1D array of projection values (dark pixel count per row)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Invert (text should be dark on light background)
        inverted = 255 - gray
        
        # Sum across width (count dark pixels per row)
        projection = np.sum(inverted, axis=1, dtype=np.float64)
        
        return projection
    
    def smooth_projection(self, projection: np.ndarray) -> np.ndarray:
        """
        Smooth projection profile dengan Gaussian filter.
        
        Args:
            projection: Raw projection profile
        
        Returns:
            Smoothed projection profile
        """
        smoothed = gaussian_filter1d(projection, sigma=self.projection_sigma)
        return smoothed
    
    def detect_text_line_regions(self, projection: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect text line regions dari projection profile.
        
        Args:
            projection: Smoothed projection profile
        
        Returns:
            List of (y_start, y_end) tuples
        """
        # Find peaks (text lines)
        max_val = np.max(projection)
        prominence_threshold = max_val * self.prominence_ratio
        
        peaks, properties = find_peaks(
            projection,
            distance=self.min_distance,
            prominence=prominence_threshold
        )
        
        if len(peaks) == 0:
            print("⚠️  No text lines detected!")
            return []
        
        print(f"📊 Detected {len(peaks)} text line peaks")
        
        # For each peak, find line extent (where projection drops)
        line_regions = []
        threshold_ratio = 0.3  # Drop to 30% of peak value
        
        for peak in peaks:
            peak_value = projection[peak]
            threshold = peak_value * threshold_ratio
            
            # Find top boundary
            top = peak
            while top > 0 and projection[top] > threshold:
                top -= 1
            
            # Find bottom boundary
            bottom = peak
            while bottom < len(projection) - 1 and projection[bottom] > threshold:
                bottom += 1
            
            # Check minimum height
            line_height = bottom - top
            if line_height >= self.min_line_height:
                line_regions.append((top, bottom))
        
        # Merge overlapping regions
        line_regions = self._merge_overlapping_regions(line_regions)
        
        print(f"✅ Final line regions: {len(line_regions)}")
        
        return line_regions
    
    def _merge_overlapping_regions(self, regions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping line regions."""
        if not regions:
            return []
        
        # Sort by y_start
        sorted_regions = sorted(regions, key=lambda x: x[0])
        
        merged = [sorted_regions[0]]
        
        for current in sorted_regions[1:]:
            last = merged[-1]
            
            # Check overlap
            if current[0] <= last[1]:
                # Merge
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        
        return merged
    
    def extract_line_strips(self, image: np.ndarray, 
                           line_regions: List[Tuple[int, int]]) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Extract full-width line strips from document.
        
        Args:
            image: Input document image
            line_regions: List of (y_start, y_end)
        
        Returns:
            List of (line_image, (y_start, y_end))
        """
        h, w = image.shape[:2]
        lines = []
        
        for y_start, y_end in line_regions:
            # Clamp to image bounds
            y_start = max(0, y_start)
            y_end = min(h, y_end)
            
            # Extract full-width strip
            line_strip = image[y_start:y_end, :].copy()
            
            lines.append((line_strip, (y_start, y_end)))
        
        return lines
    
    def preprocess_line_for_gan(self, line: np.ndarray) -> np.ndarray:
        """
        Preprocess line untuk GAN input (1024×128).
        
        Args:
            line: Line image (H×W×3)
        
        Returns:
            Preprocessed line (1024×128×3), normalized [0,1]
        """
        # Resize to GAN input size
        resized = cv2.resize(line, (1024, 128), interpolation=cv2.INTER_CUBIC)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def postprocess_gan_output(self, output: np.ndarray, 
                               target_width: int, target_height: int) -> np.ndarray:
        """
        Postprocess GAN output back to original line size.
        
        Args:
            output: GAN output (1024×128×3), range [0,1]
            target_width: Original line width
            target_height: Original line height
        
        Returns:
            Resized line (target_height×target_width×3), uint8
        """
        # Clip to [0, 1]
        output = np.clip(output, 0.0, 1.0)
        
        # Convert to uint8
        output_uint8 = (output * 255.0).astype(np.uint8)
        
        # Resize back to original dimensions
        resized = cv2.resize(
            output_uint8, 
            (target_width, target_height),
            interpolation=cv2.INTER_CUBIC
        )
        
        return resized
    
    def restore_lines_batch(self, lines: List[Tuple[np.ndarray, Tuple[int, int]]], 
                           batch_size: int = 8) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Restore multiple lines dengan batch processing.
        
        Args:
            lines: List of (line_image, coordinates)
            batch_size: Batch size untuk GAN inference
        
        Returns:
            List of (restored_line_image, coordinates)
        """
        restored_lines = []
        
        num_lines = len(lines)
        num_batches = (num_lines + batch_size - 1) // batch_size
        
        print(f"🔄 Processing {num_lines} lines in {num_batches} batches...")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_lines)
            
            batch_lines = lines[start_idx:end_idx]
            
            # Prepare batch
            batch_inputs = []
            batch_metadata = []
            
            for line_img, coords in batch_lines:
                # Store original dimensions
                orig_h, orig_w = line_img.shape[:2]
                
                # Preprocess
                preprocessed = self.preprocess_line_for_gan(line_img)
                
                batch_inputs.append(preprocessed)
                batch_metadata.append((orig_w, orig_h, coords))
            
            # Stack into batch
            batch_inputs = np.stack(batch_inputs, axis=0)
            
            # GAN inference
            batch_outputs = self.generator(batch_inputs, training=False).numpy()
            
            # Postprocess each output
            for i, output in enumerate(batch_outputs):
                orig_w, orig_h, coords = batch_metadata[i]
                
                # Resize back to original dimensions
                restored = self.postprocess_gan_output(output, orig_w, orig_h)
                
                restored_lines.append((restored, coords))
            
            print(f"  ✅ Batch {batch_idx + 1}/{num_batches} complete")
        
        return restored_lines
    
    def reconstruct_document(self, original_image: np.ndarray,
                            restored_lines: List[Tuple[np.ndarray, Tuple[int, int]]]) -> np.ndarray:
        """
        Reconstruct full document dari restored lines.
        
        Args:
            original_image: Original document image (for background)
            restored_lines: List of (restored_line, (y_start, y_end))
        
        Returns:
            Restored document (same size as original)
        """
        # Start with original image (preserve margins, backgrounds)
        canvas = original_image.copy()
        
        # Place each restored line at exact coordinates
        for restored_line, (y_start, y_end) in restored_lines:
            line_height = y_end - y_start
            
            # Ensure line matches expected height
            if restored_line.shape[0] != line_height:
                print(f"⚠️  Height mismatch: expected {line_height}, got {restored_line.shape[0]}")
                restored_line = cv2.resize(
                    restored_line,
                    (canvas.shape[1], line_height),
                    interpolation=cv2.INTER_CUBIC
                )
            
            # Place on canvas at exact y-coordinates
            canvas[y_start:y_end, :] = restored_line
        
        return canvas
    
    def visualize_projection_profile(self, image: np.ndarray, 
                                     projection: np.ndarray,
                                     line_regions: List[Tuple[int, int]],
                                     output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize projection profile dengan detected regions.
        
        Args:
            image: Original image
            projection: Projection profile
            line_regions: Detected line regions
            output_path: Optional save path
        
        Returns:
            Visualization image
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        
        # Plot image dengan line regions
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Detected Text Line Regions', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Draw line regions
        for y_start, y_end in line_regions:
            ax1.axhline(y=y_start, color='green', linewidth=2, alpha=0.7)
            ax1.axhline(y=y_end, color='red', linewidth=2, alpha=0.7)
            
            # Fill region
            ax1.axhspan(y_start, y_end, alpha=0.1, color='blue')
        
        # Plot projection profile
        y_coords = np.arange(len(projection))
        ax2.plot(projection, y_coords, linewidth=2, color='blue')
        ax2.set_xlabel('Projection Value (Dark Pixel Count)', fontsize=12)
        ax2.set_ylabel('Y Coordinate (pixels)', fontsize=12)
        ax2.set_title('Horizontal Projection Profile', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()  # Match image orientation
        
        # Mark detected regions
        for y_start, y_end in line_regions:
            ax2.axhspan(y_start, y_end, alpha=0.2, color='green')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"📊 Visualization saved: {output_path}")
        
        # Convert to image array
        fig.canvas.draw()
        vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return vis_array
    
    def process_document(self, image_path: str, output_dir: str,
                        save_visualization: bool = True) -> dict:
        """
        Process full document dengan projection profile approach.
        
        Args:
            image_path: Path to input document
            output_dir: Output directory
            save_visualization: Save projection profile visualization
        
        Returns:
            Processing metadata
        """
        start_time = time.time()
        
        print("=" * 80)
        print(f"📄 PROCESSING: {Path(image_path).name}")
        print("=" * 80)
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        doc_name = Path(image_path).stem
        
        # Load image
        print("📖 Loading image...")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        h, w = image.shape[:2]
        print(f"   Size: {w}×{h} pixels")
        
        # 1. Compute projection profile
        print("📊 Computing projection profile...")
        projection = self.compute_projection_profile(image)
        projection_smooth = self.smooth_projection(projection)
        
        # 2. Detect text lines
        print("🔍 Detecting text line regions...")
        line_regions = self.detect_text_line_regions(projection_smooth)
        
        if not line_regions:
            print("❌ No text lines detected!")
            return {"success": False, "error": "No lines detected"}
        
        # 3. Extract line strips
        print(f"✂️  Extracting {len(line_regions)} line strips...")
        lines = self.extract_line_strips(image, line_regions)
        
        # 4. Restore lines dengan GAN
        print("🎨 Restoring lines with GAN...")
        restored_lines = self.restore_lines_batch(lines, batch_size=8)
        
        # 5. Reconstruct document
        print("🔧 Reconstructing document...")
        restored_document = self.reconstruct_document(image, restored_lines)
        
        # 6. Save output
        output_path = output_dir / f"{doc_name}_restored.png"
        cv2.imwrite(str(output_path), restored_document)
        print(f"💾 Saved: {output_path}")
        
        # 7. Save visualization
        if save_visualization:
            vis_path = output_dir / f"{doc_name}_projection_profile.png"
            self.visualize_projection_profile(
                image, projection_smooth, line_regions, str(vis_path)
            )
        
        # 8. Save metadata
        processing_time = time.time() - start_time
        
        metadata = {
            "success": True,
            "document": doc_name,
            "input_size": {"width": w, "height": h},
            "num_lines_detected": len(line_regions),
            "line_regions": [(int(y1), int(y2)) for y1, y2 in line_regions],
            "processing_time_seconds": round(processing_time, 2),
            "output_path": str(output_path)
        }
        
        metadata_path = output_dir / f"{doc_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📊 Metadata saved: {metadata_path}")
        print(f"⏱️  Processing time: {processing_time:.2f}s")
        print("=" * 80)
        print()
        
        return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Document restoration using projection profile line detection"
    )
    parser.add_argument('--input', required=True, help='Input document image')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--gan-model', required=True, help='Path to GAN generator model')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Skip projection profile visualization')
    
    args = parser.parse_args()
    
    # Initialize restoration system
    restoration = ProjectionProfileRestoration(
        gan_model_path=args.gan_model,
        gpu_id=args.gpu
    )
    
    # Process document
    metadata = restoration.process_document(
        image_path=args.input,
        output_dir=args.output_dir,
        save_visualization=not args.no_visualization
    )
    
    if metadata['success']:
        print("✅ Document restoration complete!")
    else:
        print(f"❌ Restoration failed: {metadata.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
