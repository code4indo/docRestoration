#!/usr/bin/env python3
"""
Simplified Projection Profile Document Restoration

Uses existing inference pipeline but replaces Laypa with projection profile.
This avoids model loading issues and reuses tested GAN inference code.

Key Changes from Laypa approach:
1. Line Detection: Projection Profile (not Laypa)
2. Coordinates: Horizontal strips (not polyline baselines)
3. Alignment: Perfect by design (horizontal strips)
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import json
import time
from typing import List, Tuple

# Import existing inference pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent))
from inference_pipeline_hybrid_laypa import HybridLaypaInferencePipeline


class ProjectionProfileLineDetector:
    """
    Simple line detector using horizontal projection profile.
    """
    
    def __init__(self):
        self.projection_sigma = 5.0
        self.min_line_height = 30
        self.min_distance = 50
        self.prominence_ratio = 0.15
    
    def detect_lines(self, image: np.ndarray) -> List[dict]:
        """
        Detect text lines using projection profile.
        
        Returns list of dicts dengan format compatible dengan Laypa output:
        {
            'bbox': (x1, y1, x2, y2),
            'baseline_points': [(x1, y), (x2, y)]  # Horizontal baseline
        }
        """
        h, w = image.shape[:2]
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Compute projection
        inverted = 255 - gray
        projection = np.sum(inverted, axis=1, dtype=np.float64)
        
        # Smooth
        projection_smooth = gaussian_filter1d(projection, sigma=self.projection_sigma)
        
        # Find peaks
        max_val = np.max(projection_smooth)
        prominence_threshold = max_val * self.prominence_ratio
        
        peaks, _ = find_peaks(
            projection_smooth,
            distance=self.min_distance,
            prominence=prominence_threshold
        )
        
        if len(peaks) == 0:
            print("⚠️  No lines detected by projection profile!")
            return []
        
        print(f"✅ Detected {len(peaks)} text lines via projection profile")
        
        # Convert peaks to line regions
        lines = []
        threshold_ratio = 0.3
        
        for peak in peaks:
            peak_value = projection_smooth[peak]
            threshold = peak_value * threshold_ratio
            
            # Find boundaries
            top = peak
            while top > 0 and projection_smooth[top] > threshold:
                top -= 1
            
            bottom = peak
            while bottom < len(projection_smooth) - 1 and projection_smooth[bottom] > threshold:
                bottom += 1
            
            # Check minimum height
            line_height = bottom - top
            if line_height < self.min_line_height:
                continue
            
            # Create line dict (compatible format)
            line_dict = {
                'bbox': (0, top, w, bottom),  # Full width
                'baseline_points': [(0, peak), (w, peak)]  # Horizontal baseline at peak
            }
            
            lines.append(line_dict)
        
        # Merge overlapping
        lines = self._merge_overlapping(lines)
        
        print(f"✅ Final lines after merge: {len(lines)}")
        
        return lines
    
    def _merge_overlapping(self, lines: List[dict]) -> List[dict]:
        """Merge overlapping line regions."""
        if not lines:
            return []
        
        # Sort by y-coordinate
        sorted_lines = sorted(lines, key=lambda x: x['bbox'][1])
        
        merged = [sorted_lines[0]]
        
        for current in sorted_lines[1:]:
            last = merged[-1]
            
            # Check overlap
            last_top, last_bottom = last['bbox'][1], last['bbox'][3]
            curr_top, curr_bottom = current['bbox'][1], current['bbox'][3]
            
            if curr_top <= last_bottom:
                # Merge
                new_top = min(last_top, curr_top)
                new_bottom = max(last_bottom, curr_bottom)
                new_peak = (new_top + new_bottom) // 2
                
                merged[-1] = {
                    'bbox': (0, new_top, last['bbox'][2], new_bottom),
                    'baseline_points': [(0, new_peak), (last['bbox'][2], new_peak)]
                }
            else:
                merged.append(current)
        
        return merged


def process_document_projection(image_path: str, output_dir: str,
                               gan_model_dir: str, gpu_id: int = 0):
    """
    Process document using projection profile + existing GAN pipeline.
    
    Args:
        image_path: Input document
        output_dir: Output directory
        gan_model_dir: Directory containing GAN checkpoints
        gpu_id: GPU device ID
    """
    start_time = time.time()
    
    print("=" * 80)
    print(f"📄 PROJECTION PROFILE RESTORATION")
    print("=" * 80)
    print(f"Input: {image_path}")
    print(f"Output: {output_dir}")
    print("")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    h, w = image.shape[:2]
    print(f"📐 Image size: {w}×{h}")
    
    # 1. Detect lines dengan projection profile
    print("")
    print("🔍 Detecting text lines with projection profile...")
    detector = ProjectionProfileLineDetector()
    lines = detector.detect_lines(image)
    
    if not lines:
        print("❌ No lines detected!")
        return False
    
    # 2. Convert to format compatible dengan existing pipeline
    # Format: list of tuples (bbox, baseline_polyline, text_content, reading_order)
    baselines_data = []
    for i, line in enumerate(lines):
        bbox = line['bbox']
        baseline = line['baseline_points']
        
        # Create baseline_info dict (compatible dengan Laypa format)
        baseline_info = {
            'baseline_points': baseline,
            'text_content': '',  # Empty (no recognition yet)
            'reading_order': i
        }
        
        baselines_data.append((bbox, baseline_info))
    
    print(f"✅ Prepared {len(baselines_data)} lines for GAN processing")
    
    # 3. Process dengan existing GAN pipeline
    # NOTE: Kita reuse inference logic tapi skip Laypa detection
    print("")
    print("🎨 Restoring lines with GAN...")
    
    from inference_pipeline_hybrid_laypa import HybridLaypaInferencePipeline
    
    # Initialize pipeline (will load GAN model)
    pipeline = HybridLaypaInferencePipeline(
        model_dir=gan_model_dir,
        gpu_id=gpu_id
    )
    
    # Extract line regions dari detected bboxes
    line_bboxes_original = []
    line_bboxes_padded = []
    
    for bbox, _ in baselines_data:
        x1, y1, x2, y2 = bbox
        
        # Add padding (same as Laypa approach)
        padding = 10
        y1_padded = max(0, y1 - padding)
        y2_padded = min(h, y2 + padding)
        
        line_bboxes_original.append((x1, y1, x2, y2))
        line_bboxes_padded.append((x1, y1_padded, x2, y2_padded))
    
    # Extract line images
    line_images = []
    for x1, y1, x2, y2 in line_bboxes_padded:
        line_img = image[y1:y2, x1:x2]
        line_images.append(line_img)
    
    # Process dengan GAN
    restored_lines = pipeline.process_lines_batch(line_images)
    
    # 4. Reconstruct document (no alignment enforcement needed - already straight!)
    print("")
    print("🔧 Reconstructing document...")
    
    canvas = image.copy()
    
    for i, ((x1, y1, x2, y2), restored_line) in enumerate(zip(line_bboxes_original, restored_lines)):
        line_height = y2 - y1
        line_width = x2 - x1
        
        # Resize restored line to exact original size
        restored_resized = cv2.resize(
            restored_line,
            (line_width, line_height),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Place on canvas at EXACT coordinates (perfect alignment!)
        canvas[y1:y2, x1:x2] = restored_resized
    
    # 5. Save output
    output_path = Path(output_dir) / f"{Path(image_path).stem}_restored.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_path), canvas)
    print(f"💾 Saved: {output_path}")
    
    # 6. Save metadata
    processing_time = time.time() - start_time
    
    metadata = {
        "success": True,
        "document": Path(image_path).stem,
        "method": "projection_profile",
        "num_lines": len(lines),
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
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Document restoration with projection profile line detection"
    )
    parser.add_argument('--input', required=True, help='Input document')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--gan-model-dir', required=True, help='GAN checkpoints directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    
    args = parser.parse_args()
    
    success = process_document_projection(
        image_path=args.input,
        output_dir=args.output_dir,
        gan_model_dir=args.gan_model_dir,
        gpu_id=args.gpu
    )
    
    if success:
        print("✅ Document restoration complete!")
    else:
        print("❌ Restoration failed!")
        exit(1)


if __name__ == "__main__":
    main()
