#!/usr/bin/env python3
"""
Line Detection using DeepSeek-OCR
==================================

This module uses DeepSeek-OCR's grounding capability to detect text regions
and extract line-level bounding boxes from historical documents.

Note: DeepSeek-OCR requires PyTorch, which conflicts with TensorFlow.
This script is designed to run in a separate environment or use CPU inference.

Installation (in separate venv):
    pip install transformers einops addict easydict torch pillow
    pip install flash-attn --no-build-isolation  # Optional, for faster inference

Usage:
    from line_detection_deepseek import detect_lines_deepseek
    
    lines = detect_lines_deepseek('document.jpg')
    for line in lines:
        print(f"Line: {line['bbox']}, type: {line['type']}")

Author: AI Assistant + Belekok
Date: 2025-10-22
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2

try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    logging.warning("⚠️  DeepSeek-OCR dependencies not available. Install: transformers, torch, einops, addict, easydict")


class DeepSeekLineDetector:
    """
    Line detector using DeepSeek-OCR grounding capabilities.
    """
    
    def __init__(
        self,
        model_name: str = 'deepseek-ai/DeepSeek-OCR',
        device: str = 'cuda',
        use_flash_attn: bool = False
    ):
        """
        Initialize DeepSeek-OCR model.
        
        Args:
            model_name: HuggingFace model name
            device: 'cuda' or 'cpu'
            use_flash_attn: Use flash attention (requires flash-attn package)
        """
        if not DEEPSEEK_AVAILABLE:
            raise ImportError("DeepSeek-OCR dependencies not installed")
        
        logging.info(f"  Loading DeepSeek-OCR model: {model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # Load model
        attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'
        self.model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation=attn_implementation,
            trust_remote_code=True,
            use_safetensors=True
        )
        
        # Set device and dtype
        self.device = device
        if device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.cuda().to(torch.bfloat16)
            logging.info(f"  ✓ Model loaded on GPU with bfloat16")
        else:
            self.model = self.model.cpu()
            logging.info(f"  ✓ Model loaded on CPU")
        
        self.model = self.model.eval()
        
        logging.info(f"  ✓ DeepSeek-OCR initialized successfully")
    
    def detect_text_regions(
        self,
        image_path: str,
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True
    ) -> str:
        """
        Detect text regions using DeepSeek-OCR grounding.
        
        Args:
            image_path: Path to input image
            base_size: Base resolution (512/640/1024/1280)
            image_size: Image crop size (640/1024)
            crop_mode: Enable dynamic cropping
        
        Returns:
            OCR result with grounding tags
        """
        # Prompt with grounding for bounding box detection
        prompt = "<image>\n<|grounding|>Convert the document to markdown."
        
        logging.info(f"  Running DeepSeek-OCR inference...")
        logging.info(f"    Base size: {base_size}, Image size: {image_size}, Crop mode: {crop_mode}")
        
        # Run inference
        import tempfile
        import sys
        from io import StringIO
        
        # Capture stdout since DeepSeek-OCR prints output instead of returning
        captured_output = StringIO()
        old_stdout = sys.stdout
        
        try:
            sys.stdout = captured_output
            
            with tempfile.TemporaryDirectory() as tmpdir:
                with torch.no_grad():
                    result = self.model.infer(
                        self.tokenizer,
                        prompt=prompt,
                        image_file=image_path,
                        output_path=tmpdir,  # Use temp dir
                        base_size=base_size,
                        image_size=image_size,
                        crop_mode=crop_mode,
                        test_compress=False,
                        save_results=False
                    )
        finally:
            sys.stdout = old_stdout
        
        # Get captured output
        output_text = captured_output.getvalue()
        
        # If model returned something, use that; otherwise use captured output
        if result is not None and str(result).strip():
            return str(result)
        elif output_text.strip():
            logging.info(f"  Captured output from DeepSeek-OCR")
            return output_text
        else:
            logging.warning("  DeepSeek-OCR returned no output")
            return ""
    
    def parse_grounding_output(
        self,
        grounding_text: str,
        image_width: int,
        image_height: int
    ) -> List[Dict]:
        """
        Parse DeepSeek-OCR grounding output to extract bounding boxes.
        
        DeepSeek-OCR output format:
        <|ref|>text<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
        
        Args:
            grounding_text: Raw output from DeepSeek-OCR
            image_width: Image width in pixels
            image_height: Image height in pixels
        
        Returns:
            List of region dictionaries with bbox and type
        """
        # Regex pattern to match grounding tags
        pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
        matches = re.findall(pattern, grounding_text, re.DOTALL)
        
        regions = []
        for label_type, coords_str in matches:
            try:
                # Parse coordinates
                coords_list = eval(coords_str)  # [[x1, y1, x2, y2], ...]
                
                for coords in coords_list:
                    if len(coords) == 4:
                        # DeepSeek uses normalized coordinates (0-999)
                        x1, y1, x2, y2 = coords
                        
                        # Convert to pixel coordinates
                        x1_px = int(x1 / 999 * image_width)
                        y1_px = int(y1 / 999 * image_height)
                        x2_px = int(x2 / 999 * image_width)
                        y2_px = int(y2 / 999 * image_height)
                        
                        regions.append({
                            'type': label_type.strip(),
                            'bbox': (x1_px, y1_px, x2_px, y2_px),
                            'width': x2_px - x1_px,
                            'height': y2_px - y1_px
                        })
            except Exception as e:
                logging.warning(f"    Failed to parse coordinates: {coords_str}, error: {e}")
                continue
        
        logging.info(f"  ✓ Parsed {len(regions)} text regions")
        return regions
    
    def regions_to_lines(
        self,
        regions: List[Dict],
        min_line_height: int = 40,
        max_line_height: int = 200
    ) -> List[Dict]:
        """
        Convert text regions to line-level bounding boxes.
        
        DeepSeek-OCR typically detects paragraph/block level regions.
        This function filters for line-sized regions or splits larger blocks.
        
        Args:
            regions: List of detected regions
            min_line_height: Minimum acceptable line height
            max_line_height: Maximum acceptable line height
        
        Returns:
            List of line-level bounding boxes
        """
        lines = []
        
        for region in regions:
            bbox = region['bbox']
            height = region['height']
            region_type = region['type']
            
            # Filter: Only keep text-like regions
            if region_type.lower() not in ['text', 'title', 'paragraph', 'line']:
                continue
            
            # Case 1: Region is already line-sized
            if min_line_height <= height <= max_line_height:
                lines.append({
                    'bbox': bbox,
                    'height': height,
                    'width': region['width'],
                    'type': region_type,
                    'source': 'direct'
                })
            
            # Case 2: Region is too large (paragraph) - split it
            elif height > max_line_height:
                # Estimate number of lines
                num_lines = max(1, round(height / 80))  # Assume ~80px per line
                line_height = height // num_lines
                
                x1, y1, x2, y2 = bbox
                for i in range(num_lines):
                    line_y1 = y1 + i * line_height
                    line_y2 = min(y2, y1 + (i + 1) * line_height)
                    
                    lines.append({
                        'bbox': (x1, line_y1, x2, line_y2),
                        'height': line_y2 - line_y1,
                        'width': x2 - x1,
                        'type': region_type,
                        'source': 'split'
                    })
        
        logging.info(f"  ✓ Converted {len(regions)} regions to {len(lines)} lines")
        return lines
    
    def detect_lines(
        self,
        image_path: str,
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
        min_line_height: int = 40,
        max_line_height: int = 200
    ) -> List[Dict]:
        """
        Main entry point: Detect lines from document image.
        
        Args:
            image_path: Path to input image
            base_size: DeepSeek base resolution
            image_size: DeepSeek image size
            crop_mode: Enable dynamic cropping
            min_line_height: Minimum line height filter
            max_line_height: Maximum line height filter
        
        Returns:
            List of line dictionaries with bbox
        """
        # Load image to get dimensions
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        height, width = image.shape
        
        # Run DeepSeek-OCR
        grounding_output = self.detect_text_regions(
            image_path, base_size, image_size, crop_mode
        )
        
        # Parse grounding output
        regions = self.parse_grounding_output(grounding_output, width, height)
        
        # Convert regions to lines
        lines = self.regions_to_lines(regions, min_line_height, max_line_height)
        
        # Sort by vertical position
        lines.sort(key=lambda l: l['bbox'][1])
        
        return lines


def detect_lines_deepseek(
    image_path: str,
    model_name: str = 'deepseek-ai/DeepSeek-OCR',
    device: str = 'cuda',
    **kwargs
) -> List[Dict]:
    """
    Convenience function to detect lines using DeepSeek-OCR.
    
    Args:
        image_path: Path to input image
        model_name: HuggingFace model name
        device: 'cuda' or 'cpu'
        **kwargs: Additional arguments passed to detect_lines
    
    Returns:
        List of line dictionaries
    """
    detector = DeepSeekLineDetector(model_name=model_name, device=device)
    return detector.detect_lines(image_path, **kwargs)


def visualize_deepseek_lines(
    image: np.ndarray,
    lines: List[Dict],
    output_path: Path
):
    """
    Create visualization of DeepSeek-detected lines.
    
    Args:
        image: Original document image
        lines: List of line dictionaries
        output_path: Path to save visualization
    """
    # Create RGB version
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis_image = image.copy()
    
    # Draw bounding boxes
    for i, line_info in enumerate(lines):
        x1, y1, x2, y2 = line_info['bbox']
        
        # Color based on source
        if line_info.get('source') == 'split':
            color = (255, 165, 0)  # Orange for split regions
        else:
            color = (0, 255, 0)  # Green for direct detection
        
        # Draw box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw line number
        cv2.putText(vis_image, f"L{i+1}", (x1 + 10, y1 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Draw type
        line_type = line_info.get('type', 'text')
        cv2.putText(vis_image, line_type, (x1 + 10, y2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Add title
    title = f"DeepSeek-OCR Line Detection: {len(lines)} lines"
    cv2.putText(vis_image, title, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    cv2.imwrite(str(output_path), vis_image)
    logging.info(f"    ✓ Saved visualization: {output_path.name}")


# ============================================================================
# Test Script
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test DeepSeek-OCR line detection')
    parser.add_argument('--input', type=str, required=True,
                       help='Input document image')
    parser.add_argument('--output_dir', type=str, default='results/line_detection_deepseek',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--base_size', type=int, default=1024,
                       help='DeepSeek base size (512/640/1024/1280)')
    parser.add_argument('--image_size', type=int, default=640,
                       help='DeepSeek image size (640/1024)')
    parser.add_argument('--crop_mode', action='store_true',
                       help='Enable dynamic cropping')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("="*70)
    print("DeepSeek-OCR Line Detection Test")
    print("="*70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ Error: Could not load image: {args.input}")
        exit(1)
    
    print(f"✓ Loaded image: {image.shape}")
    
    # Detect lines
    print(f"\nDetecting lines with DeepSeek-OCR...")
    try:
        lines = detect_lines_deepseek(
            args.input,
            device=args.device,
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=args.crop_mode
        )
        
        print(f"\n✓ Detected {len(lines)} lines:")
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line['bbox']
            print(f"  Line {i+1}: bbox=({x1:4d}, {y1:4d}, {x2:4d}, {y2:4d}), "
                  f"h={line['height']:3d}, w={line['width']:4d}, "
                  f"type={line['type']}, source={line.get('source', 'N/A')}")
        
        # Visualize
        output_path = output_dir / f"{Path(args.input).stem}_deepseek_lines.png"
        visualize_deepseek_lines(image, lines, output_path)
        
        print(f"\n✓ Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("="*70)
