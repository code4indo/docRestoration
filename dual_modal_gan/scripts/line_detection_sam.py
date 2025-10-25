#!/usr/bin/env python3
"""
SAM (Segment Anything Model) Line Refiner

Refines Laypa baseline bounding boxes using SAM for pixel-perfect text line segmentation.
Particularly effective for cursive/paleographic scripts with variable ascenders/descenders.

Key Features:
- Content-aware segmentation (no fixed percentage assumptions)
- Captures exact text extent (ascenders, descenders, flourishes)
- Robust to detection errors (±5-10px baseline inaccuracy)
- Optional fallback to rough boxes if SAM fails

Usage:
    from line_detection_sam import SAMLineRefiner
    
    refiner = SAMLineRefiner(checkpoint_path="models/sam/sam_vit_b_01ec64.pth")
    refined_boxes = refiner.refine_boxes(image, rough_boxes)

Author: Belekok
Date: October 23, 2025
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from pathlib import Path
import logging

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logging.warning("segment-anything not installed. SAM refinement will be disabled.")


class SAMLineRefiner:
    """
    Refine rough text line bounding boxes using SAM segmentation.
    
    Converts Laypa's baseline-based boxes (with fixed percentages) into
    precise pixel-level masks that capture actual text extent.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_b",
        device: str = "cuda",
        confidence_threshold: float = 0.8,
        min_mask_area: int = 100,
        margin: int = 5
    ):
        """
        Initialize SAM refiner.
        
        Args:
            checkpoint_path: Path to SAM model weights (.pth file)
            model_type: SAM variant (vit_b, vit_l, vit_h)
            device: Device to run inference on (cuda/cpu)
            confidence_threshold: Minimum mask confidence score (0-1)
            min_mask_area: Minimum mask area in pixels
            margin: Safety margin to add around refined boxes (pixels)
        """
        if not SAM_AVAILABLE:
            raise ImportError(
                "segment-anything not installed. "
                "Install with: pip install segment-anything"
            )
        
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
        
        self.model_type = model_type
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.min_mask_area = min_mask_area
        self.margin = margin
        
        # Initialize SAM
        logging.info(f"Loading SAM model: {model_type}")
        sam = sam_model_registry[model_type](checkpoint=str(self.checkpoint_path))
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        
        # Statistics
        self.stats = {
            'total_boxes': 0,
            'refined_boxes': 0,
            'fallback_boxes': 0,
            'avg_height_reduction': []
        }
    
    def refine_boxes(
        self,
        image: np.ndarray,
        rough_boxes: List[Tuple[int, int, int, int]],
        verbose: bool = True
    ) -> List[Tuple[int, int, int, int]]:
        """
        Refine rough bounding boxes using SAM segmentation.
        
        Args:
            image: Document image (grayscale or RGB)
            rough_boxes: List of (x1, y1, x2, y2) bounding boxes from Laypa
            verbose: Print refinement statistics
            
        Returns:
            List of refined (x1, y1, x2, y2) bounding boxes
        """
        if len(rough_boxes) == 0:
            return []
        
        # Convert grayscale to RGB if needed (SAM expects RGB)
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        # Set image for SAM predictor
        self.predictor.set_image(image_rgb)
        
        refined_boxes = []
        height_reductions = []
        
        for i, bbox in enumerate(rough_boxes):
            x1, y1, x2, y2 = bbox
            orig_height = y2 - y1
            
            try:
                # Prompt SAM with bounding box
                masks, scores, logits = self.predictor.predict(
                    box=np.array([x1, y1, x2, y2]),
                    multimask_output=False
                )
                
                # Check if we got a valid mask
                if len(masks) > 0 and scores[0] >= self.confidence_threshold:
                    mask = masks[0]
                    
                    # Extract tight bounding box from mask
                    y_coords, x_coords = np.where(mask)
                    
                    if len(y_coords) >= self.min_mask_area:
                        # Calculate precise bbox
                        x_min = int(x_coords.min())
                        y_min = int(y_coords.min())
                        x_max = int(x_coords.max())
                        y_max = int(y_coords.max())
                        
                        # Add safety margin
                        image_h, image_w = image.shape[:2]
                        x_min = max(0, x_min - self.margin)
                        y_min = max(0, y_min - self.margin)
                        x_max = min(image_w, x_max + self.margin)
                        y_max = min(image_h, y_max + self.margin)
                        
                        refined_bbox = (x_min, y_min, x_max, y_max)
                        refined_boxes.append(refined_bbox)
                        
                        # Track statistics
                        refined_height = y_max - y_min
                        height_reduction = orig_height - refined_height
                        height_reductions.append(height_reduction)
                        
                        self.stats['refined_boxes'] += 1
                        
                        if verbose and i < 3:  # Show first 3 examples
                            logging.info(
                                f"  Line {i+1}: {orig_height}px → {refined_height}px "
                                f"(saved {height_reduction}px, score={scores[0]:.3f})"
                            )
                    else:
                        # Mask too small, use rough box
                        refined_boxes.append(bbox)
                        self.stats['fallback_boxes'] += 1
                        if verbose:
                            logging.debug(f"  Line {i+1}: Mask too small, using rough box")
                else:
                    # Low confidence, use rough box
                    refined_boxes.append(bbox)
                    self.stats['fallback_boxes'] += 1
                    if verbose:
                        logging.debug(
                            f"  Line {i+1}: Low confidence "
                            f"({scores[0]:.3f} < {self.confidence_threshold}), using rough box"
                        )
                        
            except Exception as e:
                # SAM failed, use rough box
                logging.warning(f"  Line {i+1}: SAM failed ({e}), using rough box")
                refined_boxes.append(bbox)
                self.stats['fallback_boxes'] += 1
        
        # Update statistics
        self.stats['total_boxes'] += len(rough_boxes)
        if height_reductions:
            self.stats['avg_height_reduction'].extend(height_reductions)
        
        # Print summary
        if verbose:
            refinement_rate = (self.stats['refined_boxes'] / len(rough_boxes)) * 100
            avg_reduction = np.mean(height_reductions) if height_reductions else 0
            logging.info(
                f"  ✓ SAM refinement: {self.stats['refined_boxes']}/{len(rough_boxes)} boxes "
                f"({refinement_rate:.1f}%), avg reduction: {avg_reduction:.1f}px"
            )
        
        return refined_boxes
    
    def get_statistics(self) -> dict:
        """Get refinement statistics."""
        stats = self.stats.copy()
        if stats['avg_height_reduction']:
            stats['avg_height_reduction'] = np.mean(stats['avg_height_reduction'])
        else:
            stats['avg_height_reduction'] = 0
        
        if stats['total_boxes'] > 0:
            stats['refinement_rate'] = stats['refined_boxes'] / stats['total_boxes']
            stats['fallback_rate'] = stats['fallback_boxes'] / stats['total_boxes']
        else:
            stats['refinement_rate'] = 0
            stats['fallback_rate'] = 0
        
        return stats
    
    def reset_statistics(self):
        """Reset refinement statistics."""
        self.stats = {
            'total_boxes': 0,
            'refined_boxes': 0,
            'fallback_boxes': 0,
            'avg_height_reduction': []
        }


def visualize_refinement(
    image: np.ndarray,
    rough_boxes: List[Tuple[int, int, int, int]],
    refined_boxes: List[Tuple[int, int, int, int]],
    output_path: str
):
    """
    Visualize refinement comparison (rough vs refined boxes).
    
    Args:
        image: Original image
        rough_boxes: Original Laypa boxes
        refined_boxes: SAM-refined boxes
        output_path: Path to save visualization
    """
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis_image = image.copy()
    
    # Draw rough boxes in red
    for bbox in rough_boxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            vis_image, "Laypa", (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        )
    
    # Draw refined boxes in green
    for bbox in refined_boxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis_image, "SAM", (x2 - 50, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
    
    cv2.imwrite(output_path, vis_image)
    logging.info(f"  ✓ Saved refinement visualization: {output_path}")


# ============================================================================
# CLI Test Interface
# ============================================================================

def main():
    """Test SAM refiner on sample images."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SAM line refiner")
    parser.add_argument('--image', type=str, required=True,
                       help='Input image path')
    parser.add_argument('--checkpoint', type=str,
                       default='models/sam/sam_vit_b_01ec64.pth',
                       help='SAM checkpoint path')
    parser.add_argument('--model_type', type=str, default='vit_b',
                       choices=['vit_b', 'vit_l', 'vit_h'],
                       help='SAM model variant')
    parser.add_argument('--output_dir', type=str, default='/tmp/sam_test',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load image
    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {args.image}")
    
    logging.info(f"Loaded image: {image.shape}")
    
    # Create dummy rough boxes for testing (you would normally get these from Laypa)
    # For testing, use simple horizontal stripes
    h, w = image.shape
    rough_boxes = []
    for i in range(0, h, 100):
        if i + 80 < h:
            rough_boxes.append((0, i, w, i + 80))
    
    logging.info(f"Created {len(rough_boxes)} rough boxes for testing")
    
    # Initialize refiner
    refiner = SAMLineRefiner(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type
    )
    
    # Refine boxes
    refined_boxes = refiner.refine_boxes(image, rough_boxes)
    
    # Print statistics
    stats = refiner.get_statistics()
    logging.info(f"\nRefinement Statistics:")
    logging.info(f"  Total boxes: {stats['total_boxes']}")
    logging.info(f"  Refined: {stats['refined_boxes']} ({stats['refinement_rate']*100:.1f}%)")
    logging.info(f"  Fallback: {stats['fallback_boxes']} ({stats['fallback_rate']*100:.1f}%)")
    logging.info(f"  Avg height reduction: {stats['avg_height_reduction']:.1f}px")
    
    # Visualize
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_refinement(
        image,
        rough_boxes,
        refined_boxes,
        str(output_dir / "sam_refinement_comparison.png")
    )


if __name__ == "__main__":
    main()
