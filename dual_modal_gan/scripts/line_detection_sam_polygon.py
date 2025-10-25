#!/usr/bin/env python3
"""
SAM-based Line Segmentation with Polygon Extraction
===================================================

This module uses Segment Anything Model (SAM) to extract text line regions
as POLYGONS/MASKS instead of simple bounding boxes. This allows capturing
cursive text with ascenders/descenders that extend beyond rectangular bounds.

Key Innovation:
--------------
Traditional: Laypa → Rectangle bbox → Fixed margins
This approach: Laypa → SAM → Polygon mask → Content-aware extraction

Benefits:
--------
✓ Captures irregular text shapes (cursive, ascenders, descenders)
✓ Follows actual text contour instead of rectangular approximation
✓ Eliminates unnecessary background in margins
✓ Preserves text integrity for complex scripts (paleography)

Usage:
------
    refiner = SAMLineSegmenter(checkpoint_path="models/sam/sam_vit_b.pth")
    line_masks, line_polygons = refiner.segment_lines(image, rough_boxes)
    
    # Extract line using mask (preserves shape)
    line_region = image.copy()
    line_region[~mask] = 255  # White background for non-text

Author: AI Assistant + Belekok
Date: 2025-10-23
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor


class SAMLineSegmenter:
    """
    SAM-based line segmenter that extracts text lines as polygon masks
    instead of rectangular bounding boxes.
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
        Initialize SAM line segmenter.
        
        Args:
            checkpoint_path: Path to SAM checkpoint (.pth file)
            model_type: SAM model variant ('vit_b', 'vit_l', 'vit_h')
            device: Device to run SAM ('cuda' or 'cpu')
            confidence_threshold: Minimum mask confidence (0-1)
            min_mask_area: Minimum mask area in pixels
            margin: Safety margin around mask in pixels
        """
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.min_mask_area = min_mask_area
        self.margin = margin
        
        # Initialize SAM
        logging.info(f"Loading SAM model: {model_type}")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        
        # Statistics
        self.stats = {
            'total_boxes': 0,
            'segmented_lines': 0,
            'fallback_boxes': 0,
            'avg_mask_coverage': []
        }
    
    def segment_lines(
        self,
        image: np.ndarray,
        rough_boxes: List[Tuple[int, int, int, int]],
        verbose: bool = True
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Segment text lines using SAM to generate polygon masks.
        
        Args:
            image: Document image (grayscale or RGB)
            rough_boxes: List of (x1, y1, x2, y2) initial bounding boxes
            verbose: Print segmentation statistics
            
        Returns:
            Tuple of (masks, polygons):
                - masks: List of binary masks (HxW arrays)
                - polygons: List of polygon contours (Nx2 arrays)
        """
        if len(rough_boxes) == 0:
            return [], []
        
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        # Set image for SAM
        self.predictor.set_image(image_rgb)
        
        line_masks = []
        line_polygons = []
        mask_coverages = []
        
        for i, bbox in enumerate(rough_boxes):
            x1, y1, x2, y2 = bbox
            bbox_area = (x2 - x1) * (y2 - y1)
            
            try:
                # Prompt SAM with bounding box
                masks, scores, _ = self.predictor.predict(
                    box=np.array([x1, y1, x2, y2]),
                    multimask_output=False
                )
                
                # Validate mask quality
                if len(masks) > 0 and scores[0] >= self.confidence_threshold:
                    mask = masks[0]
                    mask_area = np.sum(mask)
                    
                    if mask_area >= self.min_mask_area:
                        # Expand mask with margin
                        kernel = np.ones((self.margin*2+1, self.margin*2+1), np.uint8)
                        mask_expanded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
                        
                        # Extract polygon contour from mask
                        contours, _ = cv2.findContours(
                            mask_expanded.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        
                        if len(contours) > 0:
                            # Get largest contour (main text region)
                            polygon = max(contours, key=cv2.contourArea)
                            polygon = polygon.squeeze()
                            
                            # Ensure polygon is 2D
                            if len(polygon.shape) == 1:
                                polygon = polygon.reshape(-1, 2)
                            
                            line_masks.append(mask_expanded.astype(bool))
                            line_polygons.append(polygon)
                            
                            # Statistics
                            coverage = (mask_area / bbox_area) * 100
                            mask_coverages.append(coverage)
                            self.stats['segmented_lines'] += 1
                            
                            if verbose and i < 3:
                                logging.info(
                                    f"  Line {i+1}: Segmented (score={scores[0]:.3f}, "
                                    f"coverage={coverage:.1f}%, points={len(polygon)})"
                                )
                        else:
                            # No contour found, fallback
                            self._add_fallback(image, bbox, line_masks, line_polygons)
                    else:
                        # Mask too small
                        if verbose and i < 3:
                            logging.debug(f"  Line {i+1}: Mask too small ({mask_area}px)")
                        self._add_fallback(image, bbox, line_masks, line_polygons)
                else:
                    # Low confidence
                    if verbose and i < 3:
                        logging.debug(
                            f"  Line {i+1}: Low confidence ({scores[0]:.3f} < {self.confidence_threshold})"
                        )
                    self._add_fallback(image, bbox, line_masks, line_polygons)
                    
            except Exception as e:
                logging.warning(f"  Line {i+1}: SAM failed ({e})")
                self._add_fallback(image, bbox, line_masks, line_polygons)
        
        # Update statistics
        self.stats['total_boxes'] += len(rough_boxes)
        if mask_coverages:
            self.stats['avg_mask_coverage'].extend(mask_coverages)
        
        # Print summary
        if verbose:
            segmentation_rate = (self.stats['segmented_lines'] / len(rough_boxes)) * 100
            avg_coverage = np.mean(mask_coverages) if mask_coverages else 0
            logging.info(
                f"  ✓ SAM segmentation: {self.stats['segmented_lines']}/{len(rough_boxes)} lines "
                f"({segmentation_rate:.1f}%), avg coverage: {avg_coverage:.1f}%"
            )
        
        return line_masks, line_polygons
    
    def _add_fallback(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        line_masks: List[np.ndarray],
        line_polygons: List[np.ndarray]
    ):
        """Add rectangular fallback when SAM segmentation fails."""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Create rectangular mask
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = True
        
        # Create rectangular polygon
        polygon = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ])
        
        line_masks.append(mask)
        line_polygons.append(polygon)
        self.stats['fallback_boxes'] += 1
    
    def extract_line_with_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        polygon: np.ndarray,
        background_color: int = 255,
        output_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Extract text line using polygon mask (preserves irregular shape).
        
        Args:
            image: Original document image
            mask: Binary mask of text region
            polygon: Polygon contour points (Nx2 array)
            background_color: Background color for non-text pixels (default: white)
            output_size: Optional (width, height) for resize
            
        Returns:
            Extracted line image with masked background
        """
        # Get bounding box of mask
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return np.ones((128, 1024), dtype=np.uint8) * background_color
        
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()
        
        # Crop to bounding box
        line_crop = image[y_min:y_max+1, x_min:x_max+1].copy()
        mask_crop = mask[y_min:y_max+1, x_min:x_max+1]
        
        # Apply mask (set non-text pixels to background)
        if len(line_crop.shape) == 3:
            # RGB image
            line_crop[~mask_crop] = background_color
        else:
            # Grayscale
            line_crop[~mask_crop] = background_color
        
        # Resize if requested
        if output_size is not None:
            line_crop = cv2.resize(line_crop, output_size, interpolation=cv2.INTER_LANCZOS4)
        
        return line_crop
    
    def extract_for_gan_processing(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        polygon: np.ndarray,
        target_size: Tuple[int, int] = (1024, 128),
        background_color: int = 255
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Extract line for GAN processing with mask preservation.
        
        This method prepares a text line for GAN restoration while preserving
        the irregular polygon shape information. The mask is resized along with
        the image so it can be re-applied after restoration to maintain the
        original contour.
        
        Args:
            image: Original document image (grayscale)
            mask: Binary mask of text region (same size as image)
            polygon: Polygon contour points (Nx2 array)
            target_size: GAN input size (width, height), default (1024, 128)
            background_color: Background for non-text pixels (default: white)
            
        Returns:
            Tuple of (processed_line, resized_mask, metadata):
                - processed_line: Image ready for GAN (target_size, masked)
                - resized_mask: Mask resized to target_size
                - metadata: Dict with bbox, original_size, etc.
        """
        # Get bounding box of mask
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            # Empty mask, return white image
            empty_img = np.ones(target_size[::-1], dtype=np.uint8) * background_color
            empty_mask = np.zeros(target_size[::-1], dtype=bool)
            return empty_img, empty_mask, {'empty': True}
        
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()
        
        # Crop to bounding box
        line_crop = image[y_min:y_max+1, x_min:x_max+1].copy()
        mask_crop = mask[y_min:y_max+1, x_min:x_max+1]
        
        # Store metadata for reconstruction
        metadata = {
            'bbox': (x_min, y_min, x_max, y_max),
            'original_size': (x_max - x_min + 1, y_max - y_min + 1),
            'crop_size': line_crop.shape[:2][::-1],  # (width, height)
            'polygon': polygon,
            'empty': False
        }
        
        # Apply mask to crop (set non-text to background)
        line_crop[~mask_crop] = background_color
        
        # Resize to target size (for GAN input)
        target_w, target_h = target_size
        line_resized = cv2.resize(line_crop, (target_w, target_h), 
                                  interpolation=cv2.INTER_LANCZOS4)
        
        # Also resize mask (for later application)
        mask_resized = cv2.resize(mask_crop.astype(np.uint8), (target_w, target_h),
                                 interpolation=cv2.INTER_NEAREST) > 0.5
        
        return line_resized, mask_resized, metadata
    
    def reconstruct_from_gan_output(
        self,
        restored_line: np.ndarray,
        resized_mask: np.ndarray,
        metadata: Dict,
        background_color: int = 255
    ) -> np.ndarray:
        """
        Reconstruct line from GAN output, preserving original polygon shape.
        
        This method takes the GAN-restored line and:
        1. Applies the mask to preserve polygon boundary
        2. Resizes back to original crop size
        3. Maintains irregular text contour
        
        Args:
            restored_line: GAN output (1024×128)
            resized_mask: Mask at same size as restored_line
            metadata: Metadata from extract_for_gan_processing
            background_color: Background for non-text pixels
            
        Returns:
            Restored line at original crop size with polygon mask applied
        """
        if metadata.get('empty', False):
            # Return empty image at original size if available
            if 'crop_size' in metadata:
                w, h = metadata['crop_size']
                return np.ones((h, w), dtype=np.uint8) * background_color
            else:
                return np.ones((128, 1024), dtype=np.uint8) * background_color
        
        # Apply mask to restored line (preserve polygon boundary)
        restored_masked = restored_line.copy()
        restored_masked[~resized_mask] = background_color
        
        # Resize back to original crop size
        orig_w, orig_h = metadata['crop_size']
        restored_original_size = cv2.resize(
            restored_masked,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LANCZOS4
        )
        
        # Re-apply mask at original size (ensure sharp boundaries)
        bbox = metadata['bbox']
        x_min, y_min, x_max, y_max = bbox
        original_mask_crop = metadata.get('original_mask_crop')
        
        if original_mask_crop is not None:
            restored_original_size[~original_mask_crop] = background_color
        
        return restored_original_size
    
    def visualize_segmentation(
        self,
        image: np.ndarray,
        rough_boxes: List[Tuple[int, int, int, int]],
        masks: List[np.ndarray],
        polygons: List[np.ndarray],
        output_path: str
    ):
        """
        Visualize SAM segmentation results (polygons vs rectangles).
        
        Args:
            image: Original document image
            rough_boxes: Original rectangular boxes
            masks: SAM-generated masks
            polygons: Extracted polygon contours
            output_path: Path to save visualization
        """
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        # Draw rough boxes (red)
        for box in rough_boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw SAM polygons (green)
        for polygon in polygons:
            if len(polygon) > 0:
                cv2.polylines(vis, [polygon.astype(np.int32)], True, (0, 255, 0), 2)
        
        # Add legend
        cv2.putText(vis, "Red: Laypa bbox", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(vis, "Green: SAM polygon", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, vis)
        logging.info(f"Saved segmentation visualization: {output_path}")
    
    def get_statistics(self) -> Dict:
        """Get segmentation statistics."""
        stats = self.stats.copy()
        if stats['avg_mask_coverage']:
            stats['avg_mask_coverage'] = np.mean(stats['avg_mask_coverage'])
        else:
            stats['avg_mask_coverage'] = 0
        
        if stats['total_boxes'] > 0:
            stats['segmentation_rate'] = stats['segmented_lines'] / stats['total_boxes']
            stats['fallback_rate'] = stats['fallback_boxes'] / stats['total_boxes']
        else:
            stats['segmentation_rate'] = 0
            stats['fallback_rate'] = 0
        
        return stats
    
    def reset_statistics(self):
        """Reset segmentation statistics."""
        self.stats = {
            'total_boxes': 0,
            'segmented_lines': 0,
            'fallback_boxes': 0,
            'avg_mask_coverage': []
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample image
    test_image = np.random.randint(0, 255, (1000, 1500), dtype=np.uint8)
    test_boxes = [(100, 100, 500, 150), (100, 200, 600, 250)]
    
    segmenter = SAMLineSegmenter(
        checkpoint_path="models/sam/sam_vit_b_01ec64.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    masks, polygons = segmenter.segment_lines(test_image, test_boxes)
    
    print(f"Segmented {len(masks)} lines")
    print(f"Statistics: {segmenter.get_statistics()}")
