"""
Grid-based Registration System untuk Perfect Document Alignment

Konsep:
1. Tambahkan grid reference lines sebelum processing
2. Track grid distortion selama restoration
3. Warp restored lines untuk match original grid
4. Result: Zero zigzag, perfect alignment

Author: AI/ML Engineer
Date: 2025-10-07
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict
from scipy.interpolate import griddata


class GridRegistrationSystem:
    """
    System untuk memastikan alignment perfect menggunakan reference grid.
    
    Workflow:
    1. Add invisible grid markers to original document
    2. Process each line while preserving grid markers
    3. Detect grid distortion after restoration
    4. Apply inverse warp to restore perfect alignment
    """
    
    def __init__(self, grid_spacing: int = 100):
        """
        Args:
            grid_spacing: Spacing between grid lines in pixels
        """
        self.grid_spacing = grid_spacing
        self.grid_points = []
        self.grid_color = (128, 128, 128)  # Gray color for visibility but not intrusive
    
    def add_registration_grid(self, image: np.ndarray, invisible: bool = False) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Add grid lines to image sebagai reference untuk alignment.
        
        Args:
            image: Input image (BGR or grayscale)
            invisible: If True, grid embedded di high-frequency domain (not visible)
        
        Returns:
            image_with_grid: Image dengan grid
            grid_points: List of (x, y) coordinates dari grid intersections
        """
        h, w = image.shape[:2]
        
        if len(image.shape) == 2:
            image_with_grid = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_with_grid = image.copy()
        
        grid_points = []
        
        if invisible:
            # Invisible grid: embed markers di high-frequency (edges only)
            # Use subtle color difference at grid intersections
            for y in range(0, h, self.grid_spacing):
                for x in range(0, w, self.grid_spacing):
                    if 0 <= y < h and 0 <= x < w:
                        # Embed marker: slightly modify pixel value
                        image_with_grid[y, x] = image_with_grid[y, x] * 0.99  # 1% darker
                        grid_points.append((x, y))
        else:
            # Visible grid: draw thin lines (for debugging)
            # Horizontal lines
            for y in range(0, h, self.grid_spacing):
                cv2.line(image_with_grid, (0, y), (w, y), self.grid_color, 1)
            
            # Vertical lines
            for x in range(0, w, self.grid_spacing):
                cv2.line(image_with_grid, (x, 0), (x, h), self.grid_color, 1)
            
            # Grid intersection points
            for y in range(0, h, self.grid_spacing):
                for x in range(0, w, self.grid_spacing):
                    cv2.circle(image_with_grid, (x, y), 3, (255, 0, 0), -1)  # Blue dots
                    grid_points.append((x, y))
        
        self.grid_points = grid_points
        return image_with_grid, grid_points
    
    def detect_grid_distortion(self, restored_image: np.ndarray) -> Dict:
        """
        Detect how much grid has been distorted setelah restoration.
        
        Args:
            restored_image: Restored image dengan grid (potentially distorted)
        
        Returns:
            distortion_map: Dict dengan distortion vectors
        """
        # TODO: Implement grid detection in restored image
        # For now, return identity (no distortion)
        return {'distortion_vectors': [], 'max_displacement': 0}
    
    def apply_inverse_warp(self, restored_line: np.ndarray, bbox: Tuple, distortion_map: Dict) -> np.ndarray:
        """
        Apply inverse warp untuk restore perfect grid alignment.
        
        Args:
            restored_line: Restored line image
            bbox: (x_min, y_min, x_max, y_max)
            distortion_map: Distortion vectors dari detect_grid_distortion
        
        Returns:
            warped_line: Line dengan perfect alignment
        """
        # If no distortion, return as is
        if distortion_map['max_displacement'] == 0:
            return restored_line
        
        # TODO: Implement thin-plate spline warping
        return restored_line


class BaselineAlignmentEnforcer:
    """
    Alternative approach: Enforce horizontal alignment menggunakan baseline constraints.
    
    Ide:
    - Baseline HARUS tetap lurus horizontal
    - Jika ada deviation, warp line untuk straighten
    - Use affine transform atau polynomial warp
    """
    
    def __init__(self):
        self.baseline_tolerance = 2  # pixels
    
    def enforce_horizontal_alignment(
        self, 
        original_baseline: np.ndarray,
        restored_line: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Enforce bahwa baseline tetap lurus horizontal setelah restoration.
        
        Strategy:
        1. Fit line to original baseline points
        2. Check if restored line deviates
        3. Apply corrective warp if needed
        
        Args:
            original_baseline: Original baseline polyline [(x1,y1), (x2,y2), ...]
            restored_line: Restored line image
            bbox: Bounding box (x_min, y_min, x_max, y_max)
        
        Returns:
            aligned_line: Line dengan perfect horizontal alignment
        """
        x_min, y_min, x_max, y_max = bbox
        line_h, line_w = restored_line.shape[:2]
        
        # Fit linear regression ke baseline
        if len(original_baseline) < 2:
            return restored_line  # Not enough points
        
        xs = np.array([p[0] for p in original_baseline])
        ys = np.array([p[1] for p in original_baseline])
        
        # Fit y = ax + b
        if len(xs) > 1:
            coeffs = np.polyfit(xs, ys, deg=1)
            slope = coeffs[0]
            
            # Check deviation from horizontal
            angle_deg = np.degrees(np.arctan(slope))
            
            if abs(angle_deg) < self.baseline_tolerance:
                # Already aligned, no correction needed
                return restored_line
            
            # Apply rotation to straighten
            center = (line_w // 2, line_h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
            
            aligned_line = cv2.warpAffine(
                restored_line, 
                rotation_matrix, 
                (line_w, line_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            return aligned_line
        
        return restored_line


class CoordinateAnchorSystem:
    """
    Sistem anchor points untuk lock coordinates yang critical.
    
    Konsep:
    - Mark specific points yang HARUS preserved (e.g., line start/end)
    - Track anchor displacement during processing
    - Enforce anchor constraints saat reconstruction
    """
    
    def __init__(self):
        self.anchors = {}
    
    def add_anchor(self, line_id: int, point: Tuple[int, int], anchor_type: str = 'start'):
        """
        Add anchor point untuk specific line.
        
        Args:
            line_id: Line index
            point: (x, y) coordinate
            anchor_type: 'start', 'end', 'top', 'bottom'
        """
        if line_id not in self.anchors:
            self.anchors[line_id] = {}
        
        self.anchors[line_id][anchor_type] = point
    
    def compute_anchor_displacement(self, line_id: int, restored_bbox: Tuple) -> Dict:
        """
        Compute displacement dari expected anchor positions.
        
        Returns:
            displacement_map: {anchor_type: (dx, dy)}
        """
        if line_id not in self.anchors:
            return {}
        
        x_min, y_min, x_max, y_max = restored_bbox
        displacement = {}
        
        # Check start anchor
        if 'start' in self.anchors[line_id]:
            expected_x, expected_y = self.anchors[line_id]['start']
            actual_x, actual_y = x_min, y_min
            displacement['start'] = (actual_x - expected_x, actual_y - expected_y)
        
        # Check end anchor
        if 'end' in self.anchors[line_id]:
            expected_x, expected_y = self.anchors[line_id]['end']
            actual_x, actual_y = x_max, y_max
            displacement['end'] = (actual_x - expected_x, actual_y - expected_y)
        
        return displacement
    
    def apply_anchor_correction(self, canvas: np.ndarray, line_image: np.ndarray, 
                                line_id: int, bbox: Tuple) -> Tuple[np.ndarray, Tuple]:
        """
        Apply correction untuk enforce anchor constraints.
        
        Returns:
            corrected_canvas: Canvas dengan line placed at corrected position
            corrected_bbox: Corrected bounding box
        """
        displacement = self.compute_anchor_displacement(line_id, bbox)
        
        if not displacement:
            return canvas, bbox
        
        # Average displacement across anchors
        avg_dx = np.mean([d[0] for d in displacement.values()])
        avg_dy = np.mean([d[1] for d in displacement.values()])
        
        # Correct bbox
        x_min, y_min, x_max, y_max = bbox
        corrected_bbox = (
            int(x_min - avg_dx),
            int(y_min - avg_dy),
            int(x_max - avg_dx),
            int(y_max - avg_dy)
        )
        
        return canvas, corrected_bbox


class PrecisionReconstructionV7:
    """
    V7: Grid-based precision reconstruction dengan multiple alignment strategies.
    
    Improvements:
    1. Grid registration untuk global alignment
    2. Baseline straightness enforcement
    3. Anchor point constraints
    4. Sub-pixel precision placement
    """
    
    def __init__(self, use_grid: bool = True, use_baseline_align: bool = True, use_anchors: bool = True):
        self.use_grid = use_grid
        self.use_baseline_align = use_baseline_align
        self.use_anchors = use_anchors
        
        if use_grid:
            self.grid_system = GridRegistrationSystem(grid_spacing=100)
        
        if use_baseline_align:
            self.baseline_enforcer = BaselineAlignmentEnforcer()
        
        if use_anchors:
            self.anchor_system = CoordinateAnchorSystem()
    
    def reconstruct_with_precision(
        self,
        original_image: np.ndarray,
        line_bboxes: List[Tuple],
        baseline_points: List[List[Tuple]],
        restored_lines: List[np.ndarray]
    ) -> np.ndarray:
        """
        Reconstruct document dengan maximum precision alignment.
        
        Args:
            original_image: Original document
            line_bboxes: List of (x_min, y_min, x_max, y_max)
            baseline_points: List of baseline polylines
            restored_lines: List of restored line images
        
        Returns:
            reconstructed: Perfect aligned document
        """
        h, w = original_image.shape[:2]
        
        # Step 1: Add grid jika enabled (untuk debugging/validation)
        if self.use_grid:
            canvas_with_grid, grid_points = self.grid_system.add_registration_grid(
                np.ones((h, w, 3), dtype=np.uint8) * 255,
                invisible=True  # Grid tidak terlihat di output
            )
            canvas = canvas_with_grid
        else:
            canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Step 2: Register anchors
        if self.use_anchors:
            for i, bbox in enumerate(line_bboxes):
                x_min, y_min, x_max, y_max = bbox
                self.anchor_system.add_anchor(i, (x_min, y_min), 'start')
                self.anchor_system.add_anchor(i, (x_max, y_max), 'end')
        
        # Step 3: Place lines dengan alignment enforcement
        for i, (bbox, baseline, restored_line) in enumerate(
            zip(line_bboxes, baseline_points, restored_lines)
        ):
            x_min, y_min, x_max, y_max = bbox
            line_h = y_max - y_min
            line_w = x_max - x_min
            
            # Resize restored line ke target size
            line_resized = cv2.resize(restored_line, (line_w, line_h), 
                                     interpolation=cv2.INTER_CUBIC)
            
            # Enforce baseline alignment
            if self.use_baseline_align and len(baseline) > 0:
                line_aligned = self.baseline_enforcer.enforce_horizontal_alignment(
                    baseline, line_resized, bbox
                )
            else:
                line_aligned = line_resized
            
            # Convert grayscale to BGR if needed
            if len(line_aligned.shape) == 2:
                line_aligned = cv2.cvtColor(line_aligned, cv2.COLOR_GRAY2BGR)
            
            # Apply anchor correction
            if self.use_anchors:
                canvas, corrected_bbox = self.anchor_system.apply_anchor_correction(
                    canvas, line_aligned, i, bbox
                )
                x_min, y_min, x_max, y_max = corrected_bbox
            
            # Boundary check
            y_min = max(0, min(y_min, h - line_h))
            y_max = min(h, y_min + line_h)
            x_min = max(0, min(x_min, w - line_w))
            x_max = min(w, x_min + line_w)
            
            # Adjust line size jika boundary truncated
            actual_h = y_max - y_min
            actual_w = x_max - x_min
            
            if actual_h != line_h or actual_w != line_w:
                line_aligned = cv2.resize(line_aligned, (actual_w, actual_h),
                                         interpolation=cv2.INTER_CUBIC)
            
            # Paste dengan alpha blending
            alpha = self._create_fade_mask(line_aligned)
            
            canvas_region = canvas[y_min:y_max, x_min:x_max].astype(np.float32)
            line_float = line_aligned.astype(np.float32)
            
            blended = alpha * line_float + (1 - alpha) * canvas_region
            canvas[y_min:y_max, x_min:x_max] = blended.astype(np.uint8)
        
        return canvas
    
    def _create_fade_mask(self, line_image: np.ndarray, fade_pixels: int = 15) -> np.ndarray:
        """Create smooth fade mask untuk blending."""
        h, w = line_image.shape[:2]
        
        # Vertical fade (top & bottom)
        mask = np.ones((h, w), dtype=np.float32)
        
        if h > 2 * fade_pixels:
            # Top fade
            for i in range(fade_pixels):
                mask[i, :] = i / fade_pixels
            
            # Bottom fade
            for i in range(fade_pixels):
                mask[h - 1 - i, :] = i / fade_pixels
        
        # Expand to 3 channels
        if len(line_image.shape) == 3:
            mask = np.stack([mask] * 3, axis=2)
        
        return mask


# Helper function untuk easy integration
def reconstruct_with_grid_alignment(
    original_image: np.ndarray,
    line_bboxes: List[Tuple],
    baseline_points: List[List[Tuple]],
    restored_lines: List[np.ndarray],
    method: str = 'baseline'  # 'grid', 'baseline', 'anchor', 'all'
) -> np.ndarray:
    """
    Convenience function untuk reconstruction dengan alignment enforcement.
    
    Args:
        original_image: Original document
        line_bboxes: Bounding boxes
        baseline_points: Baseline polylines
        restored_lines: Restored line images
        method: Alignment method to use
    
    Returns:
        Reconstructed document dengan perfect alignment
    """
    if method == 'baseline':
        reconstructor = PrecisionReconstructionV7(
            use_grid=False,
            use_baseline_align=True,
            use_anchors=False
        )
    elif method == 'anchor':
        reconstructor = PrecisionReconstructionV7(
            use_grid=False,
            use_baseline_align=False,
            use_anchors=True
        )
    elif method == 'grid':
        reconstructor = PrecisionReconstructionV7(
            use_grid=True,
            use_baseline_align=False,
            use_anchors=False
        )
    else:  # 'all'
        reconstructor = PrecisionReconstructionV7(
            use_grid=True,
            use_baseline_align=True,
            use_anchors=True
        )
    
    return reconstructor.reconstruct_with_precision(
        original_image,
        line_bboxes,
        baseline_points,
        restored_lines
    )
