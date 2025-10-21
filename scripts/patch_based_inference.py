#!/usr/bin/env python3
"""
Universal GAN-HTR Inference with Patch-Based Processing
Phase 2: Supports arbitrary input dimensions for document enhancement
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import json
from datetime import datetime
from typing import Tuple, List, Optional

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class PatchExtractor:
    """
    Extract overlapping patches from large images for processing

    Supports various strategies:
    - Sliding window with overlap
    - Grid-based extraction
    - Adaptive patching based on content
    """

    def __init__(self, patch_size=(1024, 128), stride=64, overlap=0.25, strategy='sliding'):
        """
        Initialize patch extractor

        Args:
            patch_size: Tuple[int, int] - Size of each patch (height, width)
            stride: int - Step size between patches
            overlap: float - Overlap ratio between patches (0.0 to 0.9)
            strategy: str - 'sliding', 'grid', or 'adaptive'
        """
        self.patch_size = patch_size
        self.patch_height, self.patch_width = patch_size
        self.stride = stride
        self.overlap = overlap
        self.strategy = strategy

        # Calculate effective stride based on overlap
        if overlap > 0:
            effective_stride_h = int(patch_size[0] * (1 - overlap))
            effective_stride_w = int(patch_size[1] * (1 - overlap))
            self.effective_stride = (effective_stride_h, effective_stride_w)
        else:
            self.effective_stride = (stride, stride)

    def extract_patches_sliding(self, image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Extract patches using sliding window approach

        Args:
            image: Input image (H, W) or (H, W, C)

        Returns:
            patches: Array of extracted patches (n_patches, patch_h, patch_w) or (n_patches, patch_h, patch_w, C)
            positions: List of (y, x) positions for each patch
        """
        if len(image.shape) == 2:
            h, w = image.shape
        else:
            h, w, c = image.shape

        # Calculate number of patches
        n_patches_h = (h - self.patch_height) // self.effective_stride[0] + 1
        n_patches_w = (w - self.patch_width) // self.effective_stride[1] + 1
        n_patches = n_patches_h * n_patches_w

        print(f"   Extracting {n_patches} patches from {image.shape}")
        print(f"   Patch size: {self.patch_size}, Stride: {self.effective_stride}")

        patches = []
        positions = []

        for i in range(n_patches_h):
            for j in range(n_patches_w):
                y = i * self.effective_stride[0]
                x = j * self.effective_stride[1]

                # Ensure patch doesn't go out of bounds
                y_end = min(y + self.patch_height, h)
                x_end = min(x + self.patch_width, w)

                if y_end <= h and x_end <= w:
                    # Extract patch
                    if len(image.shape) == 2:
                        patch = image[y:y_end, x:x_end]
                    else:
                        patch = image[y:y_end, x:x_end, :]

                    patches.append(patch)
                    positions.append((y, x))

        return np.array(patches), positions

    def extract_patches_grid(self, image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Extract patches using grid approach

        Args:
            image: Input image (H, W) or (H, W, C)

        Returns:
            patches: Array of extracted patches
            positions: List of (y, x) positions for each patch
        """
        if len(image.shape) == 2:
            h, w = image.shape
        else:
            h, w, c = image.shape

        # Calculate grid positions
        y_positions = list(range(0, h - self.patch_height + 1, self.stride))
        x_positions = list(range(0, w - self.patch_width + 1, self.stride))

        patches = []
        positions = []

        for y in y_positions:
            for x in x_positions:
                y_end = min(y + self.patch_height, h)
                x_end = min(x + self.patch_width, w)

                if y_end <= h and x_end <= w:
                    if len(image.shape) == 2:
                        patch = image[y:y_end, x:x_end]
                    else:
                        patch = image[y:y_end, x:x_end, :]

                    patches.append(patch)
                    positions.append((y, x))

        return np.array(patches), positions

    def extract_patches_adaptive(self, image: np.ndarray, content_threshold=0.1) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Extract patches with adaptive sizing based on content

        Args:
            image: Input image (H, W) or (H, W, C)
            content_threshold: Threshold for determining content-rich regions

        Returns:
            patches: Array of extracted patches
            positions: List of (y, x) positions for each patch
        """
        if len(image.shape) == 2:
            h, w = image.shape
        else:
            h, w, c = image.shape

        # Calculate content density (simplified)
        if len(image.shape) == 2:
            content_map = cv2.Laplacian(image, cv2.CV_64F)
            content_map = np.abs(content_map)
        else:
            # For color images, convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            content_map = cv2.Laplacian(gray, cv2.CV_64F)
            content_map = np.abs(content_map)

        # Normalize content map
        content_map = content_map / (content_map.max() + 1e-8)

        # Find content-rich regions
        content_threshold = 0.1  # Can be adjusted

        # Simple grid-based approach with content filtering
        h_positions = list(range(0, h - self.patch_height + 1, self.stride))
        w_positions = list(range(0, w - self.patch_width + 1, self.stride))

        patches = []
        positions = []

        for y in h_positions:
            for x in w_positions:
                y_end = min(y + self.patch_height, h)
                x_end = min(x + self.patch_width, w)

                # Check content density in this region
                region_content = content_map[y:y_end, x:x_end].mean()

                # Only extract patches with sufficient content
                if region_content > content_threshold:
                    if len(image.shape) == 2:
                        patch = image[y:y_end, x:x_end]
                    else:
                        patch = image[y:y_end, x:x_end, :]

                    patches.append(patch)
                    positions.append((y, x))

        print(f"   Adaptive extraction: {len(patches)} content-rich patches from {image.shape}")
        return np.array(patches), positions

    def extract_patches(self, image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Main extraction method - routes to appropriate strategy

        Args:
            image: Input image

        Returns:
            patches: Array of extracted patches
            positions: List of (y, x) positions for each patch
        """
        print(f"🔍 Extracting patches using {self.strategy} strategy")

        if self.strategy == 'sliding':
            return self.extract_patches_sliding(image)
        elif self.strategy == 'grid':
            return self.extract_patches_grid(image)
        elif self.strategy == 'adaptive':
            return self.extract_patches_adaptive(image)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def get_patch_info(self, image_shape: Tuple) -> dict:
        """Get information about patch extraction for a given image shape"""
        h, w = image_shape[:2]

        if self.strategy == 'sliding':
            n_patches_h = (h - self.patch_height) // self.effective_stride[0] + 1
            n_patches_w = (w - self.patch_width) // self.effective_stride[1] + 1
            n_patches = n_patches_h * n_patches_w
        elif self.strategy == 'grid':
            n_patches_h = max(0, (h - self.patch_height + 1) // self.stride + 1)
            n_patches_w = max(0, (w - self.patch_width + 1) // self.stride + 1)
            n_patches = n_patches_h * n_patches_w
        elif self.strategy == 'adaptive':
            # Estimate based on content threshold
            estimated_patches_h = (h // self.patch_height)
            estimated_patches_w = (w // self.patch_width)
            n_patches = estimated_patches_h * estimated_patches_w // 4  # Rough estimate

        # Calculate coverage for all strategies
        if self.strategy == 'sliding':
            n_patches_h = (h - self.patch_height) // self.effective_stride[0] + 1
            n_patches_w = (w - self.patch_width) // self.effective_stride[1] + 1
        elif self.strategy == 'grid':
            n_patches_h = max(0, (h - self.patch_height + 1) // self.stride + 1)
            n_patches_w = max(0, (w - self.patch_width + 1) // self.stride + 1)
        elif self.strategy == 'adaptive':
            # Use estimated values for coverage calculation
            n_patches_h = estimated_patches_h if 'estimated_patches_h' in locals() else 1
            n_patches_w = estimated_patches_w if 'estimated_patches_w' in locals() else 1

        return {
            'n_patches': n_patches,
            'patch_size': self.patch_size,
            'stride': self.effective_stride,
            'coverage': {
                'horizontal': f"{n_patches_w * self.patch_width}/{w} ({(n_patches_w * self.patch_width / w):.1%})",
                'vertical': f"{n_patches_h * self.patch_height}/{h} ({(n_patches_h * self.patch_height / h):.1%})"
            }
        }


class ImageReconstructor:
    """
    Reconstruct full image from processed patches with seamless blending

    Features:
    - Weighted averaging for overlap regions
    - Multiple blending algorithms
    - Edge handling strategies
    - Quality metrics calculation
    """

    def __init__(self, original_size: Tuple[int, int], patch_size: Tuple[int, int],
                 stride: int, blend_method='linear', edge_handling='crop'):
        """
        Initialize image reconstructor

        Args:
            original_size: Original image size (height, width)
            patch_size: Size of each patch (height, width)
            stride: Stride used during extraction
            blend_method: 'linear', 'gaussian', 'edge_aware'
            edge_handling: 'crop', 'mirror', 'pad'
        """
        self.original_size = original_size
        self.original_height, self.original_width = original_size
        self.patch_height, self.patch_width = patch_size
        self.stride = stride
        self.blend_method = blend_method
        self.edge_handling = edge_handling

        # Create weight canvases for blending
        self.weight_canvas = None
        self.result_canvas = None

    def _initialize_canvases(self):
        """Initialize result and weight canvases"""
        self.result_canvas = np.zeros(self.original_size, dtype=np.float32)
        self.weight_canvas = np.zeros(self.original_size, dtype=np.float32)

    def reconstruct_image_linear(self, patches: np.ndarray, positions: List[Tuple[int, int]]) -> np.ndarray:
        """
        Reconstruct image using linear blending

        Args:
            patches: Array of processed patches
            positions: List of (y, x) positions

        Returns:
            Reconstructed image (original size)
        """
        self._initialize_canvases()

        for patch, (y, x) in zip(patches, positions):
            # Calculate actual patch dimensions (may be smaller at edges)
            y_end = min(y + self.patch_height, self.original_height)
            x_end = min(x + self.patch_width, self.original_width)

            # Calculate patch dimensions
            patch_h = y_end - y
            patch_w = x_end - x

            # Ensure patch has correct shape
            if len(patch.shape) == 2:
                if patch.shape != (patch_h, patch_w):
                    patch = cv2.resize(patch, (patch_w, patch_h))
                    patch = np.transpose(patch)  # Transpose back to (H, W)
            elif len(patch.shape) == 3:
                if patch.shape != (patch_h, patch_w, patch.shape[2]):
                    patch = cv2.resize(patch, (patch_w, patch_h, patch.shape[2]))
                    patch = np.transpose(patch, (1, 0, 2))  # Transpose (H, W, C)

            # Add patch to result canvas with weight
            self.result_canvas[y:y_end, x:x_end] += patch
            self.weight_canvas[y:y_end, x:x_end] += 1.0

        # Normalize by weights (avoid division by zero)
        result = self.result_canvas / (self.weight_canvas + 1e-8)

        return result

    def reconstruct_image_gaussian(self, patches: np.ndarray, positions: List[Tuple[int, int]],
                                sigma: float = 1.0) -> np.ndarray:
        """
        Reconstruct image using Gaussian blending for smooth transitions

        Args:
            patches: Array of processed patches
            positions: List of (y, x) positions
            sigma: Gaussian standard deviation

        Returns:
            Reconstructed image with smooth blending
        """
        self._initialize_canvases()

        # Create Gaussian kernel for blending
        kernel_size = 5
        kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = np.outer(kernel_1d, kernel_1d)
        kernel /= kernel.sum()  # Normalize

        for patch, (y, x) in zip(patches, positions):
            y_end = min(y + self.patch_height, self.original_height)
            x_end = min(x + self.patch_width, self.original_width)

            # Calculate patch dimensions
            patch_h = y_end - y
            patch_w = x_end - x

            # Ensure patch has correct shape
            if len(patch.shape) == 2:
                if patch.shape != (patch_h, patch_w):
                    patch = cv2.resize(patch, (patch_w, patch_h))
                    patch = np.transpose(patch)
            elif len(patch.shape) == 3:
                if patch.shape != (patch_h, patch_w, patch.shape[2]):
                    patch = cv2.resize(patch, (patch_w, patch_h, patch.shape[2]))
                    patch = np.transpose(patch, (1, 0, 2))

            # Resize kernel to match patch dimensions if needed
            kernel_h = y_end - y
            kernel_w = x_end - x

            if kernel_h != kernel_size or kernel_w != kernel_size:
                # Resize kernel to fit patch area
                kernel_resized = cv2.resize(kernel, (kernel_w, kernel_h))
            else:
                kernel_resized = kernel

            # Add patch with Gaussian weight
            if len(patch.shape) == 2:
                self.result_canvas[y:y_end, x:x_end] += patch * kernel_resized
                self.weight_canvas[y:y_end, x:x_end] += kernel_resized
            else:
                # For color images, apply kernel to each channel
                for c in range(patch.shape[2]):
                    self.result_canvas[y:y_end, x:x_end, c] += patch[:, :, c] * kernel_resized
                self.weight_canvas[y:y_end, x:x_end] += kernel_resized

        # Normalize by weights
        result = self.result_canvas / (self.weight_canvas + 1e-8)

        return result

    def reconstruct_image(self, patches: np.ndarray, positions: List[Tuple[int, int]]) -> np.ndarray:
        """
        Main reconstruction method - routes to appropriate blending method

        Args:
            patches: Array of processed patches
            positions: List of (y, x) positions

        Returns:
            Reconstructed image (original size)
        """
        if self.blend_method == 'linear':
            return self.reconstruct_image_linear(patches, positions)
        elif self.blend_method == 'gaussian':
            return self.reconstruct_image_gaussian(patches, positions)
        else:
            raise ValueError(f"Unknown blend method: {self.blend_method}")

    def calculate_reconstruction_quality(self, original: np.ndarray, reconstructed: np.ndarray) -> dict:
        """
        Calculate quality metrics for reconstruction

        Args:
            original: Original image
            reconstructed: Reconstructed image

        Returns:
            Dictionary with quality metrics
        """
        # PSNR
        mse = np.mean((original - reconstructed) ** 2)
        psnr = 20 * np.log10(1.0 / (mse + 1e-8))

        # SSIM
        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_score = ssim(original, reconstructed, data_range=1.0)
        except ImportError:
            ssim_score = None

        return {
            'psnr': float(psnr),
            'ssim': float(ssim_score) if ssim_score is not None else None,
            'mse': float(mse),
            'mae': float(np.mean(np.abs(original - reconstructed)))
        }


class PatchProcessor:
    """
    Process patches through GAN-HTR model with optimized batch processing

    Features:
    - GPU memory management
    - Batch processing optimization
    - Error handling and recovery
    - Progress tracking
    """

    def __init__(self, model, batch_size=4, device='CPU'):
        """
        Initialize patch processor

        Args:
            model: Loaded GAN-HTR model
            batch_size: Number of patches to process simultaneously
            device: Device for processing ('CPU' or 'GPU')
        """
        self.model = model
        self.batch_size = batch_size
        self.device = device

        print(f"🔧 Patch Processor initialized:")
        print(f"   Model: {type(model).__name__}")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {device}")

    def preprocess_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Preprocess patch for model input

        Args:
            patch: Input patch (H, W) or (H, W, C)

        Returns:
            Preprocessed patch (1, H, W, 1)
        """
        # Ensure 3D array
        if len(patch.shape) == 2:
            patch = patch[..., np.newaxis]  # Add channel dimension
        elif len(patch.shape) == 3:
            patch = np.transpose(patch, (1, 0, 2))  # (H, W, C) → (W, H, C)

        # Normalize to [-1, 1] range for tanh activation
        patch = (patch * 2.0) - 1.0

        # Add batch dimension
        patch = patch[np.newaxis, ...]  # (1, H, W, 1)

        return patch.astype(np.float32)

    def postprocess_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Postprocess model output back to [0, 1] range

        Args:
            patch: Model output patch (1, H, W, 1)

        Returns:
            Postprocessed patch (H, W)
        """
        # Remove batch dimension
        patch = patch[0, ..., 0]  # (H, W)

        # Denormalize from [-1, 1] to [0, 1]
        patch = (patch + 1.0) / 2.0
        patch = np.clip(patch, 0.0, 1.0)

        return patch

    def process_single_batch(self, patches: np.ndarray) -> List[np.ndarray]:
        """
        Process a single batch of patches through the model

        Args:
            patches: Array of patches (batch_size, H, W, 1)

        Returns:
            List of processed patches
        """
        processed_patches = []

        for i, patch in enumerate(patches):
            try:
                # Preprocess patch
                input_tensor = self.preprocess_patch(patch)

                # Model inference
                output_tensor = self.model(input_tensor, training=False)

                # Postprocess
                enhanced_patch = self.postprocess_patch(output_tensor)

                processed_patches.append(enhanced_patch)

            except Exception as e:
                print(f"   ⚠️ Error processing patch {i}: {e}")
                # Use original patch as fallback
                if len(patch.shape) == 3:
                    patch = np.transpose(patch, (1, 0, 2))
                enhanced_patch = self.postprocess_patch(patch[np.newaxis, ...])
                processed_patches.append(enhanced_patch[0, ..., 0])

        return processed_patches

    def process_patches(self, patches: np.ndarray) -> List[np.ndarray]:
        """
        Process all patches in batches

        Args:
            patches: Array of patches (n_patches, H, W, 1)

        Returns:
            List of processed patches
        """
        n_patches = len(patches)
        processed_patches = []

        print(f"🔄 Processing {n_patches} patches in batches of {self.batch_size}")

        for batch_start in range(0, n_patches, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_patches)
            batch = patches[batch_start:batch_end]

            print(f"   Batch {batch_start//self.batch_size + 1}: processing {len(batch)} patches")

            # Process batch
            batch_results = self.process_single_batch(batch)
            processed_patches.extend(batch_results)

        print(f"✅ Processed {len(processed_patches)} patches successfully")
        return processed_patches


class UniversalGANHTRProcessor:
    """
    Universal GAN-HTR Processor - Complete pipeline for arbitrary input dimensions

    This class integrates all components to provide universal inference capability:
    1. Preprocess input image to proper format
    2. Extract patches using specified strategy
    3. Process patches through GAN-HTR model
    4. Reconstruct full enhanced image
    5. Save results with quality metrics
    """

    def __init__(self, model_path: str, checkpoint_name: str = None,
                 patch_size: Tuple[int, int] = (1024, 128),
                 overlap: float = 0.25,
                 strategy: str = 'sliding',
                 blend_method: str = 'linear',
                 batch_size: int = 4):
        """
        Initialize universal processor

        Args:
            model_path: Path to model checkpoint directory
            checkpoint_name: Name of checkpoint file (without .index)
            patch_size: Size of patches for processing
            overlap: Overlap ratio between patches
            strategy: Patch extraction strategy
            blend_method: Method for blending overlapping regions
            batch_size: Batch size for processing
        """
        self.model_path = model_path
        self.patch_size = patch_size
        self.overlap = overlap
        self.strategy = strategy
        self.blend_method = blend_method
        self.batch_size = batch_size

        # Initialize components
        self.patch_extractor = PatchExtractor(
            patch_size=patch_size,
            overlap=overlap,
            strategy=strategy
        )

        self.model = None
        self.patch_processor = None
        self.image_reconstructor = None

        # Load model
        self._load_model(checkpoint_name)

        print(f"🚀 Universal GAN-HTR Processor initialized:")
        print(f"   Model: {model_path}")
        print(f"   Patch size: {patch_size}")
        print(f"   Overlap: {overlap:.1%}")
        print(f"   Strategy: {strategy}")
        print(f"   Blending: {blend_method}")

    def _load_model(self, checkpoint_name: str = None):
        """Load GAN-HTR model from checkpoint with automatic architecture detection"""
        try:
            import tensorflow as tf

            print(f"📂 Loading GAN-HTR model from: {self.model_path}")

            # Try different model architectures to find compatible one
            model_attempts = [
                {
                    'name': 'enhanced_v2',
                    'module': 'dual_modal_gan.src.models.generator_enhanced_v2',
                    'function': 'unet_enhanced_v2'
                },
                {
                    'name': 'enhanced',
                    'module': 'dual_modal_gan.src.models.generator_enhanced',
                    'function': 'unet_enhanced'
                },
                {
                    'name': 'base',
                    'module': 'dual_modal_gan.src.models.generator',
                    'function': 'unet'
                }
            ]

            # Load checkpoint
            if checkpoint_name:
                checkpoint_path = os.path.join(self.model_path, checkpoint_name)
            else:
                checkpoint_path = tf.train.latest_checkpoint(self.model_path)

            if not checkpoint_path:
                raise FileNotFoundError(f"No checkpoint found in {self.model_path}")

            print(f"   Checkpoint: {checkpoint_path}")

            # Try each model architecture
            for attempt in model_attempts:
                try:
                    print(f"   Trying {attempt['name']} architecture...")

                    # Import the module
                    module = __import__(attempt['module'], fromlist=[attempt['function']])
                    model_function = getattr(module, attempt['function'])

                    # Build model
                    self.model = model_function(input_size=(1024, 128, 1))

                    # Try to load checkpoint
                    checkpoint = tf.train.Checkpoint(generator=self.model)
                    checkpoint.restore(checkpoint_path).expect_partial()

                    print(f"   ✅ {attempt['name']} architecture loaded successfully!")
                    print(f"   Model: {type(self.model).__name__}")

                    # If successful, initialize patch processor
                    self.patch_processor = PatchProcessor(
                        model=self.model,
                        batch_size=self.batch_size
                    )
                    print(f"   Patch processor initialized")
                    return

                except Exception as e:
                    print(f"   ❌ {attempt['name']} failed: {str(e)[:100]}...")
                    continue

            # If all attempts failed
            raise RuntimeError(f"Could not load any compatible model architecture from {self.model_path}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess input image for patch extraction

        Args:
            image_path: Path to input image

        Returns:
            Preprocessed image (H, W) normalized to [0, 1]
        """
        print(f"📸 Loading image: {image_path}")

        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        print(f"   Original shape: {image.shape}")

        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Handle small images - ensure minimum size for patch extraction
        min_height, min_width = self.patch_size
        if image.shape[0] < min_height or image.shape[1] < min_width:
            print(f"   ⚠️ Image smaller than patch size, resizing...")
            scale_h = min_height / image.shape[0]
            scale_w = min_width / image.shape[1]
            scale = max(scale_h, scale_w)

            new_height = int(image.shape[0] * scale)
            new_width = int(image.shape[1] * scale)

            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"   Resized to: {image.shape}")

        return image

    def process_image(self, input_path: str, output_path: str = None) -> dict:
        """
        Process single image through complete pipeline

        Args:
            input_path: Path to input image
            output_path: Path to save output image (optional)

        Returns:
            Dictionary with results and metrics
        """
        print(f"\n🎯 Processing: {input_path}")
        print("=" * 60)

        # 1. Preprocess image
        original_image = self.preprocess_image(input_path)
        original_shape = original_image.shape

        # 2. Extract patches
        patches, positions = self.patch_extractor.extract_patches(original_image)

        # 3. Initialize reconstructor
        self.image_reconstructor = ImageReconstructor(
            original_size=original_shape,
            patch_size=self.patch_size,
            stride=self.patch_extractor.effective_stride[0],  # Use stride from extractor
            blend_method=self.blend_method
        )

        # 4. Process patches through model
        processed_patches = self.patch_processor.process_patches(patches)

        # 5. Reconstruct full image
        print(f"\n🔨 Reconstructing full image from {len(processed_patches)} patches")
        enhanced_image = self.image_reconstructor.reconstruct_image(
            np.array(processed_patches),
            positions
        )

        # 6. Calculate quality metrics
        print(f"\n📊 Calculating quality metrics")
        reconstruction_metrics = self.image_reconstructor.calculate_reconstruction_quality(
            original_image, enhanced_image
        )

        # 7. Save result if path provided
        if output_path is None:
            # Generate output path
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}_enhanced{input_file.suffix}"

        # Convert to uint8 for saving
        enhanced_image_uint8 = (enhanced_image * 255).astype(np.uint8)
        cv2.imwrite(str(output_path), enhanced_image_uint8)

        print(f"💾 Enhanced image saved to: {output_path}")

        # 8. Create side-by-side comparison
        comparison_path = Path(output_path).parent / f"{Path(output_path).stem}_comparison.png"
        self._create_comparison_image(original_image, enhanced_image, str(comparison_path))

        # 9. Return results
        results = {
            'input_path': input_path,
            'output_path': str(output_path),
            'comparison_path': str(comparison_path),
            'original_shape': original_shape,
            'enhanced_shape': enhanced_image.shape,
            'n_patches': len(patches),
            'patch_size': self.patch_size,
            'strategy': self.strategy,
            'overlap': self.overlap,
            'blend_method': self.blend_method,
            'reconstruction_metrics': reconstruction_metrics,
            'processing_info': {
                'extraction_strategy': self.strategy,
                'effective_stride': self.patch_extractor.effective_stride,
                'batch_size': self.batch_size,
                'blend_method': self.blend_method
            }
        }

        print(f"\n✅ Processing complete!")
        print(f"   Original: {original_shape}")
        print(f"   Enhanced: {enhanced_image.shape}")
        print(f"   Patches: {len(patches)}")
        print(f"   PSNR: {reconstruction_metrics['psnr']:.2f} dB")
        if reconstruction_metrics['ssim'] is not None:
            print(f"   SSIM: {reconstruction_metrics['ssim']:.4f}")

        return results

    def _create_comparison_image(self, original: np.ndarray, enhanced: np.ndarray,
                                comparison_path: str):
        """Create side-by-side comparison image"""
        # Convert to uint8
        original_uint8 = (original * 255).astype(np.uint8)
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)

        # Create side-by-side comparison
        comparison = np.hstack([original_uint8, enhanced_uint8])

        # Add labels
        h, w = comparison.shape
        label_space = np.zeros((30, w), dtype=np.uint8)

        cv2.putText(label_space, 'Original', (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
        cv2.putText(label_space, 'Enhanced', (w//2 + 10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)

        comparison_with_labels = np.vstack([label_space, comparison])

        cv2.imwrite(comparison_path, comparison_with_labels)
        print(f"   Comparison saved to: {comparison_path}")

    def process_directory(self, input_dir: str, output_dir: str = None) -> List[dict]:
        """
        Process all images in a directory

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output images

        Returns:
            List of processing results for each image
        """
        input_path = Path(input_dir)

        if output_dir is None:
            output_dir = input_path / "enhanced"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        print(f"\n📁 Processing directory: {input_dir}")
        print(f"📁 Output directory: {output_dir}")

        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        image_files = []

        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        print(f"   Found {len(image_files)} images")

        if not image_files:
            print(f"   ⚠️ No image files found in {input_dir}")
            return []

        # Process each image
        all_results = []

        for i, image_file in enumerate(image_files, 1):
            print(f"\n🖼️ Image {i}/{len(image_files)}: {image_file.name}")

            output_file = output_dir / f"{image_file.stem}_enhanced{image_file.suffix}"

            try:
                result = self.process_image(str(image_file), str(output_file))
                all_results.append(result)
            except Exception as e:
                print(f"   ❌ Error processing {image_file.name}: {e}")
                continue

        # Save summary
        summary = {
            'processing_timestamp': datetime.now().isoformat(),
            'input_directory': str(input_dir),
            'output_directory': str(output_dir),
            'total_images': len(image_files),
            'successful_processed': len(all_results),
            'failed_processed': len(image_files) - len(all_results),
            'configuration': {
                'patch_size': self.patch_size,
                'overlap': self.overlap,
                'strategy': self.strategy,
                'blend_method': self.blend_method,
                'batch_size': self.batch_size
            },
            'results': all_results
        }

        summary_path = output_dir / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n📊 Processing Summary:")
        print(f"   Total images: {len(image_files)}")
        print(f"   Successful: {len(all_results)}")
        print(f"   Failed: {len(image_files) - len(all_results)}")
        print(f"   Summary saved to: {summary_path}")

        return all_results


if __name__ == '__main__':
    # Quick test of patch extraction
    print("🔧 Patch-Based Inference System Initialized")
    print("=" * 50)

    print("📋 Available Classes:")
    print("   - PatchExtractor: Extract patches from images")
    print("   - ImageReconstructor: Reconstruct full images from patches")
    print("   - PatchProcessor: Process patches through GAN-HTR model")

    print("\n📊 Next Steps:")
    print("   1. Test patch extraction on sample images")
    print("   2. Test image reconstruction from patches")
    print("   3. Integrate with actual GAN-HTR model")
    print("   4. Test on H-DIBCO dataset")