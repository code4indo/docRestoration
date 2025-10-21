#!/usr/bin/env python3
"""
Enhanced Patch-Based GAN-HTR Inference System
Menggunakan preprocessing yang konsisten dengan training pipeline
"""

import numpy as np
import cv2
import tensorflow as tf
from typing import Tuple, List, Optional, Dict, Any
import logging
from pathlib import Path
import time
import json

# Import enhanced preprocessing
from scripts.enhanced_preprocessing import EnhancedGANHTRPreprocessor, DocumentAwarePreprocessor, create_enhanced_preprocessor

class EnhancedPatchProcessor:
    """
    Enhanced patch processor with training-consistent preprocessing
    """

    def __init__(self, model_path: str, batch_size: int = 4,
                 target_size: Tuple[int, int] = (128, 1024),
                 logger: Optional[logging.Logger] = None):
        """
        Initialize enhanced patch processor

        Args:
            model_path: Path to trained model
            batch_size: Batch size for processing
            target_size: Target size for preprocessing (height, width)
            logger: Optional logger instance
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.target_height, self.target_width = target_size
        self.logger = logger or logging.getLogger(__name__)

        # Initialize enhanced preprocessor
        self.preprocessor = EnhancedGANHTRPreprocessor(target_size, logger)

        # Load model
        self.model = self._load_model()

        self.logger.info(f"🚀 Enhanced Patch Processor initialized:")
        self.logger.info(f"   Model: {model_path}")
        self.logger.info(f"   Target size: {target_size}")
        self.logger.info(f"   Batch size: {batch_size}")

    def _load_model(self) -> tf.keras.Model:
        """Load trained GAN-HTR model with automatic architecture detection"""
        try:
            self.logger.info(f"📂 Loading GAN-HTR model from: {self.model_path}")

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
            checkpoint_path = tf.train.latest_checkpoint(self.model_path)
            if not checkpoint_path:
                raise FileNotFoundError(f"No checkpoint found in {self.model_path}")

            self.logger.info(f"   Checkpoint: {checkpoint_path}")

            # Try each model architecture
            for attempt in model_attempts:
                try:
                    self.logger.info(f"   Trying {attempt['name']} architecture...")

                    # Import the module
                    module = __import__(attempt['module'], fromlist=[attempt['function']])
                    model_function = getattr(module, attempt['function'])

                    # Build model
                    model = model_function(input_size=(1024, 128, 1))

                    # Try to load checkpoint using the working method
                    checkpoint = tf.train.Checkpoint(generator=model)
                    checkpoint.restore(checkpoint_path).expect_partial()

                    self.logger.info(f"   ✅ {attempt['name']} architecture loaded successfully!")
                    self.logger.info(f"   Model: {type(model).__name__}")
                    return model

                except Exception as e:
                    self.logger.warning(f"   ❌ {attempt['name']} failed: {str(e)[:100]}...")
                    continue

            # If all attempts failed
            raise RuntimeError(f"Could not load any compatible model architecture from {self.model_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            # As fallback, create dummy model for testing
            self.logger.warning("⚠️ Creating dummy model for testing purposes")
            from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
            model = unet_enhanced(input_size=(1024, 128, 1))
            return model

    def preprocess_patches(self, patches: np.ndarray) -> np.ndarray:
        """
        Preprocess patches using training-consistent pipeline

        Args:
            patches: Array of patches, shape (n_patches, height, width, channels)

        Returns:
            Preprocessed patches ready for model input
        """
        processed_patches = []

        for i, patch in enumerate(patches):
            try:
                # Remove batch dimension for individual processing
                if len(patch.shape) == 4:
                    patch = np.squeeze(patch, axis=0)

                # Apply enhanced preprocessing
                processed_patch = self.preprocessor.preprocess_for_inference(patch)
                processed_patches.append(processed_patch)

            except Exception as e:
                self.logger.warning(f"⚠️ Failed to preprocess patch {i}: {e}")
                # Create dummy patch
                dummy_patch = np.zeros((self.target_width, self.target_height, 1), dtype=np.float32)
                processed_patches.append(dummy_patch)

        return np.array(processed_patches)

    def process_patches_batch(self, patches: np.ndarray) -> np.ndarray:
        """
        Process patches in batches with training-consistent preprocessing

        Args:
            patches: Array of preprocessed patches

        Returns:
            Array of enhanced patches
        """
        n_patches = len(patches)
        enhanced_patches = []

        # Process in batches
        for i in range(0, n_patches, self.batch_size):
            batch_patches = patches[i:i + self.batch_size]
            batch_size_actual = len(batch_patches)

            try:
                # Model inference
                batch_output = self.model(batch_patches, training=False)

                # Handle different model output formats
                if isinstance(batch_output, (list, tuple)):
                    enhanced_batch = batch_output[0]  # Take first output (usually enhanced image)
                else:
                    enhanced_batch = batch_output

                enhanced_patches.append(enhanced_batch)
                self.logger.debug(f"   Processed batch {i//self.batch_size + 1}: "
                                f"{batch_size_actual} patches → {enhanced_batch.shape}")

            except Exception as e:
                self.logger.error(f"❌ Failed to process batch {i//self.batch_size + 1}: {e}")
                # Create dummy batch
                dummy_batch = np.zeros((batch_size_actual, self.target_width, self.target_height, 1),
                                     dtype=np.float32)
                enhanced_patches.append(dummy_batch)

        # Concatenate all batches
        if enhanced_patches:
            result = np.concatenate(enhanced_patches, axis=0)
            self.logger.info(f"✅ Processed {n_patches} patches: {result.shape}")
            return result
        else:
            self.logger.error("❌ No patches processed successfully")
            return np.zeros((n_patches, self.target_width, self.target_height, 1), dtype=np.float32)

    def postprocess_patches(self, enhanced_patches: np.ndarray) -> np.ndarray:
        """
        Postprocess enhanced patches back to displayable format

        Args:
            enhanced_patches: Model output in [-1,1] range

        Returns:
            Postprocessed patches in [0,255] range
        """
        postprocessed_patches = []

        for i, patch in enumerate(enhanced_patches):
            try:
                # Postprocess individual patch
                postprocessed_patch = self.preprocessor.postprocess_output(patch)
                postprocessed_patches.append(postprocessed_patch)

            except Exception as e:
                self.logger.warning(f"⚠️ Failed to postprocess patch {i}: {e}")
                # Create dummy patch
                dummy_patch = np.zeros((self.target_height, self.target_width), dtype=np.uint8)
                postprocessed_patches.append(dummy_patch)

        return np.array(postprocessed_patches)


class EnhancedUniversalGANHTRProcessor:
    """
    Enhanced universal processor with training-consistent preprocessing
    """

    def __init__(self, model_path: str, patch_size: Tuple[int, int] = (512, 64),
                 overlap: float = 0.25, strategy: str = 'sliding',
                 blend_method: str = 'linear', batch_size: int = 4,
                 domain_aware: bool = True, logger: Optional[logging.Logger] = None):
        """
        Initialize enhanced universal processor

        Args:
            model_path: Path to trained model
            patch_size: Size of patches (height, width)
            overlap: Overlap ratio between patches
            strategy: Patch extraction strategy
            blend_method: Blending method for reconstruction
            batch_size: Batch size for processing
            domain_aware: Whether to use domain-aware preprocessing
            logger: Optional logger instance
        """
        self.model_path = model_path
        self.patch_size = patch_size
        self.overlap = overlap
        self.strategy = strategy
        self.blend_method = blend_method
        self.batch_size = batch_size
        self.domain_aware = domain_aware
        self.logger = logger or logging.getLogger(__name__)

        # Initialize components
        self.preprocessor = create_enhanced_preprocessor(
            domain_aware=domain_aware,
            target_size=(128, 1024),  # Training target size
            logger=logger
        )

        # For document-aware processing, we use different approach
        if domain_aware:
            self.patch_processor = None  # Not used for document-aware processing
        else:
            self.patch_processor = EnhancedPatchProcessor(
                model_path, batch_size, logger=logger
            )

        # Import patch processing components for non-domain-aware mode
        if not domain_aware:
            from scripts.patch_based_inference import PatchExtractor, ImageReconstructor
            self.patch_extractor = PatchExtractor(patch_size, overlap, strategy)
            self.reconstructor = ImageReconstructor(
                original_size=(0, 0),  # Will be set per image
                patch_size=patch_size,
                stride=0,  # Will be set per image
                blend_method=blend_method
            )

        self.logger.info(f"🌟 Enhanced Universal GAN-HTR Processor initialized:")
        self.logger.info(f"   Model: {model_path}")
        self.logger.info(f"   Patch size: {patch_size}")
        self.logger.info(f"   Overlap: {overlap:.1%}")
        self.logger.info(f"   Strategy: {strategy}")
        self.logger.info(f"   Blend method: {blend_method}")
        self.logger.info(f"   Domain-aware: {domain_aware}")

    def _load_model(self) -> tf.keras.Model:
        """Load trained GAN-HTR model with automatic architecture detection"""
        try:
            self.logger.info(f"📂 Loading GAN-HTR model from: {self.model_path}")

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
            checkpoint_path = tf.train.latest_checkpoint(self.model_path)
            if not checkpoint_path:
                raise FileNotFoundError(f"No checkpoint found in {self.model_path}")

            self.logger.info(f"   Checkpoint: {checkpoint_path}")

            # Try each model architecture
            for attempt in model_attempts:
                try:
                    self.logger.info(f"   Trying {attempt['name']} architecture...")

                    # Import the module
                    module = __import__(attempt['module'], fromlist=[attempt['function']])
                    model_function = getattr(module, attempt['function'])

                    # Build model
                    model = model_function(input_size=(1024, 128, 1))

                    # Try to load checkpoint using the working method
                    checkpoint = tf.train.Checkpoint(generator=model)
                    checkpoint.restore(checkpoint_path).expect_partial()

                    self.logger.info(f"   ✅ {attempt['name']} architecture loaded successfully!")
                    self.logger.info(f"   Model: {type(model).__name__}")
                    return model

                except Exception as e:
                    self.logger.warning(f"   ❌ {attempt['name']} failed: {str(e)[:100]}...")
                    continue

            # If all attempts failed
            raise RuntimeError(f"Could not load any compatible model architecture from {self.model_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            # As fallback, create dummy model for testing
            self.logger.warning("⚠️ Creating dummy model for testing purposes")
            from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
            model = unet_enhanced(input_size=(1024, 128, 1))
            return model

    def process_image_document_aware(self, image_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process image using document-aware approach (preferred for H-DIBCO)

        Args:
            image_path: Path to input image
            output_path: Optional output path for enhanced image

        Returns:
            Processing results dictionary
        """
        start_time = time.time()

        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        original_shape = image.shape
        self.logger.info(f"📄 Processing document: {Path(image_path).name} ({original_shape})")

        # Load model if not loaded (lazy loading for document-aware mode)
        if not hasattr(self, 'model'):
            self.model = self._load_model()

        # Preprocess document (extract text lines)
        processed_lines = self.preprocessor.preprocess_document(image)

        if not processed_lines:
            raise ValueError("No text lines extracted from document")

        # Process text lines with the model
        enhanced_lines = []
        for i, line in enumerate(processed_lines):
            try:
                # Add batch dimension
                line_batch = np.expand_dims(line, axis=0)

                # Model inference
                line_output = self.model(line_batch, training=False)

                # Handle different output formats
                if isinstance(line_output, (list, tuple)):
                    enhanced_line = line_output[0]
                else:
                    enhanced_line = line_output

                # Remove batch dimension
                enhanced_line = np.squeeze(enhanced_line, axis=0)
                enhanced_lines.append(enhanced_line)

                self.logger.debug(f"   Processed line {i+1}: {line.shape} → {enhanced_line.shape}")

            except Exception as e:
                self.logger.warning(f"⚠️ Failed to process line {i+1}: {e}")
                # Create dummy enhanced line
                dummy_line = np.zeros_like(line)
                enhanced_lines.append(dummy_line)

        # Reconstruct document
        enhanced_document = self.preprocessor.reconstruct_document(enhanced_lines, original_shape)

        # Calculate metrics
        metrics = self._calculate_document_metrics(image, enhanced_document)

        # Save output if path provided
        if output_path:
            cv2.imwrite(output_path, enhanced_document)
            self.logger.info(f"💾 Enhanced document saved: {output_path}")

        processing_time = time.time() - start_time

        results = {
            'input_path': image_path,
            'output_path': output_path,
            'original_shape': original_shape,
            'enhanced_shape': enhanced_document.shape,
            'n_lines_processed': len(processed_lines),
            'processing_method': 'document_aware',
            'processing_time': processing_time,
            'reconstruction_metrics': metrics
        }

        self.logger.info(f"✅ Document processing completed in {processing_time:.2f}s")
        self.logger.info(f"   Lines processed: {len(processed_lines)}")
        self.logger.info(f"   PSNR: {metrics['psnr']:.2f} dB")
        if metrics['ssim'] is not None:
            self.logger.info(f"   SSIM: {metrics['ssim']:.4f}")

        return results

    def process_image_patch_based(self, image_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process image using patch-based approach (fallback method)

        Args:
            image_path: Path to input image
            output_path: Optional output path for enhanced image

        Returns:
            Processing results dictionary
        """
        if self.domain_aware:
            self.logger.warning("⚠️ Using patch-based processing despite domain_aware=True")

        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        original_shape = image.shape
        self.logger.info(f"🖼️ Processing image: {Path(image_path).name} ({original_shape})")

        # Resize image to be compatible with patch size if needed
        target_height, target_width = self.patch_size
        if image.shape[0] < target_height or image.shape[1] < target_width:
            scale_h = target_height / image.shape[0]
            scale_w = target_width / image.shape[1]
            scale = max(scale_h, scale_w)

            new_h = int(image.shape[0] * scale)
            new_w = int(image.shape[1] * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            self.logger.info(f"   Resized: {original_shape} → {image.shape}")

        # Extract patches
        patches, positions = self.patch_extractor.extract_patches(image)
        self.logger.info(f"   Extracted {len(patches)} patches")

        if len(patches) == 0:
            raise ValueError("No patches extracted")

        # Preprocess patches with training-consistent pipeline
        processed_patches = self.patch_processor.preprocess_patches(patches)

        # Process patches with model
        enhanced_patches = self.patch_processor.process_patches_batch(processed_patches)

        # Postprocess patches
        postprocessed_patches = self.patch_processor.postprocess_patches(enhanced_patches)

        # Reconstruct image
        self.reconstructor.original_size = image.shape
        self.reconstructor.stride = self.patch_extractor.effective_stride[0]
        enhanced_image = self.reconstructor.reconstruct_image(postprocessed_patches, positions)

        # Resize back to original dimensions if needed
        if enhanced_image.shape != original_shape:
            enhanced_image = cv2.resize(enhanced_image, (original_shape[1], original_shape[0]),
                                      interpolation=cv2.INTER_CUBIC)

        # Calculate metrics
        metrics = self._calculate_patch_metrics(image, enhanced_image)

        # Save output if path provided
        if output_path:
            cv2.imwrite(output_path, enhanced_image)
            self.logger.info(f"💾 Enhanced image saved: {output_path}")

        return {
            'input_path': image_path,
            'output_path': output_path,
            'original_shape': original_shape,
            'enhanced_shape': enhanced_image.shape,
            'n_patches': len(patches),
            'processing_method': 'patch_based',
            'reconstruction_metrics': metrics
        }

    def process_image(self, image_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process image with appropriate method based on configuration

        Args:
            image_path: Path to input image
            output_path: Optional output path

        Returns:
            Processing results dictionary
        """
        if self.domain_aware:
            return self.process_image_document_aware(image_path, output_path)
        else:
            return self.process_image_patch_based(image_path, output_path)

    def _calculate_document_metrics(self, original: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
        """Calculate reconstruction metrics for document processing"""
        # Ensure same shape
        if original.shape != enhanced.shape:
            enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]),
                                interpolation=cv2.INTER_CUBIC)

        # Convert to float [0,1]
        orig_float = original.astype(np.float32) / 255.0
        enh_float = enhanced.astype(np.float32) / 255.0

        # PSNR
        mse = np.mean((orig_float - enh_float) ** 2)
        if mse == 0:
            psnr = 100.0
        else:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))

        # SSIM
        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_value = ssim(orig_float, enh_float, data_range=1.0)
        except ImportError:
            ssim_value = None

        # Additional metrics
        mae = np.mean(np.abs(orig_float - enh_float))

        return {
            'psnr': float(psnr),
            'ssim': float(ssim_value) if ssim_value is not None else None,
            'mse': float(mse),
            'mae': float(mae)
        }

    def _calculate_patch_metrics(self, original: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
        """Calculate reconstruction metrics for patch processing"""
        return self._calculate_document_metrics(original, enhanced)


def create_enhanced_processor(model_path: str, **kwargs) -> EnhancedUniversalGANHTRProcessor:
    """
    Factory function to create enhanced processor

    Args:
        model_path: Path to trained model
        **kwargs: Additional arguments for processor

    Returns:
        Enhanced processor instance
    """
    return EnhancedUniversalGANHTRProcessor(model_path, **kwargs)


if __name__ == "__main__":
    # Test the enhanced processor
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Patch-Based GAN-HTR Inference')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--model', default='dual_modal_gan/outputs/checkpoints_fp32', help='Model path')
    parser.add_argument('--output', help='Output image path')
    parser.add_argument('--domain-aware', action='store_true', help='Use document-aware processing')
    parser.add_argument('--patch-size', type=int, nargs=2, default=[512, 64], help='Patch size')
    parser.add_argument('--overlap', type=float, default=0.25, help='Overlap ratio')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create processor
    processor = create_enhanced_processor(
        model_path=args.model,
        patch_size=tuple(args.patch_size),
        overlap=args.overlap,
        batch_size=args.batch_size,
        domain_aware=args.domain_aware,
        logger=logger
    )

    # Process image
    output_path = args.output or f"enhanced_{Path(args.image).stem}.png"
    results = processor.process_image(args.image, output_path)

    print(f"\n🎉 Processing completed!")
    print(f"   Method: {results['processing_method']}")
    print(f"   Input: {results['input_path']}")
    print(f"   Output: {results['output_path']}")
    print(f"   PSNR: {results['reconstruction_metrics']['psnr']:.2f} dB")
    if results['reconstruction_metrics']['ssim'] is not None:
        print(f"   SSIM: {results['reconstruction_metrics']['ssim']:.4f}")