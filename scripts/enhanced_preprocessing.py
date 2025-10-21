#!/usr/bin/env python3
"""
Enhanced Preprocessing Module for GAN-HTR Inference
Consistent preprocessing pipeline that matches training exactly
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
import logging

class EnhancedGANHTRPreprocessor:
    """
    Preprocessor that exactly matches the training pipeline from train_enhanced.py
    """

    def __init__(self, target_size: Tuple[int, int] = (128, 1024), logger: Optional[logging.Logger] = None):
        """
        Initialize preprocessor with target size matching training

        Args:
            target_size: (height, width) = (128, 1024) as used in training
            logger: Optional logger instance
        """
        self.target_height, self.target_width = target_size
        self.logger = logger or logging.getLogger(__name__)

        # Log initialization
        self.logger.info(f"🔧 Enhanced Preprocessor initialized:")
        self.logger.info(f"   Target size: (H,W) = {target_size}")
        self.logger.info(f"   Format: Training-consistent preprocessing")

    def preprocess_for_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image exactly like training pipeline

        This matches the preprocessing in train_enhanced.py:
        1. Convert to float32 [0,1]
        2. Resize to (128, 1024) if needed
        3. Add channel dimension if missing
        4. Transpose to (1024, 128, 1) for model input
        5. Normalize to [-1, 1] for tanh activation

        Args:
            image: Input image (grayscale or RGB)

        Returns:
            Preprocessed image ready for model input: (1024, 128, 1) in [-1,1] range
        """
        # Step 1: Convert to float32 [0,1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)

        # Step 2: Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Step 3: Ensure 2D grayscale
        if len(image.shape) == 3:
            image = np.squeeze(image, axis=-1)

        # Step 4: Resize to target size (128, 1024) - CRITICAL for model compatibility
        original_shape = image.shape
        if image.shape != (self.target_height, self.target_width):
            # Use high-quality interpolation
            image = cv2.resize(image, (self.target_width, self.target_height),
                             interpolation=cv2.INTER_CUBIC)
            self.logger.debug(f"   Resized: {original_shape} → {image.shape}")

        # Step 5: Add channel dimension (128, 1024, 1)
        if len(image.shape) == 2:
            image = image[..., np.newaxis]

        # Step 6: Transpose to match model input expectation (1024, 128, 1)
        # This EXACTLY matches training: tf.transpose(degraded_image, perm=[1, 0, 2])
        image = np.transpose(image, (1, 0, 2))  # (128, 1024, 1) → (1024, 128, 1)

        # Step 7: Normalize to [-1, 1] for tanh activation
        # Clamp to [0,1] first to handle images that don't use full range
        image = np.clip(image, 0.0, 1.0)
        image = (image * 2.0) - 1.0

        return image

    def postprocess_output(self, image: np.ndarray) -> np.ndarray:
        """
        Postprocess model output back to displayable format

        Reverse of preprocess_for_inference:
        1. Denormalize from [-1,1] to [0,1]
        2. Transpose back to (128, 1024, 1)
        3. Remove channel dimension
        4. Convert to uint8 [0,255]

        Args:
            image: Model output in [-1,1] range, shape (1024, 128, 1)

        Returns:
            Displayable image in [0,255] range, shape (128, 1024)
        """
        # Step 1: Denormalize from [-1,1] to [0,1]
        image = (image + 1.0) / 2.0

        # Step 2: Transpose back to (128, 1024, 1)
        image = np.transpose(image, (1, 0, 2))  # (1024, 128, 1) → (128, 1024, 1)

        # Step 3: Remove channel dimension
        if len(image.shape) == 3 and image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)

        # Step 4: Convert to uint8 [0,255]
        image = (image * 255).astype(np.uint8)

        return image

    def validate_input(self, image: np.ndarray) -> bool:
        """
        Validate input image compatibility

        Args:
            image: Input image to validate

        Returns:
            True if compatible, False otherwise
        """
        if image is None or image.size == 0:
            self.logger.error("❌ Invalid image: None or empty")
            return False

        if len(image.shape) not in [2, 3]:
            self.logger.error(f"❌ Invalid image dimensions: {image.shape}")
            return False

        if len(image.shape) == 3 and image.shape[-1] not in [1, 3]:
            self.logger.error(f"❌ Invalid channel count: {image.shape[-1]}")
            return False

        # Check if image is too small for meaningful processing
        min_size = min(image.shape[:2])
        if min_size < 32:
            self.logger.warning(f"⚠️ Image very small: {image.shape}, may produce poor results")

        return True


class DocumentAwarePreprocessor:
    """
    Domain-aware preprocessor for full document images
    Handles the mismatch between full documents (H-DIBCO) and training format (text lines)
    """

    def __init__(self, target_size: Tuple[int, int] = (128, 1024), logger: Optional[logging.Logger] = None):
        """
        Initialize document-aware preprocessor

        Args:
            target_size: Target size for individual text lines (height, width)
            logger: Optional logger instance
        """
        self.target_height, self.target_width = target_size
        self.logger = logger or logging.getLogger(__name__)
        self.base_preprocessor = EnhancedGANHTRPreprocessor(target_size, logger)

        self.logger.info(f"📄 Document-Aware Preprocessor initialized:")
        self.logger.info(f"   Target line size: (H,W) = {target_size}")
        self.logger.info(f"   Strategy: Extract text lines from full documents")

    def extract_text_lines(self, document: np.ndarray, min_line_height: int = 32,
                          max_line_height: int = 200, line_spacing_factor: float = 1.5) -> List[np.ndarray]:
        """
        Extract text lines from full document using projection analysis

        Args:
            document: Full document image (grayscale)
            min_line_height: Minimum expected text line height
            max_line_height: Maximum expected text line height
            line_spacing_factor: Factor for determining line spacing

        Returns:
            List of extracted text line images
        """
        # Ensure grayscale
        if len(document.shape) == 3:
            document = cv2.cvtColor(document, cv2.COLOR_RGB2GRAY)
        
        # Convert to uint8 if float32 for binarization
        if document.dtype == np.float32 or document.dtype == np.float64:
            document_uint8 = (document * 255).astype(np.uint8)
        else:
            document_uint8 = document.astype(np.uint8)

        # Binarize for projection analysis
        _, binary = cv2.threshold(document_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Horizontal projection to find text lines
        horizontal_projection = np.sum(binary == 0, axis=1)  # Count black pixels per row

        # Find peaks (text lines) using simple thresholding
        threshold = np.max(horizontal_projection) * 0.1  # 10% of max
        text_rows = horizontal_projection > threshold

        # Find contiguous text regions
        text_lines = []
        in_text = False
        line_start = 0

        for i, is_text in enumerate(text_rows):
            if is_text and not in_text:
                # Start of text line
                line_start = i
                in_text = True
            elif not is_text and in_text:
                # End of text line
                line_end = i
                line_height = line_end - line_start

                # Validate line height
                if min_line_height <= line_height <= max_line_height:
                    # Extract the text line with some padding
                    padding = int(line_height * 0.1)  # 10% padding
                    padded_start = max(0, line_start - padding)
                    padded_end = min(document.shape[0], line_end + padding)

                    text_line = document[padded_start:padded_end, :]
                    text_lines.append(text_line)

                    self.logger.debug(f"   Extracted line {len(text_lines)}: height={line_height}, "
                                    f"padded={padded_end-padded_start}")

                in_text = False

        # Handle case where document ends with text
        if in_text:
            line_end = document.shape[0]
            line_height = line_end - line_start
            if min_line_height <= line_height <= max_line_height:
                padding = int(line_height * 0.1)
                padded_start = max(0, line_start - padding)
                padded_end = min(document.shape[0], line_end + padding)
                text_line = document[padded_start:padded_end, :]
                text_lines.append(text_line)

        self.logger.info(f"   Extracted {len(text_lines)} text lines from document")
        return text_lines

    def preprocess_document(self, document: np.ndarray) -> List[np.ndarray]:
        """
        Preprocess full document by extracting and processing text lines

        Args:
            document: Full document image (grayscale or RGB)

        Returns:
            List of preprocessed text lines ready for model input
        """
        if not self.base_preprocessor.validate_input(document):
            raise ValueError("Invalid document input")

        # Extract text lines
        text_lines = self.extract_text_lines(document)

        if not text_lines:
            self.logger.warning("⚠️ No text lines extracted, using full document")
            text_lines = [document]

        # Preprocess each text line
        processed_lines = []
        for i, line in enumerate(text_lines):
            try:
                processed_line = self.base_preprocessor.preprocess_for_inference(line)
                processed_lines.append(processed_line)
                self.logger.debug(f"   Processed line {i+1}: {line.shape} → {processed_line.shape}")
            except Exception as e:
                self.logger.warning(f"   ⚠️ Failed to process line {i+1}: {e}")
                continue

        self.logger.info(f"   Successfully processed {len(processed_lines)} text lines")
        return processed_lines

    def reconstruct_document(self, processed_lines: List[np.ndarray],
                           original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Reconstruct document from processed text lines

        Args:
            processed_lines: List of model outputs (1024, 128, 1) in [-1,1] range
            original_shape: Original document shape (height, width)

        Returns:
            Reconstructed document image
        """
        if not processed_lines:
            self.logger.error("❌ No processed lines to reconstruct")
            return np.zeros(original_shape, dtype=np.uint8)

        # Postprocess each line
        postprocessed_lines = []
        for i, line in enumerate(processed_lines):
            try:
                post_line = self.base_preprocessor.postprocess_output(line)
                postprocessed_lines.append(post_line)
                self.logger.debug(f"   Postprocessed line {i+1}: {line.shape} → {post_line.shape}")
            except Exception as e:
                self.logger.warning(f"   ⚠️ Failed to postprocess line {i+1}: {e}")
                # Create dummy line to maintain structure
                dummy_line = np.zeros((128, 1024), dtype=np.uint8)
                postprocessed_lines.append(dummy_line)

        # Concatenate lines vertically
        reconstructed = np.vstack(postprocessed_lines)

        # Resize to match original document width
        if reconstructed.shape[1] != original_shape[1]:
            reconstructed = cv2.resize(reconstructed, (original_shape[1], reconstructed.shape[0]),
                                     interpolation=cv2.INTER_CUBIC)

        # Crop or pad to match original height
        if reconstructed.shape[0] > original_shape[0]:
            reconstructed = reconstructed[:original_shape[0], :]
        elif reconstructed.shape[0] < original_shape[0]:
            padding = original_shape[0] - reconstructed.shape[0]
            reconstructed = np.pad(reconstructed, ((0, padding), (0, 0)),
                                 mode='constant', constant_values=255)

        self.logger.info(f"   Reconstructed document: {original_shape} → {reconstructed.shape}")
        return reconstructed


def create_enhanced_preprocessor(domain_aware: bool = True,
                              target_size: Tuple[int, int] = (128, 1024),
                              logger: Optional[logging.Logger] = None) -> object:
    """
    Factory function to create appropriate preprocessor

    Args:
        domain_aware: Whether to use document-aware preprocessing
        target_size: Target size for text lines (height, width)
        logger: Optional logger instance

    Returns:
        Preprocessor instance
    """
    if domain_aware:
        return DocumentAwarePreprocessor(target_size, logger)
    else:
        return EnhancedGANHTRPreprocessor(target_size, logger)


if __name__ == "__main__":
    # Simple test
    import sys

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if len(sys.argv) < 2:
        print("Usage: python enhanced_preprocessing.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Test basic preprocessor
    print("🧪 Testing EnhancedGANHTRPreprocessor...")
    preprocessor = EnhancedGANHTRPreprocessor(logger=logger)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        sys.exit(1)

    processed = preprocessor.preprocess_for_inference(image)
    print(f"✅ Preprocessed: {image.shape} → {processed.shape}")

    postprocessed = preprocessor.postprocess_output(processed)
    print(f"✅ Postprocessed: {processed.shape} → {postprocessed.shape}")

    # Test document-aware preprocessor
    print("\n🧪 Testing DocumentAwarePreprocessor...")
    doc_preprocessor = DocumentAwarePreprocessor(logger=logger)

    processed_lines = doc_preprocessor.preprocess_document(image)
    print(f"✅ Document processed: {len(processed_lines)} lines")

    reconstructed = doc_preprocessor.reconstruct_document(processed_lines, image.shape)
    print(f"✅ Document reconstructed: {image.shape} → {reconstructed.shape}")

    print("\n🎉 All tests completed successfully!")