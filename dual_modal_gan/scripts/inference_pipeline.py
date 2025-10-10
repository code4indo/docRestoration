#!/usr/bin/env python3
"""
GAN-HTR Inference Pipeline - Document Restoration & Text Recognition

Pipeline untuk memproses dokumen terdegradasi ukuran penuh:
1. Segmentasi dokumen menjadi baris teks
2. Restorasi setiap baris menggunakan GAN-HTR
3. Ekstraksi teks menggunakan HTR recognizer
4. Rekonstruksi dokumen bersih dan output teks

Usage:
    poetry run python dual_modal_gan/scripts/inference_pipeline.py \
        --input_dir /path/to/degraded/documents \
        --output_dir outputs/restored_documents \
        --gan_checkpoint dual_modal_gan/outputs/checkpoints_fp32_smoke_test \
        --htr_weights models/best_htr_recognizer/best_model.weights.h5 \
        --charset_path real_data_preparation/real_data_charlist.txt \
        --gpu_id 1
"""

import argparse
import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

import tensorflow as tf
tf.config.optimizer.set_jit(False)
tf.get_logger().setLevel('ERROR')

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dual_modal_gan.src.models.generator import unet
from tensorflow.keras import layers, Model

# Constants
IMG_WIDTH = 1024
IMG_HEIGHT = 128
MAX_LABEL_LENGTH = 128


def read_charlist(file_path):
    """Load character set dari file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        charset = []
        for line in f:
            content = line.rstrip('\n')
            if content == ' ':
                charset.append(' ')
            elif content:
                charset.append(content)
    return charset


def create_htr_recognizer(charset_size, proj_dim=512, num_layers=6, num_heads=8, ff_dim=2048, dropout_rate=0.20):
    """
    Create HTR Recognizer - EXACT architecture dari training.
    Struktur harus identik dengan train_transformer_improved_v2.py
    """
    inputs = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1), name='image_input')
    x = inputs
    
    # CNN Backbone
    def conv_block(inp, filters, k=3, s=(1,1), name_prefix='cb', dropout=0.0):
        """Conv block dengan BatchNorm dan dropout"""
        y = layers.Conv2D(filters, k, strides=s, padding='same', 
                         use_bias=False, name=f'{name_prefix}_conv')(inp)
        y = layers.BatchNormalization(name=f'{name_prefix}_bn')(y)
        y = layers.Activation('gelu', name=f'{name_prefix}_gelu')(y)
        if dropout > 0:
            y = layers.Dropout(dropout, name=f'{name_prefix}_drop')(y)
        return y
    
    # Progressive feature extraction
    x = conv_block(x, 64, k=7, s=(1,2), name_prefix='s1_1', dropout=dropout_rate*0.5)
    x = conv_block(x, 64, k=3, s=(1,1), name_prefix='s1_2', dropout=dropout_rate*0.5)
    x = layers.MaxPooling2D(pool_size=(2,2), name='pool1')(x)
    
    x = conv_block(x, 128, k=3, s=(1,1), name_prefix='s2_1', dropout=dropout_rate*0.7)
    x = conv_block(x, 128, k=3, s=(1,1), name_prefix='s2_2', dropout=dropout_rate*0.7)
    x = layers.MaxPooling2D(pool_size=(2,2), name='pool2')(x)
    
    x = conv_block(x, 256, k=3, s=(1,1), name_prefix='s3_1', dropout=dropout_rate)
    x = conv_block(x, 256, k=3, s=(1,1), name_prefix='s3_2', dropout=dropout_rate)
    x = layers.MaxPooling2D(pool_size=(2,1), name='pool3')(x)
    
    x = conv_block(x, 512, k=3, s=(1,1), name_prefix='s4_1', dropout=dropout_rate)
    x = conv_block(x, 512, k=3, s=(1,1), name_prefix='s4_2', dropout=dropout_rate)
    
    # Sequence Projection
    x = layers.Lambda(
        lambda t: tf.reshape(t, (tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2]*tf.shape(t)[3])),
        name='flatten_height'
    )(x)
    
    x = layers.Dense(proj_dim, name='proj_dense')(x)
    x = layers.LayerNormalization(name='proj_ln')(x)
    x = layers.Dropout(dropout_rate, name='proj_drop')(x)
    
    # Positional Encoding
    seq_len = 128
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_embedding_layer = layers.Embedding(
        input_dim=seq_len,
        output_dim=proj_dim,
        name='positional_embedding'
    )
    x = x + pos_embedding_layer(positions)
    
    # Transformer Encoder
    for i in range(num_layers):
        attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=proj_dim // num_heads,
            dropout=dropout_rate,
            name=f'trn_attn_{i}'
        )(x, x)
        x = layers.LayerNormalization(name=f'trn_ln1_{i}')(x + attn)
        
        ffn = layers.Dense(ff_dim, activation='gelu', name=f'trn_ffn1_{i}')(x)
        ffn = layers.Dropout(dropout_rate, name=f'trn_ffn_drop_{i}')(ffn)
        ffn = layers.Dense(proj_dim, name=f'trn_ffn2_{i}')(ffn)
        x = layers.LayerNormalization(name=f'trn_ln2_{i}')(x + ffn)
        x = layers.Dropout(dropout_rate, name=f'trn_drop_{i}')(x)
    
    # CTC Output
    outputs = layers.Dense(charset_size + 1, activation=None, name='logits')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='HTR_Recognizer')
    return model


def safe_ctc_decode(logits, charset):
    """CTC decoding - convert logits ke text."""
    charset_size = len(charset)
    raw_preds = np.argmax(logits, axis=-1)[0]
    
    # Manual CTC decode
    deduped = []
    prev = -1
    for token in raw_preds:
        if token != prev:
            deduped.append(token)
            prev = token
    
    result = []
    for token in deduped:
        if token != charset_size:  # Skip blank token
            result.append(token)
    
    text = ''.join([charset[i] if 0 <= i < len(charset) else '<?>' for i in result])
    return text


class LineSegmenter:
    """
    Segmentasi dokumen dengan ADAPTIVE MULTI-MODE.
    
    Mode 1: Contour-based polygon (IDEAL - follows text curves)
    Mode 2: Projection profile (untuk dokumen dengan baris terpisah)
    Mode 3: Geometric split (last resort untuk continuous scan)
    """
    
    def __init__(self, min_line_height=10, max_line_height=300, 
                 line_spacing_threshold=3, padding=5,
                 min_lines_threshold=5,    # Fallback jika deteksi < threshold
                 geometric_height=128,     # Height untuk geometric split
                 geometric_overlap=32,     # Overlap untuk blending
                 use_contours=True):       # Enable contour-based segmentation
        self.min_line_height = min_line_height
        self.max_line_height = max_line_height
        self.line_spacing_threshold = line_spacing_threshold
        self.padding = padding
        self.min_lines_threshold = min_lines_threshold
        self.geometric_height = geometric_height
        self.geometric_overlap = geometric_overlap
        self.use_contours = use_contours
        self.segmentation_mode = None
    
    def preprocess_image(self, image):
        """Preprocessing untuk segmentasi yang lebih baik."""
        # Convert ke grayscale jika color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Binarization dengan Otsu
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return binary
    
    def detect_lines(self, image):
        """
        Deteksi baris teks dengan ADAPTIVE method.
        Priority: Contours → Projection → Geometric
        """
        # Mode 1: Try contour-based polygon detection (IDEAL)
        if self.use_contours:
            line_regions = self._detect_lines_contours(image)
            
            # Accept ANY contours found (even 2-3 lines >> 50 geometric tiles cutting strokes!)
            if len(line_regions) > 0:
                self.segmentation_mode = 'contours'
                print(f"✅ Contour mode: {len(line_regions)} natural text lines (polygon boundaries)")
                return line_regions
            else:
                print(f"⚠️  Contour found 0 lines, trying projection...")
        
        # Mode 2: Try projection profile method
        line_regions = self._detect_lines_projection(image)
        
        if len(line_regions) >= self.min_lines_threshold:
            self.segmentation_mode = 'projection'
            print(f"✓ Projection detected {len(line_regions)} lines (rectangle boxes)")
            return line_regions
        
        # Mode 3: Fallback to geometric split
        print(f"⚠️  Projection detected only {len(line_regions)} lines (< {self.min_lines_threshold})")
        print(f"   Switching to GEOMETRIC SPLIT mode for continuous scan...")
        line_regions = self._detect_lines_geometric(image)
        self.segmentation_mode = 'geometric'
        
        return line_regions
    
    def _detect_lines_contours(self, image):
        """
        MODE 1: Contour-based polygon detection (IDEAL).
        Follows actual text boundaries, not rigid rectangles.
        """
        binary = self.preprocess_image(image)
        h, w = binary.shape
        
        # ADAPTIVE kernel dengan CONSERVATIVE approach
        # Start small to avoid over-merging entire document!
        kernel_w = max(80, int(w * 0.04))  # 4% OR 80px (SMALLER - avoid giant blobs)
        kernel_h = 5  # Thin vertical (connect strokes only, not lines)
        
        print(f"   🔍 Contour detection: kernel={kernel_w}×{kernel_h}px (conservative), doc={w}×{h}px")
        
        # Horizontal kernel untuk merge karakter dalam satu baris
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_horizontal)
        
        # Vertical closing untuk tall characters (ل، ط، etc)
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
        connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel_vertical)
        
        # Find contours
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours by vertical position
        line_regions = []
        
        # RELAXED filters for handwriting (lines bisa sangat pendek/tidak penuh!)
        min_width = max(200, w * 0.05)  # 5% OR 200px (handwriting bisa pendek!)
        min_h = 20  # Minimal 20px height (skip noise)
        max_h = 400  # Allow taller lines (connected multi-lines)
        
        for cnt in contours:
            # Get bounding rect untuk initial filter
            x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
            
            # Filter berdasarkan height (allow wider range)
            if h_cnt < min_h or h_cnt > max_h:
                continue
            
            # Filter berdasarkan width (very relaxed - handwriting bisa pendek!)
            if w_cnt < min_width:
                continue
            
            # Store as polygon region
            # For now use bounding box, but store contour for future polygon extraction
            line_regions.append({
                'y_start': y,
                'y_end': y + h_cnt,
                'x_start': x,
                'x_end': x + w_cnt,
                'contour': cnt,
                'type': 'polygon'
            })
        
        # Sort by vertical position (top to bottom)
        line_regions.sort(key=lambda r: r['y_start'])
        
        # Convert to simple tuple format for compatibility
        # (Nanti bisa expand ke polygon extraction)
        simple_regions = [(r['y_start'], r['y_end']) for r in line_regions]
        
        # Detailed logging
        if len(simple_regions) > 0:
            avg_h = sum(end - start for start, end in simple_regions) / len(simple_regions)
            print(f"   ✅ Detected {len(simple_regions)} text lines (contour-based)")
            print(f"      - Avg height: {avg_h:.1f}px")
            print(f"      - Range: y={simple_regions[0][0]} to y={simple_regions[-1][1]}")
        else:
            print(f"   ❌ No valid contours found (kernel={kernel_w}×{kernel_h}, min_width={min_width:.0f}px)")
        
        return simple_regions
    
    def _detect_lines_projection(self, image):
        """MODE 2: Original projection profile method (rectangle boxes)."""
        binary = self.preprocess_image(image)
        
        # Horizontal projection (sum piksel putih per baris)
        h_projection = np.sum(binary, axis=1)
        
        # Normalize
        if h_projection.max() > 0:
            h_projection = h_projection / h_projection.max()
        
        # Deteksi transisi dengan threshold lebih rendah
        threshold = 0.01  # Lebih sensitive untuk degraded documents
        in_line = False
        line_regions = []
        start_y = 0
        
        for y, val in enumerate(h_projection):
            if not in_line and val > threshold:
                start_y = y
                in_line = True
            elif in_line and val <= threshold:
                end_y = y
                height = end_y - start_y
                
                if self.min_line_height <= height <= self.max_line_height:
                    line_regions.append((start_y, end_y))
                
                in_line = False
        
        # Handle case terakhir
        if in_line:
            end_y = len(h_projection)
            height = end_y - start_y
            if self.min_line_height <= height <= self.max_line_height:
                line_regions.append((start_y, end_y))
        
        return line_regions
    
    def _detect_lines_geometric(self, image):
        """
        MODE 3: Geometric split untuk continuous scan documents (LAST RESORT).
        WARNING: This mode CUTS through text strokes! Use only as fallback.
        Split dokumen menjadi tiles overlapping.
        """
        h, w = image.shape[:2]
        line_regions = []
        
        y = 0
        stride = self.geometric_height - self.geometric_overlap
        
        while y < h:
            y_end = min(y + self.geometric_height, h)
            
            # Ensure minimum height
            if y_end - y >= 30:  # Minimum 30px untuk tile
                line_regions.append((y, y_end))
            
            y += stride
            
            # Last tile - prevent too small remainder
            if y < h and (h - y) < 30:
                # Extend last tile to cover remainder
                if line_regions:
                    line_regions[-1] = (line_regions[-1][0], h)
                break
        
        print(f"   ⚠️  GEOMETRIC MODE (may cut through strokes!):")
        print(f"   - Generated {len(line_regions)} uniform tiles")
        print(f"   - Tile height: {self.geometric_height}px")
        print(f"   - Overlap: {self.geometric_overlap}px")
        print(f"   - Coverage: {sum(end-start for start, end in line_regions)}px / {h}px")
        print(f"   - NOT text-aware - may interrupt handwriting flow!")
        
        return line_regions
    
    def merge_close_lines(self, line_regions):
        """
        Merge baris yang terlalu dekat (kemungkinan satu baris).
        SKIP merging jika mode=geometric karena tiles sudah overlapping by design.
        """
        if not line_regions:
            return []
        
        # IMPORTANT: Don't merge geometric tiles!
        if self.segmentation_mode == 'geometric':
            print(f"   ✓ Skipping merge for geometric mode (keeping {len(line_regions)} tiles)")
            return line_regions
        
        # Merge hanya untuk projection mode
        merged = []
        current_start, current_end = line_regions[0]
        
        for start, end in line_regions[1:]:
            gap = start - current_end
            
            if gap < self.line_spacing_threshold:
                # Merge dengan baris sebelumnya
                current_end = end
            else:
                # Simpan baris sebelumnya dan mulai baris baru
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        # Tambahkan baris terakhir
        merged.append((current_start, current_end))
        
        print(f"   ✓ Merged {len(line_regions)} lines → {len(merged)} lines")
        return merged
    
    def extract_line_images(self, image, line_regions):
        """Extract image crops untuk setiap baris."""
        line_images = []
        
        for start_y, end_y in line_regions:
            # Tambahkan padding
            start_y = max(0, start_y - self.padding)
            end_y = min(image.shape[0], end_y + self.padding)
            
            # Crop baris
            line_img = image[start_y:end_y, :]
            line_images.append({
                'image': line_img,
                'y_start': start_y,
                'y_end': end_y,
                'height': end_y - start_y
            })
        
        return line_images
    
    def segment_document(self, image):
        """Main method: segment dokumen menjadi baris."""
        line_regions = self.detect_lines(image)
        line_regions = self.merge_close_lines(line_regions)
        line_images = self.extract_line_images(image, line_regions)
        
        return line_images


def preprocess_line_for_model(line_image, target_width=IMG_WIDTH, target_height=IMG_HEIGHT):
    """
    Preprocess baris teks untuk input model GAN-HTR.
    
    CRITICAL FIX: Split wide lines into horizontal tiles to preserve resolution!
    Avoid extreme downscaling that loses handwriting details.
    
    Returns: List of preprocessed tiles (each 1024×128)
    """
    # Convert ke grayscale jika perlu
    if len(line_image.shape) == 3:
        gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = line_image.copy()
    
    h, w = gray.shape
    
    # Check if width requires tiling to avoid >2x downscaling
    scale_factor = target_width / w
    
    if scale_factor < 0.5:  # Would lose >50% resolution
        # Split horizontally into tiles
        tiles = split_line_into_tiles(gray, target_width, target_height)
        return tiles
    else:
        # Original method for narrow lines
        # Calculate scaling untuk maintain aspect ratio
        scale = min(target_width / w, target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas dengan padding
        canvas = np.ones((target_height, target_width), dtype=np.uint8) * 255
        
        # Center the resized image
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Normalize ke [0, 1]
        normalized = canvas.astype(np.float32) / 255.0
        
        # Transpose ke (W, H) untuk model
        transposed = np.transpose(normalized)
        
        # Add channel dimension
        final = np.expand_dims(transposed, axis=-1)
        
        return [final]  # Return as list for consistency


def split_line_into_tiles(line_image, tile_width=IMG_WIDTH, tile_height=IMG_HEIGHT, overlap=128):
    """
    Split wide line into overlapping horizontal tiles to preserve resolution.
    
    Args:
        line_image: numpy array (H, W) grayscale
        tile_width: Target tile width (default 1024)
        tile_height: Target tile height (default 128)
        overlap: Overlap between tiles in pixels (default 128 for smooth blending)
    
    Returns:
        List of (preprocessed_tile, x_start, x_end) tuples
    """
    h, w = line_image.shape
    tiles = []
    
    # Resize height to target while preserving aspect ratio
    if h != tile_height:
        scale_h = tile_height / h
        new_w = int(w * scale_h)
        line_resized = cv2.resize(line_image, (new_w, tile_height), interpolation=cv2.INTER_CUBIC)
    else:
        line_resized = line_image
        new_w = w
    
    # Split width into tiles with overlap
    stride = tile_width - overlap
    x = 0
    
    while x < new_w:
        x_end = min(x + tile_width, new_w)
        
        # Extract tile
        tile = line_resized[:, x:x_end]
        tile_h, tile_w = tile.shape
        
        # Pad if needed
        if tile_w < tile_width:
            padded = np.ones((tile_height, tile_width), dtype=np.uint8) * 255
            padded[:tile_h, :tile_w] = tile
            tile = padded
        
        # Normalize ke [0, 1]
        normalized = tile.astype(np.float32) / 255.0
        
        # Transpose ke (W, H) untuk model
        transposed = np.transpose(normalized)
        
        # Add channel dimension
        final = np.expand_dims(transposed, axis=-1)
        
        tiles.append({
            'tile': final,
            'x_start': x,
            'x_end': x_end,
            'original_width': new_w
        })
        
        x += stride
        
        # Break if we've covered the entire width
        if x_end >= new_w:
            break
    
    return tiles


def stitch_tiles_horizontally(tiles, tile_infos, target_height, target_width):
    """
    Stitch horizontal tiles back into full line with overlap blending.
    
    Args:
        tiles: List of restored tile arrays (model output format)
        tile_infos: List of tile metadata (x_start, x_end, original_width)
        target_height: Target height of reconstructed line
        target_width: Target width of reconstructed line
    
    Returns:
        Stitched line as numpy array (H, W) uint8
    """
    if len(tiles) == 1:
        # Single tile - no stitching needed
        return postprocess_single_tile(tiles[0], target_height, target_width)
    
    # Get original width from first tile info
    original_width = tile_infos[0]['original_width']
    
    # Create canvas for stitching
    canvas = np.zeros((IMG_HEIGHT, original_width), dtype=np.float32)
    weight_map = np.zeros((IMG_HEIGHT, original_width), dtype=np.float32)
    
    overlap = 128  # Increased from 64 for better stroke continuity
    
    for tile, tile_info in zip(tiles, tile_infos):
        # Denormalize tile (model output [0, 1])
        # Tile shape: (W, H, 1) from model
        tile_2d = tile[:, :, 0].T  # (H, W)
        
        x_start = tile_info['x_start']
        x_end = tile_info['x_end']
        tile_w = x_end - x_start
        
        # Extract valid part of tile (remove padding)
        tile_valid = tile_2d[:, :tile_w]
        
        # Create alpha mask with Gaussian blending (smoother than linear)
        alpha = np.ones((IMG_HEIGHT, tile_w), dtype=np.float32)
        
        # Fade edges if not first/last tile
        if x_start > 0:  # Not first tile - fade left edge
            fade_w = min(overlap, tile_w)
            # Gaussian-like curve for natural blending
            fade_curve = np.exp(-3 * (1 - np.linspace(0, 1, fade_w))**2)
            alpha[:, :fade_w] *= fade_curve
        
        if x_end < original_width:  # Not last tile - fade right edge
            fade_w = min(overlap, tile_w)
            # Gaussian-like curve for natural blending
            fade_curve = np.exp(-3 * np.linspace(0, 1, fade_w)**2)
            alpha[:, -fade_w:] *= fade_curve
        
        # Accumulate weighted sum
        canvas[:, x_start:x_end] += tile_valid * alpha
        weight_map[:, x_start:x_end] += alpha
    
    # Normalize by weights
    weight_map = np.maximum(weight_map, 1e-6)
    stitched = canvas / weight_map
    
    # Denormalize to [0, 255]
    stitched_uint8 = (stitched * 255.0).clip(0, 255).astype(np.uint8)
    
    # Phase 2: Morphological stroke connection
    # Connect broken strokes at tile boundaries
    stitched_uint8 = connect_broken_strokes(stitched_uint8)
    
    # Resize to target dimensions
    if stitched_uint8.shape[0] != target_height or stitched_uint8.shape[1] != target_width:
        stitched_uint8 = cv2.resize(
            stitched_uint8,
            (target_width, target_height),
            interpolation=cv2.INTER_CUBIC
        )
    
    return stitched_uint8


def connect_broken_strokes(image, kernel_size=3):
    """
    Phase 2: Connect broken strokes using morphological operations.
    
    Args:
        image: numpy array (H, W) uint8 grayscale
        kernel_size: Size of morphological kernel (3 = gentle, 5 = aggressive)
    
    Returns:
        Image with connected strokes
    """
    # Invert for processing (text becomes white)
    inverted = 255 - image
    
    # Apply morphological closing to connect nearby strokes
    # Horizontal kernel prioritizes connecting horizontal strokes (handwriting)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size * 2, kernel_size))
    closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel_h)
    
    # Light vertical kernel for vertical connections
    kernel_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size * 2))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_v)
    
    # Invert back
    result = 255 - closed
    
    # Blend with original to preserve detail (70% processed, 30% original)
    # This prevents over-smoothing
    blended = cv2.addWeighted(result, 0.7, image, 0.3, 0)
    
    return blended.astype(np.uint8)


def postprocess_single_tile(tile, target_height, target_width):
    """Postprocess single tile (no stitching needed)."""
    # Tile shape: (W, H, 1) from model
    tile_2d = tile[:, :, 0].T  # (H, W)
    
    # Denormalize to [0, 255]
    tile_uint8 = (tile_2d * 255.0).clip(0, 255).astype(np.uint8)
    
    # Resize to target dimensions
    if tile_uint8.shape[0] != target_height or tile_uint8.shape[1] != target_width:
        tile_uint8 = cv2.resize(
            tile_uint8,
            (target_width, target_height),
            interpolation=cv2.INTER_CUBIC
        )
    
    return tile_uint8


def recognize_stitched_line(stitched_line, recognizer, charset):
    """
    Run HTR recognition on stitched line.
    
    Args:
        stitched_line: numpy array (H, W) uint8 - stitched line at original resolution
        recognizer: HTR model
        charset: Character list
    
    Returns:
        Recognized text string
    """
    # Normalize to [0, 1]
    normalized = stitched_line.astype(np.float32) / 255.0
    
    # Resize to HTR input size: 1024 x 128 (W x H)
    # stitched_line shape: (H, W) -> resize -> (128, 1024)
    h, w = normalized.shape
    
    # Calculate scale to preserve aspect ratio
    scale_h = IMG_HEIGHT / h
    scale_w = IMG_WIDTH / w
    scale = min(scale_h, scale_w)
    
    # Resize with aspect ratio preserved
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Ensure dimensions are at least 1
    new_h = max(1, new_h)
    new_w = min(new_w, IMG_WIDTH)  # Cap at model width
    new_h = min(new_h, IMG_HEIGHT)  # Cap at model height
    
    resized = cv2.resize(
        normalized,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )
    
    # Center pad to target size
    canvas = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    
    # Calculate padding offsets for centering
    y_offset = (IMG_HEIGHT - new_h) // 2
    x_offset = (IMG_WIDTH - new_w) // 2
    
    # Place resized image in center
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Transpose to (W, H) for model
    transposed = np.transpose(canvas)
    
    # Add channel and batch dimensions: (1, W, H, 1)
    input_tensor = transposed[np.newaxis, :, :, np.newaxis]
    
    # Run recognition
    logits = recognizer(input_tensor, training=False)
    
    # Decode text
    text = safe_ctc_decode(logits.numpy(), charset)
    
    return text


def postprocess_line_output(generated_image, original_height, original_width):
    """
    Postprocess output GAN untuk rekonstruksi dokumen.
    Resize kembali ke dimensi asli baris.
    """
    # Remove batch dan channel dimension
    if len(generated_image.shape) == 4:
        img = generated_image[0]
    else:
        img = generated_image
    
    # Remove channel dimension
    if img.shape[-1] == 1:
        img = np.squeeze(img, axis=-1)
    
    # Transpose kembali dari (W, H) ke (H, W)
    img = np.transpose(img)
    
    # Denormalize [0, 1] -> [0, 255]
    img = (img * 255.0).astype(np.uint8)
    
    # Resize ke dimensi asli baris
    resized = cv2.resize(img, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
    
    return resized


class DocumentRestorationPipeline:
    """Pipeline lengkap untuk restorasi dokumen dan ekstraksi teks."""
    
    def __init__(self, generator, recognizer, charset, segmenter, batch_size=8):
        self.generator = generator
        self.recognizer = recognizer
        self.charset = charset
        self.segmenter = segmenter
        self.batch_size = batch_size
    
    def process_document(self, document_path, output_dir, save_intermediates=True):
        """
        Process satu dokumen: segmentasi -> restorasi -> OCR.
        
        Returns:
            dict dengan restored_image, text, dan metadata
        """
        # Load dokumen
        image = cv2.imread(str(document_path))
        if image is None:
            raise ValueError(f"Cannot load image: {document_path}")
        
        doc_name = Path(document_path).stem
        print(f"\n📄 Processing: {doc_name}")
        print(f"   Original size: {image.shape[1]}x{image.shape[0]}")
        
        # Segmentasi baris
        print(f"   🔍 Segmenting lines...")
        line_data = self.segmenter.segment_document(image)
        print(f"   ✅ Found {len(line_data)} lines")
        
        if len(line_data) == 0:
            print(f"   ⚠️  No lines detected in document!")
            return None
        
        # Prepare batches untuk processing
        # IMPORTANT: preprocess_line_for_model now returns list of tiles!
        print(f"   🔧 Preprocessing lines (with horizontal tiling for wide lines)...")
        preprocessed_tiles = []
        tile_to_line_map = []  # Track which tiles belong to which line
        
        for line_idx, line_info in enumerate(line_data):
            tiles = preprocess_line_for_model(line_info['image'])
            
            for tile_info in tiles:
                preprocessed_tiles.append(tile_info['tile'])
                tile_to_line_map.append({
                    'line_idx': line_idx,
                    'tile_info': tile_info
                })
        
        print(f"   📊 Total tiles to process: {len(preprocessed_tiles)} (from {len(line_data)} lines)")
        
        # Batch processing untuk restorasi
        print(f"   🎨 Restoring tiles...")
        restored_tiles = []
        
        for i in tqdm(range(0, len(preprocessed_tiles), self.batch_size), desc="   Restoring"):
            batch = preprocessed_tiles[i:i+self.batch_size]
            batch_array = np.array(batch)
            
            # GAN restoration
            generated_batch = self.generator(batch_array, training=False)
            
            # Store restored tiles
            for generated in generated_batch:
                restored_tiles.append(generated.numpy())
        
        # Stitch tiles back into lines
        print(f"   🧩 Stitching tiles back into lines...")
        restored_lines = []
        recognized_texts = []
        
        current_line_idx = -1
        current_line_tiles = []
        current_tile_infos = []
        
        for tile_idx, (restored_tile, mapping) in enumerate(zip(restored_tiles, tile_to_line_map)):
            line_idx = mapping['line_idx']
            
            if line_idx != current_line_idx:
                # Process previous line if exists
                if current_line_idx >= 0:
                    stitched_line = stitch_tiles_horizontally(
                        current_line_tiles,
                        current_tile_infos,
                        line_data[current_line_idx]['height'],
                        line_data[current_line_idx]['image'].shape[1]
                    )
                    restored_lines.append(stitched_line)
                    
                    # HTR recognition on stitched line
                    text = recognize_stitched_line(
                        stitched_line, 
                        self.recognizer,
                        self.charset
                    )
                    recognized_texts.append(text)
                
                # Start new line
                current_line_idx = line_idx
                current_line_tiles = [restored_tile]
                current_tile_infos = [mapping['tile_info']]
            else:
                # Same line, accumulate tiles
                current_line_tiles.append(restored_tile)
                current_tile_infos.append(mapping['tile_info'])
        
        # Process last line
        if current_line_idx >= 0:
            stitched_line = stitch_tiles_horizontally(
                current_line_tiles,
                current_tile_infos,
                line_data[current_line_idx]['height'],
                line_data[current_line_idx]['image'].shape[1]
            )
            restored_lines.append(stitched_line)
            
            text = recognize_stitched_line(
                stitched_line,
                self.recognizer,
                self.charset
            )
            recognized_texts.append(text)
        
        # Rekonstruksi dokumen penuh
        print(f"   🔨 Reconstructing document...")
        restored_document = self.reconstruct_document(
            image, line_data, restored_lines
        )
        
        # Save outputs
        output_path = Path(output_dir) / doc_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save restored document
        restored_path = output_path / f"{doc_name}_restored.png"
        cv2.imwrite(str(restored_path), restored_document)
        
        # Save text
        text_path = output_path / f"{doc_name}_text.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(recognized_texts, 1):
                f.write(f"Line {i:03d}: {text}\n")
        
        # Save metadata
        metadata = {
            'document': doc_name,
            'original_size': {'width': image.shape[1], 'height': image.shape[0]},
            'num_lines': len(line_data),
            'processing_date': datetime.now().isoformat(),
            'lines': []
        }
        
        for i, (line_info, text) in enumerate(zip(line_data, recognized_texts)):
            metadata['lines'].append({
                'line_number': i + 1,
                'y_start': int(line_info['y_start']),
                'y_end': int(line_info['y_end']),
                'height': int(line_info['height']),
                'text': text,
                'text_length': len(text)
            })
        
        metadata_path = output_path / f"{doc_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save intermediates jika diminta
        if save_intermediates:
            lines_dir = output_path / "lines"
            lines_dir.mkdir(exist_ok=True)
            
            for i, (orig_line, rest_line, text) in enumerate(zip(
                line_data, restored_lines, recognized_texts
            )):
                # Get original line image
                orig_img = orig_line['image']
                
                # Ensure both images have same number of channels
                if len(orig_img.shape) == 3 and len(rest_line.shape) == 2:
                    # Original is color, restored is grayscale -> convert restored to color
                    rest_line_display = cv2.cvtColor(rest_line, cv2.COLOR_GRAY2BGR)
                elif len(orig_img.shape) == 2 and len(rest_line.shape) == 3:
                    # Original is grayscale, restored is color -> convert original to color
                    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
                    rest_line_display = rest_line
                elif len(orig_img.shape) == 2 and len(rest_line.shape) == 2:
                    # Both grayscale -> OK
                    rest_line_display = rest_line
                else:
                    # Both color -> OK
                    rest_line_display = rest_line
                
                # Save comparison
                comparison = np.hstack([orig_img, rest_line_display])
                cv2.imwrite(
                    str(lines_dir / f"line_{i+1:03d}_comparison.png"),
                    comparison
                )
        
        print(f"   ✅ Saved to: {output_path}")
        
        return {
            'restored_image': restored_document,
            'text': '\n'.join(recognized_texts),
            'metadata': metadata,
            'output_path': str(output_path)
        }
    
    def reconstruct_document(self, original_image, line_data, restored_lines):
        """Rekonstruksi dokumen dari restored lines."""
        # Create canvas dengan ukuran sama dengan original
        h, w = original_image.shape[:2]
        
        # Check if original is color or grayscale
        if len(original_image.shape) == 3:
            # Color image - create color canvas
            reconstructed = np.ones((h, w, 3), dtype=np.uint8) * 255
        else:
            # Grayscale image
            reconstructed = np.ones((h, w), dtype=np.uint8) * 255
        
        # Place setiap restored line di posisi aslinya
        for line_info, restored_line in zip(line_data, restored_lines):
            y_start = line_info['y_start']
            y_end = line_info['y_end']
            
            # Pastikan dimensi cocok
            if restored_line.shape[0] != (y_end - y_start):
                restored_line = cv2.resize(
                    restored_line,
                    (w, y_end - y_start),
                    interpolation=cv2.INTER_CUBIC
                )
            
            # Convert grayscale to color jika perlu
            if len(reconstructed.shape) == 3 and len(restored_line.shape) == 2:
                restored_line = cv2.cvtColor(restored_line, cv2.COLOR_GRAY2BGR)
            
            reconstructed[y_start:y_end, :] = restored_line
        
        return reconstructed


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # Enable GPU memory growth untuk prevent OOM
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠️  Memory growth setting failed: {e}")
    
    print("=" * 80)
    print("GAN-HTR INFERENCE PIPELINE - Document Restoration & Text Recognition")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load charset
    print(f"\n[1/5] Loading charset from {args.charset_path}...")
    charset = read_charlist(args.charset_path)
    charset_size = len(charset)
    print(f"  ✅ Charset loaded: {charset_size} characters")
    
    # Load models
    print(f"\n[2/5] Loading models...")
    print(f"  📦 Generator from: {args.gan_checkpoint}")
    print(f"  📦 HTR from: {args.htr_weights}")
    
    # Create generator
    generator = unet(input_size=(IMG_WIDTH, IMG_HEIGHT, 1))
    
    # Create recognizer
    recognizer = create_htr_recognizer(
        charset_size=charset_size,
        proj_dim=512,
        num_layers=6,
        num_heads=8,
        ff_dim=2048,
        dropout_rate=0.20
    )
    
    # Load weights
    checkpoint = tf.train.Checkpoint(generator=generator)
    
    # Handle both directory and file path
    if os.path.isdir(args.gan_checkpoint):
        checkpoint_path = tf.train.latest_checkpoint(args.gan_checkpoint)
    else:
        # Remove .index or .data suffix if present
        checkpoint_path = args.gan_checkpoint.replace('.index', '').replace('.data-00000-of-00001', '')
    
    if checkpoint_path and os.path.exists(checkpoint_path + '.index'):
        status = checkpoint.restore(checkpoint_path)
        status.expect_partial()
        print(f"  ✅ Generator loaded from: {checkpoint_path}")
    else:
        print(f"  ⚠️  No checkpoint found at {args.gan_checkpoint}")
        print(f"     Tried: {checkpoint_path}")
        return
    
    # Load HTR weights
    if os.path.exists(args.htr_weights):
        recognizer.load_weights(args.htr_weights)
        print(f"  ✅ HTR recognizer loaded")
    else:
        print(f"  ❌ HTR weights not found: {args.htr_weights}")
        return
    
    print(f"  ✅ Models ready")
    print(f"     Generator: {generator.count_params():,} parameters")
    print(f"     Recognizer: {recognizer.count_params():,} parameters")
    
    # Initialize segmenter
    print(f"\n[3/5] Initializing line segmenter...")
    segmenter = LineSegmenter(
        min_line_height=args.min_line_height,
        max_line_height=args.max_line_height,
        line_spacing_threshold=args.line_spacing_threshold,
        padding=args.line_padding
    )
    print(f"  ✅ Segmenter configured")
    
    # Create pipeline
    print(f"\n[4/5] Creating inference pipeline...")
    pipeline = DocumentRestorationPipeline(
        generator=generator,
        recognizer=recognizer,
        charset=charset,
        segmenter=segmenter,
        batch_size=args.batch_size
    )
    print(f"  ✅ Pipeline ready (batch_size={args.batch_size})")
    
    # Get input documents
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"\n❌ Input directory not found: {args.input_dir}")
        return
    
    # Find all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
    document_files = []
    for ext in image_extensions:
        document_files.extend(list(input_path.glob(f"*{ext}")))
        document_files.extend(list(input_path.glob(f"*{ext.upper()}")))
    
    print(f"\n[5/5] Processing {len(document_files)} documents...")
    
    if len(document_files) == 0:
        print(f"  ⚠️  No image files found in {args.input_dir}")
        return
    
    # Process documents
    results_summary = []
    start_time = datetime.now()
    
    for doc_path in tqdm(document_files, desc="Documents"):
        try:
            result = pipeline.process_document(
                doc_path,
                args.output_dir,
                save_intermediates=args.save_intermediates
            )
            
            if result:
                results_summary.append({
                    'document': doc_path.name,
                    'status': 'success',
                    'num_lines': result['metadata']['num_lines'],
                    'output_path': result['output_path']
                })
        except Exception as e:
            print(f"\n❌ Error processing {doc_path.name}: {str(e)}")
            results_summary.append({
                'document': doc_path.name,
                'status': 'failed',
                'error': str(e)
            })
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Save summary
    summary = {
        'processing_info': {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_time_seconds': processing_time,
            'documents_processed': len(document_files),
            'documents_success': sum(1 for r in results_summary if r['status'] == 'success'),
            'documents_failed': sum(1 for r in results_summary if r['status'] == 'failed')
        },
        'configuration': {
            'gan_checkpoint': args.gan_checkpoint,
            'htr_weights': args.htr_weights,
            'batch_size': args.batch_size,
            'min_line_height': args.min_line_height,
            'max_line_height': args.max_line_height
        },
        'results': results_summary
    }
    
    summary_path = output_dir / 'processing_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"✅ Successfully processed: {summary['processing_info']['documents_success']}/{len(document_files)}")
    print(f"❌ Failed: {summary['processing_info']['documents_failed']}/{len(document_files)}")
    print(f"⏱️  Total time: {processing_time:.2f}s ({len(document_files)/processing_time:.2f} docs/sec)")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📄 Summary saved to: {summary_path}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GAN-HTR Inference Pipeline for Document Restoration & Text Recognition'
    )
    
    # Input/Output
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing degraded document images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save restored documents and extracted text')
    
    # Model paths
    parser.add_argument('--gan_checkpoint', type=str,
                       default='dual_modal_gan/outputs/checkpoints_fp32_smoke_test',
                       help='Path to GAN-HTR checkpoint directory')
    parser.add_argument('--htr_weights', type=str,
                       default='models/best_htr_recognizer/best_model.weights.h5',
                       help='Path to HTR recognizer weights')
    parser.add_argument('--charset_path', type=str,
                       default='real_data_preparation/real_data_charlist.txt',
                       help='Path to character set file')
    
    # Processing parameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for processing (default: 8)')
    parser.add_argument('--gpu_id', type=str, default='1',
                       help='GPU ID to use (default: 1)')
    
    # Segmentation parameters
    parser.add_argument('--min_line_height', type=int, default=20,
                       help='Minimum line height in pixels (default: 20)')
    parser.add_argument('--max_line_height', type=int, default=200,
                       help='Maximum line height in pixels (default: 200)')
    parser.add_argument('--line_spacing_threshold', type=int, default=10,
                       help='Threshold for merging close lines (default: 10)')
    parser.add_argument('--line_padding', type=int, default=5,
                       help='Padding around each line (default: 5)')
    
    # Output options
    parser.add_argument('--save_intermediates', action='store_true', default=True,
                       help='Save intermediate line images (default: True)')
    parser.add_argument('--no_intermediates', dest='save_intermediates', 
                       action='store_false',
                       help='Do not save intermediate line images')
    
    args = parser.parse_args()
    main(args)
