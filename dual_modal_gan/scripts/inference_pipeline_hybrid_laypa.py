#!/usr/bin/env python3
"""
HYBRID GAN-HTR + LAYPA INFERENCE PIPELINE

Mengintegrasikan Laypa untuk segmentation dengan GAN-HTR untuk restoration & recognition.

Pipeline:
1. Laypa: Superior line segmentation (60× better: 182 vs 3 lines)
2. Parse PageXML: Extract baseline coordinates
3. GAN-HTR: Restore each line (our model)
4. HTR: Text recognition (our model)
5. Output: Restored document + recognized text

Usage:
    poetry run python dual_modal_gan/scripts/inference_pipeline_hybrid_laypa.py \
        --input_dir DokumenRusak \
        --output_dir outputs/hybrid_laypa \
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
import subprocess
import xml.etree.ElementTree as ET
from scipy.stats import norm

# Import grid registration system untuk alignment
sys.path.insert(0, str(Path(__file__).parent))
from grid_registration import BaselineAlignmentEnforcer

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


def run_loghi_segmentation(input_dir, output_dir):
    """
    Run Laypa untuk segmentasi (TANPA HTR recognition).
    Output: PageXML dengan baseline coordinates.
    """
    script_path = Path(__file__).parent.parent.parent / "loghi_integration" / "scripts" / "run_loghi_segmentation_only.sh"
    
    if not script_path.exists():
        raise FileNotFoundError(f"Loghi segmentation script not found: {script_path}")
    
    print("\n" + "="*80)
    print("STAGE 1: LAYPA SEGMENTATION")
    print("="*80)
    
    result = subprocess.run(
        [str(script_path), input_dir, output_dir],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Loghi segmentation failed!")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError("Loghi segmentation failed")
    
    print(result.stdout)
    print("✅ Laypa segmentation complete")
    
    # Return path ke PageXML directory
    return Path(output_dir) / "page"


def parse_pagexml_baselines(pagexml_path):
    """
    Parse PageXML untuk extract baseline coordinates.
    
    Returns:
        List of {
            'line_id': str,
            'baseline_points': [(x, y), ...],
            'coords_points': [(x, y), ...],  # Polygon boundary
            'bbox': (x_min, y_min, x_max, y_max)
        }
    """
    tree = ET.parse(pagexml_path)
    root = tree.getroot()
    
    # Handle namespace
    ns = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    
    baselines = []
    
    for textline in root.findall('.//pc:TextLine', ns):
        line_id = textline.get('id')
        
        # Extract baseline points
        baseline_elem = textline.find('pc:Baseline', ns)
        if baseline_elem is None:
            continue
        
        baseline_points_str = baseline_elem.get('points')
        if not baseline_points_str:
            continue
        
        baseline_points = []
        for point_str in baseline_points_str.split():
            x, y = map(int, point_str.split(','))
            baseline_points.append((x, y))
        
        # Extract coords (polygon boundary)
        coords_elem = textline.find('pc:Coords', ns)
        coords_points = []
        if coords_elem is not None:
            coords_points_str = coords_elem.get('points')
            if coords_points_str:
                for point_str in coords_points_str.split():
                    x, y = map(int, point_str.split(','))
                    coords_points.append((x, y))
        
        # Calculate bounding box
        all_points = baseline_points + coords_points
        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            bbox = (min(xs), min(ys), max(xs), max(ys))
        else:
            # Fallback jika coords tidak ada
            xs = [p[0] for p in baseline_points]
            ys = [p[1] for p in baseline_points]
            # Add padding untuk height
            bbox = (min(xs), min(ys)-20, max(xs), max(ys)+20)
        
        baselines.append({
            'line_id': line_id,
            'baseline_points': baseline_points,
            'coords_points': coords_points,
            'bbox': bbox
        })
    
    return baselines


def extract_line_image_from_baseline(image, baseline_info, padding=10):
    """
    Extract line image menggunakan polygon boundary dari Laypa.
    
    Args:
        image: Full document image
        baseline_info: Dict dengan 'bbox' dan 'coords_points'
        padding: Extra padding (pixels)
    
    Returns:
        line_img: Cropped line image (grayscale)
        original_bbox: Original bbox WITHOUT padding (x_min, y_min, x_max, y_max)
        padded_bbox: Padded bbox for extraction (x_min_p, y_min_p, x_max_p, y_max_p)
    """
    h, w = image.shape[:2]
    x_min_orig, y_min_orig, x_max_orig, y_max_orig = baseline_info['bbox']
    
    # Create padded bbox for extraction (with boundary check)
    x_min_padded = max(0, x_min_orig - padding)
    y_min_padded = max(0, y_min_orig - padding)
    x_max_padded = min(w, x_max_orig + padding)
    y_max_padded = min(h, y_max_orig + padding)
    
    # Crop dari full document (dengan padding untuk better restoration)
    line_img = image[y_min_padded:y_max_padded, x_min_padded:x_max_padded]
    
    # Convert ke grayscale jika perlu
    if len(line_img.shape) == 3:
        line_img = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    
    # Return: image, original bbox (for placement), padded bbox (for processing)
    original_bbox = (x_min_orig, y_min_orig, x_max_orig, y_max_orig)
    padded_bbox = (x_min_padded, y_min_padded, x_max_padded, y_max_padded)
    
    return line_img, original_bbox, padded_bbox


def preprocess_line_for_gan(line_image, target_width=IMG_WIDTH, target_height=IMG_HEIGHT):
    """
    Preprocess line untuk GAN input.
    Output: (1, W, H, 1) tensor
    """
    h, w = line_image.shape
    
    # Calculate scale untuk maintain aspect ratio
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(line_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
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
    
    # Add batch and channel dimension: (1, W, H, 1)
    final = transposed[np.newaxis, :, :, np.newaxis]
    
    return final


def postprocess_gan_output(generated_image, original_height, original_width):
    """
    Convert GAN output (1, W, H, 1) kembali ke image format (H, W).
    
    ASPECT RATIO PRESERVATION STRATEGY:
    - GAN output: 1024×128 (aspect 8:1)
    - If target aspect differs significantly, preserve GAN aspect with padding
    - This avoids "gepeng" (squashed) characters
    """
    # Remove batch dimension
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
    
    gan_h, gan_w = img.shape
    gan_aspect = gan_w / gan_h  # 8.0
    target_aspect = original_width / max(original_height, 1)
    
    # If aspect ratio difference > 20%, preserve GAN aspect
    if abs(gan_aspect - target_aspect) / gan_aspect > 0.2:
        # Maintain GAN aspect ratio, adjust height accordingly
        preserved_height = int(original_width / gan_aspect)
        
        # Ensure height is reasonable (tidak terlalu ekstrem)
        min_height = original_height // 2
        max_height = original_height * 2
        preserved_height = max(min_height, min(preserved_height, max_height))
        
        # Resize dengan preserved aspect
        resized = cv2.resize(img, (original_width, preserved_height), interpolation=cv2.INTER_CUBIC)
    else:
        # Aspect ratio mirip, resize langsung
        resized = cv2.resize(img, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
    
    return resized


def create_content_aware_fade_mask(line_image, fade_pixels=15, ink_threshold=240):
    """
    Create alpha mask with CONTENT-AWARE GAUSSIAN FADE.
    
    v6+ Enhancement: Gaussian fade untuk smoother blending
    - Deteksi ink di edge regions, preserve detail goresan
    - Jika ada ink di edge → Minimal fade (preserve fine strokes)
    - Jika background only → Gaussian fade (natural blend)
    
    Args:
        line_image: Line image (H, W) or (H, W, 3), range [0, 255]
        fade_pixels: Max fade width (default: 15, balanced)
        ink_threshold: Pixel value > threshold = background (default: 240)
    
    Returns:
        alpha: (H, W) float32 mask [0.0, 1.0]
               1.0 = show line fully
               Variable at edges based on ink presence
    """
    # Convert to grayscale jika color
    if len(line_image.shape) == 3:
        gray = cv2.cvtColor(line_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = line_image.astype(np.uint8)
    
    height, width = gray.shape
    alpha = np.ones((height, width), dtype=np.float32)
    
    # Ink detection: pixel < threshold = ink present
    has_ink = gray < ink_threshold
    
    # Gaussian curve (most natural, smooth blending)
    # Creates bell-shaped fade: slow start, fast middle, slow end
    t = np.linspace(-2, 2, fade_pixels)
    fade_curve = norm.cdf(t)  # Cumulative distribution function: 0→1 smoothly
    
    # TOP edge: content-aware fade
    if height > fade_pixels * 2:
        for i in range(min(fade_pixels, height // 2)):
            # Check if this row has ink
            row_has_ink = np.any(has_ink[i, :])
            if row_has_ink:
                # Preserve ink: minimal fade (90% → 100%)
                alpha[i, :] = np.minimum(alpha[i, :], 0.9 + 0.1 * fade_curve[i])
            else:
                # Background only: normal fade (0% → 100%)
                alpha[i, :] = np.minimum(alpha[i, :], fade_curve[i])
        
        # BOTTOM edge: content-aware fade
        for i in range(min(fade_pixels, height // 2)):
            row_idx = -(i+1)
            row_has_ink = np.any(has_ink[row_idx, :])
            if row_has_ink:
                alpha[row_idx, :] = np.minimum(alpha[row_idx, :], 0.9 + 0.1 * fade_curve[fade_pixels-1-i])
            else:
                alpha[row_idx, :] = np.minimum(alpha[row_idx, :], fade_curve[fade_pixels-1-i])
    
    # LEFT edge: content-aware fade
    if width > fade_pixels * 2:
        for i in range(min(fade_pixels, width // 2)):
            col_has_ink = np.any(has_ink[:, i])
            if col_has_ink:
                alpha[:, i] = np.minimum(alpha[:, i], 0.9 + 0.1 * fade_curve[i])
            else:
                alpha[:, i] = np.minimum(alpha[:, i], fade_curve[i])
        
        # RIGHT edge: content-aware fade
        for i in range(min(fade_pixels, width // 2)):
            col_idx = -(i+1)
            col_has_ink = np.any(has_ink[:, col_idx])
            if col_has_ink:
                alpha[:, col_idx] = np.minimum(alpha[:, col_idx], 0.9 + 0.1 * fade_curve[fade_pixels-1-i])
            else:
                alpha[:, col_idx] = np.minimum(alpha[:, col_idx], fade_curve[fade_pixels-1-i])
    
    return alpha


def recognize_line(line_image, recognizer, charset):
    """
    Run HTR recognition pada line image.
    Input: line_image sudah dalam format (W, H, 1)
    """
    # Ensure correct input format: (1, W, H, 1)
    if len(line_image.shape) == 3:
        input_tensor = line_image[np.newaxis, :, :, :]
    else:
        input_tensor = line_image
    
    # Run recognition
    logits = recognizer(input_tensor, training=False)
    
    # Decode text
    text = safe_ctc_decode(logits.numpy(), charset)
    
    return text


class HybridLaypaPipeline:
    """Pipeline hybrid: Laypa segmentation + GAN restoration + HTR recognition."""
    
    def __init__(self, generator, recognizer, charset, batch_size=8):
        self.generator = generator
        self.recognizer = recognizer
        self.charset = charset
        self.batch_size = batch_size
    
    def process_document(self, document_path, pagexml_path, output_dir, save_intermediates=True):
        """
        Process satu dokumen dengan Laypa baselines.
        
        Args:
            document_path: Path ke original document image
            pagexml_path: Path ke PageXML dari Laypa
            output_dir: Output directory
            save_intermediates: Save intermediate results
        """
        # Load dokumen
        image = cv2.imread(str(document_path))
        if image is None:
            raise ValueError(f"Cannot load image: {document_path}")
        
        doc_name = Path(document_path).stem
        print(f"\n📄 Processing: {doc_name}")
        print(f"   Original size: {image.shape[1]}x{image.shape[0]}")
        
        # Parse PageXML baselines
        print(f"   📖 Parsing Laypa baselines from PageXML...")
        baselines = parse_pagexml_baselines(pagexml_path)
        
        # Filter noise: Remove baselines yang terlalu kecil (likely noise/artifacts)
        MIN_LINE_WIDTH = 50  # Minimum 50 pixels width
        valid_baselines = []
        for baseline in baselines:
            # Gunakan bbox untuk cek width
            bbox = baseline.get('bbox')
            if not bbox:
                continue
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            if width >= MIN_LINE_WIDTH:
                valid_baselines.append(baseline)
        
        print(f"   ✅ Found {len(valid_baselines)} baselines from Laypa (filtered {len(baselines) - len(valid_baselines)} noise)")
        baselines = valid_baselines
        
        if len(baselines) == 0:
            print(f"   ⚠️  No valid baselines after filtering!")
            return None
        
        # Extract line images
        print(f"   ✂️  Extracting line images from baselines...")
        line_images = []
        line_bboxes_original = []  # For precise placement
        line_bboxes_padded = []    # For processing dimensions
        baseline_polylines = []    # For alignment enforcement (NEW)
        
        for baseline_info in baselines:
            line_img, orig_bbox, padded_bbox = extract_line_image_from_baseline(image, baseline_info, padding=10)
            line_images.append(line_img)
            line_bboxes_original.append(orig_bbox)
            line_bboxes_padded.append(padded_bbox)
            # Store baseline polyline untuk alignment
            baseline_polylines.append(baseline_info.get('baseline_points', []))
        
        print(f"   ✅ Extracted {len(line_images)} line images")
        
        # Preprocess untuk GAN
        print(f"   🔧 Preprocessing lines for GAN...")
        preprocessed = []
        for line_img in line_images:
            prep = preprocess_line_for_gan(line_img)
            preprocessed.append(prep[0])  # Remove batch dimension untuk stacking
        
        # Batch processing - GAN restoration
        print(f"   🎨 Restoring lines with GAN...")
        restored_lines = []
        
        for i in tqdm(range(0, len(preprocessed), self.batch_size), desc="   Restoring"):
            batch = preprocessed[i:i+self.batch_size]
            batch_array = np.array(batch)
            
            # GAN restoration
            generated_batch = self.generator(batch_array, training=False)
            
            for generated in generated_batch:
                restored_lines.append(generated.numpy())
        
        print(f"   ✅ Restored {len(restored_lines)} lines")
        
        # === NEW: SAVE INDIVIDUAL LINE IMAGES ===
        if save_intermediates:
            print(f"   💾 Saving individual line images...")
            lines_dir = Path(output_dir) / "individual_lines"
            lines_dir.mkdir(parents=True, exist_ok=True)
            
            # Save side-by-side comparison untuk setiap line
            for idx, (orig_line, restored_line, bbox_orig, bbox_pad) in enumerate(
                zip(line_images, restored_lines, line_bboxes_original, line_bboxes_padded), 
                start=1
            ):
                # Original line (extracted from document)
                orig_h, orig_w = orig_line.shape[:2]
                
                # Restored line (from GAN, needs postprocessing)
                x_min_pad, y_min_pad, x_max_pad, y_max_pad = bbox_pad
                line_h_pad = y_max_pad - y_min_pad
                line_w_pad = x_max_pad - x_min_pad
                
                # Postprocess restored line
                restored_vis = postprocess_gan_output(restored_line, line_h_pad, line_w_pad)
                
                # Resize original untuk matching height (for side-by-side)
                target_height = 128  # Standard display height
                scale = target_height / orig_h
                orig_resized = cv2.resize(orig_line, (int(orig_w * scale), target_height))
                
                # Resize restored untuk matching
                rest_h, rest_w = restored_vis.shape[:2]
                rest_scale = target_height / rest_h
                rest_resized = cv2.resize(restored_vis, (int(rest_w * rest_scale), target_height))
                
                # Convert grayscale to color jika perlu
                if len(orig_resized.shape) == 2:
                    orig_resized = cv2.cvtColor(orig_resized, cv2.COLOR_GRAY2BGR)
                if len(rest_resized.shape) == 2:
                    rest_resized = cv2.cvtColor(rest_resized, cv2.COLOR_GRAY2BGR)
                
                # Create side-by-side comparison
                gap = 20  # Pixels between images
                gap_img = np.ones((target_height, gap, 3), dtype=np.uint8) * 200  # Gray separator
                
                comparison = np.hstack([orig_resized, gap_img, rest_resized])
                
                # Add labels
                label_orig = "ORIGINAL (Degraded)"
                label_rest = "RESTORED (GAN Enhanced)"
                
                cv2.putText(comparison, label_orig, (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(comparison, label_rest, (orig_resized.shape[1] + gap + 10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                
                # Save comparison
                line_filename = f"line_{idx:03d}_comparison.png"
                line_path = lines_dir / line_filename
                cv2.imwrite(str(line_path), comparison)
                
                # Also save individual files
                cv2.imwrite(str(lines_dir / f"line_{idx:03d}_original.png"), orig_line)
                cv2.imwrite(str(lines_dir / f"line_{idx:03d}_restored.png"), restored_vis)
            
            print(f"   ✅ Saved {len(line_images)} line comparisons to: {lines_dir}")
        
        # HTR recognition
        print(f"   🔤 Running HTR recognition...")
        recognized_texts = []
        
        for restored_line in tqdm(restored_lines, desc="   Recognizing"):
            # Add batch dimension
            line_input = restored_line[np.newaxis, :, :, :]
            text = recognize_line(line_input, self.recognizer, self.charset)
            recognized_texts.append(text)
        
        print(f"   ✅ Recognized {len(recognized_texts)} lines")
        
        # Reconstruct document with baseline alignment enforcement
        print(f"   🔨 Reconstructing document...")
        restored_document = self.reconstruct_document(
            image, line_bboxes_original, line_bboxes_padded, restored_lines,
            baseline_polylines=baseline_polylines  # Pass baseline info (NEW)
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
            'num_lines': len(baselines),
            'segmentation_method': 'Laypa (Loghi)',
            'restoration_method': 'GAN-HTR',
            'recognition_method': 'HTR Transformer',
            'processing_date': datetime.now().isoformat(),
            'lines': []
        }
        
        for i, (baseline_info, bbox, text) in enumerate(zip(baselines, line_bboxes_original, recognized_texts)):
            x_min, y_min, x_max, y_max = bbox
            metadata['lines'].append({
                'line_number': i + 1,
                'line_id': baseline_info['line_id'],
                'bbox': {'x': int(x_min), 'y': int(y_min), 'width': int(x_max-x_min), 'height': int(y_max-y_min)},
                'text': text,
                'text_length': len(text)
            })
        
        metadata_path = output_path / f"{doc_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save intermediates
        if save_intermediates:
            lines_dir = output_path / "lines"
            lines_dir.mkdir(exist_ok=True)
            
            for i, (orig_line, rest_line, text) in enumerate(zip(
                line_images, restored_lines, recognized_texts
            )):
                # Postprocess restored line
                rest_line_vis = postprocess_gan_output(
                    rest_line,
                    orig_line.shape[0],
                    orig_line.shape[1]
                )
                
                # Save comparison (resize to match height if needed)
                if orig_line.shape[0] != rest_line_vis.shape[0]:
                    # Resize restored to match original height
                    rest_line_vis = cv2.resize(
                        rest_line_vis,
                        (rest_line_vis.shape[1], orig_line.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                
                comparison = np.hstack([orig_line, rest_line_vis])
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
    
    def reconstruct_document(self, original_image, line_bboxes_original, line_bboxes_padded, restored_lines, 
                           baseline_polylines=None):
        """
        Reconstruct full document dari restored lines.
        
        v7 BASELINE ALIGNMENT ENFORCEMENT:
        - Track TWO bboxes: original (for placement) & padded (for processing)
        - Enforce baseline straightness untuk prevent zigzag
        - Apply rotation correction jika baseline deviate dari horizontal
        - Result: Perfect horizontal alignment, no zigzag effect
        
        Args:
            original_image: Original document image
            line_bboxes_original: List of (x_min, y_min, x_max, y_max) - ORIGINAL baseline bboxes
            line_bboxes_padded: List of (x_min, y_min, x_max, y_max) - PADDED bboxes for GAN processing
            restored_lines: List of restored line images (1024x128 from GAN)
            baseline_polylines: List of baseline points [(x1,y1), (x2,y2), ...] untuk alignment (NEW)
        
        Returns:
            Reconstructed document image
        """
        # Initialize baseline enforcer
        baseline_enforcer = BaselineAlignmentEnforcer()
        h, w = original_image.shape[:2]
        
        # FIXED: START WITH ORIGINAL IMAGE (preserve margins, background, structure!)
        # Only replace text line regions, NOT entire document
        canvas = original_image.copy()
        
        # Render lines with precise alignment + baseline enforcement
        for i, (bbox_orig, bbox_padded, restored_line) in enumerate(zip(line_bboxes_original, line_bboxes_padded, restored_lines)):
            # Original bbox for placement (NO padding)
            x_min_orig, y_min_orig, x_max_orig, y_max_orig = bbox_orig
            line_h_orig = y_max_orig - y_min_orig
            line_w_orig = x_max_orig - x_min_orig
            
            # Padded bbox for dimensions (WITH padding, matches restored line)
            x_min_pad, y_min_pad, x_max_pad, y_max_pad = bbox_padded
            line_h_pad = y_max_pad - y_min_pad
            line_w_pad = x_max_pad - x_min_pad
            
            # Postprocess restored line to PADDED dimensions
            # NOTE: postprocess may return different height (aspect ratio preservation)
            line_vis_padded = postprocess_gan_output(
                restored_line,
                line_h_pad,
                line_w_pad
            )
            
            # Handle potential height mismatch dari aspect ratio preservation
            actual_h_pad, actual_w_pad = line_vis_padded.shape[:2]
            
            # Calculate padding offset untuk crop ke original size
            pad_left = x_min_orig - x_min_pad
            pad_top = y_min_orig - y_min_pad
            
            # Adjust pad_top jika height berbeda (center vertically)
            if actual_h_pad != line_h_pad:
                height_diff = actual_h_pad - line_h_pad
                pad_top = max(0, pad_top + height_diff // 2)
            
            # Safe crop dengan boundary check
            crop_y_start = max(0, pad_top)
            crop_y_end = min(actual_h_pad, pad_top + line_h_orig)
            crop_x_start = max(0, pad_left)
            crop_x_end = min(actual_w_pad, pad_left + line_w_orig)
            
            line_vis_cropped = line_vis_padded[
                crop_y_start:crop_y_end,
                crop_x_start:crop_x_end
            ]
            
            # Resize jika crop size tidak match (edge case)
            if line_vis_cropped.shape[0] != line_h_orig or line_vis_cropped.shape[1] != line_w_orig:
                line_vis_original = cv2.resize(line_vis_cropped, (line_w_orig, line_h_orig), 
                                               interpolation=cv2.INTER_CUBIC)
            else:
                line_vis_original = line_vis_cropped
            
            # Convert grayscale to color jika perlu
            if len(canvas.shape) == 3 and len(line_vis_original.shape) == 2:
                line_vis_original = cv2.cvtColor(line_vis_original, cv2.COLOR_GRAY2BGR)
            
            # === NEW: BASELINE ALIGNMENT ENFORCEMENT ===
            # Enforce horizontal straightness untuk prevent zigzag
            if baseline_polylines is not None and i < len(baseline_polylines):
                baseline_points = baseline_polylines[i]
                if len(baseline_points) > 0:
                    # Apply baseline straightening
                    line_vis_original = baseline_enforcer.enforce_horizontal_alignment(
                        baseline_points,
                        line_vis_original,
                        bbox_orig
                    )
            
            # Create content-aware fade mask (v5)
            # Deteksi ink di edge regions, preserve fine details
            fade_pixels = max(10, min(20, line_h_orig // 5))
            alpha = create_content_aware_fade_mask(line_vis_original, fade_pixels=fade_pixels, ink_threshold=240)
            
            # Get canvas region (EXACT original bbox, NO padding)
            canvas_region = canvas[y_min_orig:y_max_orig, x_min_orig:x_max_orig].astype(np.float32)
            line_float = line_vis_original.astype(np.float32)
            
            # FIXED: Use ORIGINAL background instead of white
            # This preserves document structure, margins, and natural appearance
            background = canvas_region.copy()
            
            # Expand alpha untuk multi-channel
            if len(canvas.shape) == 3:
                alpha_3d = alpha[:, :, np.newaxis]
            else:
                alpha_3d = alpha
            
            # Blend: restored line → original background (preserve document structure!)
            # alpha=1.0 → show restored line fully
            # alpha=0.0 → show original background
            blended = line_float * alpha_3d + background * (1.0 - alpha_3d)
            
            # Place on canvas with PRECISE original coordinates (NO padding offset)
            canvas[y_min_orig:y_max_orig, x_min_orig:x_max_orig] = blended.astype(np.uint8)
        
        # REMOVED: Line numbers (preserve clean document appearance)
        # Line numbers dapat di-enable untuk debugging jika perlu
        # Untuk production restoration: NO annotations, keep document natural!
        
        return canvas


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # Enable GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠️  Memory growth setting failed: {e}")
    
    print("=" * 80)
    print("HYBRID LAYPA + GAN-HTR INFERENCE PIPELINE")
    print("=" * 80)
    print("Segmentation: Laypa (Loghi) - 60× better detection")
    print("Restoration:  GAN-HTR - Degradation removal")
    print("Recognition:  HTR Transformer - Text extraction")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Run Laypa segmentation
    loghi_output = output_dir / "loghi_segmentation"
    pagexml_dir = run_loghi_segmentation(args.input_dir, str(loghi_output))
    
    # Load charset
    print(f"\n[2/5] Loading charset from {args.charset_path}...")
    charset = read_charlist(args.charset_path)
    charset_size = len(charset)
    print(f"  ✅ Charset loaded: {charset_size} characters")
    
    # Load models
    print(f"\n[3/5] Loading GAN-HTR models...")
    print(f"  📦 Generator: {args.gan_checkpoint}")
    print(f"  📦 HTR: {args.htr_weights}")
    
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
    
    if os.path.isdir(args.gan_checkpoint):
        checkpoint_path = tf.train.latest_checkpoint(args.gan_checkpoint)
    else:
        checkpoint_path = args.gan_checkpoint.replace('.index', '').replace('.data-00000-of-00001', '')
    
    if checkpoint_path and os.path.exists(checkpoint_path + '.index'):
        status = checkpoint.restore(checkpoint_path)
        status.expect_partial()
        print(f"  ✅ Generator loaded: {generator.count_params():,} parameters")
    else:
        print(f"  ❌ No checkpoint found at {args.gan_checkpoint}")
        return
    
    if os.path.exists(args.htr_weights):
        recognizer.load_weights(args.htr_weights)
        print(f"  ✅ HTR recognizer loaded: {recognizer.count_params():,} parameters")
    else:
        print(f"  ❌ HTR weights not found: {args.htr_weights}")
        return
    
    # Create hybrid pipeline
    print(f"\n[4/5] Creating hybrid pipeline...")
    pipeline = HybridLaypaPipeline(
        generator=generator,
        recognizer=recognizer,
        charset=charset,
        batch_size=args.batch_size
    )
    print(f"  ✅ Pipeline ready (batch_size={args.batch_size})")
    
    # Get input documents
    input_path = Path(args.input_dir)
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
            # Find corresponding PageXML
            pagexml_path = pagexml_dir / f"{doc_path.stem}.xml"
            
            if not pagexml_path.exists():
                print(f"\n⚠️  PageXML not found for {doc_path.name}: {pagexml_path}")
                results_summary.append({
                    'document': doc_path.name,
                    'status': 'failed',
                    'error': 'PageXML not found'
                })
                continue
            
            result = pipeline.process_document(
                doc_path,
                pagexml_path,
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
            import traceback
            traceback.print_exc()
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
            'segmentation': 'Laypa (Loghi)',
            'restoration': 'GAN-HTR',
            'recognition': 'HTR Transformer',
            'gan_checkpoint': args.gan_checkpoint,
            'htr_weights': args.htr_weights,
            'batch_size': args.batch_size
        },
        'results': results_summary
    }
    
    summary_path = output_dir / 'hybrid_processing_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("HYBRID PIPELINE COMPLETE")
    print("=" * 80)
    print(f"✅ Successfully processed: {summary['processing_info']['documents_success']}/{len(document_files)}")
    print(f"❌ Failed: {summary['processing_info']['documents_failed']}/{len(document_files)}")
    print(f"⏱️  Total time: {processing_time:.2f}s")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📄 Summary: {summary_path}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hybrid Laypa + GAN-HTR Inference Pipeline'
    )
    
    # Input/Output
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing degraded document images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save restored documents and text')
    
    # Model paths
    parser.add_argument('--gan_checkpoint', type=str,
                       default='dual_modal_gan/outputs/checkpoints_fp32_smoke_test',
                       help='Path to GAN-HTR checkpoint')
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
    
    # Output options
    parser.add_argument('--save_intermediates', action='store_true', default=True,
                       help='Save intermediate line images')
    parser.add_argument('--no_intermediates', dest='save_intermediates',
                       action='store_false',
                       help='Do not save intermediate line images')
    
    args = parser.parse_args()
    main(args)
