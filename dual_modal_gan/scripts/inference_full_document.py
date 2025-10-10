#!/usr/bin/env python3
"""
Inference Pipeline - FULL DOCUMENT MODE
Untuk dokumen continuous scan seperti ANRI (tidak ada pemisahan baris)
Memproses dokumen sebagai tiles dengan overlap untuk menghindari seam artifacts.
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import time
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import generator builder
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from models.generator import unet

# Constants
IMG_HEIGHT = 128
IMG_WIDTH = 1024
TILE_HEIGHT = 128
TILE_WIDTH = 1024
OVERLAP = 32  # Overlap untuk blending
BATCH_SIZE = 4  # Process multiple tiles at once


class FullDocumentRestorer:
    def __init__(self, generator_checkpoint, gpu_id=1):
        """
        Initialize full document restoration pipeline.
        
        Args:
            generator_checkpoint: Path to trained generator checkpoint
            gpu_id: GPU to use (default 1)
        """
        self.generator_checkpoint = generator_checkpoint
        self.gpu_id = gpu_id
        
        # Configure GPU
        self._configure_gpu()
        
        # Load generator
        self.generator = self._load_generator()
    
    def _configure_gpu(self):
        """Configure GPU dengan memory growth."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set visible GPU
                if self.gpu_id is not None:
                    tf.config.set_visible_devices(gpus[self.gpu_id], 'GPU')
                    print(f"✓ Using GPU {self.gpu_id}: {gpus[self.gpu_id].name}")
                else:
                    print(f"✓ Using all {len(gpus)} GPUs")
            except RuntimeError as e:
                print(f"⚠️  GPU configuration error: {e}")
        else:
            print("⚠️  No GPUs found, using CPU")
    
    def _load_generator(self):
        """Load trained generator model."""
        print(f"\n📦 Loading generator from: {self.generator_checkpoint}")
        
        # Create generator architecture
        generator = unet(input_size=(IMG_WIDTH, IMG_HEIGHT, 1))
        
        # Load weights
        checkpoint = tf.train.Checkpoint(generator=generator)
        checkpoint.restore(self.generator_checkpoint).expect_partial()
        
        print(f"✓ Generator loaded successfully ({generator.count_params():,} parameters)")
        return generator
    
    def preprocess_tile(self, tile):
        """
        Preprocess tile untuk input model.
        
        Args:
            tile: numpy array (H, W) atau (H, W, 3)
        
        Returns:
            Preprocessed tile (1, 1024, 128, 1) range [-1, 1]
            NOTE: Model expects (batch, width, height, channels) NOT (batch, height, width, channels)!
        """
        # Convert to grayscale if needed
        if len(tile.shape) == 3:
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size (H=128, W=1024)
        if tile.shape != (TILE_HEIGHT, TILE_WIDTH):
            tile = cv2.resize(tile, (TILE_WIDTH, TILE_HEIGHT), interpolation=cv2.INTER_AREA)
        
        # Transpose to (W, H) untuk match model input
        # Model expects (batch, 1024, 128, 1)
        tile = tile.T  # (1024, 128)
        
        # Normalize to [0, 1] - Model uses SIGMOID output!
        tile = tile.astype(np.float32) / 255.0
        
        # Add channel dimension
        tile = np.expand_dims(tile, axis=-1)  # (1024, 128, 1)
        
        # Add batch dimension
        tile = np.expand_dims(tile, axis=0)  # (1, 1024, 128, 1)
        
        return tile
    
    def postprocess_tile(self, tile):
        """
        Postprocess output dari model.
        
        Args:
            tile: numpy array (1, 1024, 128, 1) range [-1, 1]
        
        Returns:
            numpy array (128, 1024) range [0, 255]
        """
        # Remove batch and channel dimensions
        tile = tile[0, :, :, 0]  # (1024, 128)
        
        # Transpose back to (H, W)
        tile = tile.T  # (128, 1024)
        
        # Denormalize from [0, 1] to [0, 255] - Model uses SIGMOID!
        tile = (tile * 255.0).astype(np.uint8)
        
        return tile
    
    def create_tiles(self, image):
        """
        Split dokumen menjadi overlapping tiles.
        
        Args:
            image: numpy array (H, W) atau (H, W, 3)
        
        Returns:
            List of (tile, x_start, y_start, original_h, original_w)
        """
        doc_height, doc_width = image.shape[:2]
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        tiles = []
        
        # Calculate stride (tile size - overlap)
        stride_h = TILE_HEIGHT - OVERLAP
        stride_w = TILE_WIDTH - OVERLAP
        
        y = 0
        while y < doc_height:
            x = 0
            while x < doc_width:
                # Extract tile
                y_end = min(y + TILE_HEIGHT, doc_height)
                x_end = min(x + TILE_WIDTH, doc_width)
                
                tile = gray_image[y:y_end, x:x_end]
                original_h, original_w = tile.shape
                
                # Pad tile if needed
                if original_h < TILE_HEIGHT or original_w < TILE_WIDTH:
                    padded = np.ones((TILE_HEIGHT, TILE_WIDTH), dtype=np.uint8) * 255
                    padded[:original_h, :original_w] = tile
                    tile = padded
                
                tiles.append({
                    'tile': tile,
                    'x': x,
                    'y': y,
                    'original_h': original_h,
                    'original_w': original_w
                })
                
                x += stride_w
                if x >= doc_width:
                    break
            
            y += stride_h
            if y >= doc_height:
                break
        
        return tiles
    
    def restore_tiles(self, tiles):
        """
        Restore semua tiles menggunakan generator.
        Process dalam batch untuk efisiensi.
        
        Args:
            tiles: List of tile dictionaries
        
        Returns:
            List of restored tiles
        """
        num_tiles = len(tiles)
        restored = []
        
        print(f"  Processing {num_tiles} tiles in batches of {BATCH_SIZE}...")
        
        for i in range(0, num_tiles, BATCH_SIZE):
            batch_tiles = tiles[i:i+BATCH_SIZE]
            batch_data = []
            
            # Preprocess batch
            for tile_info in batch_tiles:
                preprocessed = self.preprocess_tile(tile_info['tile'])
                batch_data.append(preprocessed[0])  # Remove batch dim
            
            # Stack into batch
            batch_data = np.stack(batch_data, axis=0)
            
            # Restore batch
            restored_batch = self.generator.predict(batch_data, verbose=0)
            
            # Postprocess batch
            for j, tile_info in enumerate(batch_tiles):
                restored_tile = self.postprocess_tile(restored_batch[j:j+1])
                
                # Crop to original size
                original_h = tile_info['original_h']
                original_w = tile_info['original_w']
                restored_tile = restored_tile[:original_h, :original_w]
                
                restored.append({
                    'tile': restored_tile,
                    'x': tile_info['x'],
                    'y': tile_info['y'],
                    'original_h': original_h,
                    'original_w': original_w
                })
            
            if (i + BATCH_SIZE) % 40 == 0 or (i + BATCH_SIZE) >= num_tiles:
                print(f"    Progress: {min(i + BATCH_SIZE, num_tiles)}/{num_tiles} tiles")
        
        return restored
    
    def blend_tiles(self, tiles, doc_height, doc_width):
        """
        Blend restored tiles kembali menjadi full document.
        Menggunakan alpha blending di overlap regions.
        
        Args:
            tiles: List of restored tile dictionaries
            doc_height: Original document height
            doc_width: Original document width
        
        Returns:
            Restored full document (H, W) uint8
        """
        # Create output canvas
        restored_doc = np.ones((doc_height, doc_width), dtype=np.float32) * 255
        weight_map = np.zeros((doc_height, doc_width), dtype=np.float32)
        
        # Create alpha blending kernel
        alpha_h = np.ones(TILE_HEIGHT, dtype=np.float32)
        alpha_w = np.ones(TILE_WIDTH, dtype=np.float32)
        
        # Fade in/out at edges
        fade_size = OVERLAP
        alpha_h[:fade_size] = np.linspace(0, 1, fade_size)
        alpha_h[-fade_size:] = np.linspace(1, 0, fade_size)
        alpha_w[:fade_size] = np.linspace(0, 1, fade_size)
        alpha_w[-fade_size:] = np.linspace(1, 0, fade_size)
        
        # Create 2D alpha map
        alpha_map = np.outer(alpha_h, alpha_w)
        
        # Blend all tiles
        for tile_info in tiles:
            tile = tile_info['tile'].astype(np.float32)
            x, y = tile_info['x'], tile_info['y']
            h, w = tile_info['original_h'], tile_info['original_w']
            
            # Get alpha for this tile
            tile_alpha = alpha_map[:h, :w]
            
            # Accumulate weighted sum
            restored_doc[y:y+h, x:x+w] += tile * tile_alpha
            weight_map[y:y+h, x:x+w] += tile_alpha
        
        # Normalize by weights
        weight_map = np.maximum(weight_map, 1e-6)  # Avoid division by zero
        restored_doc = restored_doc / weight_map
        
        # Convert to uint8
        restored_doc = np.clip(restored_doc, 0, 255).astype(np.uint8)
        
        return restored_doc
    
    def process_document(self, image_path, output_dir):
        """
        Process full document.
        
        Args:
            image_path: Path to input image
            output_dir: Directory untuk output
        
        Returns:
            Dictionary dengan hasil processing
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"Mode: FULL DOCUMENT RESTORATION")
        print(f"{'='*60}")
        
        # Load image
        doc_img = cv2.imread(image_path)
        if doc_img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        doc_height, doc_width = doc_img.shape[:2]
        print(f"Document size: {doc_width}x{doc_height} pixels")
        
        # Create output directory
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        doc_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(doc_output_dir, exist_ok=True)
        
        # Step 1: Create tiles
        print(f"\n[1/3] Creating overlapping tiles ({TILE_WIDTH}x{TILE_HEIGHT}, overlap={OVERLAP}px)...")
        tiles = self.create_tiles(doc_img)
        print(f"  Created {len(tiles)} tiles")
        
        # Step 2: Restore tiles
        print(f"\n[2/3] Restoring tiles with GAN...")
        restored_tiles = self.restore_tiles(tiles)
        
        # Step 3: Blend tiles
        print(f"\n[3/3] Blending tiles into full document...")
        restored_doc = self.blend_tiles(restored_tiles, doc_height, doc_width)
        
        # Save restored document
        restored_path = os.path.join(doc_output_dir, f"{base_name}_restored.png")
        cv2.imwrite(restored_path, restored_doc)
        print(f"  Saved: {restored_path}")
        
        # Create comparison image
        print(f"  Creating comparison image...")
        
        # Convert original to grayscale for comparison
        if len(doc_img.shape) == 3:
            original_gray = cv2.cvtColor(doc_img, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = doc_img
        
        comparison = np.hstack([original_gray, restored_doc])
        comparison_path = os.path.join(doc_output_dir, f"{base_name}_comparison.png")
        cv2.imwrite(comparison_path, comparison)
        print(f"  Saved: {comparison_path}")
        
        # Calculate statistics
        processing_time = time.time() - start_time
        
        # Analyze output quality
        white_pixels = np.sum(restored_doc > 250)
        total_pixels = restored_doc.size
        white_percentage = white_pixels / total_pixels * 100
        mean_brightness = restored_doc.mean()
        
        # Create metadata
        result = {
            'input_path': image_path,
            'output_dir': doc_output_dir,
            'restored_document': restored_path,
            'comparison_image': comparison_path,
            'mode': 'full_document',
            'document_size': f"{doc_width}x{doc_height}",
            'num_tiles': len(tiles),
            'tile_size': f"{TILE_WIDTH}x{TILE_HEIGHT}",
            'overlap': OVERLAP,
            'processing_time': processing_time,
            'quality_metrics': {
                'mean_brightness': float(mean_brightness),
                'white_percentage': float(white_percentage)
            }
        }
        
        # Save metadata
        metadata_path = os.path.join(doc_output_dir, f"{base_name}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"✅ Processing complete!")
        print(f"   Total time: {processing_time:.2f}s")
        print(f"   Tiles processed: {len(tiles)}")
        print(f"   Mean brightness: {mean_brightness:.1f}")
        print(f"   White pixels: {white_percentage:.1f}%")
        print(f"   Output: {restored_path}")
        print(f"{'='*60}\n")
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Full Document Restoration Pipeline')
    parser.add_argument('--input', '-i', required=True, help='Input image or directory')
    parser.add_argument('--output', '-o', default='outputs/FullDocumentRestoration',
                        help='Output directory (default: outputs/FullDocumentRestoration)')
    parser.add_argument('--checkpoint', '-c',
                        default='dual_modal_gan/outputs/checkpoints_fp32_smoke_test/best_model-12',
                        help='Generator checkpoint path')
    parser.add_argument('--gpu', type=int, default=1,
                        help='GPU device ID to use (default: 1)')
    
    args = parser.parse_args()
    
    # Create restorer
    restorer = FullDocumentRestorer(
        generator_checkpoint=args.checkpoint,
        gpu_id=args.gpu
    )
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        print(f"\n📄 Processing single document...")
        result = restorer.process_document(str(input_path), args.output)
        print(f"\n✅ Done! Output saved to: {result['restored_document']}")
    
    elif input_path.is_dir():
        # Directory of images
        print(f"\n📁 Processing directory: {input_path}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        print(f"Found {len(image_files)} images")
        
        results = []
        for i, img_file in enumerate(image_files, 1):
            print(f"\n{'='*60}")
            print(f"Document {i}/{len(image_files)}")
            print(f"{'='*60}")
            
            try:
                result = restorer.process_document(str(img_file), args.output)
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing {img_file.name}: {e}")
                continue
        
        # Summary
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total processed: {len(results)}/{len(image_files)}")
        print(f"Output directory: {args.output}")
        
        # Calculate average metrics
        if results:
            avg_time = np.mean([r['processing_time'] for r in results])
            avg_brightness = np.mean([r['quality_metrics']['mean_brightness'] for r in results])
            avg_white = np.mean([r['quality_metrics']['white_percentage'] for r in results])
            
            print(f"\nAverage metrics:")
            print(f"  Processing time: {avg_time:.2f}s per document")
            print(f"  Mean brightness: {avg_brightness:.1f}")
            print(f"  White percentage: {avg_white:.1f}%")
    
    else:
        print(f"❌ Error: {input_path} is not a file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
