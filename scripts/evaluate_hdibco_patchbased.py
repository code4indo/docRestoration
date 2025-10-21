#!/usr/bin/env python3
"""
Patch-Based Evaluation on H-DIBCO 2016 Dataset
Uses DocumentAwarePreprocessor to extract text lines and preserve aspect ratio
Expected improvement: +13-23 dB over full document resize approach
"""

import os
import sys
import numpy as np
import cv2
import json
from datetime import datetime
from pathlib import Path
import tensorflow as tf
import logging

# ✅ Use GPU 1 (GPU 0 is being used for training)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from scripts.enhanced_preprocessing import DocumentAwarePreprocessor

def load_hdibco2016_dataset(dataset_root="dibco_datasets"):
    """Load H-DIBCO 2016 dataset for evaluation"""
    
    logger.info("📂 Loading H-DIBCO 2016 dataset...")
    
    # Paths
    original_path = Path(dataset_root) / "DIPCO2016_dataset"
    gt_path = Path(dataset_root) / "DIPCO2016_Dataset_GT"
    
    if not original_path.exists() or not gt_path.exists():
        logger.error("❌ H-DIBCO 2016 dataset not found!")
        logger.error(f"   Expected: {original_path}")
        logger.error(f"   Expected: {gt_path}")
        return None
    
    # Load image pairs
    image_pairs = []
    
    for i in range(1, 11):  # Images 1-10
        original_file = original_path / f"{i}.bmp"
        gt_file = gt_path / f"{i}_gt.bmp"
        
        if original_file.exists() and gt_file.exists():
            # Load images
            original = cv2.imread(str(original_file), cv2.IMREAD_GRAYSCALE)
            ground_truth = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
            
            if original is None or ground_truth is None:
                logger.warning(f"⚠️  Failed to load image {i}")
                continue
            
            # Normalize to [0, 1]
            original = original.astype(np.float32) / 255.0
            ground_truth = ground_truth.astype(np.float32) / 255.0
            
            # Binarize ground truth (ensure binary values)
            ground_truth = (ground_truth > 0.5).astype(np.float32)
            
            image_pairs.append({
                'id': i,
                'original': original,
                'ground_truth': ground_truth,
                'shape': original.shape,
                'filepath': str(original_file)
            })
    
    logger.info(f"✅ Loaded {len(image_pairs)} image pairs from H-DIBCO 2016\n")
    return image_pairs

def preprocess_text_line(line, target_size=(1024, 128)):
    """
    Preprocess extracted text line for model input
    Matches training pipeline exactly
    """
    # Resize: cv2.resize((width, height)) returns numpy (height, width)
    resized = cv2.resize(line, (target_size[0], target_size[1]), interpolation=cv2.INTER_AREA)
    # Result: numpy array (H=128, W=1024)
    
    # Add channel dimension
    resized = resized[..., np.newaxis]  # (128, 1024, 1)
    
    # Transpose to match training: [1, 0, 2] swaps H and W
    resized = np.transpose(resized, (1, 0, 2))  # (1024, 128, 1)
    
    # Add batch dimension
    processed = resized[np.newaxis, ...]  # (1, 1024, 128, 1)
    
    # Normalize to [-1,1] range
    processed = (processed * 2.0) - 1.0
    
    return processed.astype(np.float32)

def calculate_psnr(enhanced, ground_truth):
    """Calculate PSNR between enhanced and ground truth images"""
    mse = np.mean((enhanced - ground_truth) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(enhanced, ground_truth):
    """Calculate SSIM between enhanced and ground truth images"""
    try:
        from skimage.metrics import structural_similarity as ssim
        ssim_score = ssim(enhanced, ground_truth, data_range=1.0)
        return ssim_score
    except ImportError:
        logger.warning("⚠️  scikit-image not available, skipping SSIM")
        return None

def evaluate_patchbased(model, image_pairs, save_results=True, output_dir="hdibco2016_patchbased"):
    """
    Evaluate model using patch-based approach with text line extraction
    Expected to significantly improve PSNR compared to full document resize
    """
    
    logger.info("🔍 Patch-Based Evaluation on H-DIBCO 2016")
    logger.info("=" * 60)
    logger.info(f"   Strategy: Extract text lines → Process individually")
    logger.info(f"   Expected: +13-23 dB improvement over full resize\n")
    
    # Create preprocessor
    preprocessor = DocumentAwarePreprocessor(target_size=(128, 1024), logger=logger)
    
    # Create output directory
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
    
    results = []
    
    for idx, sample in enumerate(image_pairs):
        logger.info(f"📄 Processing image {sample['id']} ({idx+1}/{len(image_pairs)})")
        logger.info(f"   Original size: {sample['shape']}")
        
        original = sample['original']
        ground_truth = sample['ground_truth']
        
        try:
            # Extract text lines from document
            text_lines = preprocessor.extract_text_lines(original)
            gt_lines = preprocessor.extract_text_lines(ground_truth)
            
            if not text_lines:
                logger.warning(f"   ⚠️  No text lines extracted, using full document")
                text_lines = [original]
                gt_lines = [ground_truth]
            
            logger.info(f"   ✅ Extracted {len(text_lines)} text lines")
            
            # Process each line
            enhanced_lines = []
            psnr_per_line = []
            ssim_per_line = []
            
            for i, (line, gt_line) in enumerate(zip(text_lines, gt_lines)):
                # Preprocess line
                line_input = preprocess_text_line(line)
                
                # Generate enhanced line
                enhanced_tensor = model(line_input, training=False)
                enhanced_line = enhanced_tensor.numpy()[0, ..., 0]
                
                # Denormalize from [-1,1] to [0,1]
                enhanced_line = (enhanced_line + 1.0) / 2.0
                enhanced_line = np.clip(enhanced_line, 0, 1)
                
                # Transpose back to (H, W) for comparison
                enhanced_line = np.transpose(enhanced_line)  # (1024, 128) → (128, 1024)
                
                # Resize GT line to match enhanced
                if gt_line.shape != enhanced_line.shape:
                    gt_line_resized = cv2.resize(gt_line, 
                                                 (enhanced_line.shape[1], enhanced_line.shape[0]),
                                                 interpolation=cv2.INTER_AREA)
                else:
                    gt_line_resized = gt_line
                
                # Calculate metrics for this line
                psnr_line = calculate_psnr(enhanced_line, gt_line_resized)
                ssim_line = calculate_ssim(enhanced_line, gt_line_resized)
                
                psnr_per_line.append(psnr_line)
                if ssim_line is not None:
                    ssim_per_line.append(ssim_line)
                
                enhanced_lines.append(enhanced_line)
                
                ssim_str = f"{ssim_line:.4f}" if ssim_line is not None else "N/A"
                logger.info(f"      Line {i+1}: PSNR={psnr_line:.2f} dB, SSIM={ssim_str}")
            
            # Calculate average metrics across lines
            avg_psnr = np.mean(psnr_per_line)
            avg_ssim = np.mean(ssim_per_line) if ssim_per_line else None
            
            result = {
                'image_id': sample['id'],
                'n_lines': len(text_lines),
                'psnr': float(avg_psnr),
                'psnr_std': float(np.std(psnr_per_line)),
                'psnr_per_line': [float(p) for p in psnr_per_line],
                'ssim': float(avg_ssim) if avg_ssim else None,
                'ssim_std': float(np.std(ssim_per_line)) if ssim_per_line else None,
            }
            
            results.append(result)
            
            logger.info(f"   📊 Average: PSNR={avg_psnr:.2f}±{result['psnr_std']:.2f} dB")
            logger.info(f"              SSIM={avg_ssim:.4f if avg_ssim else 'N/A'}\n")
            
            # Save visual results
            if save_results:
                save_patchbased_results(original, enhanced_lines, ground_truth, 
                                       sample['id'], output_path)
            
        except Exception as e:
            logger.error(f"❌ Error processing image {sample['id']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate overall statistics
    if results:
        psnr_scores = [r['psnr'] for r in results]
        ssim_scores = [r['ssim'] for r in results if r['ssim'] is not None]
        
        overall_stats = {
            'n_samples': len(results),
            'psnr': {
                'mean': float(np.mean(psnr_scores)),
                'std': float(np.std(psnr_scores)),
                'min': float(np.min(psnr_scores)),
                'max': float(np.max(psnr_scores)),
                'median': float(np.median(psnr_scores))
            },
            'ssim': {
                'mean': float(np.mean(ssim_scores)) if ssim_scores else None,
                'std': float(np.std(ssim_scores)) if ssim_scores else None,
                'min': float(np.min(ssim_scores)) if ssim_scores else None,
                'max': float(np.max(ssim_scores)) if ssim_scores else None,
                'median': float(np.median(ssim_scores)) if ssim_scores else None
            }
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 OVERALL RESULTS (Patch-Based Evaluation)")
        logger.info("=" * 60)
        logger.info(f"   PSNR: {overall_stats['psnr']['mean']:.2f} ± {overall_stats['psnr']['std']:.2f} dB")
        logger.info(f"   PSNR Range: [{overall_stats['psnr']['min']:.2f}, {overall_stats['psnr']['max']:.2f}] dB")
        
        if overall_stats['ssim']['mean'] is not None:
            logger.info(f"   SSIM: {overall_stats['ssim']['mean']:.4f} ± {overall_stats['ssim']['std']:.4f}")
        
        logger.info("")
        
        # Save results
        if save_results:
            save_json_results(results, overall_stats, output_path)
        
        return results, overall_stats
    else:
        logger.error("❌ No successful evaluations!")
        return [], {}

def save_patchbased_results(original, enhanced_lines, ground_truth, image_id, output_dir):
    """Save patch-based evaluation results"""
    
    # Save individual lines
    for i, line in enumerate(enhanced_lines):
        line_uint8 = (line * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"{image_id:02d}_line_{i+1}_enhanced.png"), line_uint8)
    
    # Reconstruct full document from enhanced lines
    reconstructed = np.vstack(enhanced_lines)
    
    # Save comparison
    original_uint8 = (original * 255).astype(np.uint8)
    reconstructed_uint8 = (reconstructed * 255).astype(np.uint8)
    gt_uint8 = (ground_truth * 255).astype(np.uint8)
    
    cv2.imwrite(str(output_dir / f"{image_id:02d}_original.png"), original_uint8)
    cv2.imwrite(str(output_dir / f"{image_id:02d}_reconstructed.png"), reconstructed_uint8)
    cv2.imwrite(str(output_dir / f"{image_id:02d}_ground_truth.png"), gt_uint8)

def save_json_results(results, overall_stats, output_dir):
    """Save evaluation results to JSON"""
    
    evaluation_data = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'dataset': 'H-DIBCO 2016',
        'method': 'Patch-Based (Text Line Extraction)',
        'evaluation_summary': overall_stats,
        'detailed_results': results,
    }
    
    results_file = output_dir / "hdibco2016_patchbased_results.json"
    with open(results_file, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    logger.info(f"💾 Results saved to: {results_file}")

def load_generator_model(model_path):
    """Load generator model from checkpoint"""
    
    logger.info(f"📂 Loading model from: {model_path}")
    logger.info(f"   🎯 Using BEST MODEL from full_training_production_v1")
    
    try:
        from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
        generator = unet_enhanced(input_size=(1024, 128, 1))
        model_type = "enhanced"
        logger.info("   ✅ Using enhanced generator architecture")
    except Exception as e:
        logger.warning(f"   ⚠️ Enhanced failed: {e}, trying base...")
        from dual_modal_gan.src.models.generator import unet
        generator = unet(input_size=(1024, 128, 1))
        model_type = "base"
        logger.info("   ✅ Using base generator architecture")
    
    # Load checkpoint
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    
    if checkpoint_path:
        checkpoint.restore(checkpoint_path).expect_partial()
        logger.info(f"   ✅ Checkpoint loaded: {checkpoint_path}\n")
        return generator
    else:
        logger.error(f"   ❌ No checkpoint found in: {model_path}")
        return None

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Patch-Based Evaluation on H-DIBCO 2016')
    parser.add_argument('--model_path', type=str,
                       default='dual_modal_gan/checkpoints/full_training_production_v1/best_model',
                       help='Path to model checkpoint directory')
    parser.add_argument('--dataset_root', type=str,
                       default='dibco_datasets',
                       help='Path to H-DIBCO 2016 dataset root directory')
    parser.add_argument('--output_dir', type=str,
                       default='hdibco2016_patchbased_evaluation',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_hdibco2016_dataset(args.dataset_root)
    if not dataset:
        return
    
    # Load model
    model = load_generator_model(args.model_path)
    if not model:
        return
    
    # Evaluate with patch-based approach
    results, stats = evaluate_patchbased(
        model, dataset,
        save_results=True,
        output_dir=args.output_dir
    )
    
    if results:
        logger.info("✅ Patch-based evaluation completed successfully!")
        logger.info(f"📁 Results saved to: {args.output_dir}")
    else:
        logger.error("❌ Evaluation failed!")

if __name__ == '__main__':
    main()
