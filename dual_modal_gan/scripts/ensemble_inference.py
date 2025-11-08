"""
Ensemble Inference for DIBCO 2012 Test Set
Combines multiple checkpoints to boost PSNR performance

Expected boost: +0.5 to +0.8 dB over single best model
Target: Beat DE-GAN (22.0 dB) and approach DocEnTR (22.29 dB)
"""

import os
import sys
import argparse
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import tensorflow as tf
from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
from dual_modal_gan.src.models.generator_enhanced_v2 import unet_enhanced_v2

def load_model_from_checkpoint(checkpoint_path, generator_version='enhanced'):
    """Load generator model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Build generator (models were trained with grayscale input_size=(1024, 1024, 1))
    if generator_version == 'enhanced':
        generator = unet_enhanced(input_size=(1024, 1024, 1))
    elif generator_version == 'enhanced_v2':
        generator = unet_enhanced_v2(input_size=(1024, 1024, 1))
    else:
        raise ValueError(f"Unknown generator version: {generator_version}")
    
    # Create checkpoint object
    checkpoint = tf.train.Checkpoint(generator=generator)
    
    # Restore weights
    status = checkpoint.restore(checkpoint_path)
    status.expect_partial()  # Ignore optimizer and discriminator
    
    print(f"✅ Model loaded successfully")
    return generator

def preprocess_image(image_path):
    """Load and preprocess image for inference"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR to GRAYSCALE (models trained with grayscale)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Add channel dimension
    img = np.expand_dims(img, axis=-1)
    
    # Normalize to [-1, 1]
    img = (img.astype(np.float32) / 127.5) - 1.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def postprocess_image(img_tensor):
    """Convert model output back to image"""
    # Remove batch dimension
    img = img_tensor[0]
    
    # Remove channel dimension if grayscale
    if img.shape[-1] == 1:
        img = img[:, :, 0]
    
    # Denormalize from [-1, 1] to [0, 255]
    img = ((img + 1.0) * 127.5).astype(np.uint8)
    
    # Clip to valid range
    img = np.clip(img, 0, 255)
    
    return img

def ensemble_inference(image_path, checkpoints_config, fusion_method='weighted_average'):
    """
    Run ensemble inference on single image
    
    Args:
        image_path: Path to degraded image
        checkpoints_config: List of dict with 'path', 'weight', 'version' keys
        fusion_method: 'simple_average', 'weighted_average', or 'median'
    
    Returns:
        Restored image (numpy array)
    """
    # Preprocess input
    input_img = preprocess_image(image_path)
    
    # Collect outputs from all models
    outputs = []
    weights = []
    
    for config in checkpoints_config:
        ckpt_path = config['path']
        weight = config.get('weight', 1.0)
        version = config.get('version', 'enhanced')
        
        # Load model
        generator = load_model_from_checkpoint(ckpt_path, version)
        
        # Run inference
        output = generator(input_img, training=False)
        output_np = output.numpy()
        
        outputs.append(output_np)
        weights.append(weight)
        
        # Clean up
        del generator
        tf.keras.backend.clear_session()
    
    # Fusion
    if fusion_method == 'simple_average':
        # Simple average (equal weights)
        final_output = np.mean(outputs, axis=0)
        
    elif fusion_method == 'weighted_average':
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        final_output = np.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            final_output += weights[i] * output
            
    elif fusion_method == 'median':
        # Pixel-wise median
        final_output = np.median(outputs, axis=0)
        
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    # Postprocess
    final_img = postprocess_image(final_output)
    
    return final_img

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculate SSIM using OpenCV"""
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    # Calculate SSIM using simple formula
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    img1_gray = img1_gray.astype(float)
    img2_gray = img2_gray.astype(float)
    
    mu1 = cv2.GaussianBlur(img1_gray, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2_gray, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1_gray ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2_gray ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1_gray * img2_gray, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return float(np.mean(ssim_map))

def main(args):
    """Main ensemble inference pipeline"""
    
    # Define checkpoint ensemble configuration
    checkpoints_config = [
        {
            'name': 'DIBCO Transfer Best (Epoch 60)',
            'path': 'dual_modal_gan/checkpoints/dibco_transfer_from_production_v3_fixed/best_model/ckpt-214',
            'weight': 0.30,
            'version': 'enhanced',
            'description': 'Best DIBCO fine-tuned model (21.53 dB)'
        },
        {
            'name': 'DIBCO Transfer V2 Enhanced',
            'path': 'dual_modal_gan/checkpoints/dibco_transfer_learning_v2_enhanced/best_model/ckpt-119',
            'weight': 0.25,
            'version': 'enhanced',
            'description': 'Alternative DIBCO training (26.27 dB reported)'
        },
        {
            'name': 'Production V3 Synthetic',
            'path': 'dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model/ckpt-88',
            'weight': 0.25,
            'version': 'enhanced',
            'description': 'Strong synthetic baseline (30.91 dB on synthetic)'
        },
        {
            'name': 'DIBCO Pure SOTA',
            'path': 'dual_modal_gan/checkpoints/dibco_pure_sota_comparison_v1/best_model/ckpt-115',
            'weight': 0.20,
            'version': 'enhanced',
            'description': 'From-scratch DIBCO training (20.91 dB)'
        }
    ]
    
    # Verify all checkpoints exist
    print("=== ENSEMBLE CONFIGURATION ===\n")
    for i, config in enumerate(checkpoints_config, 1):
        ckpt_path = config['path']
        exists = os.path.exists(ckpt_path + '.index')
        status = "✅" if exists else "❌"
        print(f"{i}. {config['name']}")
        print(f"   Path: {ckpt_path}")
        print(f"   Weight: {config['weight']:.2f}")
        print(f"   Status: {status}")
        print(f"   {config['description']}")
        print()
        
        if not exists:
            print(f"ERROR: Checkpoint not found: {ckpt_path}")
            return
    
    # Get test images
    test_dir = Path(args.test_dir)
    gt_dir = Path(args.gt_dir)
    
    # Find all test images
    test_images = sorted(list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.bmp')))
    
    if len(test_images) == 0:
        print(f"ERROR: No test images found in {test_dir}")
        return
    
    print(f"=== FOUND {len(test_images)} TEST IMAGES ===\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run ensemble inference
    results = []
    
    for img_path in tqdm(test_images, desc="Ensemble Inference"):
        img_name = img_path.stem
        
        # Run ensemble
        restored_img = ensemble_inference(
            img_path, 
            checkpoints_config,
            fusion_method=args.fusion_method
        )
        
        # Save restored image
        output_path = output_dir / f"{img_name}_restored.png"
        cv2.imwrite(str(output_path), restored_img)  # Already grayscale, no conversion needed
        
        # Calculate metrics if GT available
        gt_path = gt_dir / f"{img_name}_gt.png"
        if not gt_path.exists():
            gt_path = gt_dir / f"{img_name}.png"
        
        if gt_path.exists():
            gt_img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            
            psnr = calculate_psnr(restored_img, gt_img)
            ssim = calculate_ssim(restored_img, gt_img)
            
            results.append({
                'image': img_name,
                'psnr': psnr,
                'ssim': ssim
            })
            
            print(f"\n{img_name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
    
    # Summary
    if results:
        avg_psnr = np.mean([r['psnr'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])
        
        print(f"\n=== ENSEMBLE RESULTS ===")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"\nComparison with SOTA:")
        print(f"  DE-GAN (22.00 dB):   {avg_psnr - 22.00:+.2f} dB")
        print(f"  DocEnTR (22.29 dB):  {avg_psnr - 22.29:+.2f} dB")
        print(f"  Target (22.50 dB):   {avg_psnr - 22.50:+.2f} dB")
        
        # Save results
        results_path = output_dir / 'ensemble_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'checkpoints': checkpoints_config,
                'fusion_method': args.fusion_method,
                'average_psnr': float(avg_psnr),
                'average_ssim': float(avg_ssim),
                'results': results
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble Inference for DIBCO 2012')
    parser.add_argument('--test_dir', type=str, default='dibco_datasets/2012/handwritten-images',
                        help='Directory containing test images')
    parser.add_argument('--gt_dir', type=str, default='dibco_datasets/2012/handwritten-images-gt',
                        help='Directory containing ground truth images')
    parser.add_argument('--output_dir', type=str, default='dual_modal_gan/outputs/ensemble_dibco2012',
                        help='Directory to save restored images')
    parser.add_argument('--fusion_method', type=str, default='weighted_average',
                        choices=['simple_average', 'weighted_average', 'median'],
                        help='Fusion method for combining predictions')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # Run ensemble
    main(args)
