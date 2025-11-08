#!/usr/bin/env python3
"""
DIBCO 2012 Pre-processing Experiment
Goal: Find optimal pre-processing to achieve PSNR >23 dB

Strategy:
1. Apply various pre-processing to degraded images
2. Feed to model
3. Apply binarization
4. Measure PSNR vs GT

Pre-processing techniques tested:
- Sharpening (Unsharp Masking, Laplacian)
- Contrast Enhancement (CLAHE, Histogram Equalization, Stretching)
- Denoising (Bilateral, Non-local means, Gaussian)
- Morphological operations (Opening, Closing, Top-hat)
- Combinations of above
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


class PreprocessingExperiment:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.load_model()
    
    def load_model(self):
        """Load generator"""
        print("Loading model...")
        self.generator = unet_enhanced(input_size=(128, 1024, 1))
        checkpoint = tf.train.Checkpoint(generator=self.generator)
        checkpoint.restore(self.checkpoint_path).expect_partial()
        print(f"✅ Model loaded: {self.checkpoint_path}")
    
    # ==================== PRE-PROCESSING METHODS ====================
    
    def sharpen_unsharp_mask(self, image, amount=1.5, kernel_size=5):
        """Unsharp masking for sharpening"""
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def sharpen_laplacian(self, image, alpha=0.5):
        """Laplacian sharpening"""
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        sharpened = image - alpha * laplacian
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def enhance_contrast_clahe(self, image, clip_limit=2.0, tile_size=8):
        """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(image)
    
    def enhance_contrast_hist_eq(self, image):
        """Global histogram equalization"""
        return cv2.equalizeHist(image)
    
    def enhance_contrast_stretch(self, image, percentile_low=2, percentile_high=98):
        """Contrast stretching with percentile clipping"""
        p_low = np.percentile(image, percentile_low)
        p_high = np.percentile(image, percentile_high)
        
        stretched = ((image - p_low) / (p_high - p_low) * 255)
        return np.clip(stretched, 0, 255).astype(np.uint8)
    
    def denoise_bilateral(self, image, d=9, sigma_color=75, sigma_space=75):
        """Bilateral filtering (edge-preserving denoising)"""
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def denoise_nlmeans(self, image, h=10, template_size=7, search_size=21):
        """Non-local means denoising"""
        return cv2.fastNlMeansDenoising(image, None, h, template_size, search_size)
    
    def denoise_gaussian(self, image, kernel_size=5):
        """Gaussian blur denoising"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def morph_opening(self, image, kernel_size=3):
        """Morphological opening (remove small noise)"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    def morph_closing(self, image, kernel_size=3):
        """Morphological closing (fill small holes)"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    def morph_tophat(self, image, kernel_size=15):
        """Top-hat transform (enhance bright features)"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        return cv2.add(image, tophat)
    
    def morph_blackhat(self, image, kernel_size=15):
        """Black-hat transform (enhance dark features = text)"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        return cv2.subtract(image, blackhat)
    
    def adaptive_gamma_correction(self, image, gamma=0.8):
        """Gamma correction to darken (preserve text)"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def text_emphasis(self, image, threshold=200):
        """
        Emphasize text regions (dark pixels)
        Strategy: Darken pixels below threshold, brighten above
        """
        result = image.copy().astype(float)
        
        # Darken text (pixels < threshold)
        text_mask = image < threshold
        result[text_mask] = result[text_mask] * 0.8
        
        # Brighten background (pixels >= threshold)
        bg_mask = ~text_mask
        result[bg_mask] = np.minimum(result[bg_mask] * 1.1, 255)
        
        return result.astype(np.uint8)
    
    # ==================== COMBINATION STRATEGIES ====================
    
    def preprocess_none(self, image):
        """No preprocessing (baseline)"""
        return image
    
    def preprocess_sharpen_only(self, image):
        """Sharpen only"""
        return self.sharpen_unsharp_mask(image, amount=1.5)
    
    def preprocess_contrast_only(self, image):
        """CLAHE only"""
        return self.enhance_contrast_clahe(image, clip_limit=3.0, tile_size=8)
    
    def preprocess_sharpen_contrast(self, image):
        """Sharpen + CLAHE"""
        sharpened = self.sharpen_unsharp_mask(image, amount=1.5)
        enhanced = self.enhance_contrast_clahe(sharpened, clip_limit=3.0, tile_size=8)
        return enhanced
    
    def preprocess_denoise_sharpen_contrast(self, image):
        """Denoise + Sharpen + CLAHE"""
        denoised = self.denoise_bilateral(image, d=5, sigma_color=50, sigma_space=50)
        sharpened = self.sharpen_unsharp_mask(denoised, amount=1.5)
        enhanced = self.enhance_contrast_clahe(sharpened, clip_limit=3.0, tile_size=8)
        return enhanced
    
    def preprocess_blackhat_contrast(self, image):
        """Black-hat (text enhancement) + CLAHE"""
        blackhat = self.morph_blackhat(image, kernel_size=15)
        enhanced = self.enhance_contrast_clahe(blackhat, clip_limit=3.0, tile_size=8)
        return enhanced
    
    def preprocess_text_emphasis_sharpen(self, image):
        """Text emphasis + Sharpen"""
        emphasized = self.text_emphasis(image, threshold=200)
        sharpened = self.sharpen_unsharp_mask(emphasized, amount=2.0)
        return sharpened
    
    def preprocess_aggressive_enhancement(self, image):
        """Aggressive: Denoise + Sharpen + CLAHE + Text emphasis"""
        denoised = self.denoise_bilateral(image, d=5, sigma_color=50, sigma_space=50)
        sharpened = self.sharpen_unsharp_mask(denoised, amount=2.0)
        enhanced = self.enhance_contrast_clahe(sharpened, clip_limit=4.0, tile_size=8)
        emphasized = self.text_emphasis(enhanced, threshold=210)
        return emphasized
    
    def preprocess_conservative_enhancement(self, image):
        """Conservative: Light sharpen + Stretch"""
        sharpened = self.sharpen_unsharp_mask(image, amount=1.0)
        stretched = self.enhance_contrast_stretch(sharpened, percentile_low=1, percentile_high=99)
        return stretched
    
    def preprocess_morphological_pipeline(self, image):
        """Morphological: Black-hat + Sharpen"""
        blackhat = self.morph_blackhat(image, kernel_size=11)
        sharpened = self.sharpen_unsharp_mask(blackhat, amount=1.5)
        return sharpened
    
    # ==================== MODEL INFERENCE ====================
    
    def preprocess_for_model(self, image):
        """Preprocess for generator"""
        original_shape = image.shape
        image_resized = cv2.resize(image, (1024, 128), interpolation=cv2.INTER_LINEAR)
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_tanh = image_normalized * 2.0 - 1.0
        image_batch = image_tanh[np.newaxis, :, :, np.newaxis]
        return tf.convert_to_tensor(image_batch, dtype=tf.float32), original_shape
    
    def postprocess_from_model(self, tensor, original_shape):
        """Postprocess generator output"""
        image_normalized = (tensor.numpy()[0, :, :, 0] + 1.0) / 2.0
        image_normalized = np.clip(image_normalized, 0.0, 1.0)
        image_uint8 = (image_normalized * 255).astype(np.uint8)
        image_restored = cv2.resize(image_uint8, (original_shape[1], original_shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
        return image_restored
    
    def run_inference(self, image):
        """Run model inference"""
        tensor, original_shape = self.preprocess_for_model(image)
        restored_tensor = self.generator(tensor, training=False)
        restored = self.postprocess_from_model(restored_tensor, original_shape)
        return restored
    
    # ==================== EVALUATION ====================
    
    def evaluate_strategy(self, input_dir, gt_dir, strategy_name, preprocess_func, 
                         binarize_method='otsu', save_outputs=False, output_dir=None):
        """
        Evaluate a preprocessing strategy
        
        Args:
            strategy_name: Name of strategy
            preprocess_func: Function to preprocess degraded image before model
            binarize_method: 'otsu', 'sauvola', or 'adaptive'
            save_outputs: Whether to save intermediate/final outputs
            output_dir: Where to save outputs
        """
        results = []
        
        input_dir = Path(input_dir)
        gt_dir = Path(gt_dir)
        
        if save_outputs and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = sorted(input_dir.glob("*.png"))
        
        for img_path in image_files:
            img_id = img_path.stem
            gt_path = gt_dir / f"{img_id}.png"
            
            if not gt_path.exists():
                continue
            
            # Load degraded
            degraded = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            # Apply preprocessing
            preprocessed = preprocess_func(degraded)
            
            # Run model inference
            restored = self.run_inference(preprocessed)
            
            # Apply binarization
            if binarize_method == 'otsu':
                _, binary = cv2.threshold(restored, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif binarize_method == 'sauvola':
                binary = self.sauvola_binarization(restored, window_size=25, k=0.2)
            elif binarize_method == 'adaptive':
                binary = cv2.adaptiveThreshold(restored, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 15, 2)
            else:
                _, binary = cv2.threshold(restored, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Save if requested
            if save_outputs and output_dir:
                cv2.imwrite(str(output_dir / f"{img_id}_preprocessed.png"), preprocessed)
                cv2.imwrite(str(output_dir / f"{img_id}_restored.png"), restored)
                cv2.imwrite(str(output_dir / f"{img_id}_binary.png"), binary)
            
            # Calculate metrics
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if binary.shape != gt.shape:
                binary = cv2.resize(binary, (gt.shape[1], gt.shape[0]))
            
            binary_norm = binary.astype(np.float32) / 255.0
            gt_norm = gt.astype(np.float32) / 255.0
            
            psnr_val = psnr(gt_norm, binary_norm, data_range=1.0)
            ssim_val = ssim(gt_norm, binary_norm, data_range=1.0)
            
            results.append({'image_id': img_id, 'psnr': psnr_val, 'ssim': ssim_val})
        
        # Calculate statistics
        if results:
            psnr_vals = [r['psnr'] for r in results]
            ssim_vals = [r['ssim'] for r in results]
            
            return {
                'strategy': strategy_name,
                'binarize_method': binarize_method,
                'num_images': len(results),
                'psnr_mean': float(np.mean(psnr_vals)),
                'psnr_std': float(np.std(psnr_vals, ddof=1)),
                'psnr_min': float(np.min(psnr_vals)),
                'psnr_max': float(np.max(psnr_vals)),
                'psnr_median': float(np.median(psnr_vals)),
                'ssim_mean': float(np.mean(ssim_vals)),
                'ssim_std': float(np.std(ssim_vals, ddof=1)),
                'per_image': results
            }
        
        return None
    
    def sauvola_binarization(self, image, window_size=15, k=0.2):
        """Sauvola local thresholding"""
        img_float = image.astype(np.float32)
        mean = cv2.boxFilter(img_float, -1, (window_size, window_size))
        mean_sq = cv2.boxFilter(img_float**2, -1, (window_size, window_size))
        std = np.sqrt(np.maximum(mean_sq - mean**2, 0))  # Avoid negative due to numerical error
        
        R = 128.0
        threshold = mean * (1 + k * (std / R - 1))
        binary = np.where(img_float > threshold, 255, 0).astype(np.uint8)
        return binary


def main():
    # Configuration
    CHECKPOINT = "dual_modal_gan/checkpoints/thin_stroke_preservation_v1_academic/best_model/ckpt-99"
    INPUT_DIR = "dibco_datasets/2012/imgs"
    GT_DIR = "dibco_datasets/2012/gt_imgs"
    OUTPUT_BASE = "results/dibco2012_preprocessing_exp"
    
    print("="*80)
    print("🧪 DIBCO 2012 Pre-processing Experiment")
    print("="*80)
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Target: PSNR > 23 dB")
    print()
    
    # Initialize experiment
    exp = PreprocessingExperiment(CHECKPOINT)
    
    # Define strategies to test
    strategies = [
        ("baseline_no_preprocess", exp.preprocess_none),
        ("sharpen_unsharp", exp.preprocess_sharpen_only),
        ("clahe_contrast", exp.preprocess_contrast_only),
        ("sharpen_clahe", exp.preprocess_sharpen_contrast),
        ("denoise_sharpen_clahe", exp.preprocess_denoise_sharpen_contrast),
        ("blackhat_clahe", exp.preprocess_blackhat_contrast),
        ("text_emphasis_sharpen", exp.preprocess_text_emphasis_sharpen),
        ("aggressive_enhancement", exp.preprocess_aggressive_enhancement),
        ("conservative_enhancement", exp.preprocess_conservative_enhancement),
        ("morphological_pipeline", exp.preprocess_morphological_pipeline),
    ]
    
    # Test with different binarization methods
    binarize_methods = ['otsu', 'sauvola']
    
    all_results = []
    
    print("Testing preprocessing strategies...")
    print("-"*80)
    
    for binarize_method in binarize_methods:
        print(f"\n📋 Binarization: {binarize_method.upper()}")
        print("-"*80)
        
        for strategy_name, preprocess_func in tqdm(strategies, desc=f"  {binarize_method}"):
            stats = exp.evaluate_strategy(
                INPUT_DIR, GT_DIR, strategy_name, preprocess_func,
                binarize_method=binarize_method,
                save_outputs=False  # Change to True to save outputs
            )
            
            if stats:
                all_results.append(stats)
                
                psnr_mean = stats['psnr_mean']
                ssim_mean = stats['ssim_mean']
                status = "✅" if psnr_mean > 23 else "⚠️"
                
                print(f"  {status} {strategy_name:30s}: PSNR = {psnr_mean:6.2f} dB, SSIM = {ssim_mean:.4f}")
    
    # Sort by PSNR
    all_results.sort(key=lambda x: x['psnr_mean'], reverse=True)
    
    print()
    print("="*80)
    print("🏆 TOP 10 STRATEGIES")
    print("="*80)
    
    for i, stats in enumerate(all_results[:10], 1):
        psnr_mean = stats['psnr_mean']
        ssim_mean = stats['ssim_mean']
        strategy = stats['strategy']
        binarize = stats['binarize_method']
        
        status = "✅" if psnr_mean > 23 else "⚠️"
        
        print(f"{i:2d}. {status} {strategy:30s} + {binarize:8s}: "
              f"PSNR = {psnr_mean:6.2f} ± {stats['psnr_std']:4.2f} dB, "
              f"SSIM = {ssim_mean:.4f}")
    
    # Check if target achieved
    best = all_results[0]
    print()
    print("="*80)
    print("🎯 BEST RESULT")
    print("="*80)
    print(f"Strategy:     {best['strategy']}")
    print(f"Binarization: {best['binarize_method']}")
    print(f"PSNR:         {best['psnr_mean']:.2f} ± {best['psnr_std']:.2f} dB")
    print(f"SSIM:         {best['ssim_mean']:.4f}")
    print(f"Range:        [{best['psnr_min']:.2f}, {best['psnr_max']:.2f}] dB")
    
    if best['psnr_mean'] > 23:
        print()
        print("🎉 TARGET ACHIEVED! PSNR > 23 dB")
        print("="*80)
    else:
        gap = 23 - best['psnr_mean']
        print()
        print(f"⚠️  Still below target. Gap: {gap:.2f} dB")
        print("="*80)
        print("💡 RECOMMENDATIONS:")
        print("   1. Fine-tune model on DIBCO dataset (most effective)")
        print("   2. Try ensemble of multiple preprocessing strategies")
        print("   3. Investigate per-image failures for pattern analysis")
        print("="*80)
    
    # Save results
    output_file = Path(OUTPUT_BASE) / "preprocessing_experiment_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'checkpoint': CHECKPOINT,
            'dataset': 'DIBCO 2012',
            'target_psnr': 23.0,
            'best_strategy': {
                'name': best['strategy'],
                'binarization': best['binarize_method'],
                'psnr': best['psnr_mean'],
                'ssim': best['ssim_mean']
            },
            'all_results': all_results
        }, f, indent=2)
    
    print(f"\n📄 Full results saved to: {output_file}")
    
    # Offer to save best strategy outputs
    print()
    print("="*80)
    print("💡 NEXT STEPS")
    print("="*80)
    print(f"To save outputs with best strategy ({best['strategy']} + {best['binarize_method']}):")
    print(f"  1. Modify save_outputs=True in the script")
    print(f"  2. Re-run for the best strategy only")
    print()


if __name__ == '__main__':
    main()
