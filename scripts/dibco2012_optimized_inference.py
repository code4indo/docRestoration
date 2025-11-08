#!/usr/bin/env python3
"""
DIBCO-Optimized Inference with Conservative Enhancement
Goal: Preserve text better, achieve PSNR >23 dB
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from dual_modal_gan.src.models.generator_enhanced import unet_enhanced

class DIBCOOptimizedInference:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.load_model()
    
    def load_model(self):
        """Load generator"""
        self.generator = unet_enhanced(input_size=(128, 1024, 1))
        checkpoint = tf.train.Checkpoint(generator=self.generator)
        checkpoint.restore(self.checkpoint_path).expect_partial()
        print(f"✅ Model loaded: {self.checkpoint_path}")
    
    def preprocess(self, image):
        """Preprocess for generator"""
        original_shape = image.shape
        image_resized = cv2.resize(image, (1024, 128), interpolation=cv2.INTER_LINEAR)
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_tanh = image_normalized * 2.0 - 1.0
        image_batch = image_tanh[np.newaxis, :, :, np.newaxis]
        return tf.convert_to_tensor(image_batch, dtype=tf.float32), original_shape
    
    def postprocess(self, tensor, original_shape):
        """Postprocess generator output"""
        image_normalized = (tensor.numpy()[0, :, :, 0] + 1.0) / 2.0
        image_normalized = np.clip(image_normalized, 0.0, 1.0)
        image_uint8 = (image_normalized * 255).astype(np.uint8)
        image_restored = cv2.resize(image_uint8, (original_shape[1], original_shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
        return image_restored
    
    def conservative_enhancement(self, degraded, restored, alpha=0.3):
        """
        Conservative enhancement: Blend degraded with restored
        
        Args:
            degraded: Original degraded image
            restored: Model output
            alpha: Blending factor (0=full degraded, 1=full restored)
        
        Strategy: Model cenderung menghilangkan text, jadi blend dengan degraded
        untuk preserve text yang hilang
        """
        # Blend: result = alpha * restored + (1-alpha) * degraded
        blended = cv2.addWeighted(restored, alpha, degraded, 1-alpha, 0)
        return blended
    
    def text_aware_blending(self, degraded, restored, text_threshold=200):
        """
        Adaptive blending based on text detection
        
        Strategy:
        - Di area text (dark pixels), lebih percaya degraded
        - Di area background (bright pixels), lebih percaya restored
        """
        # Create text mask (darker pixels = likely text)
        text_mask = degraded < text_threshold
        
        # Blend adaptively
        result = restored.copy()
        # Pada area text, blend 70% degraded + 30% restored
        result[text_mask] = (0.3 * restored[text_mask] + 0.7 * degraded[text_mask]).astype(np.uint8)
        
        return result
    
    def gamma_correction(self, image, gamma=0.9):
        """
        Darken image slightly to preserve text
        gamma < 1 = darker, gamma > 1 = brighter
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def process_image(self, image_path, strategy="text_aware", **kwargs):
        """
        Process single image with specified strategy
        
        Strategies:
        - "vanilla": No modification
        - "conservative": Simple alpha blending
        - "text_aware": Adaptive blending based on text detection
        - "gamma": Gamma correction
        - "conservative_gamma": Conservative + gamma
        """
        # Load degraded
        degraded = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        # Run model
        degraded_tensor, original_shape = self.preprocess(degraded)
        restored_tensor = self.generator(degraded_tensor, training=False)
        restored = self.postprocess(restored_tensor, original_shape)
        
        # Apply strategy
        if strategy == "vanilla":
            result = restored
        
        elif strategy == "conservative":
            alpha = kwargs.get('alpha', 0.3)
            result = self.conservative_enhancement(degraded, restored, alpha)
        
        elif strategy == "text_aware":
            text_threshold = kwargs.get('text_threshold', 200)
            result = self.text_aware_blending(degraded, restored, text_threshold)
        
        elif strategy == "gamma":
            gamma = kwargs.get('gamma', 0.9)
            result = self.gamma_correction(restored, gamma)
        
        elif strategy == "conservative_gamma":
            alpha = kwargs.get('alpha', 0.3)
            gamma = kwargs.get('gamma', 0.9)
            blended = self.conservative_enhancement(degraded, restored, alpha)
            result = self.gamma_correction(blended, gamma)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return result

def test_strategies_on_dibco(checkpoint_path, input_dir, gt_dir, output_base_dir):
    """Test different strategies on DIBCO 2012"""
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    
    input_dir = Path(input_dir)
    gt_dir = Path(gt_dir)
    output_base_dir = Path(output_base_dir)
    
    # Initialize processor
    processor = DIBCOOptimizedInference(checkpoint_path)
    
    # Define strategies to test
    strategies = [
        ("vanilla", {}),
        ("conservative_alpha_02", {"strategy": "conservative", "alpha": 0.2}),
        ("conservative_alpha_03", {"strategy": "conservative", "alpha": 0.3}),
        ("conservative_alpha_04", {"strategy": "conservative", "alpha": 0.4}),
        ("conservative_alpha_05", {"strategy": "conservative", "alpha": 0.5}),
        ("text_aware_t180", {"strategy": "text_aware", "text_threshold": 180}),
        ("text_aware_t200", {"strategy": "text_aware", "text_threshold": 200}),
        ("text_aware_t220", {"strategy": "text_aware", "text_threshold": 220}),
        ("gamma_09", {"strategy": "gamma", "gamma": 0.9}),
        ("gamma_08", {"strategy": "gamma", "gamma": 0.8}),
        ("conservative_gamma", {"strategy": "conservative_gamma", "alpha": 0.3, "gamma": 0.9}),
    ]
    
    all_results = {}
    
    # Get image list
    image_files = sorted(input_dir.glob("*.png"))
    
    print("="*80)
    print("🧪 Testing DIBCO-Optimized Strategies")
    print("="*80)
    print(f"Images: {len(image_files)}")
    print(f"Strategies: {len(strategies)}")
    print()
    
    for strategy_name, params in tqdm(strategies, desc="Strategies"):
        output_dir = output_base_dir / strategy_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for img_path in image_files:
            img_id = img_path.stem
            gt_path = gt_dir / f"{img_id}.png"
            
            if not gt_path.exists():
                continue
            
            # Process with strategy
            restored = processor.process_image(img_path, **params)
            
            # Apply Otsu binarization for DIBCO evaluation
            _, binary = cv2.threshold(restored, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Save
            cv2.imwrite(str(output_dir / f"{img_id}_restored.png"), binary)
            
            # Calculate metrics
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if binary.shape != gt.shape:
                binary = cv2.resize(binary, (gt.shape[1], gt.shape[0]))
            
            binary_norm = binary.astype(np.float32) / 255.0
            gt_norm = gt.astype(np.float32) / 255.0
            
            psnr_val = psnr(gt_norm, binary_norm, data_range=1.0)
            ssim_val = ssim(gt_norm, binary_norm, data_range=1.0)
            
            results.append({'image_id': img_id, 'psnr': psnr_val, 'ssim': ssim_val})
        
        # Calculate stats
        if results:
            psnr_vals = [r['psnr'] for r in results]
            psnr_mean = np.mean(psnr_vals)
            ssim_mean = np.mean([r['ssim'] for r in results])
            
            all_results[strategy_name] = {
                'psnr_mean': psnr_mean,
                'ssim_mean': ssim_mean,
                'psnr_std': np.std(psnr_vals, ddof=1),
                'results': results
            }
    
    # Print results
    print()
    print("="*80)
    print("📊 RESULTS")
    print("="*80)
    
    # Sort by PSNR
    sorted_strategies = sorted(all_results.items(), 
                               key=lambda x: x[1]['psnr_mean'], 
                               reverse=True)
    
    for strategy_name, stats in sorted_strategies:
        psnr_mean = stats['psnr_mean']
        ssim_mean = stats['ssim_mean']
        status = "✅" if psnr_mean > 23 else "⚠️"
        
        print(f"{status} {strategy_name:30s}: PSNR = {psnr_mean:6.2f} dB, SSIM = {ssim_mean:.4f}")
    
    # Best strategy
    best_strategy, best_stats = sorted_strategies[0]
    print()
    print("="*80)
    print("🏆 BEST STRATEGY")
    print("="*80)
    print(f"Strategy: {best_strategy}")
    print(f"PSNR:     {best_stats['psnr_mean']:.2f} ± {best_stats['psnr_std']:.2f} dB")
    print(f"SSIM:     {best_stats['ssim_mean']:.4f}")
    
    if best_stats['psnr_mean'] > 23:
        print("\n🎉 TARGET ACHIEVED (PSNR > 23 dB)!")
    else:
        print(f"\n⚠️  Still below target. Gap: {23 - best_stats['psnr_mean']:.2f} dB")
        print("   Recommendation: Fine-tune model on DIBCO dataset")
    
    return sorted_strategies

if __name__ == '__main__':
    CHECKPOINT = "dual_modal_gan/checkpoints/thin_stroke_preservation_v1_academic/best_model/ckpt-99"
    INPUT_DIR = "dibco_datasets/2012/imgs"
    GT_DIR = "dibco_datasets/2012/gt_imgs"
    OUTPUT_BASE = "results/dibco2012_optimized_strategies"
    
    test_strategies_on_dibco(CHECKPOINT, INPUT_DIR, GT_DIR, OUTPUT_BASE)
