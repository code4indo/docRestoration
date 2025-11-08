#!/usr/bin/env python3
"""
DIBCO 2012 Test Set Evaluation Script

Evaluates the best fine-tuned model on DIBCO 2012 official test set.

Features:
- Load DIBCO 2012 degraded-GT pairs
- Run inference using best checkpoint (epoch 49)
- Calculate comprehensive metrics: PSNR, SSIM, MSE, MAE
- Generate comparison visualizations
- Export detailed metrics report

Usage:
    python scripts/evaluate_dibco_2012.py
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dual_modal_gan.src.models.generator_enhanced import unet_enhanced as create_generator
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse


class DIBCO2012Evaluator:
    def __init__(self, checkpoint_path, dibco_root, output_dir):
        """
        Args:
            checkpoint_path: Path to best model checkpoint
            dibco_root: Root directory of DIBCO 2012 dataset
            output_dir: Directory to save results
        """
        self.checkpoint_path = checkpoint_path
        self.dibco_root = Path(dibco_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.restored_dir = self.output_dir / "restored_images"
        self.comparison_dir = self.output_dir / "comparisons"
        self.restored_dir.mkdir(exist_ok=True)
        self.comparison_dir.mkdir(exist_ok=True)
        
        self.generator = None
        self.results = []
        
    def load_model(self):
        """Load generator model from checkpoint"""
        print(f"\n{'='*80}")
        print(f"🔧 Loading Model")
        print(f"{'='*80}")
        print(f"Checkpoint: {self.checkpoint_path}")
        
        # Create generator (enhanced version)
        self.generator = create_generator(
            input_size=(128, 1024, 1)  # Will handle variable sizes via padding
        )
        
        # Create checkpoint and restore
        checkpoint = tf.train.Checkpoint(generator=self.generator)
        checkpoint.restore(self.checkpoint_path).expect_partial()
        
        print(f"✅ Model loaded successfully")
        
    def load_dibco_pairs(self):
        """Load all DIBCO 2012 degraded-GT pairs"""
        print(f"\n{'='*80}")
        print(f"📂 Loading DIBCO 2012 Test Set")
        print(f"{'='*80}")
        print(f"Root directory: {self.dibco_root}")
        
        degraded_dir = self.dibco_root / "imgs"
        gt_dir = self.dibco_root / "gt_imgs"
        
        # Get all degraded images
        degraded_files = sorted(degraded_dir.glob("*.png"))
        
        pairs = []
        for deg_path in degraded_files:
            # Find corresponding GT
            gt_path = gt_dir / deg_path.name
            
            if not gt_path.exists():
                print(f"⚠️  Warning: No GT found for {deg_path.name}")
                continue
            
            pairs.append({
                'id': deg_path.stem,
                'degraded_path': str(deg_path),
                'gt_path': str(gt_path)
            })
        
        print(f"✅ Found {len(pairs)} degraded-GT pairs")
        return pairs
    
    def preprocess_image(self, image):
        """
        Preprocess image for generator input.
        Resize to 128x1024 to match model input size.
        
        Args:
            image: Grayscale image [0, 255]
        
        Returns:
            Preprocessed image [-1, 1] with shape (1, 128, 1024, 1),
            original shape for later resizing back
        """
        # Store original shape
        original_shape = image.shape
        
        # Resize to model input size (128 height, 1024 width)
        image_resized = cv2.resize(image, (1024, 128), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Normalize to [-1, 1] for generator
        image_tanh = image_normalized * 2.0 - 1.0
        
        # Add batch and channel dimensions
        image_batch = image_tanh[np.newaxis, :, :, np.newaxis]
        
        return tf.convert_to_tensor(image_batch, dtype=tf.float32), original_shape
    
    def postprocess_image(self, image_tensor, original_shape):
        """
        Postprocess generator output to [0, 255] uint8 and resize back to original.
        
        Args:
            image_tensor: Generator output [-1, 1] with shape (1, 128, 1024, 1)
            original_shape: Original image shape (H, W)
        
        Returns:
            Image [0, 255] uint8 with original shape
        """
        # Denormalize from [-1, 1] to [0, 1]
        image_normalized = (image_tensor.numpy()[0, :, :, 0] + 1.0) / 2.0
        
        # Clip to [0, 1] and convert to [0, 255]
        image_normalized = np.clip(image_normalized, 0.0, 1.0)
        image_uint8 = (image_normalized * 255).astype(np.uint8)
        
        # Resize back to original shape (W, H) for cv2.resize
        image_restored = cv2.resize(image_uint8, (original_shape[1], original_shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
        
        return image_restored
    
    def calculate_metrics(self, restored, gt):
        """
        Calculate comprehensive metrics between restored and GT.
        
        Args:
            restored: Restored image [0, 255] uint8
            gt: Ground truth image [0, 255] uint8
        
        Returns:
            dict of metrics (all values converted to Python float)
        """
        # Ensure same size
        if restored.shape != gt.shape:
            print(f"⚠️  Size mismatch: restored {restored.shape} vs GT {gt.shape}")
            # Resize restored to match GT
            restored = cv2.resize(restored, (gt.shape[1], gt.shape[0]))
        
        # Normalize to [0, 1] for PSNR/SSIM
        restored_norm = restored.astype(np.float32) / 255.0
        gt_norm = gt.astype(np.float32) / 255.0
        
        # Calculate metrics (convert to Python float for JSON serialization)
        metrics = {
            'psnr': float(psnr(gt_norm, restored_norm, data_range=1.0)),
            'ssim': float(ssim(gt_norm, restored_norm, data_range=1.0)),
            'mse': float(mse(gt_norm, restored_norm)),
            'mae': float(np.mean(np.abs(gt_norm - restored_norm))),
            'rmse': float(np.sqrt(mse(gt_norm, restored_norm)))
        }
        
        return metrics
    
    def create_comparison_image(self, degraded, restored, gt, image_id):
        """
        Create side-by-side comparison: Degraded | Restored | GT
        
        Args:
            degraded: Degraded image
            restored: Restored image
            gt: Ground truth image
            image_id: Image identifier
        
        Returns:
            Comparison image
        """
        # Ensure all same height (use GT height as reference)
        h, w = gt.shape[:2]
        
        # Resize degraded and restored if needed
        if degraded.shape != (h, w):
            degraded = cv2.resize(degraded, (w, h))
        if restored.shape != (h, w):
            restored = cv2.resize(restored, (w, h))
        
        # Create horizontal concatenation
        comparison = np.hstack([degraded, restored, gt])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = 0  # Black text
        
        # Label positions (top-left of each section)
        label_y = 30
        cv2.putText(comparison, "Degraded", (10, label_y), font, font_scale, color, thickness)
        cv2.putText(comparison, "Restored", (w + 10, label_y), font, font_scale, color, thickness)
        cv2.putText(comparison, "Ground Truth", (2*w + 10, label_y), font, font_scale, color, thickness)
        
        return comparison
    
    def process_single_image(self, pair):
        """Process single DIBCO 2012 image"""
        image_id = pair['id']
        
        # Load images
        degraded = cv2.imread(pair['degraded_path'], cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(pair['gt_path'], cv2.IMREAD_GRAYSCALE)
        
        if degraded is None or gt is None:
            print(f"❌ Failed to load images for {image_id}")
            return None
        
        # Store original size
        original_shape = degraded.shape
        
        # Preprocess for generator (resize to 128x1024)
        degraded_tensor, _ = self.preprocess_image(degraded)
        
        # Run inference
        restored_tensor = self.generator(degraded_tensor, training=False)
        
        # Postprocess and resize back to original
        restored = self.postprocess_image(restored_tensor, original_shape)
        
        # Calculate metrics (now both are same size)
        metrics = self.calculate_metrics(restored, gt)
        metrics['image_id'] = image_id
        metrics['original_shape'] = list(original_shape)  # Convert tuple to list for JSON
        
        # Save restored image
        restored_path = self.restored_dir / f"{image_id}_restored.png"
        cv2.imwrite(str(restored_path), restored)
        
        # Create and save comparison
        comparison = self.create_comparison_image(degraded, restored, gt, image_id)
        comparison_path = self.comparison_dir / f"{image_id}_comparison.png"
        cv2.imwrite(str(comparison_path), comparison)
        
        return metrics
    
    def evaluate_all(self):
        """Evaluate all DIBCO 2012 images"""
        print(f"\n{'='*80}")
        print(f"🚀 Running Evaluation on DIBCO 2012 Test Set")
        print(f"{'='*80}")
        
        # Load model
        self.load_model()
        
        # Load dataset pairs
        pairs = self.load_dibco_pairs()
        
        # Process each image
        print(f"\n{'='*80}")
        print(f"🔄 Processing Images")
        print(f"{'='*80}")
        
        for pair in tqdm(pairs, desc="Evaluating"):
            result = self.process_single_image(pair)
            if result:
                self.results.append(result)
                print(f"  ✅ {result['image_id']}: "
                      f"PSNR={result['psnr']:.2f} dB, "
                      f"SSIM={result['ssim']:.4f}")
        
        # Calculate statistics
        self.calculate_statistics()
        
        # Generate report
        self.generate_report()
        
    def calculate_statistics(self):
        """Calculate aggregate statistics"""
        if not self.results:
            print("❌ No results to calculate statistics")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 AGGREGATE STATISTICS")
        print(f"{'='*80}")
        
        metrics_arrays = {
            'psnr': np.array([r['psnr'] for r in self.results]),
            'ssim': np.array([r['ssim'] for r in self.results]),
            'mse': np.array([r['mse'] for r in self.results]),
            'mae': np.array([r['mae'] for r in self.results]),
            'rmse': np.array([r['rmse'] for r in self.results])
        }
        
        self.statistics = {}
        for metric_name, values in metrics_arrays.items():
            self.statistics[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        # Print statistics
        print(f"\nNumber of images: {len(self.results)}")
        print(f"\nMetric           Mean ± Std        Min      Max      Median")
        print(f"{'-'*70}")
        
        for metric in ['psnr', 'ssim', 'mse', 'mae', 'rmse']:
            stats = self.statistics[metric]
            print(f"{metric.upper():15s} "
                  f"{stats['mean']:6.4f} ± {stats['std']:6.4f}  "
                  f"{stats['min']:7.4f}  "
                  f"{stats['max']:7.4f}  "
                  f"{stats['median']:7.4f}")
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        print(f"\n{'='*80}")
        print(f"📄 Generating Report")
        print(f"{'='*80}")
        
        # Create detailed report
        report = {
            'evaluation_info': {
                'dataset': 'DIBCO 2012 Test Set',
                'num_images': len(self.results),
                'model_checkpoint': str(self.checkpoint_path),
                'evaluation_date': datetime.now().isoformat()
            },
            'aggregate_statistics': self.statistics,
            'per_image_results': self.results
        }
        
        # Save JSON report
        json_path = self.output_dir / "evaluation_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ JSON report saved: {json_path}")
        
        # Generate markdown report
        self.generate_markdown_report(report)
        
        # Generate plots
        self.generate_plots()
        
        print(f"\n{'='*80}")
        print(f"✅ EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Output directory: {self.output_dir}")
        print(f"  - Restored images:   {self.restored_dir}")
        print(f"  - Comparisons:       {self.comparison_dir}")
        print(f"  - JSON report:       {json_path}")
        print(f"  - Markdown report:   {self.output_dir / 'EVALUATION_REPORT.md'}")
        print(f"  - Plots:             {self.output_dir / 'metrics_*.png'}")
    
    def generate_markdown_report(self, report):
        """Generate markdown report"""
        md_path = self.output_dir / "EVALUATION_REPORT.md"
        
        with open(md_path, 'w') as f:
            f.write("# DIBCO 2012 Test Set Evaluation Report\n\n")
            f.write(f"**Date**: {report['evaluation_info']['evaluation_date']}\n\n")
            f.write(f"**Model**: {report['evaluation_info']['model_checkpoint']}\n\n")
            f.write(f"**Dataset**: DIBCO 2012 Official Test Set ({report['evaluation_info']['num_images']} images)\n\n")
            
            f.write("---\n\n")
            f.write("## 📊 Aggregate Statistics\n\n")
            
            f.write("| Metric | Mean ± Std | Min | Max | Median |\n")
            f.write("|--------|------------|-----|-----|--------|\n")
            
            for metric in ['psnr', 'ssim', 'mse', 'mae', 'rmse']:
                stats = self.statistics[metric]
                f.write(f"| **{metric.upper()}** | "
                       f"{stats['mean']:.4f} ± {stats['std']:.4f} | "
                       f"{stats['min']:.4f} | "
                       f"{stats['max']:.4f} | "
                       f"{stats['median']:.4f} |\n")
            
            f.write("\n---\n\n")
            f.write("## 📋 Per-Image Results\n\n")
            
            f.write("| Image ID | PSNR (dB) | SSIM | MSE | MAE | RMSE |\n")
            f.write("|----------|-----------|------|-----|-----|------|\n")
            
            # Sort by PSNR descending
            sorted_results = sorted(self.results, key=lambda x: x['psnr'], reverse=True)
            
            for r in sorted_results:
                f.write(f"| {r['image_id']} | "
                       f"{r['psnr']:.2f} | "
                       f"{r['ssim']:.4f} | "
                       f"{r['mse']:.6f} | "
                       f"{r['mae']:.6f} | "
                       f"{r['rmse']:.6f} |\n")
            
            f.write("\n---\n\n")
            f.write("## 🎯 Key Findings\n\n")
            
            # PSNR analysis
            psnr_mean = self.statistics['psnr']['mean']
            ssim_mean = self.statistics['ssim']['mean']
            
            if psnr_mean >= 25:
                f.write(f"✅ **Excellent PSNR**: {psnr_mean:.2f} dB (>25 dB target achieved)\n\n")
            elif psnr_mean >= 22:
                f.write(f"✅ **Good PSNR**: {psnr_mean:.2f} dB (within SOTA range for DIBCO)\n\n")
            else:
                f.write(f"⚠️ **Moderate PSNR**: {psnr_mean:.2f} dB (below 22 dB)\n\n")
            
            if ssim_mean >= 0.95:
                f.write(f"✅ **Excellent SSIM**: {ssim_mean:.4f} (>0.95 structural similarity)\n\n")
            elif ssim_mean >= 0.90:
                f.write(f"✅ **Good SSIM**: {ssim_mean:.4f} (>0.90 structural similarity)\n\n")
            else:
                f.write(f"⚠️ **Moderate SSIM**: {ssim_mean:.4f}\n\n")
            
            # Best/worst performing images
            best = sorted_results[0]
            worst = sorted_results[-1]
            
            f.write(f"**Best performing**: {best['image_id']} (PSNR: {best['psnr']:.2f} dB, SSIM: {best['ssim']:.4f})\n\n")
            f.write(f"**Worst performing**: {worst['image_id']} (PSNR: {worst['psnr']:.2f} dB, SSIM: {worst['ssim']:.4f})\n\n")
        
        print(f"✅ Markdown report saved: {md_path}")
    
    def generate_plots(self):
        """Generate visualization plots"""
        # Extract data
        image_ids = [r['image_id'] for r in self.results]
        psnrs = [r['psnr'] for r in self.results]
        ssims = [r['ssim'] for r in self.results]
        
        # Sort by image ID
        sorted_indices = np.argsort([int(id) for id in image_ids])
        image_ids = [image_ids[i] for i in sorted_indices]
        psnrs = [psnrs[i] for i in sorted_indices]
        ssims = [ssims[i] for i in sorted_indices]
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: PSNR per image
        ax1.bar(range(len(image_ids)), psnrs, color='steelblue', alpha=0.7)
        ax1.axhline(y=np.mean(psnrs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(psnrs):.2f} dB')
        ax1.set_xlabel('Image ID', fontsize=12)
        ax1.set_ylabel('PSNR (dB)', fontsize=12)
        ax1.set_title('PSNR per Image - DIBCO 2012 Test Set', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(image_ids)))
        ax1.set_xticklabels(image_ids, rotation=0)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: SSIM per image
        ax2.bar(range(len(image_ids)), ssims, color='forestgreen', alpha=0.7)
        ax2.axhline(y=np.mean(ssims), color='red', linestyle='--',
                   label=f'Mean: {np.mean(ssims):.4f}')
        ax2.set_xlabel('Image ID', fontsize=12)
        ax2.set_ylabel('SSIM', fontsize=12)
        ax2.set_title('SSIM per Image - DIBCO 2012 Test Set', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(image_ids)))
        ax2.set_xticklabels(image_ids, rotation=0)
        ax2.set_ylim([0, 1.05])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "metrics_per_image.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Plot saved: {plot_path}")
        
        # Distribution plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # PSNR distribution
        ax1.hist(psnrs, bins=10, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(x=np.mean(psnrs), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(psnrs):.2f} dB')
        ax1.set_xlabel('PSNR (dB)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('PSNR Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # SSIM distribution
        ax2.hist(ssims, bins=10, color='forestgreen', alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.mean(ssims), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(ssims):.4f}')
        ax2.set_xlabel('SSIM', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('SSIM Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        dist_plot_path = self.output_dir / "metrics_distribution.png"
        plt.savefig(dist_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Distribution plot saved: {dist_plot_path}")


def main():
    # Configuration
    CHECKPOINT_PATH = "dual_modal_gan/checkpoints/dibco_tiled_no_palm_visual_only/best_model/ckpt-184"
    DIBCO_ROOT = "dibco_datasets/2012"
    OUTPUT_DIR = "dual_modal_gan/outputs/dibco_2012_evaluation"
    
    print(f"\n{'='*80}")
    print(f"🎯 DIBCO 2012 TEST SET EVALUATION")
    print(f"{'='*80}")
    print(f"Model: {CHECKPOINT_PATH}")
    print(f"Dataset: {DIBCO_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Create evaluator
    evaluator = DIBCO2012Evaluator(
        checkpoint_path=CHECKPOINT_PATH,
        dibco_root=DIBCO_ROOT,
        output_dir=OUTPUT_DIR
    )
    
    # Run evaluation
    evaluator.evaluate_all()
    
    print(f"\n{'='*80}")
    print(f"🎉 EVALUATION COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\nCheck results in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
