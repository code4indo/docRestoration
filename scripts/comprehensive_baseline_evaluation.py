#!/usr/bin/env python3
"""
Comprehensive Baseline Evaluation Script
Evaluates all methods on IDENTICAL test set for fair comparison:
1. Degraded Input (No restoration)
2. Otsu Thresholding
3. Sauvola Adaptive Thresholding
4. Proposed Method (Production V3)

All using same frozen recognizer and test set (n=712)
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from skimage.metrics import structural_similarity as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from train_enhanced.py location
sys.path.insert(0, str(project_root / "dual_modal_gan" / "scripts"))

from train_enhanced import (
    _parse_tfrecord_fn,
    read_charlist,
    decode_ctc_predictions,
    decode_label,
    calculate_cer,
    calculate_wer
)

# Import model architectures
from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed as load_frozen_recognizer

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Using {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")


def create_test_dataset(tfrecord_path, batch_size, train_split=0.7, val_split=0.15):
    """
    Load ONLY the test set (locked during training).
    
    Args:
        tfrecord_path: Path to TFRecord file
        batch_size: Batch size for evaluation
        train_split: Fraction used for training (to skip)
        val_split: Fraction used for validation (to skip)
        
    Returns:
        test_dataset: Test dataset (no shuffle, no repeat)
        test_size: Number of test samples
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # Get total size
    total_size = sum(1 for _ in dataset)
    
    # Calculate sizes
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    # Map parsing
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Skip train and validation, take test
    test_dataset = dataset.skip(train_size + val_size)
    
    # No shuffle, no repeat - fixed test set
    test_dataset = test_dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    print(f"\n📊 Test Set Configuration:")
    print(f"   Total samples in dataset: {total_size}")
    print(f"   Train samples (skipped):  {train_size} ({train_split*100:.0f}%)")
    print(f"   Val samples (skipped):    {val_size} ({val_split*100:.0f}%)")
    print(f"   Test samples (loaded):    {test_size} ({(1-train_split-val_split)*100:.0f}%)")
    print(f"   ✅ Test set isolated - no data leakage")
    
    return test_dataset, test_size


class ComprehensiveEvaluator:
    """Evaluates all baseline methods on identical test set"""
    
    def __init__(
        self,
        checkpoint_path: str,
        tfrecord_path: str,
        charset_path: str,
        output_dir: str,
        train_split: float = 0.7,
        val_split: float = 0.15,
        batch_size: int = 8,
        generator_version: str = 'enhanced'
    ):
        self.checkpoint_path = checkpoint_path
        self.tfrecord_path = tfrecord_path
        self.charset_path = charset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.generator_version = generator_version
        
        print("="*80)
        print("COMPREHENSIVE BASELINE EVALUATION")
        print("="*80)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"TFRecord: {tfrecord_path}")
        print(f"Charset: {charset_path}")
        print(f"Output: {output_dir}")
        print(f"Generator: {generator_version}")
        
        # Load charset
        self.charset = read_charlist(charset_path)
        self.vocab_size = len(self.charset) + 1
        print(f"   Loaded {self.vocab_size} characters (including blank)")
        
        # Load generator and recognizer
        self.generator, self.recognizer = self._load_models()
        
        # Load test dataset
        self.test_dataset, self.test_size = create_test_dataset(
            tfrecord_path, batch_size, train_split, val_split
        )
        
    def _load_models(self):
        """Load generator and recognizer from checkpoint"""
        print("\n📦 Loading Models...")
        
        # Build generator (use same input size as training: 1024x128x1)
        if self.generator_version == 'enhanced':
            generator = unet_enhanced(input_size=(1024, 128, 1))
        else:
            raise ValueError(f"Unknown generator version: {self.generator_version}")
        
        print(f"   Generator architecture: {self.generator_version}")
        print(f"   Input size: (1024, 128, 1)")
        
        # Build generator by calling it once
        dummy_input = tf.random.normal([1, 1024, 128, 1]) * 2.0 - 1.0  # [-1, 1] range
        _ = generator(dummy_input, training=False)
        
        # Load checkpoint
        ckpt = tf.train.Checkpoint(generator=generator)
        status = ckpt.restore(self.checkpoint_path).expect_partial()
        
        print(f"✅ Generator loaded from {self.checkpoint_path}")
        
        # Recognizer is embedded in generator as 'recognizer' attribute
        if hasattr(generator, 'recognizer'):
            recognizer = generator.recognizer
            print(f"✅ Recognizer (frozen) loaded from generator")
        else:
            raise ValueError("Generator does not have 'recognizer' attribute!")
        
        return generator, recognizer
    
    def otsu_thresholding(self, image: np.ndarray) -> np.ndarray:
        """Apply Otsu thresholding"""
        # Convert to uint8 [0, 255]
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        # Apply Otsu
        _, binary = cv2.threshold(
            img_uint8, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Convert back to float [-1, 1] (same range as generator output)
        return (binary.astype(np.float32) / 255.0) * 2.0 - 1.0
    
    def sauvola_thresholding(self, image: np.ndarray, window_size: int = 15, k: float = 0.2) -> np.ndarray:
        """Apply Sauvola adaptive thresholding"""
        # Convert to uint8 [0, 255]
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        # Calculate local mean and std
        mean = cv2.boxFilter(img_uint8.astype(float), -1, (window_size, window_size))
        sqmean = cv2.boxFilter(img_uint8.astype(float)**2, -1, (window_size, window_size))
        std = np.sqrt(np.maximum(sqmean - mean**2, 0))
        
        # Sauvola threshold
        R = 128  # Dynamic range of std
        threshold = mean * (1 + k * ((std / R) - 1))
        
        # Apply threshold
        binary = (img_uint8 > threshold).astype(np.uint8) * 255
        
        # Convert back to float [-1, 1]
        return (binary.astype(np.float32) / 255.0) * 2.0 - 1.0
    
    def calculate_f_measure(self, restored: np.ndarray, gt_clean: np.ndarray, threshold: float = 0.0) -> float:
        """
        Calculate F-measure for binary images
        Args:
            restored: Restored image in range [-1, 1]
            gt_clean: Ground truth in range [-1, 1]
            threshold: Binarization threshold (default 0.0 for [-1, 1] range)
        """
        # Binarize both images
        restored_bin = (restored > threshold).astype(np.uint8)
        gt_bin = (gt_clean > threshold).astype(np.uint8)
        
        # Calculate TP, FP, FN
        tp = np.sum((restored_bin == 1) & (gt_bin == 1))
        fp = np.sum((restored_bin == 1) & (gt_bin == 0))
        fn = np.sum((restored_bin == 0) & (gt_bin == 1))
        
        # Calculate precision and recall
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        
        # F-measure
        f_measure = 2 * precision * recall / (precision + recall + 1e-10)
        
        return f_measure
    
    def run_htr(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[str, str, float, float]:
        """
        Run HTR on image and calculate CER/WER
        
        Args:
            image: Input image tensor (batch_size, H, W, 1) in range [-1, 1]
            label: Ground truth label tensor
            
        Returns:
            pred_text: Predicted text
            gt_text: Ground truth text
            cer: Character error rate
            wer: Word error rate
        """
        # Run recognizer (already uses [-1, 1] range)
        logits = self.recognizer(image, training=False)
        
        # Decode predictions
        pred_texts = decode_ctc_predictions(logits, self.charset)
        pred_text = pred_texts[0] if len(pred_texts) > 0 else ''
        
        # Decode ground truth
        gt_texts = [decode_label(lbl.numpy(), self.charset) for lbl in label]
        gt_text = gt_texts[0] if len(gt_texts) > 0 else ''
        
        # Calculate CER and WER
        cer = calculate_cer(pred_text, gt_text)
        wer = calculate_wer(pred_text, gt_text)
        
        return pred_text, gt_text, cer, wer
    
    def evaluate_method(self, method_name: str) -> Dict:
        """Evaluate a specific method on test set"""
        print(f"\n{'='*80}")
        print(f"Evaluating: {method_name.upper()}")
        print(f"{'='*80}")
        
        results = {
            'psnr': [],
            'ssim': [],
            'f_measure': [],
            'cer': [],
            'wer': [],
            'cer_degraded': [],  # CER on degraded input
            'wer_degraded': []   # WER on degraded input
        }
        
        n_samples = 0
        start_time = time.time()
        
        for degraded_batch, clean_batch, label_batch in self.test_dataset:
            # Normalize to [-1, 1] for generator compatibility
            degraded_tanh = degraded_batch * 2.0 - 1.0
            clean_tanh = clean_batch * 2.0 - 1.0
            
            # Apply restoration based on method
            if method_name == 'no_restoration':
                restored_tanh = degraded_tanh
                f_measure = None  # Cannot calculate F-measure without binarization
                
            elif method_name == 'otsu':
                # Apply Otsu to each image in batch
                batch_results = []
                for i in range(degraded_batch.shape[0]):
                    # Work with [0, 1] range for classical methods
                    img_01 = degraded_batch[i, :, :, 0].numpy()
                    restored_np = self.otsu_thresholding(img_01)
                    batch_results.append(restored_np)
                
                restored_tanh = tf.constant(
                    np.array(batch_results)[:, :, :, None],
                    dtype=tf.float32
                )
                
                # Calculate F-measure for first image in batch
                f_measure = self.calculate_f_measure(
                    batch_results[0], 
                    clean_tanh.numpy()[0, :, :, 0]
                )
                
            elif method_name == 'sauvola':
                # Apply Sauvola to each image in batch
                batch_results = []
                for i in range(degraded_batch.shape[0]):
                    # Work with [0, 1] range for classical methods
                    img_01 = degraded_batch[i, :, :, 0].numpy()
                    restored_np = self.sauvola_thresholding(img_01)
                    batch_results.append(restored_np)
                
                restored_tanh = tf.constant(
                    np.array(batch_results)[:, :, :, None],
                    dtype=tf.float32
                )
                
                # Calculate F-measure for first image in batch
                f_measure = self.calculate_f_measure(
                    batch_results[0],
                    clean_tanh.numpy()[0, :, :, 0]
                )
                
            elif method_name == 'proposed':
                # Apply generator (already works with [-1, 1] range)
                restored_tanh = self.generator(degraded_tanh, training=False)
                
                # Calculate F-measure for first image in batch
                f_measure = self.calculate_f_measure(
                    restored_tanh.numpy()[0, :, :, 0],
                    clean_tanh.numpy()[0, :, :, 0]
                )
                
            else:
                raise ValueError(f"Unknown method: {method_name}")
            
            # Process each image in batch
            batch_size = degraded_batch.shape[0]
            for i in range(batch_size):
                # Extract single images
                restored_img = restored_tanh[i:i+1]
                clean_img = clean_tanh[i:i+1]
                degraded_img = degraded_tanh[i:i+1]
                label_img = label_batch[i:i+1]
                
                # Convert to [0, 1] for PSNR/SSIM calculation
                restored_01 = (restored_img.numpy()[0, :, :, 0] + 1.0) / 2.0
                clean_01 = (clean_img.numpy()[0, :, :, 0] + 1.0) / 2.0
                
                # Calculate PSNR and SSIM
                psnr_val = compute_psnr(clean_01, restored_01, data_range=1.0)
                ssim_val = compute_ssim(clean_01, restored_01, data_range=1.0)
                
                # Run HTR on restored image
                _, _, cer, wer = self.run_htr(restored_img, label_img)
                
                # Run HTR on degraded image (for baseline comparison)
                _, _, cer_deg, wer_deg = self.run_htr(degraded_img, label_img)
                
                # Store results
                results['psnr'].append(psnr_val)
                results['ssim'].append(ssim_val)
                if f_measure is not None and i == 0:  # Only first in batch
                    results['f_measure'].append(f_measure)
                results['cer'].append(cer)
                results['wer'].append(wer)
                results['cer_degraded'].append(cer_deg)
                results['wer_degraded'].append(wer_deg)
                
                n_samples += 1
            
            if n_samples % 100 == 0:
                print(f"  Processed {n_samples}/{self.test_size} samples...")
        
        elapsed = time.time() - start_time
        print(f"\n✅ Completed {method_name}: {n_samples} samples in {elapsed:.2f}s")
        
        # Calculate statistics with 95% CI
        stats_results = {}
        for metric_name, values in results.items():
            if len(values) == 0:
                continue
            
            values = np.array(values)
            mean = np.mean(values)
            std = np.std(values, ddof=1)  # Use sample std
            
            # Calculate 95% CI
            if len(values) > 1:
                ci = stats.t.interval(
                    0.95, 
                    len(values) - 1, 
                    loc=mean, 
                    scale=stats.sem(values)
                )
            else:
                ci = (mean, mean)
            
            stats_results[metric_name] = {
                'mean': float(mean),
                'std': float(std),
                'ci_lower': float(ci[0]),
                'ci_upper': float(ci[1]),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'n': len(values)
            }
        
        return {
            'method': method_name,
            'n_samples': n_samples,
            'elapsed_seconds': elapsed,
            'metrics': stats_results
        }
    
    def run_all_evaluations(self) -> Dict:
        """Run all method evaluations"""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE EVALUATION")
        print("="*80)
        
        all_results = {}
        
        # Evaluate all methods
        methods = [
            ('no_restoration', 'Tanpa Perbaikan'),
            ('otsu', 'Otsu Thresholding'),
            ('sauvola', 'Sauvola Adaptif'),
            ('proposed', 'Yang Diusulkan')
        ]
        
        for method_key, method_label in methods:
            result = self.evaluate_method(method_key)
            result['label'] = method_label
            all_results[method_key] = result
        
        # Save results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f'comprehensive_evaluation_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Print summary
        self.print_summary(all_results)
        
        # Generate LaTeX table
        self.generate_latex_table(all_results)
        
        return all_results
    
    def print_summary(self, results: Dict):
        """Print summary table"""
        print("\n" + "="*80)
        print("SUMMARY TABLE (untuk Paper)")
        print("="*80)
        
        print(f"\n{'Metode':<25} {'PSNR (dB)':<18} {'SSIM':<18} {'CER (%)':<18} {'WER (%)':<18}")
        print("-"*97)
        
        for method_key, data in results.items():
            metrics = data['metrics']
            label = data.get('label', method_key)
            
            # Format with mean ± std
            psnr_str = f"{metrics['psnr']['mean']:.2f}±{metrics['psnr']['std']:.2f}" if 'psnr' in metrics else 'N/A'
            ssim_str = f"{metrics['ssim']['mean']:.3f}±{metrics['ssim']['std']:.3f}" if 'ssim' in metrics else 'N/A'
            cer_str = f"{metrics['cer']['mean']*100:.1f}±{metrics['cer']['std']*100:.1f}" if 'cer' in metrics else 'N/A'
            wer_str = f"{metrics['wer']['mean']*100:.1f}±{metrics['wer']['std']*100:.1f}" if 'wer' in metrics else 'N/A'
            
            print(f"{label:<25} {psnr_str:<18} {ssim_str:<18} {cer_str:<18} {wer_str:<18}")
        
        print("\n" + "="*80)
        print("DEGRADED BASELINE (untuk perbandingan)")
        print("="*80)
        
        # Show degraded baseline CER/WER from any method (all should be same)
        first_method = list(results.values())[0]
        if 'cer_degraded' in first_method['metrics']:
            cer_deg = first_method['metrics']['cer_degraded']
            wer_deg = first_method['metrics']['wer_degraded']
            print(f"CER (Degraded Input): {cer_deg['mean']*100:.1f}% ± {cer_deg['std']*100:.1f}%")
            print(f"WER (Degraded Input): {wer_deg['mean']*100:.1f}% ± {wer_deg['std']*100:.1f}%")
    
    def generate_latex_table(self, results: Dict):
        """Generate LaTeX table for paper"""
        output_file = self.output_dir / 'table_latex.txt'
        
        with open(output_file, 'w') as f:
            f.write("% LaTeX Table - Copy to Paper\n")
            f.write("\\begin{table*}[!t]\n")
            f.write("\\renewcommand{\\arraystretch}{1.3}\n")
            f.write("\\caption{Hasil Kuantitatif pada Set Uji Terdegradasi Sintetis (n=712)}\n")
            f.write("\\label{table_synthetic_results}\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{|l|c|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Metode} & \\textbf{PSNR (dB) $\\uparrow$} & \\textbf{SSIM $\\uparrow$} & \\textbf{\\textit{F-measure} $\\uparrow$} & \\textbf{CER (\\%)} & \\textbf{WER (\\%)} \\\\\n")
            f.write("\\hline\n")
            
            # Write each method
            method_order = ['no_restoration', 'otsu', 'sauvola', 'proposed']
            for method_key in method_order:
                if method_key not in results:
                    continue
                
                data = results[method_key]
                metrics = data['metrics']
                label = data.get('label', method_key)
                
                # Format metrics
                psnr_str = f"{metrics['psnr']['mean']:.2f}$\\pm${metrics['psnr']['std']:.2f}" if 'psnr' in metrics else '-'
                ssim_str = f"{metrics['ssim']['mean']:.3f}$\\pm${metrics['ssim']['std']:.3f}" if 'ssim' in metrics else '-'
                
                if 'f_measure' in metrics and len(results[method_key]['metrics']['f_measure']) > 0:
                    fm = metrics['f_measure']
                    fm_str = f"{fm['mean']:.3f}$\\pm${fm['std']:.3f}"
                else:
                    fm_str = '-\\textsuperscript{a}'
                
                cer_str = f"{metrics['cer']['mean']*100:.1f}$\\pm${metrics['cer']['std']*100:.1f} $\\downarrow$"
                wer_str = f"{metrics['wer']['mean']*100:.1f}$\\pm${metrics['wer']['std']*100:.1f} $\\downarrow$"
                
                # Bold for proposed method
                if method_key == 'proposed':
                    f.write(f"\\textbf{{{label}}} & \\textbf{{{psnr_str}}} & \\textbf{{{ssim_str}}} & \\textbf{{{fm_str}}} & \\textbf{{{cer_str}}} & \\textbf{{{wer_str}}} \\\\\n")
                else:
                    f.write(f"{label} & {psnr_str} & {ssim_str} & {fm_str} & {cer_str} & {wer_str} \\\\\n")
                
                if method_key == 'sauvola':
                    f.write("\\hline\n")
            
            # Add clean GT upper bound (use proposed method's clean baseline)
            f.write("\\hline\n")
            if 'proposed' in results:
                # Calculate clean GT baseline from degraded baseline
                # Note: We don't have direct clean GT evaluation, so we note it
                f.write("\\textit{Batas Atas (GT Bersih)} & - & - & - & \\textit{[dari eval terpisah]} & \\textit{[dari eval terpisah]} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table*}\n")
        
        print(f"\n✅ LaTeX table saved to: {output_file}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive baseline evaluation')
    parser.add_argument('--checkpoint', type=str, 
                       default='dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model/ckpt-88',
                       help='Path to checkpoint')
    parser.add_argument('--tfrecord', type=str,
                       default='dual_modal_gan/data/dataset_gan.tfrecord',
                       help='Path to TFRecord file')
    parser.add_argument('--charset', type=str,
                       default='real_data_preparation/real_data_charlist.txt',
                       help='Path to charset file')
    parser.add_argument('--output_dir', type=str,
                       default='results/comprehensive_baseline_evaluation',
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--train_split', type=float, default=0.7,
                       help='Train split ratio')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation split ratio')
    parser.add_argument('--generator_version', type=str, default='enhanced',
                       choices=['enhanced'],
                       help='Generator version')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(
        checkpoint_path=args.checkpoint,
        tfrecord_path=args.tfrecord,
        charset_path=args.charset,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        batch_size=args.batch_size,
        generator_version=args.generator_version
    )
    
    # Run all evaluations
    results = evaluator.run_all_evaluations()
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
