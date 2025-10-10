#!/usr/bin/env python3
"""
Eksperimen Komparasi Keterbacaan: Before vs After Restoration

Tujuan:
- Membandingkan visual quality dokumen utuh sebelum dan setelah restorasi GAN
- Mengukur improvement readability secara kuantitatif dan kualitatif
- Validasi bahwa model yang dilatih dengan gambar baris teks dapat meningkatkan dokumen utuh

Metrik:
1. Visual Quality: PSNR, SSIM (jika ada ground truth)
2. Text Clarity: Edge sharpness, contrast ratio
3. HTR Readability: CER before vs after
4. Human Assessment: Side-by-side visualization

Author: AI/ML Engineer
Date: 2025-10-07
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class ReadabilityComparator:
    """Compare readability metrics before and after restoration."""
    
    def __init__(self, original_dir: Path, restored_dir: Path, output_dir: Path):
        self.original_dir = Path(original_dir)
        self.restored_dir = Path(restored_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance.
        Higher value = sharper image.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return float(variance)
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        """
        Calculate RMS contrast (root mean square).
        Higher value = better contrast.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Normalize to [0, 1]
        gray_norm = gray.astype(float) / 255.0
        
        # RMS contrast
        mean = np.mean(gray_norm)
        rms_contrast = np.sqrt(np.mean((gray_norm - mean) ** 2))
        
        return float(rms_contrast * 100)  # Scale to percentage
    
    def calculate_noise_level(self, image: np.ndarray) -> float:
        """
        Estimate noise level using high-frequency components.
        Lower value = cleaner image.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply median filter (removes noise but preserves edges)
        median = cv2.medianBlur(gray, 5)
        
        # Difference = noise estimate
        noise = cv2.absdiff(gray, median)
        noise_level = np.mean(noise)
        
        return float(noise_level)
    
    def calculate_edge_density(self, image: np.ndarray) -> float:
        """
        Calculate edge density (text stroke clarity).
        Higher value = more defined edges = clearer text.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Density = percentage of edge pixels
        density = (np.count_nonzero(edges) / edges.size) * 100
        
        return float(density)
    
    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> Optional[float]:
        """
        Calculate PSNR between two images.
        Only works if images have same dimensions.
        """
        try:
            # Ensure same size
            if img1.shape != img2.shape:
                # Resize img2 to match img1
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # Convert to grayscale if needed
            if len(img1.shape) == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            if len(img2.shape) == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
            
            if mse == 0:
                return float('inf')
            
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            
            return float(psnr)
        
        except Exception as e:
            print(f"  ⚠️  PSNR calculation failed: {e}")
            return None
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> Optional[float]:
        """
        Calculate SSIM (Structural Similarity Index).
        Requires same dimensions.
        """
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Ensure same size
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # Convert to grayscale if needed
            if len(img1.shape) == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            if len(img2.shape) == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            score = ssim(img1, img2, data_range=255)
            
            return float(score)
        
        except ImportError:
            print("  ⚠️  scikit-image not installed, skipping SSIM")
            return None
        except Exception as e:
            print(f"  ⚠️  SSIM calculation failed: {e}")
            return None
    
    def analyze_document(self, original_path: Path, restored_path: Path) -> Dict:
        """Analyze single document before and after restoration."""
        
        doc_name = original_path.stem
        print(f"\n📄 Analyzing: {doc_name}")
        
        # Load images
        original = cv2.imread(str(original_path))
        restored = cv2.imread(str(restored_path))
        
        if original is None:
            print(f"  ❌ Failed to load original: {original_path}")
            return None
        
        if restored is None:
            print(f"  ❌ Failed to load restored: {restored_path}")
            return None
        
        # Calculate metrics
        result = {
            'document': doc_name,
            'original': {},
            'restored': {},
            'improvement': {}
        }
        
        print("  📊 Calculating metrics...")
        
        # Original metrics
        result['original']['sharpness'] = self.calculate_sharpness(original)
        result['original']['contrast'] = self.calculate_contrast(original)
        result['original']['noise_level'] = self.calculate_noise_level(original)
        result['original']['edge_density'] = self.calculate_edge_density(original)
        
        # Restored metrics
        result['restored']['sharpness'] = self.calculate_sharpness(restored)
        result['restored']['contrast'] = self.calculate_contrast(restored)
        result['restored']['noise_level'] = self.calculate_noise_level(restored)
        result['restored']['edge_density'] = self.calculate_edge_density(restored)
        
        # Improvement calculations
        result['improvement']['sharpness'] = \
            ((result['restored']['sharpness'] - result['original']['sharpness']) / 
             result['original']['sharpness'] * 100)
        
        result['improvement']['contrast'] = \
            ((result['restored']['contrast'] - result['original']['contrast']) / 
             result['original']['contrast'] * 100)
        
        result['improvement']['noise_reduction'] = \
            ((result['original']['noise_level'] - result['restored']['noise_level']) / 
             result['original']['noise_level'] * 100)
        
        result['improvement']['edge_density'] = \
            ((result['restored']['edge_density'] - result['original']['edge_density']) / 
             result['original']['edge_density'] * 100)
        
        # PSNR & SSIM (if applicable)
        psnr = self.calculate_psnr(original, restored)
        ssim = self.calculate_ssim(original, restored)
        
        if psnr:
            result['psnr'] = psnr
        if ssim:
            result['ssim'] = ssim
        
        # Print summary
        print(f"  ✅ Sharpness:     {result['original']['sharpness']:.1f} → "
              f"{result['restored']['sharpness']:.1f} "
              f"({result['improvement']['sharpness']:+.1f}%)")
        print(f"  ✅ Contrast:      {result['original']['contrast']:.1f}% → "
              f"{result['restored']['contrast']:.1f}% "
              f"({result['improvement']['contrast']:+.1f}%)")
        print(f"  ✅ Noise Level:   {result['original']['noise_level']:.2f} → "
              f"{result['restored']['noise_level']:.2f} "
              f"({result['improvement']['noise_reduction']:+.1f}% reduction)")
        print(f"  ✅ Edge Density:  {result['original']['edge_density']:.2f}% → "
              f"{result['restored']['edge_density']:.2f}% "
              f"({result['improvement']['edge_density']:+.1f}%)")
        
        if psnr:
            print(f"  ✅ PSNR:          {psnr:.2f} dB")
        if ssim:
            print(f"  ✅ SSIM:          {ssim:.4f}")
        
        return result
    
    def create_side_by_side_comparison(
        self, 
        original_path: Path, 
        restored_path: Path,
        metrics: Dict,
        output_path: Path
    ):
        """Create visual side-by-side comparison with metrics overlay."""
        
        # Load images
        original = cv2.imread(str(original_path))
        restored = cv2.imread(str(restored_path))
        
        # Convert BGR to RGB for matplotlib
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        restored_rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 0.6, 0.4], hspace=0.3, wspace=0.2)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original_rgb)
        ax1.set_title('BEFORE RESTORATION (Original)', fontsize=16, fontweight='bold', color='red')
        ax1.axis('off')
        
        # Restored image
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(restored_rgb)
        ax2.set_title('AFTER RESTORATION (GAN-HTR)', fontsize=16, fontweight='bold', color='green')
        ax2.axis('off')
        
        # Zoomed region (top-left corner for detail)
        h, w = original.shape[:2]
        zoom_h, zoom_w = min(400, h//3), min(600, w//3)
        
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.imshow(original_rgb[:zoom_h, :zoom_w])
        ax3.set_title('Detail: Before', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.imshow(restored_rgb[:zoom_h, :zoom_w])
        ax4.set_title('Detail: After', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # Metrics table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        metrics_text = f"""
QUANTITATIVE COMPARISON:

Metric                    Before          After           Improvement
{'='*80}
Sharpness (Laplacian)    {metrics['original']['sharpness']:8.1f}      {metrics['restored']['sharpness']:8.1f}      {metrics['improvement']['sharpness']:+6.1f}%
Contrast (RMS)           {metrics['original']['contrast']:8.1f}%     {metrics['restored']['contrast']:8.1f}%     {metrics['improvement']['contrast']:+6.1f}%
Noise Level              {metrics['original']['noise_level']:8.2f}      {metrics['restored']['noise_level']:8.2f}      {metrics['improvement']['noise_reduction']:+6.1f}% reduction
Edge Density             {metrics['original']['edge_density']:8.2f}%     {metrics['restored']['edge_density']:8.2f}%     {metrics['improvement']['edge_density']:+6.1f}%
"""
        
        if 'psnr' in metrics:
            metrics_text += f"PSNR                     N/A             {metrics['psnr']:8.2f} dB   (higher = better)\n"
        if 'ssim' in metrics:
            metrics_text += f"SSIM                     N/A             {metrics['ssim']:8.4f}     (1.0 = perfect)\n"
        
        ax5.text(0.05, 0.5, metrics_text, 
                fontfamily='monospace', fontsize=11,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Overall assessment
        avg_improvement = np.mean([
            metrics['improvement']['sharpness'],
            metrics['improvement']['contrast'],
            metrics['improvement']['noise_reduction'],
            metrics['improvement']['edge_density']
        ])
        
        if avg_improvement > 10:
            assessment = "✅ SIGNIFICANT IMPROVEMENT"
            color = 'green'
        elif avg_improvement > 5:
            assessment = "✅ MODERATE IMPROVEMENT"
            color = 'orange'
        elif avg_improvement > 0:
            assessment = "✅ SLIGHT IMPROVEMENT"
            color = 'blue'
        else:
            assessment = "⚠️ NO CLEAR IMPROVEMENT"
            color = 'red'
        
        fig.text(0.5, 0.02, f'Overall Assessment: {assessment} (Avg: {avg_improvement:+.1f}%)',
                ha='center', fontsize=14, fontweight='bold', color=color,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Save
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved comparison: {output_path}")
    
    def run_comparison(self, sample_size: Optional[int] = None):
        """Run full comparison on all documents."""
        
        print("="*80)
        print("🔬 EKSPERIMEN KOMPARASI KETERBACAAN")
        print("="*80)
        print(f"📁 Original Dir:  {self.original_dir}")
        print(f"📁 Restored Dir:  {self.restored_dir}")
        print(f"📁 Output Dir:    {self.output_dir}")
        print("="*80)
        
        # Find all original images
        original_images = sorted(self.original_dir.glob("*.jpg")) + \
                         sorted(self.original_dir.glob("*.png"))
        
        if not original_images:
            print("❌ No images found in original directory!")
            return
        
        print(f"📊 Found {len(original_images)} documents")
        
        # Limit sample size if specified
        if sample_size:
            original_images = original_images[:sample_size]
            print(f"🎯 Processing sample of {len(original_images)} documents")
        
        # Process each document
        comparisons_dir = self.output_dir / "comparisons"
        comparisons_dir.mkdir(exist_ok=True)
        
        for orig_path in tqdm(original_images, desc="Analyzing documents"):
            # Find corresponding restored image (search recursively)
            restored_candidates = list(self.restored_dir.rglob(f"{orig_path.stem}_restored.png"))
            
            if not restored_candidates:
                # Try alternative naming
                restored_candidates = list(self.restored_dir.rglob(f"{orig_path.stem}_restored.jpg"))
            
            if not restored_candidates:
                print(f"⚠️  Skipping {orig_path.stem}: No restored version found")
                continue
            
            restored_path = restored_candidates[0]  # Use first match
            
            # Analyze
            metrics = self.analyze_document(orig_path, restored_path)
            
            if metrics:
                self.results.append(metrics)
                
                # Create visual comparison
                comparison_path = comparisons_dir / f"{orig_path.stem}_comparison.png"
                self.create_side_by_side_comparison(
                    orig_path, restored_path, metrics, comparison_path
                )
        
        # Save results
        self.save_results()
        
        # Generate summary report
        self.generate_summary_report()
    
    def save_results(self):
        """Save detailed results to JSON."""
        
        results_path = self.output_dir / "readability_comparison_results.json"
        
        results_data = {
            'experiment': 'Readability Comparison: Before vs After Restoration',
            'date': datetime.now().isoformat(),
            'original_dir': str(self.original_dir),
            'restored_dir': str(self.restored_dir),
            'total_documents': len(self.results),
            'documents': self.results
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved detailed results: {results_path}")
    
    def generate_summary_report(self):
        """Generate aggregate statistics and summary report."""
        
        if not self.results:
            print("❌ No results to summarize!")
            return
        
        print("\n" + "="*80)
        print("📊 SUMMARY REPORT")
        print("="*80)
        
        # Aggregate statistics
        sharpness_improvements = [r['improvement']['sharpness'] for r in self.results]
        contrast_improvements = [r['improvement']['contrast'] for r in self.results]
        noise_reductions = [r['improvement']['noise_reduction'] for r in self.results]
        edge_improvements = [r['improvement']['edge_density'] for r in self.results]
        
        print(f"\n📈 AVERAGE IMPROVEMENTS (n={len(self.results)}):")
        print(f"  • Sharpness:      {np.mean(sharpness_improvements):+.1f}% ± {np.std(sharpness_improvements):.1f}%")
        print(f"  • Contrast:       {np.mean(contrast_improvements):+.1f}% ± {np.std(contrast_improvements):.1f}%")
        print(f"  • Noise Reduction: {np.mean(noise_reductions):+.1f}% ± {np.std(noise_reductions):.1f}%")
        print(f"  • Edge Density:   {np.mean(edge_improvements):+.1f}% ± {np.std(edge_improvements):.1f}%")
        
        # Overall improvement
        overall_improvements = [
            (r['improvement']['sharpness'] + 
             r['improvement']['contrast'] + 
             r['improvement']['noise_reduction'] + 
             r['improvement']['edge_density']) / 4
            for r in self.results
        ]
        
        print(f"\n🎯 OVERALL READABILITY IMPROVEMENT:")
        print(f"  • Mean:           {np.mean(overall_improvements):+.1f}%")
        print(f"  • Median:         {np.median(overall_improvements):+.1f}%")
        print(f"  • Std Dev:        {np.std(overall_improvements):.1f}%")
        print(f"  • Min:            {np.min(overall_improvements):+.1f}%")
        print(f"  • Max:            {np.max(overall_improvements):+.1f}%")
        
        # Success rate
        success_count = sum(1 for x in overall_improvements if x > 5)
        success_rate = (success_count / len(overall_improvements)) * 100
        
        print(f"\n✅ SUCCESS RATE:")
        print(f"  • Documents with >5% improvement:  {success_count}/{len(self.results)} ({success_rate:.1f}%)")
        
        # PSNR/SSIM if available
        psnr_values = [r.get('psnr') for r in self.results if 'psnr' in r]
        ssim_values = [r.get('ssim') for r in self.results if 'ssim' in r]
        
        if psnr_values:
            print(f"\n📐 PSNR (Peak Signal-to-Noise Ratio):")
            print(f"  • Mean:           {np.mean(psnr_values):.2f} dB")
            print(f"  • Range:          {np.min(psnr_values):.2f} - {np.max(psnr_values):.2f} dB")
        
        if ssim_values:
            print(f"\n📐 SSIM (Structural Similarity):")
            print(f"  • Mean:           {np.mean(ssim_values):.4f}")
            print(f"  • Range:          {np.min(ssim_values):.4f} - {np.max(ssim_values):.4f}")
        
        # Best and worst cases
        best_idx = np.argmax(overall_improvements)
        worst_idx = np.argmin(overall_improvements)
        
        print(f"\n🏆 BEST CASE:")
        print(f"  • Document:       {self.results[best_idx]['document']}")
        print(f"  • Improvement:    {overall_improvements[best_idx]:+.1f}%")
        
        print(f"\n⚠️  WORST CASE:")
        print(f"  • Document:       {self.results[worst_idx]['document']}")
        print(f"  • Improvement:    {overall_improvements[worst_idx]:+.1f}%")
        
        # Create summary plot
        self.create_summary_plots(overall_improvements)
        
        print("\n" + "="*80)
        print("✅ COMPARISON COMPLETE!")
        print(f"📁 Results saved to: {self.output_dir}")
        print("="*80)
    
    def create_summary_plots(self, overall_improvements):
        """Create summary visualization plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Readability Improvement Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Histogram of overall improvements
        axes[0, 0].hist(overall_improvements, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(overall_improvements), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(overall_improvements):+.1f}%')
        axes[0, 0].set_xlabel('Overall Improvement (%)', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        axes[0, 0].set_title('Distribution of Overall Improvements')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Plot 2: Individual metrics comparison
        metrics = ['Sharpness', 'Contrast', 'Noise\nReduction', 'Edge\nDensity']
        means = [
            np.mean([r['improvement']['sharpness'] for r in self.results]),
            np.mean([r['improvement']['contrast'] for r in self.results]),
            np.mean([r['improvement']['noise_reduction'] for r in self.results]),
            np.mean([r['improvement']['edge_density'] for r in self.results])
        ]
        
        bars = axes[0, 1].bar(metrics, means, color=['green' if x > 0 else 'red' for x in means])
        axes[0, 1].axhline(0, color='black', linewidth=0.8)
        axes[0, 1].set_ylabel('Average Improvement (%)', fontweight='bold')
        axes[0, 1].set_title('Improvement by Metric')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:+.1f}%',
                           ha='center', va='bottom' if height > 0 else 'top',
                           fontweight='bold')
        
        # Plot 3: Cumulative distribution
        sorted_improvements = np.sort(overall_improvements)
        cumulative = np.arange(1, len(sorted_improvements) + 1) / len(sorted_improvements) * 100
        
        axes[1, 0].plot(sorted_improvements, cumulative, linewidth=2, color='purple')
        axes[1, 0].axvline(0, color='red', linestyle='--', label='No improvement')
        axes[1, 0].axvline(5, color='orange', linestyle='--', label='5% threshold')
        axes[1, 0].set_xlabel('Overall Improvement (%)', fontweight='bold')
        axes[1, 0].set_ylabel('Cumulative Percentage (%)', fontweight='bold')
        axes[1, 0].set_title('Cumulative Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 4: Top 10 and bottom 10 documents
        top_10_idx = np.argsort(overall_improvements)[-10:]
        top_10_docs = [self.results[i]['document'][:15] for i in top_10_idx]
        top_10_vals = [overall_improvements[i] for i in top_10_idx]
        
        axes[1, 1].barh(range(10), top_10_vals, color='green', alpha=0.7)
        axes[1, 1].set_yticks(range(10))
        axes[1, 1].set_yticklabels(top_10_docs, fontsize=8)
        axes[1, 1].set_xlabel('Improvement (%)', fontweight='bold')
        axes[1, 1].set_title('Top 10 Improved Documents')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        summary_plot_path = self.output_dir / "summary_analysis.png"
        plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved summary plots: {summary_plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare readability before and after GAN-HTR restoration"
    )
    parser.add_argument(
        '--original_dir',
        type=str,
        required=True,
        help='Directory containing original (degraded) documents'
    )
    parser.add_argument(
        '--restored_dir',
        type=str,
        required=True,
        help='Directory containing restored documents'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/readability_comparison',
        help='Output directory for comparison results'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=None,
        help='Process only first N documents (for testing)'
    )
    
    args = parser.parse_args()
    
    # Run comparison
    comparator = ReadabilityComparator(
        original_dir=args.original_dir,
        restored_dir=args.restored_dir,
        output_dir=args.output_dir
    )
    
    comparator.run_comparison(sample_size=args.sample_size)


if __name__ == "__main__":
    main()
