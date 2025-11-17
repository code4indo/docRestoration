#!/usr/bin/env python3
"""
IMPROVED VERSION: Analisis blur relatif pada region iron gall corrosion
dengan membandingkan sharpness antara region iron gall vs region normal.

Konsep: "Slight blur" = region iron gall memiliki sharpness lebih rendah 
dibandingkan region normal pada gambar yang sama.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from typing import List, Tuple, Dict
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RelativeBlurResult:
    """Hasil analisis blur relatif"""
    filename: str
    iron_gall_blur_score: float
    normal_region_blur_score: float
    blur_ratio: float  # iron_gall / normal (< 1 = iron gall lebih blur)
    has_relative_blur: bool
    severity: str
    iron_gall_coverage: float
    sample_regions: List[Dict]  # [{type, bbox, blur_score}]


class RelativeBlurAnalyzer:
    """Analyzer untuk mendeteksi blur relatif pada region iron gall"""
    
    def __init__(
        self,
        degraded_dir: Path,
        restored_dir: Path,
        output_dir: Path,
        blur_ratio_threshold: float = 0.85,  # Iron gall < 85% normal = slight blur
        sample_size: int = 256
    ):
        self.degraded_dir = Path(degraded_dir)
        self.restored_dir = Path(restored_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.blur_ratio_threshold = blur_ratio_threshold
        self.sample_size = sample_size
        self.results: List[RelativeBlurResult] = []
    
    def detect_iron_gall_regions(self, degraded_img: np.ndarray) -> np.ndarray:
        """Deteksi iron gall corrosion regions"""
        if len(degraded_img.shape) == 3:
            gray = cv2.cvtColor(degraded_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = degraded_img.copy()
        
        # Multi-threshold approach untuk iron gall
        # Iron gall sangat gelap (0-60) dan membentuk cluster
        _, very_dark = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        iron_gall_mask = cv2.morphologyEx(very_dark, cv2.MORPH_CLOSE, kernel)
        iron_gall_mask = cv2.morphologyEx(iron_gall_mask, cv2.MORPH_OPEN, 
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        
        return iron_gall_mask
    
    def calculate_local_blur(self, image: np.ndarray) -> float:
        """Hitung blur score dengan Laplacian variance"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    def sample_regions(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        is_iron_gall: bool = True,
        n_samples: int = 10
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Sample random regions dari mask dan hitung blur score.
        
        Returns:
            List of (bbox, blur_score)
        """
        h, w = mask.shape
        samples = []
        
        # Erode mask sedikit untuk avoid edge effects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        safe_mask = cv2.erode(mask, kernel)
        
        # Find valid positions
        coords = np.argwhere(safe_mask > 0)
        
        if len(coords) < n_samples:
            n_samples = max(1, len(coords))
        
        if len(coords) == 0:
            return []
        
        # Random sampling
        sampled_indices = np.random.choice(len(coords), size=n_samples, replace=False)
        
        for idx in sampled_indices:
            y, x = coords[idx]
            
            # Extract patch centered at (x, y)
            x1 = max(0, x - self.sample_size // 2)
            y1 = max(0, y - self.sample_size // 2)
            x2 = min(w, x1 + self.sample_size)
            y2 = min(h, y1 + self.sample_size)
            
            # Adjust jika patch terlalu kecil
            if (x2 - x1) < self.sample_size // 2 or (y2 - y1) < self.sample_size // 2:
                continue
            
            patch = image[y1:y2, x1:x2]
            patch_mask = safe_mask[y1:y2, x1:x2]
            
            # Hitung coverage
            coverage = np.sum(patch_mask > 0) / patch_mask.size
            
            # Skip jika coverage terlalu rendah
            if is_iron_gall and coverage < 0.3:
                continue
            elif not is_iron_gall and coverage < 0.7:  # Normal region harus mostly clear
                continue
            
            blur_score = self.calculate_local_blur(patch)
            samples.append(((x1, y1, x2-x1, y2-y1), blur_score))
        
        return samples
    
    def analyze_single_image(self, degraded_path: Path, restored_path: Path) -> RelativeBlurResult:
        """Analisis blur relatif pada satu gambar"""
        logger.info(f"Analyzing: {degraded_path.name}")
        
        degraded = cv2.imread(str(degraded_path))
        restored = cv2.imread(str(restored_path))
        
        if degraded is None or restored is None:
            logger.error(f"Failed to load {degraded_path.name}")
            return None
        
        # Deteksi iron gall regions
        iron_gall_mask = self.detect_iron_gall_regions(degraded)
        iron_gall_coverage = np.sum(iron_gall_mask > 0) / iron_gall_mask.size
        
        # Skip jika tidak ada iron gall yang cukup
        if iron_gall_coverage < 0.05:
            logger.info(f"  Skipped: iron gall coverage too low ({iron_gall_coverage:.2%})")
            return None
        
        # Buat normal region mask (inverse of iron gall)
        normal_mask = cv2.bitwise_not(iron_gall_mask)
        
        # Sample regions dari iron gall dan normal
        iron_gall_samples = self.sample_regions(restored, iron_gall_mask, is_iron_gall=True, n_samples=15)
        normal_samples = self.sample_regions(restored, normal_mask, is_iron_gall=False, n_samples=15)
        
        if len(iron_gall_samples) == 0 or len(normal_samples) == 0:
            logger.info(f"  Skipped: insufficient samples (iron_gall={len(iron_gall_samples)}, normal={len(normal_samples)})")
            return None
        
        # Hitung average blur scores
        iron_gall_blur_scores = [s[1] for s in iron_gall_samples]
        normal_blur_scores = [s[1] for s in normal_samples]
        
        avg_iron_gall_blur = np.mean(iron_gall_blur_scores)
        avg_normal_blur = np.mean(normal_blur_scores)
        
        # Hitung blur ratio
        blur_ratio = avg_iron_gall_blur / avg_normal_blur if avg_normal_blur > 0 else 1.0
        
        # Klasifikasi
        has_relative_blur = blur_ratio < self.blur_ratio_threshold
        
        if blur_ratio < 0.7:
            severity = 'moderate'
        elif blur_ratio < self.blur_ratio_threshold:
            severity = 'slight'
        else:
            severity = 'none'
        
        logger.info(f"  Iron gall blur: {avg_iron_gall_blur:.2f}, Normal blur: {avg_normal_blur:.2f}")
        logger.info(f"  Blur ratio: {blur_ratio:.3f}, Severity: {severity}")
        logger.info(f"  Iron gall coverage: {iron_gall_coverage:.2%}")
        
        # Prepare sample regions for visualization
        sample_regions = []
        for bbox, score in iron_gall_samples[:5]:
            sample_regions.append({
                'type': 'iron_gall',
                'bbox': [int(x) for x in bbox],  # Convert to int
                'blur_score': float(score)
            })
        for bbox, score in normal_samples[:5]:
            sample_regions.append({
                'type': 'normal',
                'bbox': [int(x) for x in bbox],  # Convert to int
                'blur_score': float(score)
            })
        
        result = RelativeBlurResult(
            filename=degraded_path.name,
            iron_gall_blur_score=float(avg_iron_gall_blur),
            normal_region_blur_score=float(avg_normal_blur),
            blur_ratio=float(blur_ratio),
            has_relative_blur=bool(has_relative_blur),
            severity=severity,
            iron_gall_coverage=float(iron_gall_coverage),
            sample_regions=sample_regions
        )
        
        return result
    
    def visualize_comparison(
        self, 
        degraded_path: Path, 
        restored_path: Path, 
        result: RelativeBlurResult
    ):
        """Buat visualisasi komparatif dengan sample regions highlighted"""
        degraded = cv2.imread(str(degraded_path))
        restored = cv2.imread(str(restored_path))
        iron_gall_mask = self.detect_iron_gall_regions(degraded)
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Row 1: Full images
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
        ax1.set_title('Degraded Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 2:])
        restored_rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
        ax2.imshow(restored_rgb)
        ax2.set_title('Restored Image with Sample Regions', fontsize=14, fontweight='bold')
        
        # Draw sample regions
        for region in result.sample_regions:
            x, y, w, h = region['bbox']
            color = 'red' if region['type'] == 'iron_gall' else 'blue'
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                    edgecolor=color, facecolor='none')
            ax2.add_patch(rect)
        ax2.axis('off')
        
        # Row 2 & 3: Sample crops comparison
        iron_gall_samples = [r for r in result.sample_regions if r['type'] == 'iron_gall']
        normal_samples = [r for r in result.sample_regions if r['type'] == 'normal']
        
        # Iron gall samples (row 2)
        for i in range(4):
            ax = fig.add_subplot(gs[1, i])
            if i < len(iron_gall_samples):
                region = iron_gall_samples[i]
                x, y, w, h = region['bbox']
                crop = restored[y:y+h, x:x+w]
                ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                ax.set_title(f"Iron Gall #{i+1}\nBlur: {region['blur_score']:.1f}", 
                           fontsize=10, color='red')
            ax.axis('off')
        
        # Normal samples (row 3)
        for i in range(4):
            ax = fig.add_subplot(gs[2, i])
            if i < len(normal_samples):
                region = normal_samples[i]
                x, y, w, h = region['bbox']
                crop = restored[y:y+h, x:x+w]
                ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                ax.set_title(f"Normal #{i+1}\nBlur: {region['blur_score']:.1f}", 
                           fontsize=10, color='blue')
            ax.axis('off')
        
        # Add statistics
        stats_text = (
            f"Filename: {result.filename}\n"
            f"Iron Gall Blur Score: {result.iron_gall_blur_score:.2f}\n"
            f"Normal Region Blur Score: {result.normal_region_blur_score:.2f}\n"
            f"Blur Ratio: {result.blur_ratio:.3f} ({'<' if result.blur_ratio < self.blur_ratio_threshold else '>='} {self.blur_ratio_threshold})\n"
            f"Severity: {result.severity.upper()}\n"
            f"Iron Gall Coverage: {result.iron_gall_coverage:.2%}\n\n"
            f"Interpretation: {'Iron gall regions are BLURRIER than normal' if result.has_relative_blur else 'No significant relative blur'}"
        )
        
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=11, 
                family='monospace', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save
        output_path = self.output_dir / f"relative_blur_{result.filename.replace('.jpg', '.png')}"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved visualization: {output_path.name}")
    
    def extract_comparison_crops(
        self, 
        degraded_path: Path, 
        restored_path: Path, 
        result: RelativeBlurResult
    ):
        """Ekstrak crops untuk paper - side by side degraded vs restored"""
        degraded = cv2.imread(str(degraded_path))
        restored = cv2.imread(str(restored_path))
        
        crops_dir = self.output_dir / 'comparison_crops'
        crops_dir.mkdir(exist_ok=True)
        
        base_name = result.filename.replace('.jpg', '')
        
        # Ekstrak iron gall samples
        iron_gall_samples = [r for r in result.sample_regions if r['type'] == 'iron_gall']
        
        for idx, region in enumerate(iron_gall_samples[:3]):  # Top 3
            x, y, w, h = region['bbox']
            
            # Add margin
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(restored.shape[1], x + w + margin)
            y2 = min(restored.shape[0], y + h + margin)
            
            crop_degraded = degraded[y1:y2, x1:x2]
            crop_restored = restored[y1:y2, x1:x2]
            
            # Save side-by-side
            combined = np.hstack([crop_degraded, crop_restored])
            
            # Add text annotation
            combined_annotated = combined.copy()
            cv2.putText(combined_annotated, f"Degraded | Restored", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(combined_annotated, f"Blur Score: {region['blur_score']:.1f}", 
                       (10, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 0, 255), 2)
            
            output_path = crops_dir / f"{base_name}_iron_gall_{idx+1}_comparison.png"
            cv2.imwrite(str(output_path), combined_annotated)
        
        logger.info(f"  Extracted {len(iron_gall_samples[:3])} comparison crops")
    
    def run_analysis(self):
        """Jalankan analisis untuk semua gambar"""
        logger.info("=" * 60)
        logger.info("RELATIVE BLUR ANALYSIS: Iron Gall vs Normal Regions")
        logger.info("=" * 60)
        
        degraded_files = sorted(self.degraded_dir.glob("*.jpg"))
        
        for degraded_path in degraded_files:
            base_name = degraded_path.stem
            restored_candidates = [
                self.restored_dir / f"{base_name}_restored.tiff",
                self.restored_dir / f"{base_name}_restored.tif",
                self.restored_dir / f"{base_name}.tiff",
            ]
            
            restored_path = None
            for candidate in restored_candidates:
                if candidate.exists():
                    restored_path = candidate
                    break
            
            if restored_path is None:
                logger.warning(f"No restored image for {degraded_path.name}")
                continue
            
            result = self.analyze_single_image(degraded_path, restored_path)
            
            if result is None:
                continue
            
            self.results.append(result)
            
            # Visualize jika ada relative blur
            if result.has_relative_blur:
                self.visualize_comparison(degraded_path, restored_path, result)
                self.extract_comparison_crops(degraded_path, restored_path, result)
        
        self.generate_report()
    
    def generate_report(self):
        """Generate summary report"""
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY REPORT")
        logger.info("=" * 60)
        
        total = len(self.results)
        slight_cases = [r for r in self.results if r.severity == 'slight']
        moderate_cases = [r for r in self.results if r.severity == 'moderate']
        
        logger.info(f"Total images analyzed: {total}")
        logger.info(f"  - No relative blur: {total - len(slight_cases) - len(moderate_cases)}")
        logger.info(f"  - Slight relative blur: {len(slight_cases)}")
        logger.info(f"  - Moderate relative blur: {len(moderate_cases)}")
        
        if len(slight_cases) > 0:
            logger.info("\n📊 SLIGHT BLUR CASES (untuk justifikasi paper):")
            for r in slight_cases:
                logger.info(f"  • {r.filename}")
                logger.info(f"    - Blur ratio: {r.blur_ratio:.3f} (iron_gall/normal)")
                logger.info(f"    - Iron gall score: {r.iron_gall_blur_score:.2f}")
                logger.info(f"    - Normal score: {r.normal_region_blur_score:.2f}")
                logger.info(f"    - Coverage: {r.iron_gall_coverage:.2%}")
        
        if len(moderate_cases) > 0:
            logger.info("\n⚠️  MODERATE BLUR CASES:")
            for r in moderate_cases:
                logger.info(f"  • {r.filename}: ratio={r.blur_ratio:.3f}")
        
        # Save JSON report
        report = {
            'methodology': {
                'description': 'Relative blur analysis comparing iron gall regions vs normal regions',
                'blur_ratio_threshold': self.blur_ratio_threshold,
                'sample_size': self.sample_size,
                'interpretation': 'Blur ratio < threshold indicates iron gall regions are blurrier than normal'
            },
            'summary': {
                'total_analyzed': total,
                'no_blur': total - len(slight_cases) - len(moderate_cases),
                'slight_blur': len(slight_cases),
                'moderate_blur': len(moderate_cases)
            },
            'cases': [asdict(r) for r in self.results]
        }
        
        report_path = self.output_dir / 'relative_blur_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✅ Report saved: {report_path}")
        logger.info("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze relative blur: iron gall regions vs normal regions'
    )
    parser.add_argument('--degraded-dir', type=str, default='DokumenRusak/forPaper')
    parser.add_argument('--restored-dir', type=str, default='DokumenRusak/forPaper_results')
    parser.add_argument('--output-dir', type=str, default='analysis_results/relative_blur')
    parser.add_argument('--blur-ratio-threshold', type=float, default=0.85)
    parser.add_argument('--sample-size', type=int, default=256)
    
    args = parser.parse_args()
    
    analyzer = RelativeBlurAnalyzer(
        degraded_dir=args.degraded_dir,
        restored_dir=args.restored_dir,
        output_dir=args.output_dir,
        blur_ratio_threshold=args.blur_ratio_threshold,
        sample_size=args.sample_size
    )
    
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
