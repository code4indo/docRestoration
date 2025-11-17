#!/usr/bin/env python3
"""
Skrip untuk mengidentifikasi dan memvisualisasikan kasus 'slight blur' 
pada region dengan tinta iron gall terkorosi dalam hasil restorasi dokumen.

Tujuan: Memberikan justifikasi visual untuk pernyataan dalam paper:
"Pada 2-3 kasus sedang, terdapat slight blur pada region dengan tinta iron gall 
terkorosi, namun ini tidak mengurangi utilitas untuk transkripsi semantik."

Author: ML Engineer
Date: 2025-11-17
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from typing import List, Tuple, Dict
from dataclasses import dataclass, asdict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BlurAnalysisResult:
    """Data class untuk menyimpan hasil analisis blur"""
    filename: str
    blur_score: float  # Laplacian variance (lower = more blur)
    iron_gall_coverage: float  # Persentase area dengan iron gall corrosion
    has_slight_blur: bool
    severity: str  # 'none', 'slight', 'moderate', 'severe'
    crop_regions: List[Tuple[int, int, int, int]]  # (x, y, w, h)


class IronGallBlurAnalyzer:
    """Analyzer untuk mendeteksi slight blur pada region iron gall corrosion"""
    
    def __init__(
        self,
        degraded_dir: Path,
        restored_dir: Path,
        output_dir: Path,
        blur_threshold_slight: float = 100.0,
        blur_threshold_moderate: float = 50.0,
        min_iron_gall_coverage: float = 0.05,
        crop_size: int = 512
    ):
        self.degraded_dir = Path(degraded_dir)
        self.restored_dir = Path(restored_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thresholds
        self.blur_threshold_slight = blur_threshold_slight
        self.blur_threshold_moderate = blur_threshold_moderate
        self.min_iron_gall_coverage = min_iron_gall_coverage
        self.crop_size = crop_size
        
        # Results storage
        self.results: List[BlurAnalysisResult] = []
        
    def detect_iron_gall_regions(self, degraded_img: np.ndarray) -> np.ndarray:
        """
        Deteksi region dengan iron gall ink corrosion pada gambar degradasi.
        
        Iron gall ink biasanya tampak sebagai area gelap/kehitaman dengan lubang-lubang
        karena korosi tinta yang merusak kertas.
        
        Returns:
            Binary mask dengan region iron gall corrosion
        """
        # Convert to grayscale
        if len(degraded_img.shape) == 3:
            gray = cv2.cvtColor(degraded_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = degraded_img.copy()
        
        # Iron gall corrosion biasanya sangat gelap (pixel value rendah)
        # dan membentuk cluster dengan tekstur kasar
        _, dark_regions = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations untuk mendapatkan region yang koheren
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        iron_gall_mask = cv2.morphologyEx(dark_regions, cv2.MORPH_CLOSE, kernel)
        iron_gall_mask = cv2.morphologyEx(iron_gall_mask, cv2.MORPH_OPEN, 
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        
        return iron_gall_mask
    
    def calculate_blur_score(self, image: np.ndarray, mask: np.ndarray = None) -> float:
        """
        Hitung blur score menggunakan Laplacian variance.
        
        Lower score = more blur
        Higher score = sharper image
        
        Args:
            image: Input image (grayscale or BGR)
            mask: Optional mask untuk fokus pada region tertentu
            
        Returns:
            Blur score (Laplacian variance)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Compute Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        if mask is not None:
            # Hanya hitung pada region yang di-mask
            laplacian = laplacian[mask > 0]
        
        # Variance of Laplacian = blur metric
        blur_score = laplacian.var()
        
        return blur_score
    
    def find_blur_regions(
        self, 
        restored_img: np.ndarray, 
        iron_gall_mask: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Temukan region dengan blur pada area iron gall corrosion.
        
        Returns:
            List of bounding boxes (x, y, w, h) untuk region yang blur
        """
        # Konversi ke grayscale
        if len(restored_img.shape) == 3:
            gray = cv2.cvtColor(restored_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = restored_img.copy()
        
        # Hitung local blur score dengan sliding window
        h, w = gray.shape
        window_size = 128
        stride = 64
        
        blur_regions = []
        
        for y in range(0, h - window_size, stride):
            for x in range(0, w - window_size, stride):
                # Extract window
                window = gray[y:y+window_size, x:x+window_size]
                mask_window = iron_gall_mask[y:y+window_size, x:x+window_size]
                
                # Skip jika tidak ada iron gall corrosion di window ini
                iron_gall_coverage = np.sum(mask_window > 0) / (window_size * window_size)
                if iron_gall_coverage < 0.1:
                    continue
                
                # Hitung blur score untuk window ini
                blur_score = self.calculate_blur_score(window, mask_window)
                
                # Jika blur score rendah (blur), simpan region ini
                if blur_score < self.blur_threshold_slight:
                    blur_regions.append((x, y, window_size, window_size))
        
        # Merge overlapping regions
        blur_regions = self._merge_overlapping_boxes(blur_regions)
        
        return blur_regions
    
    def _merge_overlapping_boxes(
        self, 
        boxes: List[Tuple[int, int, int, int]], 
        overlap_threshold: float = 0.3
    ) -> List[Tuple[int, int, int, int]]:
        """Merge bounding boxes yang overlap"""
        if not boxes:
            return []
        
        # Simple merging: combine boxes that overlap significantly
        merged = []
        used = [False] * len(boxes)
        
        for i, box1 in enumerate(boxes):
            if used[i]:
                continue
                
            x1, y1, w1, h1 = box1
            group = [box1]
            used[i] = True
            
            for j, box2 in enumerate(boxes[i+1:], start=i+1):
                if used[j]:
                    continue
                    
                x2, y2, w2, h2 = box2
                
                # Check overlap
                x_overlap = max(0, min(x1+w1, x2+w2) - max(x1, x2))
                y_overlap = max(0, min(y1+h1, y2+h2) - max(y1, y2))
                overlap_area = x_overlap * y_overlap
                
                if overlap_area > overlap_threshold * min(w1*h1, w2*h2):
                    group.append(box2)
                    used[j] = True
            
            # Merge group into single box
            xs = [b[0] for b in group]
            ys = [b[1] for b in group]
            x_min = min(xs)
            y_min = min(ys)
            x_max = max(x + w for x, y, w, h in group)
            y_max = max(y + h for x, y, w, h in group)
            
            merged.append((x_min, y_min, x_max - x_min, y_max - y_min))
        
        return merged
    
    def classify_blur_severity(self, blur_score: float) -> str:
        """Klasifikasi tingkat keparahan blur"""
        if blur_score >= self.blur_threshold_slight:
            return 'none'
        elif blur_score >= self.blur_threshold_moderate:
            return 'slight'
        elif blur_score >= 25.0:
            return 'moderate'
        else:
            return 'severe'
    
    def analyze_single_image(self, degraded_path: Path, restored_path: Path) -> BlurAnalysisResult:
        """Analisis satu pasang gambar (degraded + restored)"""
        logger.info(f"Analyzing: {degraded_path.name}")
        
        # Load images
        degraded = cv2.imread(str(degraded_path))
        restored = cv2.imread(str(restored_path))
        
        if degraded is None or restored is None:
            logger.error(f"Failed to load images for {degraded_path.name}")
            return None
        
        # Deteksi iron gall regions
        iron_gall_mask = self.detect_iron_gall_regions(degraded)
        iron_gall_coverage = np.sum(iron_gall_mask > 0) / iron_gall_mask.size
        
        # Skip jika tidak ada iron gall corrosion yang signifikan
        if iron_gall_coverage < self.min_iron_gall_coverage:
            logger.info(f"  Skipped: iron gall coverage too low ({iron_gall_coverage:.2%})")
            return None
        
        # Hitung blur score pada region iron gall di gambar restored
        blur_score = self.calculate_blur_score(restored, iron_gall_mask)
        severity = self.classify_blur_severity(blur_score)
        
        # Temukan region-region yang blur
        blur_regions = self.find_blur_regions(restored, iron_gall_mask)
        
        has_slight_blur = severity in ['slight', 'moderate']
        
        logger.info(f"  Blur score: {blur_score:.2f}, Severity: {severity}, "
                   f"Iron gall coverage: {iron_gall_coverage:.2%}, "
                   f"Blur regions: {len(blur_regions)}")
        
        result = BlurAnalysisResult(
            filename=degraded_path.name,
            blur_score=blur_score,
            iron_gall_coverage=iron_gall_coverage,
            has_slight_blur=has_slight_blur,
            severity=severity,
            crop_regions=blur_regions
        )
        
        return result
    
    def visualize_comparison(
        self, 
        degraded_path: Path, 
        restored_path: Path, 
        result: BlurAnalysisResult
    ):
        """Buat visualisasi side-by-side dengan highlight pada region blur"""
        degraded = cv2.imread(str(degraded_path))
        restored = cv2.imread(str(restored_path))
        
        # Deteksi iron gall mask untuk visualisasi
        iron_gall_mask = self.detect_iron_gall_regions(degraded)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Row 1: Full images
        axes[0, 0].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Degraded Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Restored Image', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Iron gall mask overlay
        overlay = restored.copy()
        overlay[iron_gall_mask > 0] = overlay[iron_gall_mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5
        axes[0, 2].imshow(cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Iron Gall Regions (Red)', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Blur region crops (up to 3)
        crop_titles = ['Crop 1', 'Crop 2', 'Crop 3']
        for idx in range(3):
            if idx < len(result.crop_regions):
                x, y, w, h = result.crop_regions[idx]
                # Expand crop slightly for context
                margin = 50
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(restored.shape[1], x + w + margin)
                y2 = min(restored.shape[0], y + h + margin)
                
                crop_degraded = degraded[y1:y2, x1:x2]
                crop_restored = restored[y1:y2, x1:x2]
                
                # Side by side crop
                combined = np.hstack([crop_degraded, crop_restored])
                axes[1, idx].imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
                axes[1, idx].set_title(f'{crop_titles[idx]}: Degraded | Restored', 
                                      fontsize=12, fontweight='bold')
                
                # Draw rectangle on blur region
                rect_x = w // 2
                rect = plt.Rectangle((rect_x + margin, margin), w, h, 
                                    fill=False, edgecolor='red', linewidth=2)
                axes[1, idx].add_patch(rect)
            else:
                axes[1, idx].axis('off')
            
            axes[1, idx].axis('off')
        
        # Add metadata
        metadata_text = (
            f"Filename: {result.filename}\n"
            f"Blur Score: {result.blur_score:.2f}\n"
            f"Severity: {result.severity}\n"
            f"Iron Gall Coverage: {result.iron_gall_coverage:.2%}\n"
            f"Blur Regions Found: {len(result.crop_regions)}"
        )
        fig.text(0.5, 0.02, metadata_text, ha='center', fontsize=11, 
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        # Save
        output_path = self.output_dir / f"analysis_{result.filename.replace('.jpg', '.png')}"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved visualization: {output_path.name}")
    
    def extract_crops(self, degraded_path: Path, restored_path: Path, result: BlurAnalysisResult):
        """Ekstrak crops dari region blur untuk paper"""
        degraded = cv2.imread(str(degraded_path))
        restored = cv2.imread(str(restored_path))
        
        crops_dir = self.output_dir / 'crops'
        crops_dir.mkdir(exist_ok=True)
        
        for idx, (x, y, w, h) in enumerate(result.crop_regions[:5]):  # Max 5 crops
            # Expand untuk konteks
            margin = 30
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(restored.shape[1], x + w + margin)
            y2 = min(restored.shape[0], y + h + margin)
            
            crop_degraded = degraded[y1:y2, x1:x2]
            crop_restored = restored[y1:y2, x1:x2]
            
            # Save individual crops
            base_name = result.filename.replace('.jpg', '')
            cv2.imwrite(str(crops_dir / f"{base_name}_crop{idx+1}_degraded.png"), crop_degraded)
            cv2.imwrite(str(crops_dir / f"{base_name}_crop{idx+1}_restored.png"), crop_restored)
            
            # Save side-by-side
            combined = np.hstack([crop_degraded, crop_restored])
            cv2.imwrite(str(crops_dir / f"{base_name}_crop{idx+1}_comparison.png"), combined)
        
        logger.info(f"  Extracted {len(result.crop_regions)} crops")
    
    def run_analysis(self):
        """Jalankan analisis untuk semua gambar"""
        logger.info("Starting Iron Gall Blur Analysis")
        logger.info(f"Degraded dir: {self.degraded_dir}")
        logger.info(f"Restored dir: {self.restored_dir}")
        logger.info(f"Output dir: {self.output_dir}")
        
        # Find matching pairs
        degraded_files = sorted(self.degraded_dir.glob("*.jpg"))
        
        for degraded_path in degraded_files:
            # Find corresponding restored file
            base_name = degraded_path.stem
            restored_candidates = [
                self.restored_dir / f"{base_name}_restored.tiff",
                self.restored_dir / f"{base_name}_restored.tif",
                self.restored_dir / f"{base_name}_restored.png",
                self.restored_dir / f"{base_name}.tiff",
            ]
            
            restored_path = None
            for candidate in restored_candidates:
                if candidate.exists():
                    restored_path = candidate
                    break
            
            if restored_path is None:
                logger.warning(f"No restored image found for {degraded_path.name}")
                continue
            
            # Analyze
            result = self.analyze_single_image(degraded_path, restored_path)
            
            if result is None:
                continue
            
            self.results.append(result)
            
            # Visualize jika ada slight blur
            if result.has_slight_blur and len(result.crop_regions) > 0:
                self.visualize_comparison(degraded_path, restored_path, result)
                self.extract_crops(degraded_path, restored_path, result)
        
        # Generate summary report
        self.generate_report()
    
    def generate_report(self):
        """Generate summary report"""
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*60)
        
        total = len(self.results)
        slight_blur_cases = [r for r in self.results if r.severity == 'slight']
        moderate_blur_cases = [r for r in self.results if r.severity == 'moderate']
        severe_blur_cases = [r for r in self.results if r.severity == 'severe']
        
        logger.info(f"Total images analyzed: {total}")
        logger.info(f"  - No blur: {total - len(slight_blur_cases) - len(moderate_blur_cases) - len(severe_blur_cases)}")
        logger.info(f"  - Slight blur: {len(slight_blur_cases)}")
        logger.info(f"  - Moderate blur: {len(moderate_blur_cases)}")
        logger.info(f"  - Severe blur: {len(severe_blur_cases)}")
        
        logger.info("\nSlight blur cases (untuk justifikasi paper):")
        for r in slight_blur_cases:
            logger.info(f"  - {r.filename}: blur_score={r.blur_score:.2f}, "
                       f"iron_gall_coverage={r.iron_gall_coverage:.2%}, "
                       f"regions={len(r.crop_regions)}")
        
        # Save JSON report
        report = {
            'summary': {
                'total_analyzed': total,
                'no_blur': total - len(slight_blur_cases) - len(moderate_blur_cases) - len(severe_blur_cases),
                'slight_blur': len(slight_blur_cases),
                'moderate_blur': len(moderate_blur_cases),
                'severe_blur': len(severe_blur_cases)
            },
            'cases': [asdict(r) for r in self.results]
        }
        
        report_path = self.output_dir / 'analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nReport saved: {report_path}")
        logger.info("="*60)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze slight blur on iron gall corrosion regions'
    )
    parser.add_argument(
        '--degraded-dir',
        type=str,
        default='DokumenRusak/forPaper',
        help='Directory containing degraded images'
    )
    parser.add_argument(
        '--restored-dir',
        type=str,
        default='DokumenRusak/forPaper_results',
        help='Directory containing restored images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_results/iron_gall_blur',
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--blur-threshold-slight',
        type=float,
        default=100.0,
        help='Threshold for slight blur detection (Laplacian variance)'
    )
    parser.add_argument(
        '--blur-threshold-moderate',
        type=float,
        default=50.0,
        help='Threshold for moderate blur detection'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = IronGallBlurAnalyzer(
        degraded_dir=args.degraded_dir,
        restored_dir=args.restored_dir,
        output_dir=args.output_dir,
        blur_threshold_slight=args.blur_threshold_slight,
        blur_threshold_moderate=args.blur_threshold_moderate
    )
    
    # Run analysis
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
