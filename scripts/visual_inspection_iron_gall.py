#!/usr/bin/env python3
"""
PRACTICAL APPROACH: Manual visual inspection tool
untuk menemukan kasus 'slight blur' pada iron gall regions.

Strategy:
1. Load semua gambar degraded + restored
2. Deteksi dark regions (potential iron gall)
3. Ekstrak multiple crops dari berbagai area
4. Buat grid visualization untuk manual inspection
5. Hitung blur metrics untuk setiap crop
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class VisualInspectionTool:
    """Tool untuk inspeksi visual mencari kasus slight blur"""
    
    def __init__(self, degraded_dir: Path, restored_dir: Path, output_dir: Path):
        self.degraded_dir = Path(degraded_dir)
        self.restored_dir = Path(restored_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def calculate_blur_score(self, img: np.ndarray) -> float:
        """Hitung blur score menggunakan Laplacian variance"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    def calculate_local_contrast(self, img: np.ndarray) -> float:
        """Hitung local contrast"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return gray.std()
    
    def find_dark_regions(self, degraded: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Temukan dark regions (potential iron gall) dan return bounding boxes.
        
        Returns:
            List of (x, y, w, h) untuk dark regions
        """
        if len(degraded.shape) == 3:
            gray = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
        else:
            gray = degraded
        
        # Multiple thresholds untuk capture different levels of degradation
        dark_regions = []
        
        for threshold in [40, 60, 80]:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (tidak terlalu kecil, tidak terlalu besar)
                if 500 < area < 50000:
                    x, y, w, h = cv2.boundingRect(contour)
                    dark_regions.append((x, y, w, h, threshold, area))
        
        # Remove duplicates and sort by area
        dark_regions = sorted(set(dark_regions), key=lambda x: x[5], reverse=True)
        
        return [(x, y, w, h) for x, y, w, h, _, _ in dark_regions[:50]]  # Top 50
    
    def extract_diverse_crops(
        self, 
        degraded: np.ndarray, 
        restored: np.ndarray,
        n_crops: int = 20,
        crop_size: int = 384
    ) -> List[Dict]:
        """
        Ekstrak diverse crops dari berbagai region (dark, medium, light)
        untuk visual inspection.
        """
        h, w = degraded.shape[:2]
        crops = []
        
        # 1. Dark regions (potential iron gall)
        dark_boxes = self.find_dark_regions(degraded)
        
        for bbox in dark_boxes[:n_crops//2]:
            x, y, bw, bh = bbox
            
            # Center crop around dark region with margin
            cx = x + bw // 2
            cy = y + bh // 2
            
            x1 = max(0, cx - crop_size // 2)
            y1 = max(0, cy - crop_size // 2)
            x2 = min(w, x1 + crop_size)
            y2 = min(h, y1 + crop_size)
            
            if (x2 - x1) < crop_size // 2 or (y2 - y1) < crop_size // 2:
                continue
            
            crop_deg = degraded[y1:y2, x1:x2]
            crop_res = restored[y1:y2, x1:x2]
            
            # Calculate metrics
            blur_score = self.calculate_blur_score(crop_res)
            contrast = self.calculate_local_contrast(crop_res)
            
            # Check jika ini area gelap
            mean_intensity = cv2.cvtColor(crop_deg, cv2.COLOR_BGR2GRAY).mean()
            
            crops.append({
                'degraded': crop_deg,
                'restored': crop_res,
                'bbox': (x1, y1, x2-x1, y2-y1),
                'blur_score': blur_score,
                'contrast': contrast,
                'mean_intensity': mean_intensity,
                'type': 'dark_region'
            })
        
        # 2. Random sampling untuk comparison
        for _ in range(n_crops - len(crops)):
            x = np.random.randint(0, max(1, w - crop_size))
            y = np.random.randint(0, max(1, h - crop_size))
            
            crop_deg = degraded[y:y+crop_size, x:x+crop_size]
            crop_res = restored[y:y+crop_size, x:x+crop_size]
            
            blur_score = self.calculate_blur_score(crop_res)
            contrast = self.calculate_local_contrast(crop_res)
            mean_intensity = cv2.cvtColor(crop_deg, cv2.COLOR_BGR2GRAY).mean()
            
            crops.append({
                'degraded': crop_deg,
                'restored': crop_res,
                'bbox': (x, y, crop_size, crop_size),
                'blur_score': blur_score,
                'contrast': contrast,
                'mean_intensity': mean_intensity,
                'type': 'random'
            })
        
        # Sort by blur score (ascending = more blur first)
        crops = sorted(crops, key=lambda x: x['blur_score'])
        
        return crops
    
    def create_inspection_grid(
        self, 
        filename: str,
        crops: List[Dict]
    ):
        """Buat grid visualization untuk manual inspection"""
        n_crops = min(len(crops), 12)  # Max 12 crops per grid
        n_cols = 4
        n_rows = (n_crops + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(20, 5 * n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.4, wspace=0.3)
        
        for idx, crop_data in enumerate(crops[:n_crops]):
            row = idx // n_cols
            col = idx % n_cols
            
            ax = fig.add_subplot(gs[row, col])
            
            # Side by side: degraded | restored
            deg = cv2.cvtColor(crop_data['degraded'], cv2.COLOR_BGR2RGB)
            res = cv2.cvtColor(crop_data['restored'], cv2.COLOR_BGR2RGB)
            
            combined = np.hstack([deg, res])
            ax.imshow(combined)
            
            # Title dengan metrics
            title = (
                f"#{idx+1} [{crop_data['type']}]\n"
                f"Blur: {crop_data['blur_score']:.1f} | "
                f"Contrast: {crop_data['contrast']:.1f}\n"
                f"Intensity: {crop_data['mean_intensity']:.1f}"
            )
            ax.set_title(title, fontsize=9, fontfamily='monospace')
            ax.axis('off')
            
            # Add vertical line separator
            h = combined.shape[0]
            w = combined.shape[1]
            ax.axvline(x=w//2, color='red', linewidth=2, linestyle='--')
        
        # Overall title
        fig.suptitle(f"Visual Inspection: {filename}\n"
                    f"Left=Degraded | Right=Restored | Sorted by Blur Score (low=more blur)",
                    fontsize=14, fontweight='bold')
        
        # Save
        output_path = self.output_dir / f"inspection_{filename.replace('.jpg', '.png')}"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved inspection grid: {output_path.name}")
        
        return output_path
    
    def create_summary_report(self, all_results: List[Dict]):
        """Buat summary report dengan kandidat slight blur"""
        logger.info("\n" + "="*60)
        logger.info("INSPECTION SUMMARY")
        logger.info("="*60)
        
        # Find potential slight blur cases
        # Definisi: blur_score < 150 dan mean_intensity < 100 (dark region)
        potential_cases = []
        
        for result in all_results:
            filename = result['filename']
            crops = result['crops']
            
            slight_blur_crops = [
                c for c in crops 
                if c['blur_score'] < 150 and c['mean_intensity'] < 100 and c['type'] == 'dark_region'
            ]
            
            if len(slight_blur_crops) > 0:
                potential_cases.append({
                    'filename': filename,
                    'n_crops': len(slight_blur_crops),
                    'min_blur': min(c['blur_score'] for c in slight_blur_crops),
                    'avg_blur': np.mean([c['blur_score'] for c in slight_blur_crops]),
                    'crops': slight_blur_crops
                })
        
        logger.info(f"\n📊 Total images processed: {len(all_results)}")
        logger.info(f"🔍 Potential slight blur cases: {len(potential_cases)}")
        
        if len(potential_cases) > 0:
            logger.info("\n🎯 KANDIDAT SLIGHT BLUR (untuk justifikasi paper):")
            for case in potential_cases:
                logger.info(f"\n  • {case['filename']}")
                logger.info(f"    - Dark region crops dengan blur: {case['n_crops']}")
                logger.info(f"    - Min blur score: {case['min_blur']:.1f}")
                logger.info(f"    - Avg blur score: {case['avg_blur']:.1f}")
                logger.info(f"    → Check: inspection_{case['filename'].replace('.jpg', '.png')}")
        else:
            logger.info("\n✅ Tidak ditemukan slight blur yang signifikan")
            logger.info("   Semua hasil restorasi memiliki kualitas sharp yang baik")
        
        # Save JSON
        report = {
            'total_images': len(all_results),
            'potential_slight_blur_cases': len(potential_cases),
            'cases': potential_cases,
            'all_results': all_results
        }
        
        report_path = self.output_dir / 'inspection_report.json'
        
        # Convert numpy types to native Python types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean report from numpy types
        import json
        clean_report = json.loads(json.dumps(report, default=convert))
        
        with open(report_path, 'w') as f:
            json.dump(clean_report, f, indent=2)
        
        logger.info(f"\n✅ Report saved: {report_path}")
        logger.info("="*60)
    
    def run_inspection(self):
        """Jalankan visual inspection untuk semua gambar"""
        logger.info("="*60)
        logger.info("VISUAL INSPECTION TOOL FOR SLIGHT BLUR DETECTION")
        logger.info("="*60)
        
        degraded_files = sorted(self.degraded_dir.glob("*.jpg"))
        all_results = []
        
        for degraded_path in degraded_files:
            logger.info(f"\nProcessing: {degraded_path.name}")
            
            # Find restored image
            base_name = degraded_path.stem
            restored_candidates = [
                self.restored_dir / f"{base_name}_restored.tiff",
                self.restored_dir / f"{base_name}_restored.tif",
            ]
            
            restored_path = None
            for candidate in restored_candidates:
                if candidate.exists():
                    restored_path = candidate
                    break
            
            if restored_path is None:
                logger.warning(f"  No restored image found")
                continue
            
            # Load images
            degraded = cv2.imread(str(degraded_path))
            restored = cv2.imread(str(restored_path))
            
            if degraded is None or restored is None:
                logger.error(f"  Failed to load images")
                continue
            
            # Extract crops
            crops = self.extract_diverse_crops(degraded, restored, n_crops=20)
            
            logger.info(f"  Extracted {len(crops)} crops")
            logger.info(f"  Blur score range: {crops[0]['blur_score']:.1f} - {crops[-1]['blur_score']:.1f}")
            
            # Create inspection grid
            self.create_inspection_grid(degraded_path.name, crops)
            
            # Store results (without image data)
            result_data = {
                'filename': degraded_path.name,
                'n_crops': len(crops),
                'blur_scores': [float(c['blur_score']) for c in crops],
                'mean_intensities': [float(c['mean_intensity']) for c in crops],
                'crops': [{
                    'bbox': c['bbox'],
                    'blur_score': float(c['blur_score']),
                    'contrast': float(c['contrast']),
                    'mean_intensity': float(c['mean_intensity']),
                    'type': c['type']
                } for c in crops]
            }
            
            all_results.append(result_data)
        
        # Generate summary
        self.create_summary_report(all_results)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visual inspection tool for slight blur detection')
    parser.add_argument('--degraded-dir', type=str, default='DokumenRusak/forPaper')
    parser.add_argument('--restored-dir', type=str, default='DokumenRusak/forPaper_results')
    parser.add_argument('--output-dir', type=str, default='analysis_results/visual_inspection')
    
    args = parser.parse_args()
    
    tool = VisualInspectionTool(
        degraded_dir=args.degraded_dir,
        restored_dir=args.restored_dir,
        output_dir=args.output_dir
    )
    
    tool.run_inspection()


if __name__ == '__main__':
    main()
