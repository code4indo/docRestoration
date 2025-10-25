#!/usr/bin/env python3
"""
Laypa-based Line Detection for Document Images
Uses pretrained Laypa baseline detection model via Docker
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
import subprocess
import tempfile
import shutil
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET
import json


class LaypaLineDetector:
    """
    Laypa baseline detector using pretrained model via Docker
    
    Laypa is a state-of-the-art layout analysis tool that detects:
    - Baselines (text lines) via segmentation mask
    - Regions (paragraphs, page numbers, etc.)
    - Start/end points of baselines
    
    Pipeline (following Loghi pattern):
    1. Laypa: Generate segmentation mask (background=0, baseline=255)
    2. MinionExtractBaselines: Convert mask to baseline coordinates in PageXML
    
    This wrapper uses the Docker images:
    - loghi/docker.laypa for baseline segmentation
    - loghi/docker.loghi-tooling for coordinate extraction
    """
    
    def __init__(self, 
                 docker_image: str = "loghi/docker.laypa:2.2.14",
                 tooling_image: str = "loghi/docker.loghi-tooling:2.2.14",
                 config_path: str = None,
                 model_weights: str = None,
                 gpu_id: int = -1,
                 min_line_height: int = 20,
                 max_line_height: int = 200):
        """
        Initialize Laypa line detector
        
        Args:
            docker_image: Docker image name for Laypa segmentation
            tooling_image: Docker image name for Loghi tooling (MinionExtractBaselines)
            config_path: Path to Laypa config yaml file (full path on host)
            model_weights: Path to pretrained weights pth file (full path on host)
            gpu_id: GPU ID to use (-1 for CPU)
            min_line_height: Minimum line height for filtering
            max_line_height: Maximum line height for filtering
        """
        self.docker_image = docker_image
        self.tooling_image = tooling_image
        self.tooling_image = tooling_image
        
        # Use default paths if not provided
        if config_path is None:
            # Default: assume model in workspace
            workspace_root = Path(__file__).resolve().parent.parent.parent
            config_path = workspace_root / "laypa_models" / "config.yaml"
        
        if model_weights is None:
            workspace_root = Path(__file__).resolve().parent.parent.parent
            model_weights = workspace_root / "laypa_models" / "model_best_mIoU.pth"
        
        self.config_path = str(Path(config_path).resolve())
        self.model_weights = str(Path(model_weights).resolve())
        self.gpu_id = gpu_id
        self.min_line_height = min_line_height
        self.max_line_height = max_line_height
        
        # Verify files exist
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        if not Path(self.model_weights).exists():
            raise FileNotFoundError(f"Model weights not found: {self.model_weights}")
        
        print(f"✓ LaypaLineDetector initialized")
        print(f"  Laypa Docker: {docker_image}")
        print(f"  Tooling Docker: {tooling_image}")
        print(f"  Config: {self.config_path}")
        print(f"  Weights: {self.model_weights}")
        print(f"  GPU: {gpu_id if gpu_id >= 0 else 'CPU'}")
    
    def detect_baselines_docker(self, 
                                image_path: str,
                                output_dir: str) -> str:
        """
        Run Laypa baseline detection via Docker (2-step pipeline following Loghi)
        
        Step 1: Laypa segmentation - generates mask PNG (background=0, baseline=255)
        Step 2: MinionExtractBaselines - converts mask to baseline coordinates in PageXML
        
        Args:
            image_path: Path to input image
            output_dir: Directory for output PageXML
            
        Returns:
            Path to generated PageXML file with baseline coordinates
        """
        # Prepare paths
        image_path = Path(image_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_name = image_path.stem
        
        # Create temporary directory for processing
        # Use /tmp to avoid cross-device link issues
        import shutil
        import tempfile
        temp_root = tempfile.mkdtemp(prefix="laypa_")
        temp_input_dir = Path(temp_root) / "input"
        temp_output_dir = Path(temp_root) / "output"
        temp_input_dir.mkdir(parents=True)
        temp_output_dir.mkdir(parents=True)
        
        # Copy image to temp directory
        temp_image_path = temp_input_dir / image_path.name
        shutil.copy2(image_path, temp_image_path)
        
        # Get model directory (need to mount it)
        model_dir = Path(self.model_weights).parent
        
        try:
            # ========================================
            # STEP 1: Run Laypa segmentation
            # ========================================
            print(f"\n[Step 1/2] Running Laypa segmentation...")
            
            # Build docker command for Laypa
            # Use single parent mount (/workspace) containing input + output subdirs
            docker_cmd = ["docker", "run"]
            
            # GPU configuration
            if self.gpu_id >= 0:
                docker_cmd.extend(["--gpus", "all"])
            
            # Docker parameters (following Loghi pattern)
            docker_cmd.extend([
                "--rm",
                "-u", f"{subprocess.check_output(['id', '-u']).decode().strip()}:{subprocess.check_output(['id', '-g']).decode().strip()}",
                "-m", "32000m",
                "--shm-size", "10240m",
                "-v", f"{model_dir}:/models",  # Model directory
                "-v", f"{temp_root}:/workspace",  # Single parent mount
                self.docker_image,
                "python", "run.py",
                "-c", "/models/config.yaml",
                "-i", "/workspace/input",
                "-o", "/workspace/output",
                "--opts",
                "MODEL.WEIGHTS", '""',  # Empty string
                "TEST.WEIGHTS", "/models/model_best_mIoU.pth"
            ])
            
            # Run Laypa Docker
            process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output (only important lines)
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                if any(keyword in line.lower() for keyword in ['error', 'warning', 'predicting', '%', 'loading']):
                    print(f"  Laypa: {line.strip()}")
            
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"Laypa segmentation failed with code {return_code}")
            
            print(f"✓ Step 1 complete: Segmentation mask generated")
            
            # ========================================
            # STEP 2: Run MinionExtractBaselines
            # ========================================
            print(f"[Step 2/2] Extracting baseline coordinates...")
            
            # Build docker command for MinionExtractBaselines
            extract_cmd = [
                "docker", "run",
                "--rm",
                "-u", f"{subprocess.check_output(['id', '-u']).decode().strip()}:{subprocess.check_output(['id', '-g']).decode().strip()}",
                "-v", f"{temp_root}:/workspace",
                self.tooling_image,
                "/src/loghi-tooling/minions/target/appassembler/bin/MinionExtractBaselines",
                "-input_path_image", "/workspace/input",
                "-input_path_png", "/workspace/output/page/",
                "-input_path_page", "/workspace/output/page/",
                "-output_path_page", "/workspace/output/page/",
                "-recalculate_textline_contours_from_baselines",
                "-as_single_region"
            ]
            
            # Run MinionExtractBaselines
            process = subprocess.Popen(
                extract_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output (show extraction info)
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                if any(keyword in line.lower() for keyword in ['error', 'warning', 'extracted', 'textlines', 'found labels']):
                    print(f"  Minion: {line.strip()}")
            
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"MinionExtractBaselines failed with code {return_code}")
            
            print(f"✓ Step 2 complete: Baseline coordinates extracted")
            
            # Find generated PageXML in temp directory
            temp_page_dir = temp_output_dir / "page"
            temp_xml_file = temp_page_dir / f"{image_name}.xml"
            
            if temp_xml_file.exists():
                print(f"✓ PageXML ready: {temp_xml_file}")
                return str(temp_xml_file), temp_root
            else:
                print(f"❌ PageXML not found: {temp_xml_file}")
                print(f"   Files in {temp_page_dir}:")
                if temp_page_dir.exists():
                    for f in temp_page_dir.iterdir():
                        print(f"     - {f.name}")
                
                shutil.rmtree(temp_root, ignore_errors=True)
                return None, None
                
        except subprocess.TimeoutExpired:
            print(f"❌ Docker command timed out")
            shutil.rmtree(temp_root, ignore_errors=True)
            return None, None
        except Exception as e:
            print(f"❌ Error in pipeline: {e}")
            import traceback
            traceback.print_exc()
            shutil.rmtree(temp_root, ignore_errors=True)
            return None, None
    
    def parse_pagexml_baselines(self, xml_path: str) -> List[np.ndarray]:
        """
        Parse PageXML to extract baseline coordinates
        
        Args:
            xml_path: Path to PageXML file
            
        Returns:
            List of baseline coordinate arrays (Nx2)
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # PageXML namespace - support both 2013 and 2019 versions
            # MinionExtractBaselines outputs 2019 version
            ns_2013 = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
            ns_2019 = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'}
            
            baselines = []
            
            # Try 2019 namespace first (MinionExtractBaselines output), then 2013
            for ns in [ns_2019, ns_2013]:
                baseline_elems = root.findall('.//pc:Baseline', ns)
                if baseline_elems:
                    break
            
            # Also try wildcard namespace if above fail
            if not baseline_elems:
                baseline_elems = root.findall('.//{*}Baseline')
            
            # Parse baseline elements
            for baseline_elem in baseline_elems:
                points_str = baseline_elem.get('points')
                
                if points_str:
                    # Parse points: "x1,y1 x2,y2 x3,y3 ..."
                    points = []
                    for point_str in points_str.strip().split():
                        try:
                            x, y = map(int, point_str.split(','))
                            points.append([x, y])
                        except ValueError:
                            continue
                    
                    if len(points) >= 2:  # Need at least 2 points for a baseline
                        baselines.append(np.array(points))
            
            print(f"✓ Parsed {len(baselines)} baseline(s) from PageXML")
            return baselines
            
        except Exception as e:
            print(f"❌ Error parsing PageXML: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def calculate_adaptive_line_height(self, 
                                       baselines: List[np.ndarray],
                                       image_height: int) -> int:
        """
        Calculate adaptive line height based on baseline spacing.
        Essential for cursive/paleographic scripts with tall ascenders and long descenders.
        
        Args:
            baselines: List of baseline coordinate arrays
            image_height: Height of the image
            
        Returns:
            Adaptive line height estimate
        """
        if len(baselines) < 2:
            # Single line: use 15% of image height as safe default
            return max(100, int(image_height * 0.15))
        
        # Extract y-coordinates of all baselines
        y_coords = [np.mean(baseline[:, 1]) for baseline in baselines if len(baseline) > 0]
        
        if len(y_coords) < 2:
            return max(100, int(image_height * 0.15))
        
        # Calculate spacing between consecutive baselines
        y_coords.sort()
        spacings = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
        
        # Use median spacing (more robust than mean)
        median_spacing = np.median(spacings)
        
        # Line height = 1.3x spacing to accommodate ascenders/descenders
        # Cursive scripts need extra room for flourishes
        adaptive_height = int(median_spacing * 1.3)
        
        # Clamp to reasonable range: 60-250px
        return max(60, min(adaptive_height, 250))
    
    def baselines_to_boxes(self, 
                          baselines: List[np.ndarray],
                          line_height_estimate: int = 80,
                          image_height: int = 0) -> List[Tuple[int, int, int, int]]:
        """
        Convert baselines to bounding boxes with cursive-aware margins.
        
        Args:
            baselines: List of baseline coordinate arrays
            line_height_estimate: Estimated height of text lines (default 80, will be overridden if image_height provided)
            image_height: Height of image for adaptive calculation
            
        Returns:
            List of bounding boxes (x1, y1, x2, y2)
        """
        # Use adaptive height if image dimensions available
        if image_height > 0:
            line_height_estimate = self.calculate_adaptive_line_height(baselines, image_height)
            print(f"  ℹ Adaptive line height: {line_height_estimate}px (based on baseline spacing)")
        boxes = []
        
        for i, baseline in enumerate(baselines):
            if len(baseline) == 0:
                continue
            
            # Get baseline extent
            x_min = int(np.min(baseline[:, 0]))
            x_max = int(np.max(baseline[:, 0]))
            y_baseline = int(np.mean(baseline[:, 1]))
            
            # CURSIVE SCRIPT OPTIMIZATION:
            # Baseline is BOTTOM reference line. Cursive/paleographic scripts have:
            # - Tall ascenders (b,d,f,h,k,l,t): ~100% ABOVE baseline
            # - Long descenders (g,j,p,q,y): ~30% BELOW baseline  
            # - Flourishes and ligatures need extra margins
            y_top = y_baseline - int(line_height_estimate * 1.0)
            y_bottom = y_baseline + int(line_height_estimate * 0.3)
            
            # Safety margins for cursive flourishes
            CURSIVE_TOP_MARGIN = 10
            CURSIVE_BOTTOM_MARGIN = 10
            
            # Add extra margin for first line (headers + ornamental capitals)
            if i == 0:
                y_top = max(0, y_top - 20 - CURSIVE_TOP_MARGIN)
            else:
                y_top = max(0, y_top - CURSIVE_TOP_MARGIN)
            
            # Add bottom margin for all lines
            y_bottom = y_bottom + CURSIVE_BOTTOM_MARGIN
            
            box_height = y_bottom - y_top
            
            # Filter by height
            if self.min_line_height <= box_height <= self.max_line_height:
                boxes.append((x_min, y_top, x_max, y_bottom))
        
        # Sort by vertical position
        boxes.sort(key=lambda b: b[1])
        
        print(f"✓ Converted to {len(boxes)} bounding box(es)")
        return boxes
    
    def detect_lines(self, 
                    image_path: str,
                    output_dir: Optional[str] = None) -> List[Tuple[int, int, int, int]]:
        """
        Complete pipeline: detect baselines and convert to boxes
        
        Args:
            image_path: Path to input image
            output_dir: Optional output directory (uses temp if None)
            
        Returns:
            List of line bounding boxes (x1, y1, x2, y2)
        """
        # Create temporary directory if needed
        temp_dir = None
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="laypa_")
            output_dir = temp_dir
        
        temp_root_to_cleanup = None
        
        try:
            # Run Laypa baseline detection
            result = self.detect_baselines_docker(image_path, output_dir)
            
            if result is None or result[0] is None:
                print("❌ Laypa detection failed")
                return []
            
            xml_path, temp_root_to_cleanup = result
            
            # Parse PageXML
            baselines = self.parse_pagexml_baselines(xml_path)
            
            if not baselines:
                print("❌ No baselines found in PageXML")
                return []
            
            # Get image dimensions for adaptive height calculation
            import cv2
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image_height = img.shape[0] if img is not None else 0
            
            # Convert to bounding boxes with adaptive height
            boxes = self.baselines_to_boxes(baselines, image_height=image_height)
            
            return boxes
            
        finally:
            # Cleanup temp directories
            if temp_root_to_cleanup:
                import shutil
                shutil.rmtree(temp_root_to_cleanup, ignore_errors=True)
            if temp_dir and Path(temp_dir).exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def visualize_detections(self,
                           image_path: str,
                           boxes: List[Tuple[int, int, int, int]],
                           baselines: Optional[List[np.ndarray]] = None,
                           output_path: Optional[str] = None,
                           title: str = "Laypa Baseline Detection"):
        """
        Visualize detected lines and baselines
        
        Args:
            image_path: Path to input image
            boxes: List of bounding boxes
            baselines: Optional list of baseline coordinates
            output_path: Optional path to save visualization
            title: Title for the plot
        """
        # Load image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(img_rgb)
        
        # Draw boxes
        for idx, (x1, y1, x2, y2) in enumerate(boxes, 1):
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='cyan', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add line number
            ax.text(x1 + 5, y1 + 20, f'L{idx}',
                   color='cyan', fontsize=10, weight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Draw baselines if provided
        if baselines:
            for baseline in baselines:
                ax.plot(baseline[:, 0], baseline[:, 1], 
                       color='red', linewidth=2, alpha=0.7)
        
        ax.set_title(f"{title} ({len(boxes)} lines)", fontsize=14, weight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved: {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Laypa-based Line Detection (2-step pipeline)')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output_dir', type=str, default='results/line_detection_laypa',
                       help='Output directory')
    parser.add_argument('--gpu_id', type=int, default=-1,
                       help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--docker_image', type=str, default='loghi/docker.laypa:2.2.14',
                       help='Docker image for Laypa segmentation')
    parser.add_argument('--tooling_image', type=str, default='loghi/docker.loghi-tooling:2.2.14',
                       help='Docker image for Loghi tooling (MinionExtractBaselines)')
    parser.add_argument('--min_line_height', type=int, default=20,
                       help='Minimum line height')
    parser.add_argument('--max_line_height', type=int, default=200,
                       help='Maximum line height')
    parser.add_argument('--keep_xml', action='store_true',
                       help='Keep intermediate XML files for debugging')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    print(f"\n{'='*70}")
    print(f"Laypa Baseline Detection")
    print(f"{'='*70}")
    print(f"Input: {args.input}")
    print(f"GPU: {args.gpu_id if args.gpu_id >= 0 else 'CPU'}")
    print(f"{'='*70}\n")
    
    detector = LaypaLineDetector(
        docker_image=args.docker_image,
        tooling_image=args.tooling_image,
        gpu_id=args.gpu_id,
        min_line_height=args.min_line_height,
        max_line_height=args.max_line_height
    )
    
    # Detect lines
    print("\n📊 Detecting baselines...")
    
    # Use persistent output if keep_xml is True
    laypa_output = output_dir / "laypa_output" if args.keep_xml else None
    
    boxes = detector.detect_lines(args.input, laypa_output)
    
    if not boxes:
        print("\n❌ No lines detected!")
        return
    
    # Print results
    print(f"\n{'='*70}")
    print(f"Detection Results")
    print(f"{'='*70}")
    print(f"Total lines: {len(boxes)}")
    print(f"\nLine details:")
    for idx, (x1, y1, x2, y2) in enumerate(boxes, 1):
        width = x2 - x1
        height = y2 - y1
        print(f"  Line {idx:2d}: ({x1:4d}, {y1:4d}) -> ({x2:4d}, {y2:4d})  "
              f"[{width:4d}×{height:3d}px]")
    
    # Visualize
    input_path = Path(args.input)
    output_path = output_dir / f"{input_path.stem}_laypa_lines.png"
    
    print(f"\n📸 Generating visualization...")
    
    # Get baselines for visualization if xml exists
    baselines = None
    if laypa_output:
        xml_path = laypa_output / "page" / f"{input_path.stem}.xml"
        if xml_path.exists():
            baselines = detector.parse_pagexml_baselines(str(xml_path))
    
    detector.visualize_detections(
        args.input,
        boxes,
        baselines,
        str(output_path),
        title="Laypa Baseline Detection"
    )
    
    print(f"\n{'='*70}")
    print(f"Done!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
