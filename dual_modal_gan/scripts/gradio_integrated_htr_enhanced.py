#!/usr/bin/env python3
"""
Enhanced Integrated Document Restoration + Full Loghi HTR Pipeline
===================================================================
Full Pipeline:
1. Document Restoration using Dual-Modal GAN
2. LAYPA: Layout Analysis & Line Segmentation
3. TOOLING: Baseline Extraction & XML Processing
4. HTR: Handwritten Text Recognition per line

Author: belekok
Date: 2025-11-24
Version: 2.0 (Full Pipeline)
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
import logging
import requests
import pandas as pd
from PIL import Image
import io
import tempfile
import time
import xml.etree.ElementTree as ET
from typing import Tuple, Optional, List, Dict
import json
import subprocess
import shutil
import glob
import re

# Setup paths
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.append(str(current_dir))

# Import the inference logic
try:
    import inference_portrait_overlap_experiment as inference
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "inference_portrait_overlap_experiment",
        str(current_dir / "inference_portrait_overlap_experiment.py")
    )
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IntegratedApp")

# Global variables
GENERATOR = None
CHECKPOINT_DIR = "production_full_coverage_vgg_v1"
CHECKPOINT_NAME = "ckpt-94"

# Service Configuration
LAYPA_ADDRESS = os.getenv("LAYPA_ADDRESS", "http://localhost:5002")
HTR_ADDRESS = os.getenv("LOGHI_ADDRESS", "http://localhost:5001")
TOOLING_ADDRESS = os.getenv("TOOLING_ADDRESS", "http://localhost:8082")


def load_generator():
    """Load document restoration model"""
    global GENERATOR
    if GENERATOR is None:
        logger.info("Loading restoration model...")
        ckpt_dir = parent_dir / "checkpoints" / CHECKPOINT_DIR
        
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        
        # GPU configuration (use GPU 1 to avoid conflict with Loghi)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus and len(gpus) > 1:
            try:
                # Use second GPU if available
                tf.config.set_visible_devices(gpus[1], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[1], True)
                logger.info(f"Using GPU: {gpus[1].name}")
            except RuntimeError as e:
                logger.warning(f"GPU setup failed: {e}. Using first GPU.")
                tf.config.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
        elif gpus:
            logger.info(f"Using GPU: {gpus[0].name}")
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        else:
            logger.info("No GPU detected. Using CPU.")
            tf.config.set_visible_devices([], 'GPU')
            
        GENERATOR = inference.load_model(str(ckpt_dir), CHECKPOINT_NAME, gpu_id=1)
        logger.info("Model loaded successfully.")
    return GENERATOR


def restore_document(image, alpha, post_processing, aggressive, thin_strokes, gamma):
    """
    Document restoration function
    Returns: (numpy_image, tiff_file_path)
    """
    if image is None:
        return None, None, "No image provided"
    
    try:
        generator = load_generator()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image
            
        logger.info(f"Processing image shape: {image.shape}")
        
        # Run restoration
        restored = inference.process_portrait_document(
            image_gray,
            generator,
            logger,
            alpha=alpha,
            post_processing=post_processing,
            aggressive=aggressive,
            thin_strokes=thin_strokes,
            gamma=gamma
        )
        
        # Create TIFF file
        tiff_fd, tiff_path = tempfile.mkstemp(suffix='.tiff', prefix='restored_')
        os.close(tiff_fd)
        
        restored_pil = Image.fromarray(restored)
        restored_pil.save(tiff_path, format='TIFF', compression='tiff_lzw', dpi=(300, 300))
        
        # Create JPG file (high quality)
        jpg_fd, jpg_path = tempfile.mkstemp(suffix='.jpg', prefix='restored_')
        os.close(jpg_fd)
        restored_pil.save(jpg_path, format='JPEG', quality=95, dpi=(300, 300))
        
        logger.info(f"Restoration complete. TIFF: {tiff_path}, JPG: {jpg_path}")
        
        return restored, tiff_path, jpg_path, "✓ Restoration complete"
        
    except Exception as e:
        logger.error(f"Error during restoration: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, f"✗ Restoration failed: {str(e)}"


def laypa_segmentation(image, model_path="general/baseline"):
    """
    LAYPA: Layout analysis and baseline segmentation
    
    Returns:
    --------
    tuple: (segmentation_image, xml_content, status)
    """
    if image is None:
        return None, None, "No image provided"
    
    try:
        logger.info("Starting LAYPA segmentation...")
        
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Prepare image bytes
        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        
        # Create temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.png', prefix='laypa_input_')
        os.close(temp_fd)
        pil_image.save(temp_path, format='PNG')
        
        # Call LAYPA service
        url = f"{LAYPA_ADDRESS}/predict"
        identifier = f"gan_htr_{int(time.time())}"
        
        with open(temp_path, 'rb') as f:
            files = {'image': ('document.png', f, 'image/png')}
            data = {
                'identifier': identifier,
                'model': model_path
            }
            
            logger.info(f"Sending LAYPA request: {url}")
            response = requests.post(url, files=files, data=data, timeout=120)
        
        os.unlink(temp_path)
        
        if response.ok:
            result = response.json()
            logger.info(f"LAYPA successful: {result}")
            
            # TODO: Get segmented image and XML from LAYPA output directory
            # For now, return placeholder
            return image, "XML placeholder", "✓ LAYPA segmentation complete"
        else:
            error_msg = f"LAYPA failed: HTTP {response.status_code}"
            logger.error(error_msg)
            return None, None, f"✗ {error_msg}"
            
    except requests.exceptions.ConnectionError:
        error_msg = f"Cannot connect to LAYPA at {LAYPA_ADDRESS}"
        logger.error(error_msg)
        return None, None, f"✗ {error_msg}"
    except Exception as e:
        error_msg = f"LAYPA error: {str(e)}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return None, None, f"✗ {error_msg}"


def parse_pagexml_with_coords(xml_path: str) -> List[Dict]:
    """
    Parse PageXML and extract text lines with their coordinates.
    
    Parameters:
    -----------
    xml_path : str
        Path to PageXML file
        
    Returns:
    --------
    list : List of dicts with 'text', 'coords', 'line_id'
    """
    lines_data = []
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Detect namespace
        ns = None
        for possible_ns in [
            'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15',
            'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'
        ]:
            if root.find(f'.//{{{possible_ns}}}TextLine') is not None:
                ns = possible_ns
                break
        
        if ns:
            text_lines = root.findall(f'.//{{{ns}}}TextLine')
        else:
            # Try without namespace
            text_lines = root.findall('.//TextLine')
        
        for idx, line in enumerate(text_lines):
            line_data = {
                'line_id': idx + 1,
                'text': '',
                'coords': [],
                'baseline': []
            }
            
            # Get text
            if ns:
                unicode_elem = line.find(f'.//{{{ns}}}Unicode')
                coords_elem = line.find(f'./{{{ns}}}Coords')
                baseline_elem = line.find(f'./{{{ns}}}Baseline')
            else:
                unicode_elem = line.find('.//Unicode')
                coords_elem = line.find('./Coords')
                baseline_elem = line.find('./Baseline')
            
            if unicode_elem is not None and unicode_elem.text:
                line_data['text'] = unicode_elem.text.strip()
            
            # Get coordinates (polygon points)
            if coords_elem is not None:
                points_str = coords_elem.get('points', '')
                if points_str:
                    points = []
                    for point in points_str.split():
                        try:
                            x, y = map(int, point.split(','))
                            points.append((x, y))
                        except:
                            pass
                    line_data['coords'] = points
            
            # Get baseline
            if baseline_elem is not None:
                points_str = baseline_elem.get('points', '')
                if points_str:
                    points = []
                    for point in points_str.split():
                        try:
                            x, y = map(int, point.split(','))
                            points.append((x, y))
                        except:
                            pass
                    line_data['baseline'] = points
            
            if line_data['text']:  # Only add lines with text
                lines_data.append(line_data)
        
        logger.info(f"Parsed {len(lines_data)} text lines from XML")
        
    except Exception as e:
        logger.error(f"Error parsing PageXML: {e}")
    
    return lines_data


def create_annotated_image(image: np.ndarray, lines_data: List[Dict], 
                          highlight_line: int = None) -> np.ndarray:
    """
    Create annotated image with bounding boxes for each text line.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Original image (RGB)
    lines_data : list
        List of dicts with 'text', 'coords', 'line_id'
    highlight_line : int, optional
        Line number to highlight (1-based)
        
    Returns:
    --------
    numpy.ndarray : Annotated image
    """
    if image is None or not lines_data:
        return image
    
    # Make a copy
    annotated = image.copy()
    
    # Define colors (BGR for cv2, but we're in RGB)
    colors = [
        (255, 100, 100),  # Red
        (100, 255, 100),  # Green
        (100, 100, 255),  # Blue
        (255, 255, 100),  # Yellow
        (255, 100, 255),  # Magenta
        (100, 255, 255),  # Cyan
    ]
    
    for line_data in lines_data:
        line_id = line_data['line_id']
        coords = line_data['coords']
        
        if not coords:
            continue
        
        # Choose color
        color_idx = (line_id - 1) % len(colors)
        color = colors[color_idx]
        
        # Determine if this line should be highlighted
        is_highlighted = (highlight_line is not None and line_id == highlight_line)
        
        # Draw polygon
        pts = np.array(coords, dtype=np.int32)
        
        if is_highlighted:
            # Fill with semi-transparent yellow
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], (255, 255, 0))
            annotated = cv2.addWeighted(annotated, 0.5, overlay, 0.5, 0)
            cv2.polylines(annotated, [pts], True, (255, 0, 0), 3)
        else:
            # Light overlay
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], color)
            annotated = cv2.addWeighted(annotated, 0.85, overlay, 0.15, 0)
            cv2.polylines(annotated, [pts], True, color, 1)
        
        # Draw line number
        if coords:
            min_x = min(p[0] for p in coords)
            min_y = min(p[1] for p in coords)
            
            # Draw background for number
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text = str(line_id)
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            cv2.rectangle(annotated, 
                         (min_x - 2, min_y - text_h - 4), 
                         (min_x + text_w + 2, min_y), 
                         (0, 0, 0), -1)
            cv2.putText(annotated, text, (min_x, min_y - 2), 
                       font, font_scale, (255, 255, 255), thickness)
    
    return annotated


def format_htr_text_with_lines(lines_data: List[Dict]) -> str:
    """
    Format HTR text with line numbers for display.
    
    Parameters:
    -----------
    lines_data : list
        List of dicts with 'text', 'line_id'
        
    Returns:
    --------
    str : Formatted text with line numbers
    """
    if not lines_data:
        return ""
    
    formatted_lines = []
    for line_data in lines_data:
        line_id = line_data['line_id']
        text = line_data['text']
        formatted_lines.append(f"{line_id:3d}. {text}")
    
    return "\n".join(formatted_lines)


def perform_htr_via_pipeline(image):
    """
    Run full Loghi HTR pipeline using inference-pipeline.sh
    This is the PROVEN method that works for real HTR.
    
    Parameters:
    -----------
    image : numpy.ndarray or PIL.Image
        Restored document image to process
        
    Returns:
    --------
    tuple : (text_result, confidence, status_message, lines_data)
        lines_data is a list of dicts with text, coords, and line_id
    """
    if image is None:
        return "", 0.0, "No image provided", []
    
    try:
        logger.info("Starting HTR via inference-pipeline.sh...")
        
        # Create temp directory for processing
        temp_dir = tempfile.mkdtemp(prefix='htr_pipeline_')
        logger.info(f"Created temp directory: {temp_dir}")
        
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Save image as JPG (required by pipeline)
        img_path = os.path.join(temp_dir, "document.jpg")
        pil_image.convert('RGB').save(img_path, format='JPEG', quality=95)
        logger.info(f"Saved image: {img_path}")
        
        # Clean any existing XML
        page_dir = os.path.join(temp_dir, "page")
        os.makedirs(page_dir, exist_ok=True)
        
        # Run inference-pipeline.sh
        pipeline_script = "/home/lambda_one/tesis/loghi/scripts/inference-pipeline.sh"
        
        logger.info(f"Running pipeline: {pipeline_script} {temp_dir}")
        
        result = subprocess.run(
            [pipeline_script, temp_dir],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd="/home/lambda_one/tesis/loghi"
        )
        
        logger.info(f"Pipeline stdout: {result.stdout[-500:] if result.stdout else 'empty'}")
        if result.stderr:
            logger.warning(f"Pipeline stderr: {result.stderr[-500:]}")
        
        # Check for output XML with text
        xml_files = glob.glob(os.path.join(page_dir, "*.xml"))
        
        if not xml_files:
            logger.error("No XML output found from pipeline")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return "", 0.0, "✗ HTR pipeline failed (no XML output)", []
        
        # Parse XML to extract text WITH COORDINATES
        xml_path = xml_files[0]
        logger.info(f"Parsing XML: {xml_path}")
        
        # Use new function that extracts coordinates
        lines_data = parse_pagexml_with_coords(xml_path)
        
        # Also get simple text list for backward compatibility
        text_lines = [line['text'] for line in lines_data]
        full_text = "\n".join(text_lines)
        
        logger.info(f"Extracted {len(text_lines)} lines of text with coordinates")
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        if full_text:
            return full_text, 1.0, f"✓ HTR Complete ({len(text_lines)} lines)", lines_data
        else:
            return "", 0.0, "✗ HTR completed but no text extracted", []
            
    except subprocess.TimeoutExpired:
        logger.error("HTR pipeline timeout")
        return "", 0.0, "✗ HTR pipeline timeout (>5 min)", []
    except Exception as e:
        logger.error(f"HTR pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return "", 0.0, f"✗ HTR error: {str(e)}", []


def perform_htr_via_pipeline_with_save(image, save_to_viewer: bool = True):
    """
    Run full Loghi HTR pipeline and optionally save results for web viewer.
    
    Parameters:
    -----------
    image : numpy.ndarray or PIL.Image
        Restored document image to process
    save_to_viewer : bool
        If True, save results to web_viewer/results folder
        
    Returns:
    --------
    tuple : (text_result, confidence, status_message, lines_data, viewer_path)
    """
    if image is None:
        return "", 0.0, "No image provided", [], None
    
    try:
        logger.info("Starting HTR via inference-pipeline.sh (with viewer save)...")
        
        # Create output directory for viewer
        timestamp = int(time.time())
        doc_id = f"doc_{timestamp}"
        
        if save_to_viewer:
            viewer_results_dir = Path(__file__).parent.parent.parent / "web_viewer" / "results" / doc_id
            viewer_results_dir.mkdir(parents=True, exist_ok=True)
            page_dir_viewer = viewer_results_dir / "page"
            page_dir_viewer.mkdir(exist_ok=True)
        else:
            viewer_results_dir = None
        
        # Create temp directory for processing
        temp_dir = tempfile.mkdtemp(prefix='htr_pipeline_')
        logger.info(f"Created temp directory: {temp_dir}")
        
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Save image as JPG (required by pipeline)
        img_path = os.path.join(temp_dir, f"{doc_id}.jpg")
        pil_image.convert('RGB').save(img_path, format='JPEG', quality=95)
        logger.info(f"Saved image: {img_path}")
        
        # Also save to viewer directory
        if save_to_viewer and viewer_results_dir:
            viewer_img_path = viewer_results_dir / f"{doc_id}.jpg"
            pil_image.convert('RGB').save(viewer_img_path, format='JPEG', quality=95)
            logger.info(f"Saved image for viewer: {viewer_img_path}")
        
        # Clean any existing XML
        page_dir = os.path.join(temp_dir, "page")
        os.makedirs(page_dir, exist_ok=True)
        
        # Run inference-pipeline.sh
        pipeline_script = "/home/lambda_one/tesis/loghi/scripts/inference-pipeline.sh"
        
        logger.info(f"Running pipeline: {pipeline_script} {temp_dir}")
        
        result = subprocess.run(
            [pipeline_script, temp_dir],
            capture_output=True,
            text=True,
            timeout=300,
            cwd="/home/lambda_one/tesis/loghi"
        )
        
        logger.info(f"Pipeline stdout: {result.stdout[-500:] if result.stdout else 'empty'}")
        if result.stderr:
            logger.warning(f"Pipeline stderr: {result.stderr[-500:]}")
        
        # Check for output XML
        xml_files = glob.glob(os.path.join(page_dir, "*.xml"))
        
        if not xml_files:
            logger.error("No XML output found from pipeline")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return "", 0.0, "✗ HTR pipeline failed (no XML output)", [], None
        
        # Parse and copy XML
        xml_path = xml_files[0]
        logger.info(f"Parsing XML: {xml_path}")
        
        # Copy XML to viewer directory with correct name
        if save_to_viewer and viewer_results_dir:
            viewer_xml_path = page_dir_viewer / f"{doc_id}.xml"
            shutil.copy2(xml_path, viewer_xml_path)
            logger.info(f"Saved XML for viewer: {viewer_xml_path}")
        
        # Parse XML
        lines_data = parse_pagexml_with_coords(xml_path)
        text_lines = [line['text'] for line in lines_data]
        full_text = "\n".join(text_lines)
        
        logger.info(f"Extracted {len(text_lines)} lines of text with coordinates")
        
        # Cleanup temp
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        viewer_path = str(viewer_results_dir) if viewer_results_dir else None
        
        if full_text:
            return full_text, 1.0, f"✓ HTR Complete ({len(text_lines)} lines)", lines_data, viewer_path
        else:
            return "", 0.0, "✗ HTR completed but no text extracted", [], viewer_path
            
    except subprocess.TimeoutExpired:
        logger.error("HTR pipeline timeout")
        return "", 0.0, "✗ HTR pipeline timeout (>5 min)", [], None
    except Exception as e:
        logger.error(f"HTR pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return "", 0.0, f"✗ HTR error: {str(e)}", [], None


def perform_htr_simple(image):
    """
    Simple HTR: Send full image to HTR service (no LAYPA/TOOLING)
    
    Parameters:
    -----------
    image : numpy.ndarray or PIL.Image
        Image to process
        
    Returns:
    --------
    tuple : (text_result, confidence, status_message)
    """
    if image is None:
        return "", 0.0, "No image provided"
    
    try:
        logger.info("Starting simple HTR process...")
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Create temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.png', prefix='htr_input_')
        os.close(temp_fd)
        pil_image.save(temp_path, format='PNG')
        
        # Call HTR service
        url = f"{HTR_ADDRESS}/predict"
        identifier = f"htr_{int(time.time())}"
        
        with open(temp_path, 'rb') as f:
            files = {'image': ('document.png', f, 'image/png')}
            data = {
                'group_id': 'gan_htr_group',
                'identifier': identifier
            }
            
            logger.info(f"Sending HTR request to: {url} (identifier: {identifier})")
            response = requests.post(url, files=files, data=data, timeout=180)
        
        os.unlink(temp_path)
        
        if response.ok or response.status_code == 202:
            result = response.json()
            logger.info(f"HTR request accepted: {result}")
            
            # HTR API is async - poll for result file
            output_base = "/home/lambda_one/tesis/loghi/loghi_output"
            group_id = data['group_id']
            result_file = Path(output_base) / group_id / identifier / f"{identifier}.txt"
            
            # Poll for result (max 60 seconds)
            max_wait = 60
            poll_interval = 2
            waited = 0
            
            logger.info(f"Polling for result: {result_file}")
            while waited < max_wait:
                if result_file.exists():
                    text = result_file.read_text().strip()
                    logger.info(f"HTR complete: {text[:100]}...")
                    return text, 1.0, "✓ HTR Complete"
                time.sleep(poll_interval)
                waited += poll_interval
                logger.info(f"Waiting for HTR result... ({waited}s)")
            
            return "", 0.0, "✗ HTR timeout (result file not found)"
        else:
            error_msg = f"HTR failed: HTTP {response.status_code}"
            logger.error(error_msg)
            return "", 0.0, f"✗ {error_msg}"
            
    except requests.exceptions.ConnectionError:
        error_msg = f"Cannot connect to HTR service at {HTR_ADDRESS}"
        logger.error(error_msg)
        return "", 0.0, f"✗ {error_msg}"
    except Exception as e:
        error_msg = f"HTR error: {str(e)}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return "", 0.0, f"✗ {error_msg}"


def integrated_pipeline_simple(image, alpha, post_processing, aggressive, thin_strokes, gamma, run_htr):
    """
    Simple Pipeline: Restoration + Direct HTR (no LAYPA/TOOLING)
    
    This is the FAST path for quick results.
    """
    # Step 1: Restore document
    restored, tiff_path, jpg_path, restore_status = restore_document(
        image, alpha, post_processing, aggressive, thin_strokes, gamma
    )
    
    if restored is None:
        return None, None, None, "", 0.0, restore_status, None, None, ""
    
    # Step 2: Run HTR if requested
    if run_htr:
        # Use the full pipeline method that saves to viewer
        text, confidence, htr_status, lines_data, viewer_path = perform_htr_via_pipeline_with_save(restored, save_to_viewer=True)
        
        # Create annotated image with line regions
        annotated_img = create_annotated_image(restored, lines_data) if lines_data else restored
        
        # Format text with line numbers
        formatted_text = format_htr_text_with_lines(lines_data) if lines_data else text
        
        # Create line results dataframe
        df_lines = pd.DataFrame({
            'No': [ld['line_id'] for ld in lines_data],
            'Teks': [ld['text'] for ld in lines_data],
            'Confidence': [confidence] * len(lines_data)
        }) if lines_data else pd.DataFrame({'No': [], 'Teks': [], 'Confidence': []})
        
        # Create viewer link
        viewer_info = f"📺 Hasil disimpan. Buka HTR Viewer: http://localhost:7863" if viewer_path else ""
        
        final_status = f"{restore_status} | {htr_status}"
        return restored, tiff_path, jpg_path, formatted_text, confidence, final_status, annotated_img, df_lines, viewer_info
    else:
        return restored, tiff_path, jpg_path, "", 0.0, f"{restore_status} (HTR skipped)", None, None, ""


def integrated_pipeline_full(image, alpha, post_processing, aggressive, thin_strokes, gamma, run_htr, laypa_model):
    """
    Full Pipeline: Restoration + Full Loghi Pipeline (LAYPA + Cut + HTR + Merge)
    
    This is the COMPLETE path using inference-pipeline.sh
    """
    # Step 1: Restore document
    restored, tiff_path, jpg_path, restore_status = restore_document(
        image, alpha, post_processing, aggressive, thin_strokes, gamma
    )
    
    if restored is None:
        return None, None, None, "", 0.0, restore_status, None, None, ""
    
    if not run_htr:
        return restored, tiff_path, jpg_path, "", 0.0, f"{restore_status} (HTR skipped)", None, None, ""
    
    # Step 2: Run full HTR pipeline with save to viewer
    text, confidence, htr_status, lines_data, viewer_path = perform_htr_via_pipeline_with_save(restored, save_to_viewer=True)
    
    # Create annotated image with line regions
    annotated_img = create_annotated_image(restored, lines_data) if lines_data else restored
    
    # Format text with line numbers
    formatted_text = format_htr_text_with_lines(lines_data) if lines_data else text
    
    # Create line results dataframe
    df_lines = pd.DataFrame({
        'No': [ld['line_id'] for ld in lines_data],
        'Teks': [ld['text'] for ld in lines_data],
        'Confidence': [confidence] * len(lines_data)
    }) if lines_data else pd.DataFrame({'No': [], 'Teks': [], 'Confidence': []})
    
    # Create viewer link
    viewer_info = f"📺 Hasil disimpan. Buka HTR Viewer: http://localhost:7863" if viewer_path else ""
    
    final_status = f"{restore_status} | {htr_status}"
    
    return restored, tiff_path, jpg_path, formatted_text, confidence, final_status, annotated_img, df_lines, viewer_info


def integrated_pipeline_full_OLD(image, alpha, post_processing, aggressive, thin_strokes, gamma, run_htr, laypa_model):
    """
    [DEPRECATED] Old Full Pipeline using separate API calls - doesn't work properly
    """
    # Step 1: Restore document
    restored, tiff_path, jpg_path, restore_status = restore_document(
        image, alpha, post_processing, aggressive, thin_strokes, gamma
    )
    
    if restored is None:
        return None, None, None, "", 0.0, restore_status, None, None
    
    if not run_htr:
        return restored, tiff_path, jpg_path, "", 0.0, f"{restore_status} (HTR skipped)", None, None
    
    # Step 2: LAYPA Segmentation
    seg_image, xml_content, laypa_status = laypa_segmentation(restored, laypa_model)
    
    if seg_image is None:
        final_status = f"{restore_status} | {laypa_status}"
        return restored, tiff_path, jpg_path, "", 0.0, final_status, None, None
    
    # Step 3: TODO - TOOLING baseline extraction
    # Step 4: TODO - HTR per line
    
    # For now, fallback to simple HTR
    text, confidence, htr_status = perform_htr_simple(restored)
    final_status = f"{restore_status} | {laypa_status} | {htr_status}"
    
    # Create placeholder dataframe for line results
    df_lines = pd.DataFrame({
        'Line': ['Full Document'],
        'Text': [text],
        'Confidence': [confidence]
    })
    
    return restored, tiff_path, jpg_path, text, confidence, final_status, seg_image, df_lines


# Define Gradio Interface
with gr.Blocks(title="Document Restoration + HTR") as demo:
    gr.Markdown(
        """
        # 📜 Document Restoration + Loghi HTR System
        
        **Pipeline:** Document Restoration (Dual-Modal GAN) → HTR (Loghi)
        
        📺 **[Buka HTR Viewer](http://localhost:7863)** - Hasil akan otomatis muncul di viewer setelah proses selesai
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input & Settings")
            input_image = gr.Image(
                label="📸 Degraded Document", 
                type="numpy",
                height=400
            )
            
            with gr.Accordion("⚙️ Restoration Settings", open=True):
                alpha = gr.Slider(
                    minimum=0.0, maximum=0.5, value=0.0, step=0.05, 
                    label="Alpha Blending",
                    info="0.0 = Fully restored, 0.5 = 50% original"
                )
                gamma = gr.Slider(
                    minimum=0.5, maximum=2.0, value=1.0, step=0.1, 
                    label="Gamma Correction",
                    info="< 1.0 Darker, > 1.0 Brighter"
                )
                
                with gr.Row():
                    post_processing = gr.Checkbox(value=True, label="Post-processing")
                    aggressive = gr.Checkbox(value=False, label="Aggressive")
                    thin_strokes = gr.Checkbox(value=False, label="Thin Strokes")
            
            with gr.Accordion("🔤 HTR Settings", open=True):
                run_htr = gr.Checkbox(value=True, label="Enable HTR")
                htr_mode = gr.Radio(
                    choices=["Simple (Fast)", "Full Pipeline (Detailed)"],
                    value="Simple (Fast)",
                    label="HTR Mode"
                )
                laypa_model = gr.Textbox(
                    value="general/baseline",
                    label="LAYPA Model Path",
                    info="Used in Full Pipeline mode"
                )
            
            with gr.Row():
                submit_btn = gr.Button("✨ Process Document", variant="primary", size="lg")
                clear_btn = gr.ClearButton()
            
        with gr.Column(scale=1):
            gr.Markdown("### 🎨 Restoration Result")
            output_image = gr.Image(label="Restored Document", type="numpy", height=400)
            with gr.Row():
                download_tiff = gr.File(label="📥 Download TIFF (300 DPI)")
                download_jpg = gr.File(label="📥 Download JPG (High Quality)")
            
            status_box = gr.Textbox(label="📊 Status", value="Ready", lines=2)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 HTR Result")
            htr_text = gr.Textbox(
                label="Extracted Text",
                lines=8,
                placeholder="HTR results will appear here..."
            )
            htr_confidence = gr.Number(label="Confidence", precision=3)
            viewer_link = gr.Markdown(
                value="",
                label="Viewer Link"
            )
    
    with gr.Accordion("ℹ️ Info", open=False):
        gr.Markdown(
            f"""
            **Services:** Restoration `{CHECKPOINT_DIR}` | HTR `{HTR_ADDRESS}` | **[Viewer](http://localhost:7863)**
            """
        )
    
    # Event handlers
    def process_document(img, a, pp, ag, ts, g, rh, mode, lm):
        if mode == "Simple (Fast)":
            result = integrated_pipeline_simple(img, a, pp, ag, ts, g, rh)
        else:
            result = integrated_pipeline_full(img, a, pp, ag, ts, g, rh, lm)
        
        # result: restored, tiff, jpg, text, conf, status, annotated, df, viewer_info
        # Return simplified: restored, tiff, jpg, text, conf, status, viewer_link
        viewer_md = ""
        if result[8]:  # viewer_info exists
            viewer_md = "📺 **[Lihat hasil di HTR Viewer](http://localhost:7863)** - Auto refresh aktif"
        return result[0], result[1], result[2], result[3], result[4], result[5], viewer_md
    
    submit_btn.click(
        fn=process_document,
        inputs=[
            input_image, alpha, post_processing, aggressive, thin_strokes, gamma,
            run_htr, htr_mode, laypa_model
        ],
        outputs=[output_image, download_tiff, download_jpg, htr_text, htr_confidence, status_box, viewer_link]
    )
    
    clear_btn.add([input_image, output_image, htr_text, htr_confidence, status_box, download_tiff, download_jpg])


if __name__ == "__main__":
    print("="*80)
    print("Enhanced Integrated Document Restoration + Loghi HTR System")
    print("="*80)
    print(f"Restoration Model: {CHECKPOINT_DIR}/{CHECKPOINT_NAME}")
    print(f"LAYPA Service: {LAYPA_ADDRESS}")
    print(f"HTR Service: {HTR_ADDRESS}")
    print(f"TOOLING Service: {TOOLING_ADDRESS}")
    print("="*80)
    print("Starting Gradio interface...")
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False)
