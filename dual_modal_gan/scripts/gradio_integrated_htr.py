#!/usr/bin/env python3
"""
Integrated Document Restoration + HTR System
============================================
Combines:
1. Document Restoration using Dual-Modal GAN
2. HTR using Loghi system

Author: belekok
Date: 2025-11-24
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntegratedApp")

# Global variables
GENERATOR = None
CHECKPOINT_DIR = "production_full_coverage_vgg_v1"
CHECKPOINT_NAME = "ckpt-94"

# HTR Configuration (from environment or defaults)
HTR_ADDRESS = os.getenv("LOGHI_ADDRESS", "http://10.13.0.4:5001")
LAYPA_ADDRESS = os.getenv("LAYPA_ADDRESS", "http://10.13.0.4:5000")


def load_generator():
    """Load document restoration model"""
    global GENERATOR
    if GENERATOR is None:
        logger.info("Loading restoration model...")
        ckpt_dir = parent_dir / "checkpoints" / CHECKPOINT_DIR
        
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        
        # GPU configuration
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
                logger.info(f"Using GPU: {gpus[0].name}")
            except RuntimeError as e:
                logger.warning(f"GPU setup failed: {e}. Falling back to CPU.")
                tf.config.set_visible_devices([], 'GPU')
        else:
            logger.info("No GPU detected. Using CPU.")
            tf.config.set_visible_devices([], 'GPU')
            
        GENERATOR = inference.load_model(str(ckpt_dir), CHECKPOINT_NAME, gpu_id=0)
        logger.info("Model loaded successfully.")
    return GENERATOR


def restore_document(image, alpha, post_processing, aggressive, thin_strokes, gamma):
    """
    Document restoration function
    Returns: (numpy_image, tiff_file_path)
    """
    if image is None:
        return None, None
    
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
        
        logger.info(f"Restoration complete. TIFF saved: {tiff_path}")
        
        return restored, tiff_path
        
    except Exception as e:
        logger.error(f"Error during restoration: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def perform_htr(image, htr_address):
    """
    Perform HTR on restored image using Loghi service
    
    Parameters:
    -----------
    image : numpy.ndarray
        Restored image
    htr_address : str
        Address of Loghi HTR service
        
    Returns:
    --------
    tuple : (text_result, confidence, status_message)
    """
    if image is None:
        return "", 0.0, "No image provided"
    
    try:
        logger.info("Starting HTR process...")
        
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
            
        # Convert to bytes
        image_byte_array = io.BytesIO()
        pil_image.save(image_byte_array, format='PNG')
        image_byte_array.seek(0)
        
        # Create temporary file for upload
        temp_fd, temp_path = tempfile.mkstemp(suffix='.png', prefix='htr_input_')
        os.close(temp_fd)
        pil_image.save(temp_path, format='PNG')
        
        # Call HTR service
        url = f"{htr_address}/predict"
        
        # Send as multipart file upload
        with open(temp_path, 'rb') as f:
            files = {'image': ('document.png', f, 'image/png')}
            
            logger.info(f"Sending HTR request to: {url}")
            response = requests.post(url, files=files, timeout=120)
        
        # Cleanup temp file
        os.unlink(temp_path)
        
        if response.ok:
            result = response.json()
            logger.info(f"HTR successful: {result}")
            
            # Extract text and confidence from response
            text = result.get('text', result.get('prediction', ''))
            confidence = result.get('confidence', 0.0)
            
            return text, confidence, "HTR Complete ✓"
        else:
            error_msg = f"HTR failed: HTTP {response.status_code}"
            logger.error(error_msg)
            return "", 0.0, error_msg
            
    except requests.exceptions.ConnectionError:
        error_msg = f"Cannot connect to HTR service at {htr_address}. Please ensure Loghi HTR service is running."
        logger.error(error_msg)
        return "", 0.0, error_msg
    except Exception as e:
        error_msg = f"HTR error: {str(e)}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return "", 0.0, error_msg


def integrated_pipeline(image, alpha, post_processing, aggressive, thin_strokes, gamma, run_htr, htr_address):
    """
    Integrated pipeline: Restoration + HTR
    """
    # Step 1: Restore document
    restored, tiff_path = restore_document(image, alpha, post_processing, aggressive, thin_strokes, gamma)
    
    if restored is None:
        return None, None, "", 0.0, "Restoration failed"
    
    # Step 2: Run HTR if requested
    if run_htr:
        text, confidence, status = perform_htr(restored, htr_address)
        return restored, tiff_path, text, confidence, status
    else:
        return restored, tiff_path, "", 0.0, "Restoration complete (HTR skipped)"


# Define Gradio Interface
with gr.Blocks(title="Document Restoration + HTR System") as demo:
    gr.Markdown(
        """
        # 📜 Integrated Document Restoration + HTR System
        
        **Pipeline:**
        1. **Document Restoration** using Dual-Modal GAN
        2. **Handwritten Text Recognition (HTR)** using Loghi system
        
        Upload a degraded document image to restore and optionally extract text.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            input_image = gr.Image(
                label="Degraded Document", 
                type="numpy",
                interactive=True
            )
            
            with gr.Accordion("Restoration Settings", open=True):
                alpha = gr.Slider(
                    minimum=0.0, maximum=0.5, value=0.0, step=0.05, 
                    label="Alpha Blending (Mix with Original)",
                    info="0.0 = Fully Restored, 0.5 = 50% Original"
                )
                gamma = gr.Slider(
                    minimum=0.5, maximum=2.0, value=1.0, step=0.1, 
                    label="Gamma Correction",
                    info="< 1.0 Darker, > 1.0 Brighter"
                )
                
                post_processing = gr.Checkbox(value=True, label="Enable Post-processing")
                aggressive = gr.Checkbox(value=False, label="Aggressive Mode (Fix Broken Strokes)")
                thin_strokes = gr.Checkbox(value=False, label="Thin Strokes (Fix Boldness)")
            
            with gr.Accordion("HTR Settings", open=True):
                run_htr = gr.Checkbox(value=False, label="Enable HTR (Handwritten Text Recognition)")
                # HTR address is hidden, using default: 10.13.0.4:5001
            
            submit_btn = gr.Button("✨ Process Document", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            gr.Markdown("### Restoration Result")
            output_image = gr.Image(
                label="Restored Document", 
                type="numpy"
            )
            download_file = gr.File(
                label="📥 Download TIFF (High Quality, 300 DPI)",
                file_count="single"
            )
            
            status_message = gr.Textbox(
                label="Status",
                value="Ready"
            )
    
    with gr.Row():
        gr.Markdown(
            f"""
            **Model Information:**
            - Restoration Model: `{CHECKPOINT_DIR}/{CHECKPOINT_NAME}`
            - HTR Service: Loghi (configurable)
            
            **Note:** Ensure Loghi HTR service is running if you want to use HTR functionality.
            """
        )
            
    # Create a wrapper function that uses default HTR address
    def pipeline_wrapper(image, alpha, post_processing, aggressive, thin_strokes, gamma, run_htr):
        restored, tiff_path, text, confidence, status = integrated_pipeline(
            image, alpha, post_processing, aggressive, thin_strokes, gamma, run_htr, HTR_ADDRESS
        )
        return restored, tiff_path, status
    
    submit_btn.click(
        fn=pipeline_wrapper,
        inputs=[
            input_image, alpha, post_processing, aggressive, thin_strokes, gamma,
            run_htr
        ],
        outputs=[output_image, download_file, status_message]
    )

if __name__ == "__main__":
    print("="*80)
    print("Starting Integrated Document Restoration + HTR System")
    print("="*80)
    print(f"Restoration Model: {CHECKPOINT_DIR}/{CHECKPOINT_NAME}")
    print(f"HTR Service: {HTR_ADDRESS}")
    print("="*80)
    demo.launch(server_name="0.0.0.0", share=False)
