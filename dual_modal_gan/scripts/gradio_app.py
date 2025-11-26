import sys
import os
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
import logging

# Setup paths
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.append(str(current_dir))

# Import the inference logic
try:
    import inference_portrait_overlap_experiment as inference
except ImportError:
    # Try importing by file path if module import fails
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "inference_portrait_overlap_experiment",
        str(current_dir / "inference_portrait_overlap_experiment.py")
    )
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GradioApp")

# Global variables
GENERATOR = None
CHECKPOINT_DIR = "production_full_coverage_vgg_v1"
CHECKPOINT_NAME = "ckpt-94"

def load_generator():
    global GENERATOR
    if GENERATOR is None:
        logger.info("Loading model...")
        # Full path: checkpoints/production_full_coverage_vgg_v1/
        ckpt_dir = parent_dir / "checkpoints" / CHECKPOINT_DIR
        
        # Ensure checkpoint directory exists
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        
        # Check GPU availability and configure
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Try to use GPU 0
                tf.config.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
                logger.info(f"Using GPU: {gpus[0].name}")
                gpu_id = 0
            except RuntimeError as e:
                logger.warning(f"GPU setup failed: {e}. Falling back to CPU.")
                tf.config.set_visible_devices([], 'GPU')
                gpu_id = -1
        else:
            logger.info("No GPU detected. Using CPU.")
            tf.config.set_visible_devices([], 'GPU')
            gpu_id = -1
            
        GENERATOR = inference.load_model(str(ckpt_dir), CHECKPOINT_NAME, gpu_id=0)
        logger.info("Model loaded successfully.")
    return GENERATOR

def predict(image, alpha, post_processing, aggressive, thin_strokes, gamma):
    """
    Inference function for Gradio
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
        
        # Run inference using the imported function
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
        
        # Create TIFF file for download
        from PIL import Image
        import tempfile
        
        # Create temporary TIFF file
        tiff_fd, tiff_path = tempfile.mkstemp(suffix='.tiff', prefix='restored_')
        os.close(tiff_fd)  # Close file descriptor
        
        # Save as TIFF with LZW compression
        restored_pil = Image.fromarray(restored)
        restored_pil.save(tiff_path, format='TIFF', compression='tiff_lzw', dpi=(300, 300))
        
        logger.info(f"TIFF saved: {tiff_path}")
        
        return restored, tiff_path
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

# Define Gradio Interface
with gr.Blocks(title="Document Restoration AI") as demo:
    gr.Markdown(
        """
        # 📜 Document Restoration AI
        Restores degraded historical documents using Dual-Modal GAN.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Input Degraded Document", 
                type="numpy",
                interactive=True
            )
            
            with gr.Accordion("Advanced Settings", open=True):
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
            
            submit_btn = gr.Button("✨ Restore Document", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Restored Result", 
                type="numpy"
            )
            download_file = gr.File(
                label="📥 Download TIFF (High Quality, 300 DPI)",
                file_count="single"
            )
    
    gr.Markdown(f"**Model Checkpoint:** `{CHECKPOINT_DIR}/{CHECKPOINT_NAME}`")
            
    submit_btn.click(
        fn=predict,
        inputs=[input_image, alpha, post_processing, aggressive, thin_strokes, gamma],
        outputs=[output_image, download_file]
    )

if __name__ == "__main__":
    print("Starting Gradio Server...")
    demo.launch(server_name="0.0.0.0", share=False)
