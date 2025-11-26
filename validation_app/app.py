#!/usr/bin/env python3
"""
Document Restoration Validation System
=======================================
Validates CER improvement by comparing HTR results before vs after restoration.
Uses LLM (Grok via OpenRouter) as "judge" to analyze transcription quality.

Author: belekok
Date: 2025-11-25
"""

import os
import sys
import json
import base64
import tempfile
import time
import subprocess
import shutil
import glob
import re
from pathlib import Path
from datetime import datetime
import logging

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import requests
import numpy as np
from PIL import Image
import cv2
import xml.etree.ElementTree as ET

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "dual_modal_gan" / "scripts"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ValidationApp")

app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
OPENROUTER_API_KEY = Path(__file__).parent.parent / "api_key.txt"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "x-ai/grok-4.1-fast"  # Grok 4.1 Fast via OpenRouter

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "validation_app" / "results"
UPLOAD_DIR = BASE_DIR / "validation_app" / "uploads"
LOGHI_PIPELINE = "/home/lambda_one/tesis/loghi/scripts/inference-pipeline.sh"

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Load restoration model (lazy load)
GENERATOR = None


def get_api_key():
    """Load OpenRouter API key"""
    if OPENROUTER_API_KEY.exists():
        return OPENROUTER_API_KEY.read_text().strip()
    return os.environ.get("OPENROUTER_API_KEY", "")


def load_restoration_model():
    """Load the GAN restoration model"""
    global GENERATOR
    if GENERATOR is not None:
        return GENERATOR
    
    try:
        import tensorflow as tf
        
        # GPU config
        gpus = tf.config.list_physical_devices('GPU')
        if gpus and len(gpus) > 1:
            tf.config.set_visible_devices(gpus[1], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[1], True)
        elif gpus:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        
        # Import inference module
        scripts_dir = BASE_DIR / "dual_modal_gan" / "scripts"
        sys.path.insert(0, str(scripts_dir))
        import inference_portrait_overlap_experiment as inference
        
        # Load model
        ckpt_dir = BASE_DIR / "dual_modal_gan" / "checkpoints" / "production_full_coverage_vgg_v1"
        GENERATOR = inference.load_model(str(ckpt_dir), "ckpt-94", gpu_id=1)
        logger.info("Restoration model loaded successfully")
        return GENERATOR
    except Exception as e:
        logger.error(f"Failed to load restoration model: {e}")
        return None


def restore_image(image_array):
    """
    Restore a degraded document image using the GAN model.
    
    Parameters:
    -----------
    image_array : numpy.ndarray
        Input degraded image (grayscale or RGB)
        
    Returns:
    --------
    numpy.ndarray : Restored image (grayscale)
    """
    try:
        generator = load_restoration_model()
        if generator is None:
            raise RuntimeError("Model not loaded")
        
        import inference_portrait_overlap_experiment as inference
        
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image_array
        
        # Process
        restored = inference.process_portrait_document(
            image_gray,
            generator,
            logger,
            alpha=0.0,
            post_processing=True,
            aggressive=False,
            thin_strokes=False,
            gamma=1.0
        )
        
        return restored
    except Exception as e:
        logger.error(f"Restoration failed: {e}")
        raise


def run_htr_pipeline(image_path: str) -> dict:
    """
    Run Loghi HTR pipeline on an image.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
        
    Returns:
    --------
    dict : {text, lines, xml_path, status}
    """
    try:
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix='htr_val_')
        
        # Copy image to temp dir
        img_name = Path(image_path).stem
        dest_path = Path(temp_dir) / f"{img_name}.jpg"
        
        # Load and save as JPG
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(dest_path, format='JPEG', quality=95)
        
        # Create page directory
        page_dir = Path(temp_dir) / "page"
        page_dir.mkdir(exist_ok=True)
        
        # Run pipeline
        logger.info(f"Running HTR pipeline on {dest_path}")
        result = subprocess.run(
            [LOGHI_PIPELINE, temp_dir],
            capture_output=True,
            text=True,
            timeout=300,
            cwd="/home/lambda_one/tesis/loghi"
        )
        
        # Find output XML
        xml_files = list(page_dir.glob("*.xml"))
        
        if not xml_files:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return {"text": "", "lines": [], "xml_path": None, "status": "No XML output"}
        
        xml_path = xml_files[0]
        
        # Parse XML
        lines = parse_pagexml(str(xml_path))
        full_text = "\n".join([l['text'] for l in lines if l['text']])
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return {
            "text": full_text,
            "lines": lines,
            "line_count": len(lines),
            "status": "success"
        }
        
    except subprocess.TimeoutExpired:
        return {"text": "", "lines": [], "status": "HTR timeout"}
    except Exception as e:
        logger.error(f"HTR pipeline error: {e}")
        return {"text": "", "lines": [], "status": f"Error: {str(e)}"}


def parse_pagexml(xml_path: str) -> list:
    """Parse PageXML and extract text lines (full line text, not word-level)"""
    lines = []
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
            text_lines = root.findall('.//TextLine')
        
        for idx, line in enumerate(text_lines):
            text = ""
            coords = ""
            
            # Try to get TextEquiv directly under TextLine (full line text)
            # NOT from Word/TextEquiv (word-level text)
            if ns:
                # Look for direct child TextEquiv, not descendant
                text_equiv = None
                for child in line:
                    local_name = child.tag.replace(f'{{{ns}}}', '')
                    if local_name == 'TextEquiv':
                        text_equiv = child
                        break
                
                if text_equiv is not None:
                    unicode_elem = text_equiv.find(f'{{{ns}}}Unicode')
                    if unicode_elem is not None and unicode_elem.text:
                        text = unicode_elem.text.strip()
                
                # Get coords
                coords_elem = line.find(f'{{{ns}}}Coords')
            else:
                # Non-namespaced version
                text_equiv = None
                for child in line:
                    if child.tag == 'TextEquiv':
                        text_equiv = child
                        break
                
                if text_equiv is not None:
                    unicode_elem = text_equiv.find('Unicode')
                    if unicode_elem is not None and unicode_elem.text:
                        text = unicode_elem.text.strip()
                
                coords_elem = line.find('Coords')
            
            if coords_elem is not None:
                coords = coords_elem.get('points', '')
            
            # Only add if we have text
            if text:
                lines.append({
                    'line_id': idx + 1,
                    'text': text,
                    'coords': coords
                })
        
        logger.info(f"Parsed {len(lines)} text lines from {xml_path}")
        
    except Exception as e:
        logger.error(f"XML parse error: {e}")
        import traceback
        traceback.print_exc()
    
    return lines


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) using Levenshtein distance.
    
    CER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=reference length
    """
    if not reference:
        return 1.0 if hypothesis else 0.0
    
    ref = reference.lower().strip()
    hyp = hypothesis.lower().strip()
    
    # Levenshtein distance
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    edit_distance = dp[m][n]
    cer = edit_distance / len(ref) if ref else 0.0
    
    return min(cer, 1.0)  # Cap at 100%


def call_llm_judge(degraded_text: str, restored_text: str) -> dict:
    """
    Call LLM to:
    1. Generate estimated Ground Truth from combining both HTR results
    2. Evaluate each HTR result against the estimated GT
    3. Calculate improvement metrics
    
    Parameters:
    -----------
    degraded_text : str
        HTR result from degraded image
    restored_text : str
        HTR result from restored image
        
    Returns:
    --------
    dict : LLM analysis result with estimated_ground_truth
    """
    api_key = get_api_key()
    if not api_key:
        return {"error": "No API key configured"}
    
    # Build prompt - SIMPLIFIED for reliable JSON output
    system_prompt = """Anda adalah ahli paleografi dan penerjemah dokumen VOC abad 16-18.

TUGAS:
1. Rekonstruksi GROUND TRUTH dari dua hasil HTR (pertahankan ejaan historis Belanda Kuno)
2. Terjemahkan ke Bahasa Indonesia
3. Evaluasi kualitas kedua sumber HTR

ATURAN PENTING:
- PERTAHANKAN ejaan historis (schip, coopman, compagnie, zynde, haere)
- Perbaiki error OCR: rn→m, cl→d, li→h, vv→w
- Output HARUS JSON valid tanpa komentar

CONFIDENCE SCORE (0-100):
- 90-100: Kedua HTR identik/mirip, makna jelas
- 75-89: Satu sumber lebih baik, konteks kuat  
- 50-74: Perbedaan signifikan, inferensi semantik
- 25-49: Banyak tidak terbaca, spekulatif
- 0-24: Sebagian besar tidak terbaca"""

    user_prompt = f"""Analisis HTR dari dokumen VOC:

SUMBER A (Degraded):
{degraded_text if degraded_text else "[KOSONG]"}

SUMBER B (Restored):
{restored_text if restored_text else "[KOSONG]"}

Output JSON (TANPA komentar, TANPA markdown):
{{
    "document_context": "jenis dokumen singkat",
    "estimated_ground_truth": "teks GT Belanda (baris dipisah newline)",
    "estimated_ground_truth_indonesian": "terjemahan Indonesia (baris dipisah newline)",
    "reconstruction_notes": "catatan singkat rekonstruksi",
    "confidence_explanation": "alasan confidence score",
    "degraded_analysis": {{
        "cer_estimated": 0.0,
        "overall_score": 0,
        "issues": []
    }},
    "restored_analysis": {{
        "cer_estimated": 0.0,
        "overall_score": 0,
        "improvements": []
    }},
    "comparison": {{
        "cer_improvement_percent": 0,
        "winner": "restored",
        "confidence": 0,
        "confidence_level": "medium"
    }},
    "verdict_indonesian": "kesimpulan dalam bahasa Indonesia",
    "recommendation_indonesian": "rekomendasi dalam bahasa Indonesia"
}}"""

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:7864",
            "X-Title": "GAN-HTR Validation System"
        }
        
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 6000,
            "response_format": {"type": "json_object"}
        }
        
        logger.info(f"Calling OpenRouter API with model: {LLM_MODEL}")
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=180)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            logger.info(f"LLM raw response length: {len(content)} chars")
            
            # Parse JSON response with multiple fallbacks
            try:
                analysis = json.loads(content)
                analysis['llm_model'] = LLM_MODEL
                analysis['status'] = 'success'
                return analysis
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                logger.error(f"Raw content (first 1000 chars): {content[:1000]}")
                
                # Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
                if json_match:
                    try:
                        analysis = json.loads(json_match.group(1))
                        analysis['llm_model'] = LLM_MODEL
                        analysis['status'] = 'success'
                        logger.info("Successfully parsed JSON from code block")
                        return analysis
                    except:
                        pass
                
                # Try to find JSON object pattern
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    try:
                        analysis = json.loads(json_match.group(0))
                        analysis['llm_model'] = LLM_MODEL
                        analysis['status'] = 'success'
                        logger.info("Successfully parsed JSON from pattern match")
                        return analysis
                    except:
                        pass
                
                # Return fallback with raw content for debugging
                return {
                    "status": "parse_error",
                    "raw_response": content[:2000],
                    "error": f"Failed to parse LLM JSON response: {str(e)}",
                    "estimated_ground_truth": "",
                    "estimated_ground_truth_indonesian": "",
                    "comparison": {"confidence": 0, "winner": "unknown"},
                    "verdict": "Gagal parsing response LLM",
                    "verdict_indonesian": "Gagal parsing response LLM"
                }
        else:
            logger.error(f"API error: {response.status_code} - {response.text[:500]}")
            return {
                "status": "api_error",
                "error": f"API returned {response.status_code}: {response.text[:500]}"
            }
            
    except requests.exceptions.Timeout:
        return {"status": "timeout", "error": "LLM API timeout (180s)"}
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


# ============== API Routes ==============

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('static', 'index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)


@app.route('/api/validate', methods=['POST'])
def validate_document():
    """
    Main validation endpoint.
    
    Accepts:
    - degraded_image: File upload (degraded document)
    
    Returns:
    - Comparison results, LLM-generated GT, CER metrics
    """
    try:
        if 'degraded_image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files['degraded_image']
        
        # Generate session ID
        session_id = f"val_{int(time.time())}"
        session_dir = RESULTS_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded image
        degraded_path = session_dir / "degraded.jpg"
        img = Image.open(file.stream)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        img.save(degraded_path, format='JPEG', quality=95)
        
        logger.info(f"Session {session_id}: Uploaded image {img_array.shape}")
        
        # Step 1: Run HTR on degraded image
        logger.info(f"Session {session_id}: Running HTR on degraded image...")
        degraded_htr = run_htr_pipeline(str(degraded_path))
        
        # Step 2: Restore image
        logger.info(f"Session {session_id}: Restoring image...")
        restored_array = restore_image(img_array)
        
        # Save restored image
        restored_path = session_dir / "restored.jpg"
        Image.fromarray(restored_array).save(restored_path, format='JPEG', quality=95)
        
        # Step 3: Run HTR on restored image
        logger.info(f"Session {session_id}: Running HTR on restored image...")
        restored_htr = run_htr_pipeline(str(restored_path))
        
        # Step 4: Call LLM judge - LLM will generate Ground Truth and evaluate
        logger.info(f"Session {session_id}: Calling LLM judge to generate GT and evaluate...")
        llm_analysis = call_llm_judge(
            degraded_htr['text'],
            restored_htr['text']
        )
        
        # Extract CER from LLM analysis
        cer_degraded = None
        cer_restored = None
        cer_improvement = None
        estimated_gt = None
        
        if llm_analysis.get('status') == 'success':
            # Get estimated ground truth from LLM
            estimated_gt = llm_analysis.get('estimated_ground_truth', '')
            
            # Get CER estimates from LLM
            deg_analysis = llm_analysis.get('degraded_analysis', {})
            rest_analysis = llm_analysis.get('restored_analysis', {})
            
            cer_degraded = deg_analysis.get('cer_estimated', 0) * 100  # Convert to percentage
            cer_restored = rest_analysis.get('cer_estimated', 0) * 100
            
            # Also calculate actual CER if we have estimated GT
            if estimated_gt:
                actual_cer_degraded = calculate_cer(estimated_gt, degraded_htr['text'])
                actual_cer_restored = calculate_cer(estimated_gt, restored_htr['text'])
                
                # Use actual calculated CER
                cer_degraded = round(actual_cer_degraded * 100, 2)
                cer_restored = round(actual_cer_restored * 100, 2)
            
            # Calculate improvement
            if cer_degraded and cer_degraded > 0:
                cer_improvement = ((cer_degraded - cer_restored) / cer_degraded) * 100
            
            # Get from LLM comparison if available
            comparison = llm_analysis.get('comparison', {})
            if 'cer_improvement_percent' in comparison:
                cer_improvement = comparison['cer_improvement_percent']
        
        # Prepare response
        result = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "degraded": {
                "image_url": f"/api/image/{session_id}/degraded.jpg",
                "htr_text": degraded_htr['text'],
                "lines": degraded_htr.get('lines', []),
                "line_count": degraded_htr.get('line_count', 0),
                "htr_status": degraded_htr['status']
            },
            "restored": {
                "image_url": f"/api/image/{session_id}/restored.jpg",
                "htr_text": restored_htr['text'],
                "lines": restored_htr.get('lines', []),
                "line_count": restored_htr.get('line_count', 0),
                "htr_status": restored_htr['status']
            },
            "estimated_ground_truth": estimated_gt,
            "metrics": {
                "cer_degraded": round(cer_degraded, 2) if cer_degraded is not None else None,
                "cer_restored": round(cer_restored, 2) if cer_restored is not None else None,
                "cer_improvement_percent": round(cer_improvement, 2) if cer_improvement is not None else None
            },
            "llm_analysis": llm_analysis,
            "status": "success"
        }
        
        # Save result
        with open(session_dir / "result.json", 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Session {session_id}: Validation complete")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route('/api/image/<session_id>/<filename>')
def serve_image(session_id, filename):
    """Serve result images"""
    image_path = RESULTS_DIR / session_id / filename
    if image_path.exists():
        return send_file(image_path, mimetype='image/jpeg')
    return jsonify({"error": "Image not found"}), 404


@app.route('/api/history')
def get_history():
    """Get validation history"""
    history = []
    
    for session_dir in sorted(RESULTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if session_dir.is_dir():
            result_file = session_dir / "result.json"
            if result_file.exists():
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                        history.append({
                            "session_id": data.get("session_id"),
                            "timestamp": data.get("timestamp"),
                            "cer_improvement": data.get("metrics", {}).get("cer_improvement_percent"),
                            "llm_verdict": data.get("llm_analysis", {}).get("verdict", "")[:100]
                        })
                except:
                    pass
    
    return jsonify({"history": history[:20]})  # Last 20


@app.route('/api/session/<session_id>')
def get_session(session_id):
    """Get specific session result"""
    result_file = RESULTS_DIR / session_id / "result.json"
    if result_file.exists():
        with open(result_file) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Session not found"}), 404


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": GENERATOR is not None,
        "api_key_configured": bool(get_api_key()),
        "llm_model": LLM_MODEL
    })


if __name__ == '__main__':
    print("=" * 70)
    print("  Document Restoration Validation System")
    print("=" * 70)
    print(f"  LLM Model: {LLM_MODEL}")
    print(f"  API Key: {'Configured' if get_api_key() else 'NOT CONFIGURED'}")
    print(f"  Results Dir: {RESULTS_DIR}")
    print("=" * 70)
    print("\n  Starting server on http://0.0.0.0:7864\n")
    
    app.run(host='0.0.0.0', port=7864, debug=False, threaded=True)
