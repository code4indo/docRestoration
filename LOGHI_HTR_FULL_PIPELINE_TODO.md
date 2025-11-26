# Loghi HTR Full Pipeline - Implementation Guide

## 📋 Status Summary

### ✅ What's Working
1. **Document Restoration** - Dual-Modal GAN (Complete ✓)
2. **LAYPA Service** - Layout analysis running on port 5002 with GPU ✓
3. **HTR Service** - Text recognition running on port 5001 with GPU (NVIDIA RTX A4000) ✓
4. **TOOLING Service** - Running on port 8082 ✓
5. **Gradio UI** - Running on port 7862 ✓

### ❌ What's NOT Working
**HTR tidak menghasilkan text** karena workflow tidak sesuai dengan design Loghi.

## 🔍 Root Cause Analysis

### Kesalahan Fundamental
Kita mengirim **FULL DOCUMENT IMAGE** langsung ke HTR service, padahal:
- Loghi HTR dirancang untuk **LINE-LEVEL recognition**
- HTR expects individual text line images, not full page
- Tanpa line cutting, HTR tidak tahu cara memproses full document

### Workflow Yang Benar (Dari inference-pipeline.sh)
```
1. LAYPA 
   ↓ baseline detection
   ↓ output: PageXML with baseline coordinates
   
2. MinionExtractBaselines (TOOLING)
   ↓ Extract baseline info from PNG/XML
   ↓ output: Updated PageXML with text regions
   
3. MinionCutFromImageBasedOnPageXMLNew (TOOLING)
   ↓ Cut individual text lines from full image
   ↓ output: Individual line images (PNG)
   
4. Loghi HTR
   ↓ Recognize text for EACH line
   ↓ output: results.txt (one line per image)
   
5. MinionLoghiHTRMergePageXML (TOOLING)
   ↓ Merge HTR results back to PageXML
   ↓ output: Final PageXML with transcriptions
```

## 🛠️ Implementation TODO

### Step 1: Update `perform_htr_simple()` Function

**Current Implementation (WRONG):**
```python
def perform_htr_simple(image):
    # Send full document to HTR
    response = requests.post(HTR_ADDRESS/predict, files={'image': full_doc})
    # Poll for result (NEVER COMES because wrong input!)
```

**Correct Implementation Needed:**
```python
def perform_htr_with_line_cutting(restored_image, laypa_xml_path):
    """
    Full HTR pipeline with line cutting
    
    Steps:
    1. Get PageXML from LAYPA output
    2. Call TOOLING to cut lines from image
    3. Send each line to HTR
    4. Aggregate results
    """
    
    # Step 1: Parse LAYPA output to get PageXML path
    # LAYPA writes output to: /home/lambda_one/tesis/loghi/laypa_output/<identifier>/page/<identifier>.xml
    
    # Step 2: Call TOOLING - Cut lines
    # POST http://localhost:8082/cut-from-image-based-on-page-xml-new
    # Form data:
    #   - image: full restored document
    #   - page: PageXML from LAYPA
    #   - identifier: unique_id
    #   - output_type: 'png'
    #   - channels: 4
    # Output: Individual line images in /loghi_output/<identifier>/
    
    # Step 3: Send each line to HTR
    # For each line image:
    #   POST http://localhost:5001/predict
    #   Form data:
    #     - image: line_image.png
    #     - group_id: document_id
    #     - identifier: line_id
    
    # Step 4: Poll for each line result
    # Check: /loghi_output/<group_id>/<line_id>/<line_id>.txt
    
    # Step 5: Aggregate all line texts
    # Combine in reading order (from PageXML)
    
    return full_text, confidence, status
```

### Step 2: Environment Configuration

Add to `.env` or check current setup:
```bash
LAYPA_OUTPUT_PATH=/home/lambda_one/tesis/loghi/laypa_output
LOGHI_OUTPUT_PATH=/home/lambda_one/tesis/loghi/loghi_output
TOOLING_ADDRESS=http://localhost:8082
```

### Step 3: Update LAYPA Integration

**Current Code:**
```python
def laypa_segmentation(image, model_path="general/baseline"):
    # ... sends image ...
    # returns: (image, "XML placeholder", status)  # ❌ WRONG!
```

**Needs:**
```python
def laypa_segmentation(image, model_path="general/baseline"):
    # ... sends image ...
    
    # After LAYPA completes, get actual XML path
    xml_path = f"{LAYPA_OUTPUT_PATH}/{identifier}/page/{identifier}.xml"
    
    # Wait for file to be created
    while not os.path.exists(xml_path) and timeout < max_wait:
        time.sleep(1)
        timeout += 1
    
    if os.path.exists(xml_path):
        xml_content = Path(xml_path).read_text()
        return image, xml_content, xml_path, "✓ LAYPA complete"
    else:
        return None, None, None, "✗ LAYPA timeout"
```

### Step 4: Implement Line Cutting via TOOLING

```python
def cut_lines_from_document(image_path, xml_path, identifier):
    """
    Call TOOLING service to cut text lines from document
    
    Returns:
    --------
    list: Paths to cut line images
    """
    url = "http://localhost:8082/cut-from-image-based-on-page-xml-new"
    
    with open(image_path, 'rb') as img_file, open(xml_path, 'rb') as xml_file:
        files = {
            'image': img_file,
            'page': xml_file
        }
        data = {
            'identifier': identifier,
            'output_type': 'png',
            'channels': 4
        }
        
        response = requests.post(url, files=files, data=data, timeout=120)
    
    if response.ok:
        # TOOLING writes output to: /loghi_output/<identifier>/
        output_dir = Path(f"/home/lambda_one/tesis/loghi/loghi_output/{identifier}")
        
        # Poll for line images
        max_wait = 30
        waited = 0
        while waited < max_wait:
            if output_dir.exists():
                line_images = list(output_dir.glob("*.png"))
                if line_images:
                    return sorted(line_images)
            time.sleep(2)
            waited += 2
        
        raise TimeoutError("Line cutting timeout")
    else:
        raise Exception(f"TOOLING failed: {response.status_code}")
```

### Step 5: Implement Per-Line HTR

```python
def recognize_line(line_image_path, group_id):
    """
    Send single line image to HTR for recognition
    
    Returns:
    --------
    str: Recognized text for this line
    """
    line_id = line_image_path.stem  # filename without extension
    url = "http://localhost:5001/predict"
    
    with open(line_image_path, 'rb') as f:
        files = {'image': f}
        data = {
            'group_id': group_id,
            'identifier': line_id
        }
        
        response = requests.post(url, files=files, data=data, timeout=30)
    
    if response.status_code == 202:
        # Poll for result
        result_file = Path(f"/home/lambda_one/tesis/loghi/loghi_output/{group_id}/{line_id}/{line_id}.txt")
        
        max_wait = 60
        waited = 0
        while waited < max_wait:
            if result_file.exists():
                return result_file.read_text().strip()
            time.sleep(2)
            waited += 2
        
        raise TimeoutError(f"HTR timeout for line {line_id}")
    else:
        raise Exception(f"HTR request failed: {response.status_code}")
```

### Step 6: Complete Integrated Pipeline

```python
def integrated_pipeline_full_correct(image, alpha, post_processing, aggressive, thin_strokes, gamma, run_htr, laypa_model):
    """
    CORRECT full pipeline implementation
    """
    # Step 1: Restore document
    restored, tiff_path, restore_status = restore_document(
        image, alpha, post_processing, aggressive, thin_strokes, gamma
    )
    
    if restored is None or not run_htr:
        return restored, tiff_path, "", 0.0, restore_status, None, None
    
    # Step 2: LAYPA segmentation
    seg_image, xml_content, xml_path, laypa_status = laypa_segmentation(restored, laypa_model)
    
    if xml_path is None:
        return restored, tiff_path, "", 0.0, f"{restore_status} | {laypa_status}", None, None
    
    # Step 3: Cut lines via TOOLING
    identifier = f"doc_{int(time.time())}"
    try:
        line_images = cut_lines_from_document(tiff_path, xml_path, identifier)
        cut_status = f"✓ Cut {len(line_images)} lines"
    except Exception as e:
        return restored, tiff_path, "", 0.0, f"{restore_status} | {laypa_status} | ✗ Cutting failed: {e}", seg_image, None
    
    # Step 4: HTR for each line
    line_results = []
    for line_img in line_images:
        try:
            text = recognize_line(line_img, identifier)
            line_results.append({
                'Line': line_img.stem,
                'Text': text,
                'Confidence': 1.0  # TODO: get actual confidence
            })
        except Exception as e:
            logger.error(f"HTR failed for {line_img}: {e}")
            line_results.append({
                'Line': line_img.stem,
                'Text': f"[ERROR: {e}]",
                'Confidence': 0.0
            })
    
    # Step 5: Aggregate results
    full_text = "\n".join([r['Text'] for r in line_results])
    avg_confidence = sum([r['Confidence'] for r in line_results]) / len(line_results) if line_results else 0.0
    
    df_lines = pd.DataFrame(line_results)
    
    final_status = f"{restore_status} | {laypa_status} | {cut_status} | ✓ HTR Complete"
    
    return restored, tiff_path, full_text, avg_confidence, final_status, seg_image, df_lines
```

## 📝 Alternative: Simpler Solution

If full pipeline is too complex, consider:

### Option A: Use Tesseract OCR for Full Document
```python
import pytesseract

def perform_ocr_tesseract(image):
    """Simple OCR using Tesseract"""
    text = pytesseract.image_to_string(image, lang='nld')  # Dutch
    return text, 1.0, "✓ OCR Complete"
```

### Option B: Disable HTR Mode, Document Only
Keep "Simple Mode" as restoration-only, implement "Full Pipeline Mode" properly.

## 🎯 Recommendation

**For immediate functionality:**
1. Implement Tesseract fallback for Simple Mode
2. Label it as "OCR (Tesseract)" vs "HTR (Loghi - Full Pipeline)"

**For research/publication:**
1. Implement full Loghi pipeline as documented above
2. Use for line-level accuracy analysis
3. Benchmark against Tesseract

## 📚 References

- Loghi inference pipeline: `/home/lambda_one/tesis/loghi/scripts/inference-pipeline.sh`
- Webservice scripts: `/home/lambda_one/tesis/loghi/webservice/webservice-scripts/`
- Docker compose: `scripts/loghi_services/docker-compose.yml`

## 🚀 Next Steps

1. **Decide**: Full Loghi pipeline vs Tesseract fallback?
2. **Implement**: Based on decision above
3. **Test**: With real historical documents
4. **Benchmark**: Compare accuracy (CER/WER)
5. **Document**: For thesis Chapter 4

---

**Status**: HTR service is working correctly with GPU. Implementation needs workflow correction.

**Date**: 2025-11-24
**Author**: Analysis by GitHub Copilot (Claude Sonnet 4.5)
