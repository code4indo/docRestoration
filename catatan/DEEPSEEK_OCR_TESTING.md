# DeepSeek-OCR Line Detection Testing Report
**Date**: 2025-10-22  
**Status**: In Progress

## Setup Process

### 1. Environment Configuration
- Created separate virtual environment: `.venv_deepseek`
- Reason: PyTorch conflicts with TensorFlow in main environment

### 2. Dependencies Installation
**Initial (CPU-only):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.46.3 einops addict easydict pillow opencv-python matplotlib numpy scipy
```

**Issue**: DeepSeek-OCR model has hard-coded `.cuda()` calls - cannot run on CPU

**Current (GPU with CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
Status: **Installing (in progress)** - Large download (~2.7GB total)

### 3. Model Information
- **Model**: deepseek-ai/DeepSeek-OCR
- **Size**: 3B parameters (~1.6GB download)
- **Architecture**: DeepseekOCRForCausalLM (Vision-Language Model)
- **HuggingFace**: https://huggingface.co/deepseek-ai/DeepSeek-OCR

### 4. Implementation
Created: `dual_modal_gan/scripts/line_detection_deepseek.py`

**Key Features:**
- Grounding-based text region detection
- Converts paragraph/block regions to line-level bounding boxes
- Post-processing to split large blocks into individual lines
- Visualization support

**Prompt Used:**
```python
prompt = "<image>\n<|grounding|>Convert the document to markdown."
```

## Testing Status

### Attempted Tests
1. **CPU Inference**: ❌ Failed
   - Error: `AssertionError: Torch not compiled with CUDA enabled`
   - Reason: Model code hard-coded `.cuda()` calls

2. **GPU Inference**: ⏳ Pending
   - Installing PyTorch with CUDA support
   - Will test on GPU 1 (NVIDIA RTX A4000)

## Findings So Far

### ✅ Advantages
1. **Pre-trained VLM**: State-of-the-art OCR + layout understanding
2. **Grounding Support**: Can detect bounding boxes for text regions
3. **Multi-format Output**: Markdown, structured data, etc.

### ⚠️ Challenges
1. **Environment Conflicts**: Requires separate venv (PyTorch vs TensorFlow)
2. **GPU Required**: Cannot run on CPU due to hard-coded CUDA calls
3. **Large Model**: 3B parameters, significant memory footprint
4. **Granularity**: Detects paragraph/block level, needs post-processing for lines
5. **Setup Complexity**: Multiple dependency issues, version pinning required

### 📊 Expected Performance
- **For Line Detection**: Unknown (paragraph-level by design)
- **Need Manual Splitting**: Large blocks → individual lines
- **Accuracy**: TBD after successful GPU test

## Next Steps

1. ⏳ **Wait for PyTorch CUDA installation** (~5-10 minutes)
2. ✅ **Test on GPU**: Run inference on DIBCO image 1
3. 📊 **Evaluate Results**: Compare with Robust Morphological method
4. 🎯 **Decision**: Keep or abandon based on:
   - Line detection accuracy
   - Inference speed
   - Complexity vs benefit

## Comparison: DeepSeek-OCR vs Current Methods

| Metric | DeepSeek-OCR | Robust Morphological | Projection Profile |
|--------|--------------|---------------------|-------------------|
| **Setup** | Complex (separate env) | Simple (OpenCV only) | Simple (OpenCV only) |
| **Dependencies** | PyTorch, Transformers, etc. | OpenCV, NumPy, SciPy | OpenCV, NumPy, SciPy |
| **Model Size** | 3B params (~1.6GB) | None (algorithm-based) | None (algorithm-based) |
| **GPU Required** | Yes (hard requirement) | No | No |
| **Speed** | Slow (VLM inference) | Fast (CV operations) | Fast (CV operations) |
| **Granularity** | Paragraph/Block level | Line level ✅ | Line level ✅ |
| **Accuracy** | TBD | Good (+1.65 dB PSNR) | Baseline |
| **Success Rate** | TBD | 70% (7/10 DIBCO) | 90% (9/10 DIBCO) |

## Preliminary Recommendation

**IF GPU test succeeds AND accuracy is significantly better:**
- Consider as **optional enhancement** for challenging documents
- Keep Robust Morphological as primary method
- Use DeepSeek-OCR as fallback for failed cases

**IF GPU test fails OR accuracy not better:**
- **Abandon** DeepSeek-OCR approach
- Focus on **Hybrid Method**: Robust → Projection fallback
- Optimize Robust parameters for better success rate

---

**Update Log:**
- 15:32 - Started setup
- 15:35 - CPU test failed (CUDA required)
- 15:42 - Installing PyTorch with CUDA (in progress)
- 15:45 - Awaiting installation completion...
