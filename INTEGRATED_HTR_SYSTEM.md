# Integrated Document Restoration + HTR System

## Overview

Sistem terintegrasi yang menggabungkan:
1. **Document Restoration** menggunakan Dual-Modal GAN
2. **Handwritten Text Recognition (HTR)** menggunakan sistem Loghi

## Architecture

```
┌─────────────────┐
│  User Upload    │
│  (Degraded Doc) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Document Restoration   │
│  (Dual-Modal GAN)       │
│  - Adaptive Tiling      │
│  - Post-processing      │
└────────┬────────────────┘
         │
         ├─► TIFF Download (300 DPI)
         │
         ▼
    [Optional HTR]
         │
         ▼
┌─────────────────────────┐
│  Loghi HTR Service      │
│  (HTTP API)             │
└────────┬────────────────┘
         │
         ▼
┌─────────────────┐
│  Text Output    │
│  + Confidence   │
└─────────────────┘
```

## Prerequisites

### 1. Document Restoration Model
- Model checkpoint: `dual_modal_gan/checkpoints/production_full_coverage_vgg_v1/ckpt-94`
- GPU recommended (fallback to CPU available)

### 2. Loghi HTR Service (Optional)
If you want HTR functionality, you need to run Loghi HTR service separately:

```bash
# Navigate to loghi directory
cd /home/lambda_one/tesis/loghi

# Start Loghi HTR service (example)
# Check loghi documentation for exact command
```

Default HTR service address: `http://localhost:5001`

## Installation

Already installed in the current environment via Poetry:
- gradio
- tensorflow
- opencv-python
- pillow
- requests

## Usage

### 1. Start the Integrated System

```bash
cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration

# With GPU support
CUDA_VISIBLE_DEVICES=0 poetry run python dual_modal_gan/scripts/gradio_integrated_htr.py

# Without GPU (CPU only)
poetry run python dual_modal_gan/scripts/gradio_integrated_htr.py
```

### 2. Access the Web Interface

Open browser: `http://localhost:7860`

### 3. Workflow

**Restoration Only:**
1. Upload degraded document image
2. Adjust restoration settings (optional)
3. Leave "Enable HTR" unchecked
4. Click "✨ Process Document"
5. Download TIFF result

**Full Pipeline (Restoration + HTR):**
1. Upload degraded document image
2. Adjust restoration settings (optional)
3. Check "Enable HTR"
4. Verify HTR service address
5. Click "✨ Process Document"
6. View restored image + extracted text

## Restoration Settings

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Alpha Blending | 0.0 - 0.5 | 0.0 | Mix with original (0.0 = fully restored) |
| Gamma Correction | 0.5 - 2.0 | 1.0 | Brightness adjustment |
| Post-processing | On/Off | On | CLAHE + morphological ops |
| Aggressive Mode | On/Off | Off | Fix broken strokes |
| Thin Strokes | On/Off | Off | Reduce stroke thickness |

## HTR Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| Enable HTR | Off | Toggle HTR processing |
| HTR Service Address | http://localhost:5001 | Loghi HTR endpoint |

## API Integration

The system integrates with Loghi HTR via REST API:

```python
# HTR Request Format
POST {htr_address}/predict
Content-Type: multipart/form-data

Files:
- image: (filename, image_bytes, 'image/png')

# Expected Response
{
    "text": "extracted text...",
    "confidence": 0.95,
    "prediction": "alternative key for text"
}
```

## Outputs

1. **Restored Image Preview** - Display in browser with zoom capability
2. **TIFF File** - High-quality download (300 DPI, LZW compression)
3. **Extracted Text** - HTR result (if enabled)
4. **Confidence Score** - HTR confidence (if enabled)

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| "Checkpoint not found" | Missing model | Check checkpoint path |
| "Cannot connect to HTR service" | Loghi not running | Start Loghi service |
| "GPU not found" | No CUDA device | System will fallback to CPU |
| "Restoration failed" | Invalid image | Check image format |

## Performance

### Restoration Speed
- GPU (RTX A4000): ~5-10 seconds for 2000x1500 image
- CPU: ~30-60 seconds for same image

### HTR Speed
Depends on Loghi service configuration and image size.

## File Structure

```
dual_modal_gan/scripts/
├── gradio_integrated_htr.py        # Main integrated app
├── gradio_app.py                   # Restoration-only app (original)
├── inference_portrait_overlap_experiment.py  # Core restoration logic
└── ...

dual_modal_gan/checkpoints/
└── production_full_coverage_vgg_v1/
    └── ckpt-94                     # Latest checkpoint
```

## Troubleshooting

### HTR Service Not Responding

1. Check Loghi service is running:
   ```bash
   curl http://localhost:5001/health
   ```

2. Verify port is correct in UI settings

3. Check firewall/network settings

### GPU Not Detected

```bash
# Verify GPU availability
nvidia-smi

# Run with explicit GPU
CUDA_VISIBLE_DEVICES=0 poetry run python dual_modal_gan/scripts/gradio_integrated_htr.py
```

### Memory Issues

- Reduce input image size
- Use CPU instead of GPU
- Close other applications

## Comparison with Standalone Apps

| Feature | gradio_app.py | gradio_integrated_htr.py |
|---------|---------------|--------------------------|
| Document Restoration | ✓ | ✓ |
| TIFF Download | ✓ | ✓ |
| HTR Integration | ✗ | ✓ |
| Loghi Compatible | ✗ | ✓ |
| Standalone | ✓ | ✓ (with Loghi optional) |

## Notes

- HTR functionality is **optional** - restoration works without Loghi service
- For production deployment, consider containerizing both services
- Ensure sufficient disk space for temporary files
- TIFF files are automatically cleaned up after download
