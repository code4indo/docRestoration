# Quick Start Guide - Integrated System

## 🚀 Quick Start (Restoration Only)

```bash
cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration
./launch_integrated_system.sh
```

Then open: `http://localhost:7860`

## 🔧 Advanced Options

### Use Specific GPU
```bash
./launch_integrated_system.sh --gpu 1
```

### Use CPU Only
```bash
./launch_integrated_system.sh --cpu
```

### Custom Port
```bash
./launch_integrated_system.sh --port 8080
```

## 📋 HTR Integration (Full Pipeline)

### Step 1: Start Loghi HTR Service
```bash
# In separate terminal
cd /home/lambda_one/tesis/loghi
# Run your Loghi HTR service command here
```

### Step 2: Launch Integrated App
```bash
cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration
./launch_integrated_system.sh
```

### Step 3: Use in Browser
1. Upload image
2. ✅ Check "Enable HTR"
3. Verify HTR address (default: http://localhost:5001)
4. Click "Process Document"

## 🎯 Common Use Cases

### Case 1: Just Restore Document
- Leave "Enable HTR" unchecked
- Click process
- Download TIFF

### Case 2: Restore + Extract Text
- Check "Enable HTR"
- Click process
- Get both restored TIFF and extracted text

## 📊 Comparison: Simple vs Integrated

| App | File | HTR | Best For |
|-----|------|-----|----------|
| Simple | `gradio_app.py` | ❌ | Just restoration |
| Integrated | `gradio_integrated_htr.py` | ✅ | Full pipeline |

### Run Simple App (without HTR):
```bash
CUDA_VISIBLE_DEVICES=0 poetry run python dual_modal_gan/scripts/gradio_app.py
```

### Run Integrated App (with HTR option):
```bash
./launch_integrated_system.sh
```

## 🔍 Troubleshooting

### Problem: "Cannot connect to HTR service"
**Solution:** 
- Check Loghi is running: `curl http://localhost:5001`
- Or uncheck "Enable HTR" to skip HTR

### Problem: "Checkpoint not found"
**Solution:**
- Verify: `ls dual_modal_gan/checkpoints/production_full_coverage_vgg_v1/ckpt-94*`

### Problem: Slow processing
**Solution:**
- Use GPU: `./launch_integrated_system.sh --gpu 0`
- Reduce image size before upload

## 📝 Example Workflow

```bash
# Terminal 1: Start Loghi HTR (optional)
cd /home/lambda_one/tesis/loghi
# ... start loghi service ...

# Terminal 2: Start Integrated App
cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration
./launch_integrated_system.sh --gpu 0

# Browser:
# 1. Open http://localhost:7860
# 2. Upload degraded document
# 3. Enable HTR if Loghi is running
# 4. Process and download results
```

## 🎛️ Parameter Recommendations

### For Clean Documents with Light Degradation
```
Alpha Blending: 0.0
Gamma: 1.0
Post-processing: On
Aggressive: Off
Thin Strokes: Off
```

### For Heavy Degradation with Broken Strokes
```
Alpha Blending: 0.0
Gamma: 1.2 (brighter)
Post-processing: On
Aggressive: On ← Fix breaks
Thin Strokes: Off
```

### For Documents with Thick Strokes
```
Alpha Blending: 0.0
Gamma: 1.0
Post-processing: On
Aggressive: Off
Thin Strokes: On ← Reduce thickness
```

## 📦 Output Files

- **Restored Preview**: In-browser (zoom available)
- **TIFF Download**: High quality (300 DPI, LZW)
- **HTR Text**: Copy from text field
- **Confidence**: Numerical score

## 🧪 Test Without HTR

If Loghi is not available, system still works:
1. Leave "Enable HTR" unchecked
2. Use for restoration only
3. All restoration features work normally
