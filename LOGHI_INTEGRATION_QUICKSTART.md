# Quick Start: Integrated Document Restoration + Loghi HTR System

## 🚀 Quick Start Guide

### Prerequisites
✅ All Loghi Docker services must be running
✅ Python virtual environment activated
✅ Model checkpoints available

### Step 1: Start Loghi Services

```bash
cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration
./scripts/start_loghi_services.sh
```

This will start:
- **LAYPA** (port 5000) - Layout analysis
- **HTR** (port 5001) - Text recognition  
- **TOOLING** (port 8082) - XML processing

### Step 2: Check Services Status

```bash
./scripts/check_loghi_services.sh
```

Expected output:
```
✓ LAYPA (port 5000): Running
✓ HTR (port 5001): Running
✓ TOOLING (port 8082): Running
```

### Step 3: Launch Integrated UI

```bash
./launch_integrated_system.sh
```

Or use direct command:
```bash
poetry run python dual_modal_gan/scripts/gradio_integrated_htr_enhanced.py
```

### Step 4: Access Web Interface

Open browser: **http://localhost:7861**

## 🎯 Usage Modes

### Simple Mode (Fast)
- Direct HTR on full restored image
- Quick text extraction
- **Use case**: Quick OCR, demo, testing

### Full Pipeline Mode (Detailed)
- Complete Loghi pipeline
- Line-by-line segmentation
- Individual line HTR with confidence scores
- **Use case**: Research, detailed analysis, publication

## 📊 Features

### 1. Document Restoration
- **Model**: Dual-Modal GAN with Discriminator Enhanced V2
- **Output**: High-quality TIFF (300 DPI)
- **Customizable**: Alpha blending, gamma correction, post-processing

### 2. HTR Integration
- **LAYPA**: Automatic line detection and baseline extraction
- **HTR**: State-of-the-art handwriting recognition (Loghi)
- **Export**: PageXML format with coordinates

### 3. Results
- Full document text
- Per-line confidence scores
- Segmentation visualization
- Downloadable TIFF and XML

## 🛠️ Management Commands

### Start Services
```bash
./scripts/start_loghi_services.sh
```

### Stop Services
```bash
./scripts/stop_loghi_services.sh
```

### Check Status
```bash
./scripts/check_loghi_services.sh
```

### View Logs
```bash
cd scripts/loghi_services
docker compose logs -f
```

### Restart Individual Service
```bash
cd scripts/loghi_services
docker compose restart htr     # Restart HTR only
docker compose restart laypa   # Note: LAYPA runs externally
docker compose restart loghi-tooling
```

## 🐛 Troubleshooting

### HTR Service Not Responding
```bash
# Check logs
docker logs gan-htr-htr --tail 50

# Restart
cd scripts/loghi_services
docker compose restart htr
```

### Port Already in Use
```bash
# Check what's using the port
netstat -tlnp | grep 5001

# Or use different port by editing docker-compose.yml
```

### Out of Memory
```bash
# Reduce batch size in .env
LOGHI_BATCH_SIZE: 32  # Default: 64
```

## 📁 Important Files

### Configuration
- `scripts/loghi_services/.env` - Service configuration
- `scripts/loghi_services/docker-compose.yml` - Docker setup

### Scripts
- `launch_integrated_system.sh` - Main launcher
- `scripts/start_loghi_services.sh` - Start Loghi
- `scripts/stop_loghi_services.sh` - Stop Loghi
- `scripts/check_loghi_services.sh` - Check status

### Application
- `dual_modal_gan/scripts/gradio_integrated_htr_enhanced.py` - Main UI (Enhanced)
- `dual_modal_gan/scripts/gradio_integrated_htr.py` - Simple version

## 🎓 Research Integration

### For Publications
Use **Full Pipeline Mode** to get:
- PageXML output with line coordinates
- Per-line confidence metrics
- Baseline detection results
- Complete processing pipeline documentation

### For Benchmarking
- Compare restoration quality (PSNR, SSIM)
- Measure HTR accuracy (CER, WER)
- Track per-line confidence scores
- Export results to CSV/JSON

## 🔗 Service Endpoints

### For API Access
```python
# LAYPA Segmentation
POST http://localhost:5000/predict
Files: image (multipart/form-data)
Data: model=general/baseline, identifier=unique_id

# HTR Recognition
POST http://localhost:5001/predict
Files: image (multipart/form-data)

# TOOLING Baseline Extraction
POST http://localhost:8082/api/extract_baselines
```

## 📞 Support

For issues or questions:
1. Check logs: `docker compose logs`
2. Verify services: `./scripts/check_loghi_services.sh`
3. Review this guide

## 🎉 Ready to Use!

System is now fully integrated and ready for:
- ✅ Document restoration
- ✅ Handwritten text recognition
- ✅ Research and analysis
- ✅ Production deployment

Start processing your historical documents! 📜✨
