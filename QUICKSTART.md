# 🚀 Quick Start Guide - Docker Training

## Start Training (Local)

```bash
cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration

# === CPU Mode (Default - Slow but works on any server) ===
docker-compose up -d gan-htr-prod

# === GPU Mode (Fast - Requires NVIDIA driver) ===
docker-compose --profile gpu up -d gan-htr-prod-gpu

# Monitor logs
docker logs -f gan-htr-prod

# Monitor GPU usage (GPU mode only)
watch -n 1 nvidia-smi
```

## Manual Training Start

Container starts in idle mode. Start training manually:

```bash
# Smoke test (5 epochs - 10-15 menit)
docker exec -d gan-htr-prod bash /workspace/docRestoration/scripts/train32_smoke_test.sh

# Production (100 epochs - 3-5 jam)
docker exec -d gan-htr-prod bash /workspace/docRestoration/scripts/train32_production.sh

# Monitor progress
docker logs -f gan-htr-prod
tail -f logbook/training_*.log
```

## Stop Training

```bash
# Stop container
docker-compose down

# Stop dan remove volumes (HATI-HATI: data hilang!)
docker-compose down -v
```

## View Results

```bash
# Training outputs
ls -lh outputs/checkpoints_fp32_production/
ls -lh outputs/samples_fp32_production/

# Training logs
tail -f logbook/*.md

# MLflow UI (start service first)
docker-compose up -d mlflow
# Browse: http://localhost:5000
```

## Enable GPU (MUST untuk production)

```bash
# Install nvidia runtime (run once di host)
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU
docker run --rm --gpus all tensorflow/tensorflow:2.18.0-gpu \
  python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"

# Update docker-compose.yml
# Uncomment line: runtime: nvidia

# Restart training
docker-compose restart gan-htr-prod
```

## Troubleshooting

### Container restart loop
```bash
# Check logs untuk error
docker logs gan-htr-prod | tail -50

# Common issues:
# - Data not downloaded: Check HF_TOKEN
# - Path error: Verify volumes mounted
# - Syntax error: Check shell script syntax
```

### Training lambat (CPU mode)
```bash
# Check GPU detection
docker exec gan-htr-prod python3 -c "import tensorflow as tf; print('GPU:', len(tf.config.list_physical_devices('GPU')))"

# Expected: GPU: 1 (atau lebih)
# Actual: GPU: 0 → Enable nvidia runtime!
```

### Out of memory
```bash
# Reduce batch size di training script
# Edit: docRestoration/scripts/train32_production.sh
# Change: --batch_size 4  →  --batch_size 2
```

## Cloud Deployment (RunPod)

```bash
# 1. Push image
docker push jatnikonm/gan-htr:latest

# 2. Create pod di RunPod.io
# - Select GPU: RTX 4090 / A100
# - Container image: jatnikonm/gan-htr:latest
# - Volume: 50GB persistent
# - Environment:
#     HF_TOKEN=<token>
#     TRAINING_SCRIPT=scripts/train32_production.sh
#     MODE=production

# 3. SSH ke pod
ssh -p <port> root@<pod-ip>

# 4. Monitor
cd /workspace/docRestoration
tail -f logbook/*.md
docker logs -f gan-htr-prod
```

## Key Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Container orchestration |
| `scripts/train32_production.sh` | Training script (100 epochs) |
| `scripts/train32_smoke_test.sh` | Quick test (2 epochs) |
| `entrypoint.sh` | Container initialization |
| `outputs/` | Training results |
| `logbook/` | Training logs & reports |

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TRAINING_SCRIPT` | `scripts/train32_production.sh` | Script to run |
| `MODE` | `production` | Operating mode |
| `HF_TOKEN` | (none) | HuggingFace access token |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device ID |
| `PYTHON_BIN` | `python3` | Python executable |

## Important Paths

```
/workspace/
├── docRestoration/          # Code (bind mount)
│   ├── dual_modal_gan/
│   ├── scripts/
│   ├── outputs/            → Training results
│   └── logbook/            → Logs
├── dual_modal_gan/data/    # Dataset (volume)
├── models/                 # Pretrained models (volume)
└── real_data_preparation/  # Charlist (volume)
```
