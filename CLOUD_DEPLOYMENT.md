# Cloud Deployment Guide - GAN-HTR Training

## 📋 Overview

Docker Compose ini dirancang untuk deployment di cloud provider (RunPod, AWS, GCP, Azure, dll) dengan fitur:
- ✅ **Auto-download dataset** dari HuggingFace
- ✅ **Zero manual setup** - tinggal run
- ✅ **Persistent volumes** untuk checkpoint dan logs
- ✅ **GPU support** dengan TensorFlow
- ✅ **MLflow tracking** untuk monitoring

---

## 🚀 Quick Start (Cloud Provider)

### 1. Clone Repository di Server Cloud
```bash
git clone https://github.com/knaw-huc/loghi-htr.git
cd loghi-htr/docRestoration
```

### 2. (Optional) Konfigurasi Environment
```bash
# Edit docker-compose.yml jika perlu custom config
nano docker-compose.yml

# Ubah environment variables:
# - HF_USERNAME: username HuggingFace Anda
# - HF_REPO_NAME: nama repository dataset
# - CUDA_VISIBLE_DEVICES: GPU yang digunakan (0, 0,1, dll)
```

### 3. Start Training
```bash
# Production mode (auto-download + train)
docker-compose up -d gan-htr-prod

# Monitor logs
docker-compose logs -f gan-htr-prod
```

### 4. Stop Training
```bash
# Graceful stop (save checkpoint)
docker-compose stop gan-htr-prod

# Remove container (data tetap tersimpan di volume)
docker-compose down
```

---

## 📦 Volume Structure

```
docRestoration/
├── outputs/              # Training outputs (mounted dari host)
├── logbook/             # Experiment logs (mounted dari host)
├── mlruns/              # MLflow tracking (mounted dari host)
└── [Docker Volumes]
    ├── gan_data/        # Dataset dari HuggingFace (persistent)
    ├── htr_models/      # Model weights dari HuggingFace (persistent)
    ├── charlist_data/   # Character list dari HuggingFace (persistent)
    ├── huggingface_cache/ # HF cache (persistent)
    └── poetry_cache/    # Poetry dependencies (persistent)
```

**Keuntungan Named Volumes:**
- ✅ Data persisten meskipun container dihapus
- ✅ Tidak perlu re-download jika container di-recreate
- ✅ Lebih cepat I/O dibanding bind mounts

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODE` | `production` | Mode: `production`, `development`, `test` |
| `HF_USERNAME` | `jatnikonm` | Username HuggingFace |
| `HF_REPO_NAME` | `HTR_VOC` | Nama repository dataset |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU devices (0, 0,1, dll) |
| `TF_FORCE_GPU_ALLOW_GROWTH` | `true` | Allow GPU memory growth |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Enable fast download |

---

## 📊 Monitoring & Logs

### View Logs Real-time
```bash
docker-compose logs -f gan-htr-prod
```

### Check Training Progress
```bash
# Lihat logs terakhir
docker logs gan-htr-prod --tail 100

# Check MLflow UI (dari host)
# Akses: http://<server-ip>:5001
```

### Check GPU Usage
```bash
# Masuk ke container
docker exec -it gan-htr-prod bash

# Check GPU
nvidia-smi

# Check TensorFlow GPU
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 🔄 Resume Training

Training otomatis resume dari checkpoint terakhir:

```bash
# Start container (akan auto-resume)
docker-compose up -d gan-htr-prod

# Logs akan menunjukkan:
# "Checkpoint loaded from: /workspace/outputs/checkpoints_xxx/ckpt-xxx"
```

---

## 🧹 Clean Up & Re-train

### Hapus Checkpoint (Fresh Training)
```bash
# Stop container
docker-compose stop gan-htr-prod

# Hapus outputs
rm -rf outputs/checkpoints_*

# Start fresh training
docker-compose up -d gan-htr-prod
```

### Hapus Semua Data (Termasuk Download)
```bash
# Stop & remove containers
docker-compose down

# Hapus volumes
docker volume rm docrestoration_gan_data
docker volume rm docrestoration_htr_models
docker volume rm docrestoration_charlist_data

# Hapus HF cache (optional)
docker volume rm docrestoration_huggingface_cache

# Fresh start
docker-compose up -d gan-htr-prod
```

---

## 🐛 Troubleshooting

### 1. Container Restart Loop
```bash
# Check logs
docker logs gan-htr-prod --tail 50

# Common issues:
# - Download failed → Check HF credentials
# - OOM → Reduce batch size di config
# - GPU error → Check CUDA_VISIBLE_DEVICES
```

### 2. Download Failed
```bash
# Check HF credentials
docker exec -it gan-htr-prod bash
huggingface-cli whoami

# Login (jika private repo)
docker exec -it gan-htr-prod bash
huggingface-cli login
```

### 3. Out of Memory (OOM)
```bash
# Edit train config
nano dual_modal_gan/scripts/train.py

# Kurangi batch_size atau ubah args
```

### 4. Port Already in Use
```bash
# Edit docker-compose.yml
# Ubah port mapping:
ports:
  - "5001:5000"  # MLflow
  - "6007:6006"  # TensorBoard
  - "8889:8888"  # Jupyter
```

---

## 📈 Production Checklist

- [ ] Repository sudah di clone
- [ ] Docker & Docker Compose terinstall
- [ ] GPU drivers terinstall (nvidia-docker)
- [ ] Environment variables sudah dikonfigurasi
- [ ] Port tidak conflict dengan service lain
- [ ] Storage cukup untuk dataset (minimal 50GB)
- [ ] HuggingFace credentials configured (jika private)

---

## 🎯 Cloud Provider Specific

### RunPod
```bash
# Setup sudah include Docker + NVIDIA drivers
docker-compose up -d gan-htr-prod
```

### AWS EC2 (GPU)
```bash
# Install Docker
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Start training
docker-compose up -d gan-htr-prod
```

### GCP Compute Engine
```bash
# Similar dengan AWS, install Docker + NVIDIA drivers
# Atau gunakan Deep Learning VM Image (sudah pre-configured)
```

---

## 📝 Notes

1. **First Run**: Download dataset ~2-5GB, bisa memakan waktu 5-30 menit tergantung bandwidth
2. **Checkpoint**: Disimpan setiap N steps (configurable), bisa resume anytime
3. **MLflow**: Tracking metrics, accessible via browser di port 5001
4. **Logs**: Semua logs tersimpan di `logbook/` dengan timestamp

---

## 🆘 Support

Jika ada masalah:
1. Check logs: `docker logs gan-htr-prod --tail 100`
2. Check volumes: `docker volume ls`
3. Check GPU: `nvidia-smi`
4. Check disk space: `df -h`
