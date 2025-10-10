# 🎯 Docker Compose - Cloud Deployment Summary

## ✅ Perubahan yang Dilakukan

### 1. **Volume Structure** (UTAMA)
**SEBELUM** (Local Development):
```yaml
volumes:
  - ../network:/workspace/network           # ❌ Dependency parent directory
  - ../models:/workspace/models             # ❌ Tidak ada di cloud
  - ../real_data_preparation:/workspace/... # ❌ Tidak ada di cloud
```

**SESUDAH** (Cloud Ready):
```yaml
volumes:
  - .:/workspace/docRestoration            # ✅ Only docRestoration folder needed
  - gan_data:/workspace/dual_modal_gan/data         # ✅ Named volume
  - htr_models:/workspace/models                     # ✅ Named volume
  - charlist_data:/workspace/real_data_preparation   # ✅ Named volume
  - huggingface_cache:/root/.cache/huggingface      # ✅ Cache persistent
```

**Keuntungan:**
- ✅ Tidak perlu folder parent (`../network`, `../models`, dll)
- ✅ Data persistent meskipun container dihapus
- ✅ Auto-download dari HuggingFace saat pertama kali
- ✅ Tidak perlu re-download jika recreate container

---

### 2. **Auto-Download Script**
File: `docRestoration/scripts/download_from_huggingface.py`

**Fitur:**
- ✅ Support environment variables (`HF_USERNAME`, `HF_REPO_NAME`)
- ✅ Download ke path yang benar (`/workspace/...`)
- ✅ Check file existence sebelum download
- ✅ Fast download dengan `HF_HUB_ENABLE_HF_TRANSFER`

**Files yang di-download:**
1. `dataset_gan.tfrecord` → `/workspace/dual_modal_gan/data/`
2. `best_model.weights.h5` → `/workspace/models/best_htr_recognizer/`
3. `real_data_charlist.txt` → `/workspace/real_data_preparation/`

---

### 3. **Entrypoint Logic**
File: `docRestoration/entrypoint.sh`

**Perubahan:**
- ✅ Check file existence (bukan directory)
- ✅ Call download script dengan path yang benar
- ✅ Validasi file setelah download
- ✅ Exit dengan error jika download gagal

**Flow:**
```
1. Setup environment (mkdir, chmod, etc)
2. Check if data files exist
3. If not exist → Download from HuggingFace
4. Validate installation (Python, TF, GPU)
5. Start training
```

---

### 4. **Environment Variables**
**Ditambahkan:**
```yaml
environment:
  - HF_USERNAME=jatnikonm      # Username HuggingFace
  - HF_REPO_NAME=HTR_VOC       # Nama repository dataset
  - MODE=production             # Mode: production/development/test
```

---

## 🚀 Cara Pakai di Cloud

### Skenario 1: Fresh Deployment (Pertama Kali)
```bash
# 1. Clone repo di server cloud
git clone https://github.com/knaw-huc/loghi-htr.git
cd loghi-htr/docRestoration

# 2. Start training (auto-download dataset)
docker-compose up -d gan-htr-prod

# 3. Monitor logs
docker-compose logs -f gan-htr-prod
```

**Yang Terjadi:**
1. Container start
2. Entrypoint check: data files tidak ada
3. Auto-download dari HuggingFace (~2-5GB, 5-30 menit)
4. Validasi installation
5. Training dimulai
6. Data disimpan di named volumes (persistent)

---

### Skenario 2: Resume Training
```bash
# Container stopped, tapi volumes masih ada
docker-compose up -d gan-htr-prod

# Training auto-resume dari checkpoint
```

**Yang Terjadi:**
1. Container start
2. Entrypoint check: data files sudah ada (di volume)
3. Skip download (cepat!)
4. Training resume dari checkpoint terakhir

---

### Skenario 3: Recreate Container
```bash
# Remove container tapi keep volumes
docker-compose down

# Recreate
docker-compose up -d gan-htr-prod
```

**Yang Terjadi:**
1. Container baru dibuat
2. Mount volumes yang sudah ada (data masih ada!)
3. Skip download karena files sudah exist
4. Training resume

---

### Skenario 4: Fresh Training (Hapus Data)
```bash
# Stop container
docker-compose down

# Hapus volumes
docker volume rm docrestoration_gan_data
docker volume rm docrestoration_htr_models
docker volume rm docrestoration_charlist_data

# Fresh start
docker-compose up -d gan-htr-prod
```

**Yang Terjadi:**
1. Container start dengan empty volumes
2. Auto-download ulang dari HuggingFace
3. Training dari scratch (epoch 1)

---

## 📦 Data Persistence

### Volume Mapping:
| Volume Name | Path di Container | Content | Persistent? |
|-------------|-------------------|---------|-------------|
| `gan_data` | `/workspace/dual_modal_gan/data/` | Dataset TFRecord | ✅ Yes |
| `htr_models` | `/workspace/models/` | Model weights | ✅ Yes |
| `charlist_data` | `/workspace/real_data_preparation/` | Character list | ✅ Yes |
| `huggingface_cache` | `/root/.cache/huggingface/` | HF cache | ✅ Yes |
| `./outputs` | `/workspace/outputs/` | Checkpoints | ✅ Yes (bind mount) |
| `./logbook` | `/workspace/logbook/` | Logs | ✅ Yes (bind mount) |

### Keuntungan Named Volumes:
1. **Persistent** - Data tetap ada meskipun container dihapus
2. **Performance** - Faster I/O dibanding bind mounts
3. **Portability** - Bisa di-backup dan restore dengan `docker volume`
4. **Isolation** - Tidak depend on host filesystem structure

---

## 🔧 Customization

### Ganti HuggingFace Repository:
```yaml
environment:
  - HF_USERNAME=your_username
  - HF_REPO_NAME=your_repo
```

### Ganti GPU:
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # Multiple GPUs
```

### Ganti Port:
```yaml
ports:
  - "5001:5000"  # MLflow
  - "6007:6006"  # TensorBoard
  - "8889:8888"  # Jupyter
```

---

## 🆘 Troubleshooting

### 1. Download Failed
```bash
# Check logs
docker logs gan-htr-prod | grep -i error

# Common issues:
# - Network issue → Retry: docker-compose restart gan-htr-prod
# - Wrong credentials → Check HF_USERNAME, HF_REPO_NAME
# - Private repo → Need HF token: docker exec -it gan-htr-prod huggingface-cli login
```

### 2. Container Restart Loop
```bash
# Check what's failing
docker logs gan-htr-prod --tail 100

# Validate entrypoint
docker exec -it gan-htr-prod bash
ls -la /workspace/docRestoration/entrypoint.sh
```

### 3. Out of Memory
```bash
# Check disk space
docker exec -it gan-htr-prod df -h

# Check GPU memory
docker exec -it gan-htr-prod nvidia-smi

# Solution: Reduce batch size or use smaller model
```

---

## 📊 Monitoring

### Logs:
```bash
# Real-time
docker-compose logs -f gan-htr-prod

# Last 100 lines
docker logs gan-htr-prod --tail 100

# Search errors
docker logs gan-htr-prod | grep -E "Error|Exception"
```

### MLflow:
```bash
# Access from browser
http://<server-ip>:5001
```

### GPU:
```bash
# From host
nvidia-smi

# From container
docker exec -it gan-htr-prod nvidia-smi
```

---

## ✅ Checklist Deployment

- [ ] Server cloud ready (Docker + GPU drivers)
- [ ] Repository cloned
- [ ] Docker Compose config reviewed
- [ ] Environment variables set (jika perlu custom)
- [ ] Storage cukup (minimal 50GB free)
- [ ] Port tidak conflict
- [ ] GPU terdeteksi (`nvidia-smi`)
- [ ] Start container: `docker-compose up -d gan-htr-prod`
- [ ] Monitor logs: `docker-compose logs -f gan-htr-prod`
- [ ] Wait for download complete (~5-30 menit first time)
- [ ] Training started

---

## 🎯 Summary

**SEBELUM:** Butuh setup manual, download manual, struktur folder complex
**SESUDAH:** One-command deployment, auto-download, clean structure

**Key Benefits:**
1. ✅ Zero manual setup
2. ✅ Auto-download from HuggingFace
3. ✅ Persistent data (no re-download)
4. ✅ Cloud-ready (no parent directory dependency)
5. ✅ Easy resume training
6. ✅ Production-grade logging & monitoring

**Perfect for:**
- RunPod deployment
- AWS EC2 GPU instances
- GCP Compute Engine
- Azure VM
- Any Docker + GPU cloud provider
