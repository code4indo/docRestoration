# 🚀 GAN-HTR Cloud Deployment - Quick Start

> Docker Compose untuk training GAN-HTR di cloud provider dengan auto-download dataset dari HuggingFace

## ⚡ Quick Start (3 Steps)

```bash
# 1. Clone repository
git clone https://github.com/knaw-huc/loghi-htr.git
cd loghi-htr/docRestoration

# 2. Verify setup
./verify_cloud_deployment.sh

# 3. Start training
docker-compose up -d gan-htr-prod
```

## 📊 Monitor Training

```bash
# View logs (real-time)
docker-compose logs -f gan-htr-prod

# Check status
docker-compose ps

# View MLflow
# Browser: http://<server-ip>:5001
```

## 🛑 Stop Training

```bash
# Graceful stop (save checkpoint)
docker-compose stop gan-htr-prod

# Resume
docker-compose start gan-htr-prod
```

---

## 📚 Documentation

| File | Description |
|------|-------------|
| **CLOUD_DEPLOYMENT.md** | Panduan lengkap deployment cloud |
| **DOCKER_COMPOSE_CLOUD_SUMMARY.md** | Arsitektur dan perubahan detail |
| **docker-compose-commands.sh** | Reference command Docker Compose |
| **verify_cloud_deployment.sh** | Script validasi pre-deployment |

---

## 🎯 Features

- ✅ **Auto-download** dataset dari HuggingFace (pertama kali)
- ✅ **Zero manual setup** - tinggal run
- ✅ **Persistent volumes** - data tidak hilang meskipun container dihapus
- ✅ **Resume training** - otomatis dari checkpoint terakhir
- ✅ **GPU support** - TensorFlow + CUDA
- ✅ **MLflow tracking** - monitoring metrics real-time

---

## 🔧 Configuration

Edit `docker-compose.yml` untuk customize:

```yaml
environment:
  - HF_USERNAME=jatnikonm      # Your HuggingFace username
  - HF_REPO_NAME=HTR_VOC       # Dataset repository name
  - CUDA_VISIBLE_DEVICES=0     # GPU devices (0, 0,1, dll)
```

---

## 📦 Volume Structure

```
Named Volumes (Persistent):
├── gan_data/           # Dataset TFRecord (~2-5GB)
├── htr_models/         # Model weights
├── charlist_data/      # Character list
└── huggingface_cache/  # HF cache (faster re-download)

Bind Mounts:
├── ./outputs/          # Training checkpoints
├── ./logbook/          # Experiment logs
└── ./mlruns/           # MLflow tracking data
```

---

## 🆘 Troubleshooting

### Container restart loop?
```bash
docker logs gan-htr-prod --tail 50
```

### Download failed?
```bash
# Check HF credentials in docker-compose.yml
# Retry: docker-compose restart gan-htr-prod
```

### Port conflict?
```bash
# Edit ports in docker-compose.yml
ports:
  - "5001:5000"  # Change 5001 to other port
```

---

## ✅ Deployment Checklist

- [ ] Docker + Docker Compose installed
- [ ] GPU drivers installed (nvidia-docker)
- [ ] Run verification: `./verify_cloud_deployment.sh`
- [ ] Review config: `docker-compose config`
- [ ] Start training: `docker-compose up -d gan-htr-prod`
- [ ] Monitor logs: `docker-compose logs -f gan-htr-prod`

---

## 🎓 Architecture

**Cloud-Ready Design:**
1. No parent directory dependency (`../network`, `../models` removed)
2. Named volumes for persistent data
3. Auto-download from HuggingFace on first run
4. Smart caching - no re-download if container recreated
5. Production-grade logging and monitoring

**Perfect for:** RunPod, AWS EC2, GCP, Azure, any Docker + GPU cloud

---

## 📞 Support

- Full docs: `CLOUD_DEPLOYMENT.md`
- Commands: `docker-compose-commands.sh`
- Verify: `./verify_cloud_deployment.sh`

**Ready to train!** 🚀
