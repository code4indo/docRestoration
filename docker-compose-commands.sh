#!/bin/bash
# Quick Commands for Docker Compose GAN-HTR

# ===================================
# 🚀 START TRAINING
# ===================================

# Production mode (auto-download + train)
docker-compose up -d gan-htr-prod

# Smoke test (quick 2-epoch validation)
docker-compose up -d gan-htr-smoke-test

# Development mode (interactive shell)
docker-compose up -d gan-htr-dev

# Test mode (validate installation)
docker-compose up gan-htr-test

# Custom script via environment variable
TRAINING_SCRIPT=scripts/train32_smoke_test.sh docker-compose up -d gan-htr-prod

# Custom script via .env file
echo "TRAINING_SCRIPT=scripts/train32_smoke_test.sh" > .env
docker-compose up -d gan-htr-prod

# ===================================
# 📊 MONITORING
# ===================================

# View logs (real-time)
docker-compose logs -f gan-htr-prod

# View logs (last 100 lines)
docker logs gan-htr-prod --tail 100

# Check container status
docker-compose ps

# Check GPU usage inside container
docker exec -it gan-htr-prod nvidia-smi

# ===================================
# 🛑 STOP TRAINING
# ===================================

# Graceful stop (save checkpoint)
docker-compose stop gan-htr-prod

# Force stop
docker-compose kill gan-htr-prod

# Stop all services
docker-compose down

# ===================================
# 🔄 RESTART & RESUME
# ===================================

# Restart container (resume training)
docker-compose restart gan-htr-prod

# Start stopped container
docker-compose start gan-htr-prod

# Recreate container (fresh start, keep volumes)
docker-compose up -d --force-recreate gan-htr-prod

# ===================================
# 🐛 DEBUGGING
# ===================================

# Enter container shell
docker exec -it gan-htr-prod bash

# Check Python environment
docker exec -it gan-htr-prod python3 -c "import tensorflow as tf; print(tf.__version__)"

# Check GPU detection
docker exec -it gan-htr-prod python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Inspect volumes
docker volume ls
docker volume inspect docrestoration_gan_data

# ===================================
# 🧹 CLEANUP
# ===================================

# Remove containers only (keep volumes)
docker-compose down

# Remove containers + volumes (DELETE ALL DATA!)
docker-compose down -v

# Remove specific volume
docker volume rm docrestoration_gan_data

# Clean up Docker system
docker system prune -a

# ===================================
# 📦 DATA MANAGEMENT
# ===================================

# Check data in volumes
docker exec -it gan-htr-prod ls -lh /workspace/dual_modal_gan/data/
docker exec -it gan-htr-prod ls -lh /workspace/models/best_htr_recognizer/

# Check disk usage
docker exec -it gan-htr-prod df -h

# Copy files from container to host
docker cp gan-htr-prod:/workspace/outputs/checkpoints_xxx ./local_backup/

# Copy files from host to container
docker cp ./local_file gan-htr-prod:/workspace/some_path/

# ===================================
# 🔧 REBUILD IMAGE
# ===================================

# Pull latest image
docker-compose pull

# Rebuild local image (if Dockerfile changed)
docker-compose build --no-cache

# Force recreate with new image
docker-compose up -d --force-recreate --build

# ===================================
# 📈 MLflow UI
# ===================================

# Access MLflow (dari browser)
# http://<server-ip>:5001

# Or run MLflow UI manually
docker exec -it gan-htr-prod poetry run mlflow ui --host 0.0.0.0 --port 5000

# ===================================
# 🆘 EMERGENCY
# ===================================

# Kill all containers
docker kill $(docker ps -q)

# Remove all stopped containers
docker container prune

# Check logs for errors
docker-compose logs gan-htr-prod | grep -E "Error|Exception|Traceback"

# Check resource usage
docker stats gan-htr-prod
