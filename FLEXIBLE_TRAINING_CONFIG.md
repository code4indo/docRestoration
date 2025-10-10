# 🎯 Flexible Training Configuration Guide

## Overview

Docker Compose sekarang mendukung **konfigurasi training fleksibel** dengan multiple cara menjalankan training.

---

## 🚀 Option 1: Pre-configured Services

### 1. Production Training (Full Training)
```bash
docker-compose up -d gan-htr-prod
```
- Script: `dual_modal_gan/scripts/train.py`
- Mode: Full training dengan checkpoint resume
- Restart: Auto-restart on failure

### 2. Smoke Test (Quick Validation)
```bash
docker-compose up -d gan-htr-smoke-test
```
- Script: `scripts/train32_smoke_test.sh`
- Mode: 2 epochs untuk validasi cepat
- Restart: No auto-restart (one-time run)

### 3. Development Mode (Interactive)
```bash
docker-compose up -d gan-htr-dev
```
- Mode: Interactive shell dengan Poetry
- Use case: Debugging, testing, custom commands

### 4. Test Mode (Validation Only)
```bash
docker-compose up gan-htr-test
```
- Mode: Validate installation (TensorFlow, GPU, etc)
- No training, just validation

---

## ⚙️ Option 2: Environment Variable Override

### Method A: Inline Environment Variable
```bash
# Run smoke test menggunakan production service
TRAINING_SCRIPT=scripts/train32_smoke_test.sh docker-compose up -d gan-htr-prod

# Run custom script
TRAINING_SCRIPT=scripts/my_custom_training.sh docker-compose up -d gan-htr-prod
```

### Method B: .env File
```bash
# 1. Create .env file
cp .env.example .env

# 2. Edit .env
nano .env

# Set your training script:
TRAINING_SCRIPT=scripts/train32_smoke_test.sh

# 3. Run
docker-compose up -d gan-htr-prod
```

### Method C: Docker Compose Override
```bash
# Edit docker-compose.yml temporarily or create docker-compose.override.yml
services:
  gan-htr-prod:
    environment:
      - TRAINING_SCRIPT=scripts/train32_smoke_test.sh
```

---

## 🎨 Option 3: Custom Training Scripts

### Step 1: Create Your Script
```bash
# Create new training script
cat > docRestoration/scripts/my_training.sh <<'EOF'
#!/bin/bash
echo "🚀 My Custom Training"

poetry run python dual_modal_gan/scripts/train32.py \
  --tfrecord_path dual_modal_gan/data/dataset_gan.tfrecord \
  --charset_path real_data_preparation/real_data_charlist.txt \
  --recognizer_weights models/best_htr_recognizer/best_model.weights.h5 \
  --checkpoint_dir dual_modal_gan/outputs/my_custom_checkpoint \
  --epochs 100 \
  --batch_size 8 \
  --gpu_id 0
EOF

chmod +x docRestoration/scripts/my_training.sh
```

### Step 2: Run with Docker
```bash
# Method 1: Environment variable
TRAINING_SCRIPT=scripts/my_training.sh docker-compose up -d gan-htr-prod

# Method 2: .env file
echo "TRAINING_SCRIPT=scripts/my_training.sh" > .env
docker-compose up -d gan-htr-prod

# Method 3: Direct command override
docker-compose run --rm gan-htr-prod bash /workspace/docRestoration/scripts/my_training.sh
```

---

## 🔧 Option 4: Direct Command Override

### Execute Custom Commands
```bash
# Run Python script directly
docker-compose run --rm gan-htr-prod python3 /workspace/docRestoration/dual_modal_gan/scripts/train.py

# Run shell script
docker-compose run --rm gan-htr-prod bash /workspace/docRestoration/scripts/train32_smoke_test.sh

# Interactive shell for debugging
docker-compose run --rm gan-htr-prod bash
```

---

## 📊 Available Training Scripts

| Script | Path | Purpose |
|--------|------|---------|
| **train.py** | `dual_modal_gan/scripts/train.py` | Full training (mixed precision) |
| **train32.py** | `dual_modal_gan/scripts/train32.py` | Pure FP32 training |
| **train32_smoke_test.sh** | `scripts/train32_smoke_test.sh` | Smoke test (2 epochs) |
| **train32_production.sh** | `scripts/train32_production.sh` | Production FP32 |

---

## 🎯 Common Use Cases

### Use Case 1: Quick Smoke Test
```bash
# Before full training, run quick validation
docker-compose up gan-htr-smoke-test

# Check results
docker logs gan-htr-smoke-test
```

### Use Case 2: Full Production Training
```bash
# Default production training
docker-compose up -d gan-htr-prod

# Monitor
docker-compose logs -f gan-htr-prod
```

### Use Case 3: Custom Configuration Testing
```bash
# Create test script
cat > scripts/test_config.sh <<'EOF'
#!/bin/bash
poetry run python dual_modal_gan/scripts/train32.py \
  --epochs 5 \
  --batch_size 2 \
  --no_restore
EOF

# Run
TRAINING_SCRIPT=scripts/test_config.sh docker-compose up gan-htr-prod
```

### Use Case 4: Multi-GPU Training
```bash
# Set GPU environment
cat > .env <<EOF
CUDA_VISIBLE_DEVICES=0,1
TRAINING_SCRIPT=scripts/train32_production.sh
EOF

# Run
docker-compose up -d gan-htr-prod
```

---

## 🔍 Verification

### Check What Script is Running
```bash
# View logs to see which script is executed
docker logs gan-htr-prod | grep "Starting training with"

# Output will show:
# 🏃 Starting training with: scripts/train32_smoke_test.sh
```

### Inspect Environment
```bash
# Check environment variables in container
docker exec -it gan-htr-prod env | grep TRAINING_SCRIPT

# Output:
# TRAINING_SCRIPT=scripts/train32_smoke_test.sh
```

---

## 📝 Configuration Priority

Order of precedence (highest to lowest):

1. **Direct command override** (`docker-compose run ... command`)
2. **Inline environment variable** (`TRAINING_SCRIPT=... docker-compose up`)
3. **`.env` file** (local `.env` file)
4. **docker-compose.yml** (service-level environment)
5. **Default** (`dual_modal_gan/scripts/train.py`)

---

## 🧪 Examples

### Example 1: Smoke Test with Custom GPU
```bash
# Use GPU 1 for smoke test
cat > .env <<EOF
CUDA_VISIBLE_DEVICES=1
TRAINING_SCRIPT=scripts/train32_smoke_test.sh
EOF

docker-compose up gan-htr-smoke-test
```

### Example 2: Production Training on Multiple GPUs
```bash
# Use both GPUs
CUDA_VISIBLE_DEVICES=0,1 \
TRAINING_SCRIPT=scripts/train32_production.sh \
docker-compose up -d gan-htr-prod
```

### Example 3: Quick Test Different Batch Sizes
```bash
# Create test scripts
for bs in 2 4 8; do
  cat > scripts/test_bs_${bs}.sh <<EOF
#!/bin/bash
poetry run python dual_modal_gan/scripts/train32.py \\
  --batch_size $bs \\
  --epochs 2 \\
  --checkpoint_dir outputs/test_bs_${bs}
EOF
  chmod +x scripts/test_bs_${bs}.sh
done

# Test each
for bs in 2 4 8; do
  echo "Testing batch size: $bs"
  TRAINING_SCRIPT=scripts/test_bs_${bs}.sh docker-compose run --rm gan-htr-prod
done
```

---

## ✅ Benefits

1. ✅ **Flexibility** - Choose training configuration on-the-fly
2. ✅ **No code changes** - Switch scripts without editing docker-compose.yml
3. ✅ **Reusability** - Same container image, different configs
4. ✅ **Easy testing** - Quick smoke tests before full training
5. ✅ **Cloud-ready** - Works same way in local and cloud environments

---

## 🆘 Troubleshooting

### Script Not Found
```bash
# Check if script exists
docker exec -it gan-htr-prod ls -la /workspace/docRestoration/scripts/

# Make sure script is executable
docker exec -it gan-htr-prod chmod +x /workspace/docRestoration/scripts/train32_smoke_test.sh
```

### Wrong Script Running
```bash
# Check environment
docker exec -it gan-htr-prod env | grep TRAINING_SCRIPT

# Check .env file
cat .env

# Force specific script
docker-compose down
TRAINING_SCRIPT=scripts/train32_smoke_test.sh docker-compose up -d gan-htr-prod
```

### Poetry Command Not Found
```bash
# Make sure script uses full path or poetry run
# Inside container, poetry is available via: poetry run python ...

# Or activate environment first:
docker exec -it gan-htr-prod poetry shell
```

---

## 📚 See Also

- `CLOUD_DEPLOYMENT.md` - Cloud deployment guide
- `docker-compose-commands.sh` - Common Docker commands
- `.env.example` - Environment variable examples
- `entrypoint.sh` - Container initialization logic
