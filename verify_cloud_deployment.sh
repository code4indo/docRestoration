#!/bin/bash
# Verification Script - Docker Compose Cloud Deployment
# Cek apakah semua file yang diperlukan sudah ada dan konfigurasi benar

set -e

echo "========================================"
echo "🔍 Docker Compose Cloud Deployment Check"
echo "========================================"
echo ""

# Check 1: Docker Compose file
echo "📋 Check 1: Docker Compose Configuration"
if [ -f "docker-compose.yml" ]; then
    echo "  ✅ docker-compose.yml found"
    
    # Validate syntax
    if docker-compose config > /dev/null 2>&1; then
        echo "  ✅ docker-compose.yml syntax valid"
    else
        echo "  ❌ docker-compose.yml syntax error"
        exit 1
    fi
else
    echo "  ❌ docker-compose.yml not found"
    exit 1
fi
echo ""

# Check 2: Entrypoint script
echo "📋 Check 2: Entrypoint Script"
if [ -f "entrypoint.sh" ]; then
    echo "  ✅ entrypoint.sh found"
    
    # Check executable
    if [ -x "entrypoint.sh" ]; then
        echo "  ✅ entrypoint.sh is executable"
    else
        echo "  ⚠️ entrypoint.sh not executable (will be fixed by Docker)"
    fi
    
    # Check shebang
    if head -n 1 entrypoint.sh | grep -q "#!/bin/bash"; then
        echo "  ✅ entrypoint.sh uses bash (correct)"
    else
        echo "  ❌ entrypoint.sh should use #!/bin/bash"
        exit 1
    fi
else
    echo "  ❌ entrypoint.sh not found"
    exit 1
fi
echo ""

# Check 3: Download script
echo "📋 Check 3: Download Script"
if [ -f "scripts/download_from_huggingface.py" ]; then
    echo "  ✅ download_from_huggingface.py found"
    
    # Check Python syntax
    if python3 -m py_compile scripts/download_from_huggingface.py 2>/dev/null; then
        echo "  ✅ Python syntax valid"
    else
        echo "  ❌ Python syntax error"
        exit 1
    fi
    
    # Check for required imports
    if grep -q "from huggingface_hub import hf_hub_download" scripts/download_from_huggingface.py; then
        echo "  ✅ HuggingFace Hub import found"
    else
        echo "  ❌ Missing huggingface_hub import"
        exit 1
    fi
else
    echo "  ❌ download_from_huggingface.py not found"
    exit 1
fi
echo ""

# Check 4: Required directories
echo "📋 Check 4: Directory Structure"
for dir in "outputs" "logbook" "mlruns"; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir/ exists"
    else
        echo "  ⚠️ $dir/ will be created automatically"
    fi
done
echo ""

# Check 5: Docker image
echo "📋 Check 5: Docker Image"
if docker images | grep -q "jatnikonm/gan-htr"; then
    echo "  ✅ Docker image jatnikonm/gan-htr found locally"
else
    echo "  ⚠️ Docker image will be pulled on first run"
fi
echo ""

# Check 6: GPU (if available)
echo "📋 Check 6: GPU Availability"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --list-gpus | wc -l)
        echo "  ✅ NVIDIA GPU detected: $gpu_count device(s)"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | while read line; do
            echo "     - $line"
        done
    else
        echo "  ⚠️ nvidia-smi command failed"
    fi
else
    echo "  ⚠️ nvidia-smi not found (GPU might not be available)"
fi
echo ""

# Check 7: Network ports
echo "📋 Check 7: Port Availability"
for port in 5001 6007 8889; do
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        echo "  ⚠️ Port $port is already in use"
    else
        echo "  ✅ Port $port is available"
    fi
done
echo ""

# Check 8: Disk space
echo "📋 Check 8: Disk Space"
available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$available_space" -gt 50 ]; then
    echo "  ✅ Sufficient disk space: ${available_space}GB available"
else
    echo "  ⚠️ Low disk space: ${available_space}GB (recommended: >50GB)"
fi
echo ""

# Check 9: Docker daemon
echo "📋 Check 9: Docker Service"
if docker info > /dev/null 2>&1; then
    echo "  ✅ Docker daemon is running"
else
    echo "  ❌ Docker daemon is not running"
    exit 1
fi
echo ""

# Check 10: Docker Compose version
echo "📋 Check 10: Docker Compose Version"
compose_version=$(docker-compose version --short 2>/dev/null || echo "unknown")
echo "  ℹ️ Docker Compose version: $compose_version"
echo ""

echo "========================================"
echo "✅ All Checks Completed"
echo "========================================"
echo ""
echo "🚀 Ready to Deploy!"
echo ""
echo "Quick Start Commands:"
echo "  • Start training:  docker-compose up -d gan-htr-prod"
echo "  • View logs:       docker-compose logs -f gan-htr-prod"
echo "  • Stop training:   docker-compose stop gan-htr-prod"
echo ""
echo "📚 Documentation:"
echo "  • CLOUD_DEPLOYMENT.md - Full deployment guide"
echo "  • DOCKER_COMPOSE_CLOUD_SUMMARY.md - Architecture summary"
echo "  • docker-compose-commands.sh - Quick command reference"
echo ""
