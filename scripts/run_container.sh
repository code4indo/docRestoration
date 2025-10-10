#!/bin/bash
# Quick Check dokter setup dan jalankan GAN-HTR container

echo "🔍 Checking Docker Environment"
echo "===================="
echo ""

# Check Docker daemon
if ! docker info &> /dev/null; then
    echo "❌ Docker daemon not running"
    echo "Please start Docker first"
    exit 1
fi

# Check GPU availability
echo ""
echo "🚪 Checking GPU support..."
if nvidia-smi 2>/dev/null | grep -q "NVIDIA"; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -5
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "   Available GPU: $GPU_COUNT"
else
    echo "❌ No NVIDIA GPU detected"
    echo "This will work but training will be slow (CPU only)"
fi

echo ""
echo "🐳 Checking Docker Hub access..."
if ! docker info 2>/dev/null | grep -q "Username:"; then
    DOCKER_USER=$(docker info 2>/dev/null | grep "Username:" | awk '{print $2}')
    echo "✅ Docker Hub user: $DOCKER_USER"
else
    echo "❌ Not logged in to Docker Hub"
    echo "Please login: docker login"
    exit 1
fi

# Check local image
echo ""
echo "📋 Checking GAN-HTR image..."
echo "Target: jatnikonm/gan-htr:latest"
if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "jatnikonm/gan-htr"; then
    IMAGE_SIZE=$(docker images "jatnikonm/gan-htr:latest" --format "{{.Size}}")
    echo "✅ Image found: $IMAGE_SIZE"
else
    echo "❌ Image not found locally"
    echo "Building image first..."
    echo "  docker build -t jatnikonm/gan-htr:latest ."
fi

echo ""
echo "🚀 Starting GAN-HTR container..."
echo "=================="
echo ""

# Default command if no arguments given
CMD=${@:-docker run --gpus all -it jatnikonm/gan-htr:latest}

# Add any arguments passed after docker run
if [ $# -gt 0 ]; then
    CMD="${CMD} $@"
fi

echo "Command: $CMD"
echo ""

# Execute with GPU support
echo "Container will use GPU 0 (CUDA_VISIBLE_DEVICES=0)"
echo "GPU memory will be allocated as needed"
echo ""

# Add volume mounts for outputs if they exist
VOLUME_MOUNTS=""

if [ -d "./outputs" ]; then
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v $(pwd)/outputs:/workspace/outputs"
fi

if [ -d "./logbook" ]; then
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v $(pwd)/logbook:/workspace/logbook"
fi

if [ -d "./outputs" ]; then
    echo ""
    echo "📁 Volume mounts added for outputs"
fi
if [ -d "./logbook" ]; then
    echo "📜 Volume mounts added for logbook"
fi

# Add development mode volumes if docRestoration exists
if [ -d "./docRestoration" ]; then
    if [ -z "$VOLUME_MOUNTS" ]; then
        VOLUME_MOUNTS="$VOLUME_MOUNTS -v ./docRestoration:/workspace/working"
    else
        VOLUME_MOUNTS="$VOLUME_MOUNTS -v ./docRestoration:/workspace/docRestoration"
    fi
    echo ""
    echo "📁 Development mode with docRestoration mount"
else
    echo "📁 Using image contents (Production mode)"
fi

# Run the container
echo ""
echo "🚀 Executing: docker run $CMD $VOLUME_MOUNTS"
echo ""
docker run --gpus all $VOLUME_MOUNTS "$CMD"

echo ""
echo "🎉 Container stopped"
echo "=================="
echo ""
