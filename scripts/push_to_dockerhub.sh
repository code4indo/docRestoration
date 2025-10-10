#!/bin/bash
# Push GAN-HTR Docker Image to Docker Hub
# Simple script to push pre-built image to Docker Hub
# Usage: ./push_to_dockerhub.sh

set -e

# Configuration
IMAGE_NAME="jatnikonm/gan-htr"
TAG="${1:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${TAG}"

echo "🐳 GAN-HTR Docker Image Push"
echo "=================================="
echo ""
echo "Target Image: ${FULL_IMAGE}"
echo ""

# Check if Docker Hub login required
echo "🔍 Checking Docker Hub access..."
if ! docker info 2>/dev/null | grep -q "Username:"; then
    echo "❌ Not logged in to Docker Hub"
    echo "Please login first:"
    echo "  docker login"
    echo ""
    exit 1
fi

# Get current Docker Hub user
DOCKER_USER=$(docker info 2>/dev/null | grep "Username:" | awk '{print $2}')
echo "✅ Docker Hub User: ${DOCKER_USER}"
echo ""

# Check if image exists locally
echo "🔍 Checking local image..."
if ! docker images "${FULL_IMAGE}" --format "{{.Repository}}:{{.Tag}}" | grep -q "${FULL_IMAGE}"; then
    echo "❌ Image ${FULL_IMAGE} not found locally"
    echo ""
    echo "Build image first:"
    echo "  docker build -t ${FULL_IMAGE} ."
    echo ""
    exit 1
fi

echo "✅ Local image found"
docker images "${FULL_IMAGE}" --format "{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
echo ""

# Show image size
IMAGE_SIZE=$(docker images "${FULL_IMAGE}" --format "{{.Size}}")
echo "📊 Image Size: ${IMAGE_SIZE}"
echo ""

# Push to Docker Hub
echo "🚀 Pushing to Docker Hub..."
echo "This may take 5-25 minutes depending on image size and connection"
echo ""

if docker push "${FULL_IMAGE}"; then
    echo ""
    echo "✅ Push completed successfully!"
    echo ""
    echo "🌐 Docker Hub URL:"
    echo "   https://hub.docker.com/r/${IMAGE_NAME}"
    echo ""
    echo "📥 Pull command for others:"
    echo "   docker pull ${FULL_IMAGE}"
    echo ""
    echo "🏃 Run command:"
    echo "   docker run --gpus all -it ${FULL_IMAGE}"
    echo ""
    
    # Also tag and push latest if not already
    if [ "${TAG}" != "latest" ]; then
        echo ""
        echo "🏷️ Also tagging as latest..."
        docker tag "${FULL_IMAGE}" "${IMAGE_NAME}:latest"
        docker push "${IMAGE_NAME}:latest"
        echo "✅ Also pushed as ${IMAGE_NAME}:latest"
    fi
    
else
    echo "❌ Push failed!"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check internet connection"
    echo "2. Verify DockerHub login: docker login"
    echo "3. Check image exists: docker images ${FULL_IMAGE}"
    echo "4. Check disk space and permissions"
    echo ""
    exit 1
fi

echo ""
echo "🎉 Push operation completed!"
echo "================================"
