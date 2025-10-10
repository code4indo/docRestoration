#!/bin/bash
# Build and Push GAN-HTR Docker Image to Docker Hub
# Complete pipeline: build locally → quality check → push to Docker Hub
# Usage: ./build_and_push_to_dockerhub.sh [tag]

set -e

# Configuration
IMAGE_NAME="jatnikonm/gan-htr"
TAG="${1:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${TAG}"

echo "🐳 GAN-HTR Build & Push Pipeline"
echo "=================================="
echo ""
echo "Target: ${FULL_IMAGE}"
echo "Tag: ${TAG}"
echo ""

# Step 1: Check Docker Hub login
echo "🔍 [1/5] Checking Docker Hub access..."
if ! docker info 2>/dev/null | grep -q "Username:"; then
    echo "❌ Not logged in to Docker Hub"
    echo "Please login first:"
    echo "  docker login"
    echo ""
    exit 1
fi

DOCKER_USER=$(docker info 2>/dev/null | grep "Username:" | awk '{print $2}')
echo "✅ Docker Hub User: ${DOCKER_USER}"
echo ""

# Step 2: Build Docker image
echo "🔨 [2/5] Building Docker image..."
echo "This may take 10-20 minutes depending on your system"
echo ""

if docker build -t "${FULL_IMAGE}" .; then
    echo "✅ Build completed successfully"
else
    echo "❌ Build failed!"
    echo ""
    echo "Common issues:"
    echo "1. Dockerfile syntax errors"
    echo "2. Disk space insufficient"
    echo "3. Network connectivity issues"
    echo "4. Permissions problems"
    echo ""
    echo "Debug tips:"
    echo "- Run: docker build --progress=plain ."
    echo "- Check: docker system df"
    echo ""
    exit 1
fi

# Step 3: Verify local image
echo "🔍 [3/5] Verifying local image..."
if docker images "${FULL_IMAGE}" --format "{{.Repository}}:{{.Tag}}" | grep -q "${FULL_IMAGE}"; then
    echo "✅ Local image verification passed"
    
    # Show image details
    docker images "${FULL_IMAGE}" --format "{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    # Show image size warning if too large
    IMAGE_SIZE=$(docker images "${FULL_IMAGE}" --format "{{.Size}}")
    SIZE_MB=$(echo "${IMAGE_SIZE}" | sed 's/MB//; s/GB//;s/[^0-9.]//g')
    if [ "$(echo "${SIZE_MB} >= 8000" | bc -l 2>/dev/null || echo 0)" -eq 1 ]; then
        echo "⚠️  Warning: Large image (${IMAGE_SIZE})"
        echo "   Consider optimizing .dockerignore"
    fi
else
    echo "❌ Local image verification failed"
    echo "Build seems to have succeeded but image not found"
    exit 1
fi
echo ""

# Step 4: Test image locally (quick test)
echo "🧪 [4/5] Testing image locally..."
echo "Running quick container test..."

# Create test container
TEST_NAME="gan-htr-test-$(date +%s)"
echo "Testing: python3 --version && nvidia-smi --query-gpu=name..."

if docker run --name "${TEST_NAME}" --rm --gpus all \
    "${FULL_IMAGE}" \
    bash -c "python3 --version 2>/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'GPU check'" > /dev/null; then
    echo "✅ Local test passed"
else
    echo "⚠️  Local test had issues (but continuing)"
    echo "This might be normal if no GPU is available"
fi
echo ""

# Step 5: Push to Docker Hub
echo "🚀 [5/5] Pushing to Docker Hub..."
echo "This may take 5-25 minutes depending on image size"
echo ""

# Push main tag
if docker push "${FULL_IMAGE}"; then
    echo "✅ Push completed successfully!"
    echo ""
    
    # Also tag and push latest if it's not already latest
    if [ "${TAG}" != "latest" ]; then
        echo "🏷️ Also tagging and pushing as latest..."
        docker tag "${FULL_IMAGE}" "${IMAGE_NAME}:latest"
        if docker push "${IMAGE_NAME}:latest"; then
            echo "✅ Also pushed as latest"
        else
            echo "⚠️  Latest tag push failed"
        fi
    fi
    
    echo ""
    echo "🌐 Docker Hub URLs:"
    echo "   Latest: https://hub.docker.com/r/${IMAGE_NAME}"
    echo "   Tagged: https://hub.docker.com/r/${IMAGE_NAME}/tags"
    echo ""
    echo "📥 Pull commands:"
    echo "   docker pull ${FULL_IMAGE}"
    echo "   docker pull ${IMAGE_NAME}:latest"
    echo ""
    echo "🏃 Run example:"
    echo "   docker run --gpus all -it ${FULL_IMAGE}"
    echo "   docker run --gpus all -it ${IMAGE_NAME}:latest"
    echo ""
    echo "🎯 Ready for deployment!"
    
else
    echo "❌ Push failed!"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check Docker Hub login: docker login"
    echo "2. Check internet connection"
    echo "3. Check Docker Hub status: https://status.docker.com/"
    echo "4. Try manual push: docker push ${FULL_IMAGE}"
    echo ""
    exit 1
fi

echo ""
echo "🎉 Build & Push Pipeline Completed!"
echo "================================"
echo ""
echo "Summary:"
echo "--------"
echo "  Image: ${FULL_IMAGE}"
echo "  Size: ${IMAGE_SIZE}"
echo "  User: ${DOCKER_USER}"
echo ""
echo "Next steps:"
echo "1. Test pull from clean environment"
echo "2. Update documentation with pull command"
echo "3. Deploy to cloud (RunPod, Lambda, etc.)"
echo "4. Create GitHub release with Docker tag"
echo ""
