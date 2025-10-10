#!/bin/bash
# Test script for docker-compose configuration validation

set -e

echo "🧪 Testing Docker Compose Configuration"
echo "======================================"

# Function to test config
test_compose_config() {
    local config_file="$1"
    echo "📋 Testing $config_file..."

    if docker-compose -f "$config_file" config --quiet; then
        echo "✅ $config_file: Configuration is valid"
        return 0
    else
        echo "❌ $config_file: Configuration has errors"
        return 1
    fi
}

# Function to test basic services
test_service_availability() {
    local config_file="$1"
    echo "🔍 Checking services in $config_file..."

    # List services
    echo "📦 Available services:"
    docker-compose -f "$config_file" config --services

    echo ""
}

# Test main configuration
echo "1. Testing main docker-compose.yml"
test_compose_config "docker-compose.yml"
test_service_availability "docker-compose.yml"

echo ""

# Test dev configuration
echo "2. Testing development docker-compose.dev.yml"
test_compose_config "docker-compose.dev.yml"
test_service_availability "docker-compose.dev.yml"

echo ""

# Check for required files
echo "3. Checking required files"
required_files=(
    "entrypoint.sh"
    "Dockerfile"
    "Dockerfile.dev"
    "scripts/download_data.sh"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ Found: $file"
    else
        echo "⚠️ Missing: $file"
    fi
done

echo ""

# Check permissions
echo "4. Checking file permissions"
if [ -x "entrypoint.sh" ]; then
    echo "✅ entrypoint.sh is executable"
else
    echo "❌ entrypoint.sh is not executable"
    echo "💡 Fix: chmod +x entrypoint.sh"
fi

echo ""

# Validate directory structure
echo "5. Checking directory structure"
required_dirs=(
    "scripts"
    "network"
    "dual_modal_gan"
    "real_data_preparation"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "../$dir" ]; then
        echo "✅ Directory exists: ../$dir"
    else
        echo "⚠️ Directory missing: ../$dir"
    fi
done

echo ""
echo "🎯 Configuration test completed!"
echo "================================"