#!/bin/bash
# Complete GAN-HTR setup - install dependencies + download data

echo "🔧 GAN-HTR Complete Setup Script"
echo "================================"
echo ""
echo "This script performs complete setup:"
echo "1. Install Python dependencies (Poetry)"
echo "2. Download required data from HuggingFace"
echo "3. Verify installation"
echo ""

# Function to check if command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "❌ Error: $1 not installed"
        return 1
    fi
    return 0
}

# Check Python and Poetry
echo "🔍 Checking requirements..."

if ! check_command python3; then
    echo "Please install Python 3.10+ first"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

if ! check_command poetry; then
    echo "❌ Poetry not found. Installing..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
    
    if ! check_command poetry; then
        echo "❌ Poetry installation failed"
        echo "Manual install: pip install poetry"
        exit 1
    fi
fi

echo "✅ Poetry found: $(poetry --version)"
echo ""

# Install dependencies
echo "📦 Installing project dependencies..."
echo "This may take 5-10 minutes..."

if poetry install; then
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
# Download data
echo "📥 Now downloading required data..."
echo "This may take 15-20 minutes..."

if ./scripts/download_data.sh; then
    echo "✅ Data download completed!"
else
    echo "❌ Data download failed"
    exit 1
fi

echo ""
# Final verification
echo "🔍 Final verification..."

if ./scripts/check_required_data.sh | grep -q "All required data files present"; then
    echo ""
    echo "🎉 GAN-HTR setup completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Activate virtual environment: poetry shell"
    echo "2. Run smoke test: ./scripts/train32_smoke_test.sh"
    echo "3. View MLflow UI: poetry run mlflow ui"
    echo ""
    echo "🚀 Ready to start training!"
else
    echo ""
    echo "⚠️  Setup completed with warnings"
    echo "Some files may be missing. Check the output above."
    echo ""
    echo "Manual verification:"
    echo "./scripts/check_required_data.sh"
fi
