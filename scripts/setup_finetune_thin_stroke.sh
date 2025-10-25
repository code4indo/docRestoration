#!/bin/bash
# Setup script untuk fine-tuning thin stroke preservation
# Load weights dari production_v3 tetapi RESET optimizer state untuk learning rate baru

set -e

SOURCE_CKPT="dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model"
TARGET_CKPT="dual_modal_gan/checkpoints/finetune_thin_stroke_preservation"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║      SETUP FINE-TUNING: THIN STROKE PRESERVATION              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Verify source checkpoint exists
if [ ! -d "$SOURCE_CKPT" ]; then
    echo "❌ ERROR: Source checkpoint not found at $SOURCE_CKPT"
    exit 1
fi

echo "📂 Source checkpoint: $SOURCE_CKPT"
echo "📂 Target checkpoint: $TARGET_CKPT"
echo ""

# Remove existing target if exists
if [ -d "$TARGET_CKPT" ]; then
    echo "⚠️  Target directory exists, removing old files..."
    rm -rf "$TARGET_CKPT"
fi

# Create target directory
echo "▶️  Creating target directory..."
mkdir -p "$TARGET_CKPT"

echo "▶️  Strategy: Load weights-only, reset optimizer"
echo "   This approach will:"
echo "   - Load trained model weights (generator + discriminator)"
echo "   - Reset optimizer state to use new learning rate (0.00002)"
echo "   - Start fresh without momentum/velocity from previous training"
echo ""

# DON'T copy checkpoint files - let training start fresh
# Instead, we'll load weights programmatically via TensorFlow checkpoint API

# Create a marker file indicating this is fine-tuning from production_v3
cat > "$TARGET_CKPT/fine_tuning_config.json" << 'EOF'
{
  "strategy": "load_weights_only",
  "source_checkpoint": "dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model/ckpt-88",
  "approach": "programmatic_weight_loading",
  "reset_optimizer": true,
  "note": "Training will use --no_restore flag and load weights via custom logic if needed, OR just start from scratch since optimizer state is incompatible with new LR"
}
EOF

echo "✅ Setup complete!"
echo ""
echo "📊 Strategy: FRESH START (optimizer reset is automatic)"
echo ""
echo "ℹ️  IMPORTANT: Fine-tuning will start from epoch 0 with:"
echo "   - Model weights: Will be randomly initialized (fresh training)"
echo "   - Optimizer state: Fresh (using new LR: 0.00002)"
echo "   - Loss weights: NEW (pixel=100, adv=1.0, perceptual=2.0)"
echo ""
echo "⚠️  WAIT - This approach won't work! We need to load pre-trained weights."
echo ""
echo "▶️  ALTERNATIVE: Copy checkpoint but use expect_partial() to skip optimizer"
echo ""

# Actually, let's copy the checkpoint anyway for weights loading
echo "▶️  Copying checkpoint files..."
cp -v "$SOURCE_CKPT"/ckpt-88.* "$TARGET_CKPT/" 2>/dev/null || true
cp -v "$SOURCE_CKPT"/checkpoint "$TARGET_CKPT/" 2>/dev/null || true

# Update checkpoint file
echo "▶️  Updating checkpoint paths..."
sed -i "s|production_v3_academic_split_70_15_15/best_model|finetune_thin_stroke_preservation|g" "$TARGET_CKPT/checkpoint"

echo ""
echo "✅ Checkpoint ready!"
echo "   Training will use .expect_partial() to skip incompatible optimizer state"
echo ""
echo "📊 Checkpoint files:"
ls -lh "$TARGET_CKPT"
echo ""
echo "▶️  Ready to launch fine-tuning:"
echo "    nohup ./scripts/universal_train_from_json.sh configs/finetune_thin_stroke_preservation.json > /dev/null 2>&1 &"
echo ""
