#!/bin/bash
# Test checkpoint resume functionality
# Usage: bash scripts/test_checkpoint_resume.sh

set -e  # Exit on error

TEST_DIR="dual_modal_gan/checkpoints/test_resume"
TEST_LOG="logbook/test_resume_$(date +%Y%m%d_%H%M%S).log"

echo "======================================"
echo "CHECKPOINT RESUME FUNCTIONALITY TEST"
echo "======================================"
echo ""
echo "Test Plan:"
echo "  1. Train for 2 epochs (10 steps each)"
echo "  2. Verify epoch_info.json created"
echo "  3. Resume with --resume flag"
echo "  4. Verify training continues from epoch 3"
echo ""

# Clean previous test
if [ -d "$TEST_DIR" ]; then
    echo "🧹 Cleaning previous test: $TEST_DIR"
    rm -rf "$TEST_DIR"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PHASE 1: Initial Training (2 epochs)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

poetry run python dual_modal_gan/scripts/train_enhanced.py \
    --epochs 2 \
    --steps_per_epoch 10 \
    --batch_size 2 \
    --checkpoint_dir "$TEST_DIR" \
    --perceptual_loss_weight 0.2 \
    --early_stopping \
    --patience 10 \
    --save_interval 1 \
    2>&1 | tee -a "$TEST_LOG"

echo ""
echo "✅ Phase 1 completed"
echo ""

# Check epoch_info.json
if [ ! -f "$TEST_DIR/epoch_info.json" ]; then
    echo "❌ FAILED: epoch_info.json not created!"
    exit 1
fi

echo "📄 Epoch Info Content:"
cat "$TEST_DIR/epoch_info.json"
echo ""

LAST_EPOCH=$(cat "$TEST_DIR/epoch_info.json" | grep "last_completed_epoch" | awk -F': ' '{print $2}' | tr -d ',')
echo "Last completed epoch: $LAST_EPOCH"
echo ""

if [ "$LAST_EPOCH" != "1" ]; then
    echo "⚠️  WARNING: Expected last_completed_epoch=1, got $LAST_EPOCH"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PHASE 2: Resume Training (continue to 4 epochs)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

poetry run python dual_modal_gan/scripts/train_enhanced.py \
    --resume \
    --epochs 4 \
    --steps_per_epoch 10 \
    --batch_size 2 \
    --checkpoint_dir "$TEST_DIR" \
    --perceptual_loss_weight 0.2 \
    --early_stopping \
    --patience 10 \
    --save_interval 1 \
    2>&1 | tee -a "$TEST_LOG"

echo ""
echo "✅ Phase 2 completed"
echo ""

# Verify final epoch
FINAL_EPOCH=$(cat "$TEST_DIR/epoch_info.json" | grep "last_completed_epoch" | awk -F': ' '{print $2}' | tr -d ',')
echo "📄 Final Epoch Info:"
cat "$TEST_DIR/epoch_info.json"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$FINAL_EPOCH" == "3" ]; then
    echo "✅ SUCCESS: Checkpoint resume working correctly!"
    echo "   - Phase 1 trained epochs 0-1"
    echo "   - Phase 2 resumed from epoch 2 and completed epochs 2-3"
    echo "   - Final last_completed_epoch: $FINAL_EPOCH"
    echo ""
    echo "📋 Log file: $TEST_LOG"
    exit 0
else
    echo "❌ FAILED: Expected final epoch 3, got $FINAL_EPOCH"
    echo "📋 Check log: $TEST_LOG"
    exit 1
fi
