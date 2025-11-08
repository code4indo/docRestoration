#!/bin/bash
# UNIVERSAL TRAINING LAUNCHER FROM JSON CONFIG
# Usage: ./universal_train_from_json.sh <path/to/config.json>

set -e

# Check if config file is provided
if [ -z "$1" ]; then
    echo "❌ Error: Config file not provided"
    echo "Usage: $0 <path/to/config.json>"
    exit 1
fi

CONFIG_FILE="$1"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         UNIVERSAL TRAINING LAUNCHER FROM JSON CONFIG          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📄 Config file: $CONFIG_FILE"
echo "🕐 Timestamp: $TIMESTAMP"
echo ""

# Parse JSON and build command using Python parser script
CMD_OUTPUT=$(poetry run python scripts/parse_training_config.py "$CONFIG_FILE" 2>&1)

# Check if parsing was successful
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to parse config file"
    echo "$CMD_OUTPUT"
    exit 1
fi

# Extract experiment name and command args
EXPERIMENT_NAME=$(echo "$CMD_OUTPUT" | tail -1 | cut -d'|' -f1)
CMD_ARGS=$(echo "$CMD_OUTPUT" | tail -1 | cut -d'|' -f2-)

# Display configuration summary (everything except last line)
echo "$CMD_OUTPUT" | head -n -1

# Log file
LOG_FILE="logbook/${EXPERIMENT_NAME}_${TIMESTAMP}.log"

echo "▶️  Starting training..."
echo "📝 Log file: $LOG_FILE"
echo ""

# Execute training with --config flag to enable JSON config parsing
# This is CRITICAL for triple validation (Base + ANRI + DIBCO)
poetry run python dual_modal_gan/scripts/train_enhanced.py \
    --config "$CONFIG_FILE" \
    $CMD_ARGS \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # Extract final metrics
    echo ""
    echo "📊 FINAL METRICS:"
    grep -E "📊.*PSNR:|SSIM:|CER:" "$LOG_FILE" | tail -3
    
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo ""
    echo "📋 Last 20 lines of log:"
    tail -20 "$LOG_FILE"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

exit $EXIT_CODE
