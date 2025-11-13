#!/bin/bash

# Monitor GradNorm Production Training
# Usage: ./scripts/monitor_gradnorm_production.sh

CHECKPOINT_DIR="dual_modal_gan/checkpoints/production_v4_gradnorm"
EXPERIMENT_NAME="production_v4_gradnorm_adaptive"

echo "=========================================="
echo "🔍 MONITORING GRADNORM PRODUCTION TRAINING"
echo "=========================================="
echo ""

# Check process
echo "📊 Training Process:"
ps aux | grep -E "train_enhanced.*gradnorm_production" | grep -v grep | awk '{print "   PID:", $2, "| CPU:", $3"%", "| MEM:", $4"%", "| TIME:", $10}'
echo ""

# Check checkpoints
echo "💾 Checkpoints:"
if [ -d "$CHECKPOINT_DIR" ]; then
    ls -lth "$CHECKPOINT_DIR" | head -10
    echo ""
    LATEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/ckpt-*.index 2>/dev/null | head -1 | grep -o 'ckpt-[0-9]*' | grep -o '[0-9]*')
    if [ ! -z "$LATEST_CKPT" ]; then
        CURRENT_EPOCH=$((LATEST_CKPT / 50))  # 50 steps per epoch
        echo "   📈 Current Progress: ~Epoch $CURRENT_EPOCH/50"
    fi
else
    echo "   ⏳ Waiting for first checkpoint..."
fi
echo ""

# Check MLflow metrics
echo "📈 Latest Metrics (from MLflow):"
poetry run python -c "
import mlflow
import sys
try:
    mlflow.set_tracking_uri('file:///home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/mlruns')
    client = mlflow.tracking.MlflowClient()
    
    # Find experiment
    experiment = client.get_experiment_by_name('$EXPERIMENT_NAME')
    if not experiment:
        print('   ⏳ Experiment not started yet...')
        sys.exit(0)
    
    # Get latest run
    runs = client.search_runs(experiment.experiment_id, order_by=['start_time DESC'], max_results=1)
    if not runs:
        print('   ⏳ No runs found yet...')
        sys.exit(0)
    
    run = runs[0]
    metrics = run.data.metrics
    
    # Print key metrics
    if 'val_psnr' in metrics:
        print(f\"   PSNR: {metrics.get('val_psnr', 0):.2f} dB\")
    if 'val_ssim' in metrics:
        print(f\"   SSIM: {metrics.get('val_ssim', 0):.4f}\")
    if 'val_cer' in metrics:
        print(f\"   CER:  {metrics.get('val_cer', 0):.4f}\")
    if 'val_wer' in metrics:
        print(f\"   WER:  {metrics.get('val_wer', 0):.4f}\")
    
    # Print GradNorm weights
    print('')
    print('   🎯 GradNorm Weights:')
    for loss in ['pixel', 'adversarial', 'rec_feat', 'perceptual', 'ctc']:
        key = f'gradnorm/weight_{loss}'
        if key in metrics:
            print(f\"      {loss}: {metrics[key]:.2f}\")
    
except Exception as e:
    print(f'   ⚠️ Error reading metrics: {e}')
" 2>/dev/null || echo "   ⏳ MLflow not ready yet..."

echo ""
echo "=========================================="
echo "💡 Tips:"
echo "   - Training runs in background (nohup)"
echo "   - Check process: ps aux | grep train_enhanced"
echo "   - Kill training: pkill -f gradnorm_production"
echo "   - Full monitoring: watch -n 10 ./scripts/monitor_gradnorm_production.sh"
echo "=========================================="
