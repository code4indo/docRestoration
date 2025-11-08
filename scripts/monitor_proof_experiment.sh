#!/bin/bash

# Real-time monitoring dashboard for proof-of-concept experiment
# Shows: Epoch progress, PSNR, CER, Loss values

LOG_FILE="logs/exp_proof_ground_truth_20251106_102944.log"
CHECKPOINT_DIR="dual_modal_gan/checkpoints/exp_proof_ground_truth"

echo "=============================================="
echo "PROOF-OF-CONCEPT MONITORING DASHBOARD"
echo "=============================================="
echo "Experiment: Ground Truth Mode"
echo "Log file: ${LOG_FILE}"
echo "=============================================="
echo ""

# Check if training is still running
PID=857813
if kill -0 $PID 2>/dev/null; then
    echo "✅ Training is RUNNING (PID: ${PID})"
else
    echo "⚠️  Training has STOPPED (PID: ${PID})"
fi
echo ""

# Show last 40 lines with epoch/metric highlights
echo "=== RECENT TRAINING LOG ==="
tail -n 40 "${LOG_FILE}" | grep -E "Epoch|PSNR|CER|Loss|WARNING|ERROR" || tail -n 40 "${LOG_FILE}"
echo ""

# Show epoch info if available
if [ -f "${CHECKPOINT_DIR}/epoch_info.json" ]; then
    echo "=== CURRENT BEST METRICS ==="
    python3 -c "
import json
try:
    data = json.load(open('${CHECKPOINT_DIR}/epoch_info.json'))
    print(f\"Last Epoch:  {data.get('last_completed_epoch', 'N/A')}/10\")
    print(f\"Best Epoch:  {data.get('best_epoch', 'N/A')}\")
    print(f\"Best PSNR:   {data.get('best_psnr', 0):.2f} dB\")
    print(f\"Best CER:    {data.get('best_cer', 1)*100:.1f}%\")
    print(f\"Best SSIM:   {data.get('best_ssim', 0):.4f}\")
    print(f\"Combined:    {data.get('best_combined_score', 0):.2f}\")
    
    # Show improvement trend
    current_epoch = data.get('last_completed_epoch', 0)
    if current_epoch >= 10:
        print(f\"\\n✅ EXPERIMENT COMPLETED!\")
        if data.get('best_psnr', 0) >= 30.0:
            print(f\"✅ SUCCESS: PSNR >= 30 dB (hypothesis CONFIRMED!)\")
        else:
            print(f\"⚠️  Below target: PSNR < 30 dB (need analysis)\")
except Exception as e:
    print(f\"Error reading metrics: {e}\")
" 
else
    echo "⏳ No epoch_info.json yet (training just started)"
fi

echo ""
echo "=============================================="
echo "COMMANDS:"
echo "  Watch live: tail -f ${LOG_FILE}"
echo "  Refresh:    ./scripts/monitor_proof_experiment.sh"
echo "  Stop:       kill ${PID}"
echo "=============================================="
