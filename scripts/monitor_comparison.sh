#!/bin/bash

# Monitor PREDICTED mode experiment progress
# Compare with GROUND_TRUTH results

PRED_LOG=$(ls -t logs/exp_proof_predicted_*.log 2>/dev/null | head -1)
GT_CHECKPOINT="dual_modal_gan/checkpoints/exp_proof_ground_truth"
PRED_CHECKPOINT="dual_modal_gan/checkpoints/exp_proof_predicted"

echo "=============================================="
echo "PROOF-OF-CONCEPT COMPARISON DASHBOARD"
echo "=============================================="
echo ""

# Check if predicted training is running
PRED_PID=$(pgrep -f "exp_proof_predicted_mode.json")
if [ -n "$PRED_PID" ]; then
    echo "✅ PREDICTED MODE: RUNNING (PID: ${PRED_PID})"
else
    echo "⚠️  PREDICTED MODE: STOPPED"
fi
echo ""

# Ground truth results (COMPLETED)
echo "=== GROUND TRUTH MODE (COMPLETED) ==="
if [ -f "${GT_CHECKPOINT}/epoch_info.json" ]; then
    python3 -c "
import json
data = json.load(open('${GT_CHECKPOINT}/epoch_info.json'))
print(f\"✅ Status:       COMPLETED\")
print(f\"   Epochs:       {data['last_completed_epoch']}/10\")
print(f\"   Best Epoch:   {data['best_epoch']}\")
print(f\"   Best PSNR:    {data['best_psnr']:.2f} dB\")
print(f\"   Best CER:     {data['best_cer']*100:.1f}%\")
print(f\"   Combined:     {data['best_combined_score']:.2f}\")
"
else
    echo "❌ No results found"
fi
echo ""

# Predicted results (RUNNING/COMPLETED)
echo "=== PREDICTED MODE (CURRENT) ==="
if [ -f "${PRED_CHECKPOINT}/epoch_info.json" ]; then
    python3 -c "
import json
data = json.load(open('${PRED_CHECKPOINT}/epoch_info.json'))
status = 'RUNNING' if data['last_completed_epoch'] < 10 else 'COMPLETED'
print(f\"📊 Status:       {status}\")
print(f\"   Epochs:       {data['last_completed_epoch']}/10\")
print(f\"   Best Epoch:   {data['best_epoch']}\")
print(f\"   Best PSNR:    {data['best_psnr']:.2f} dB\")
print(f\"   Best CER:     {data['best_cer']*100:.1f}%\")
print(f\"   Combined:     {data['best_combined_score']:.2f}\")
"
else
    echo "⏳ Not started yet (waiting for first epoch)"
fi
echo ""

# Comparison (if predicted has results)
if [ -f "${PRED_CHECKPOINT}/epoch_info.json" ]; then
    echo "=== COMPARISON (Ground Truth vs Predicted) ==="
    python3 -c "
import json
gt = json.load(open('${GT_CHECKPOINT}/epoch_info.json'))
pred = json.load(open('${PRED_CHECKPOINT}/epoch_info.json'))

delta_psnr = gt['best_psnr'] - pred['best_psnr']
delta_cer = (pred['best_cer'] - gt['best_cer']) * 100
delta_combined = gt['best_combined_score'] - pred['best_combined_score']

print(f\"ΔPSNR:        {delta_psnr:+.2f} dB\", end='')
if delta_psnr >= 2.0:
    print(' ✅ SIGNIFICANT! (GT >> Pred)')
elif delta_psnr >= 1.0:
    print(' ✓ Moderate (GT > Pred)')
elif delta_psnr >= 0.5:
    print(' ~ Small (GT ≈ Pred)')
else:
    print(' ⚠️ Negligible (GT ≈ Pred) - HYPOTHESIS REJECTED!')

print(f\"ΔCER:         {delta_cer:+.1f}%\", end='')
if abs(delta_cer) >= 10.0:
    print(' ✅ SIGNIFICANT!')
elif abs(delta_cer) >= 5.0:
    print(' ✓ Moderate')
else:
    print(' ~ Small')

print(f\"ΔCombined:    {delta_combined:+.2f}\")

# Verdict
print()
if delta_psnr >= 2.0:
    print('🎉 VERDICT: HYPOTHESIS CONFIRMED!')
    print('   Ground truth mode gives SIGNIFICANT PSNR improvement!')
    print('   ✅ Proceed with full 150 epoch training')
elif delta_psnr >= 1.0:
    print('✅ VERDICT: HYPOTHESIS PARTIALLY CONFIRMED')
    print('   Ground truth mode gives moderate improvement')
    print('   Consider: Further optimization or proceed with caution')
else:
    print('⚠️  VERDICT: HYPOTHESIS REJECTED!')
    print('   Ground truth ≈ Predicted (no significant difference)')
    print('   ❌ Need diagnostic tests (see ANALISIS_JIKA_HYPOTHESIS_REJECTED.md)')
"
    echo ""
fi

# Recent log
if [ -f "$PRED_LOG" ]; then
    echo "=== RECENT PREDICTED LOG (Last 20 lines) ==="
    tail -n 20 "$PRED_LOG" | grep -E "Epoch|PSNR|CER|Loss|completed" || tail -n 20 "$PRED_LOG"
fi

echo ""
echo "=============================================="
echo "COMMANDS:"
echo "  Watch predicted: tail -f ${PRED_LOG}"
echo "  Refresh:         ./scripts/monitor_comparison.sh"
echo "  Full analysis:   poetry run python scripts/compare_proof_results.py"
echo "=============================================="
