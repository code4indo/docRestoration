#!/bin/bash
# Quick status check for ablation experiments

echo "========================================="
echo "  ABLATION EXPERIMENTS QUICK STATUS"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
echo ""

for i in {1..5}; do
    log_file="logs/ablation_$(printf "%02d" $i)_training.log"
    
    # Get experiment name
    case $i in
        1) exp_name="Pixel Only              " ;;
        2) exp_name="Pixel + Adv             " ;;
        3) exp_name="Pixel + Adv + Perc      " ;;
        4) exp_name="Pixel + Adv + Perc + CTC" ;;
        5) exp_name="FULL (+ RecFeat)        " ;;
    esac
    
    if [ ! -f "$log_file" ]; then
        echo "⏳ Exp $i [$exp_name]: NOT STARTED"
        continue
    fi
    
    # Get epoch progress
    current_epoch=$(grep -oP "Epoch \K\d+(?=/\d+)" "$log_file" | tail -1)
    
    if [ -z "$current_epoch" ]; then
        echo "⏳ Exp $i [$exp_name]: INITIALIZING..."
        continue
    fi
    
    # Get latest PSNR
    latest_psnr=$(grep -oP "Val PSNR:\s*\K[\d.]+" "$log_file" | tail -1)
    latest_cer=$(grep -oP "Val CER:\s*\K[\d.]+" "$log_file" | tail -1)
    
    if [ "$current_epoch" -eq 15 ]; then
        echo "✅ Exp $i [$exp_name]: DONE (PSNR: ${latest_psnr:-N/A} dB, CER: ${latest_cer:-N/A}%)"
    else
        echo "🔄 Exp $i [$exp_name]: Epoch $current_epoch/15 (PSNR: ${latest_psnr:-N/A} dB)"
    fi
done

echo ""
echo "========================================="
echo "Commands:"
echo "  Monitor live: ./scripts/monitor_ablation.sh"
echo "  Extract results: poetry run python scripts/extract_ablation_metrics.py"
echo "========================================="
