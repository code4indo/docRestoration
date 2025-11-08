#!/bin/bash
#############################################################################
# ABLATION MONITORING SCRIPT
# Monitors all 5 ablation experiments and auto-extracts metrics when complete
#############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXTRACTOR="$SCRIPT_DIR/extract_ablation_metrics.py"

echo "========================================="
echo "  ABLATION STUDY LIVE MONITOR          "
echo "========================================="
echo ""

while true; do
    clear
    echo "========================================="
    echo "  ABLATION EXPERIMENTS STATUS"
    echo "  $(date)"
    echo "========================================="
    echo ""
    
    # Check each experiment
    for i in {1..5}; do
        log_file="$PROJECT_ROOT/logs/ablation_$(printf "%02d" $i)_training.log"
        
        if [ ! -f "$log_file" ]; then
            echo "⏳ Experiment $i: NOT STARTED YET"
            continue
        fi
        
        # Check if training is running
        if pgrep -f "ablation_$(printf "%02d" $i)" > /dev/null; then
            status="🟢 RUNNING"
        else
            status="⚪ COMPLETED/STOPPED"
        fi
        
        # Extract current progress
        current_epoch=$(grep -oP "Epoch \K\d+(?=/)" "$log_file" | tail -1)
        total_epochs=$(grep -oP "Epoch \d+/\K\d+" "$log_file" | tail -1)
        
        # Extract latest metrics
        latest_psnr=$(grep -oP "Val PSNR:\s*\K[\d.]+" "$log_file" | tail -1)
        latest_ssim=$(grep -oP "Val SSIM:\s*\K[\d.]+" "$log_file" | tail -1)
        latest_cer=$(grep -oP "Val CER:\s*\K[\d.]+" "$log_file" | tail -1)
        
        # Get loss config name
        case $i in
            1) exp_name="Pixel Only" ;;
            2) exp_name="Pixel + Adv" ;;
            3) exp_name="Pixel + Adv + Perc" ;;
            4) exp_name="Pixel + Adv + Perc + CTC" ;;
            5) exp_name="FULL (+ RecFeat)" ;;
        esac
        
        echo "$status Exp $i: $exp_name"
        
        if [ -n "$current_epoch" ]; then
            echo "   Epoch: $current_epoch/$total_epochs"
            if [ -n "$latest_psnr" ]; then
                echo "   Latest: PSNR ${latest_psnr} dB, SSIM ${latest_ssim}, CER ${latest_cer}%"
            fi
        fi
        
        echo ""
    done
    
    echo "========================================="
    echo "Quick Commands:"
    echo "  - View Exp 1 log: tail -f logs/ablation_01_training.log"
    echo "  - Extract metrics: poetry run python scripts/extract_ablation_metrics.py"
    echo "  - Stop monitoring: Ctrl+C"
    echo "========================================="
    
    # Check if all experiments completed
    all_done=true
    for i in {1..5}; do
        log_file="$PROJECT_ROOT/logs/ablation_$(printf "%02d" $i)_training.log"
        if [ -f "$log_file" ]; then
            final_epoch=$(grep -c "Epoch 15/" "$log_file")
            if [ "$final_epoch" -eq 0 ]; then
                all_done=false
                break
            fi
        else
            all_done=false
            break
        fi
    done
    
    if [ "$all_done" = true ]; then
        echo ""
        echo "🎉 ALL EXPERIMENTS COMPLETED!"
        echo "🔍 Auto-extracting metrics..."
        cd "$PROJECT_ROOT"
        poetry run python "$EXTRACTOR"
        echo ""
        echo "✅ Results saved to: results/ablation_study/"
        break
    fi
    
    # Refresh every 30 seconds
    sleep 30
done
