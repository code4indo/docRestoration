#!/usr/bin/env python3
"""
Quick check hasil grid search yang sudah selesai
"""

import re
from pathlib import Path
import pandas as pd

results_dir = Path("dual_modal_gan/docs/grid_search_results")

print("="*70)
print("GRID SEARCH RESULTS CHECKER")
print("="*70)
print()

# Check for CSV results
csv_file = results_dir / "grid_search_results_raw.csv"
if csv_file.exists():
    print("📊 CSV Results found!")
    df = pd.read_csv(csv_file)
    print(f"   Configs completed: {len(df)}")
    print()
    print(df.to_string(index=False))
else:
    print("⏳ CSV results not yet generated (training in progress)")
    print()
    
    # Parse from training logs
    print("📝 Parsing training logs manually...")
    print()
    
    results = []
    
    for log_file in sorted(results_dir.glob("training_log_*.txt")):
        config_id = int(log_file.stem.split("_")[-1])
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Check if training complete
            if "Training Summary" in content or "Epoch 3 completed" in content:
                # Extract final metrics (last validation)
                psnr_matches = re.findall(r'PSNR:\s+([\d.]+)', content)
                cer_matches = re.findall(r'CER:\s+([\d.]+)', content)
                ssim_matches = re.findall(r'SSIM:\s+([\d.]+)', content)
                
                if psnr_matches and cer_matches:
                    psnr = float(psnr_matches[-1])
                    cer = float(cer_matches[-1])
                    ssim = float(ssim_matches[-1]) if ssim_matches else None
                    score = psnr - 0.2 * cer * 100
                    
                    results.append({
                        'config': config_id,
                        'psnr': psnr,
                        'cer': cer,
                        'ssim': ssim,
                        'score': score,
                        'status': '✅ Complete'
                    })
                else:
                    results.append({
                        'config': config_id,
                        'status': '⚠️ Complete but no metrics'
                    })
            else:
                # Check current epoch
                epoch_matches = re.findall(r'Epoch (\d+)/3', content)
                if epoch_matches:
                    current_epoch = max(map(int, epoch_matches))
                    results.append({
                        'config': config_id,
                        'status': f'🔄 Running (Epoch {current_epoch}/3)'
                    })
                else:
                    results.append({
                        'config': config_id,
                        'status': '🚀 Starting'
                    })
        
        except Exception as e:
            results.append({
                'config': config_id,
                'status': f'❌ Error: {e}'
            })
    
    # Display results
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        print()
        
        # Summary
        completed = df[df['status'].str.contains('Complete', na=False)]
        if len(completed) > 0:
            print(f"✅ Completed configs: {len(completed)}/27")
            if 'score' in completed.columns:
                best = completed.loc[completed['score'].idxmax()]
                print(f"🏆 Best so far: Config {int(best['config'])} - Score: {best['score']:.2f}")
                print(f"   PSNR: {best['psnr']:.2f}, CER: {best['cer']*100:.1f}%")
        else:
            print("⏳ No configs completed yet")
    else:
        print("  No training logs found")

print()
print("="*70)
print("💡 Tip: Run 'bash scripts/monitor_grid_search.sh' for live monitoring")
print("="*70)
