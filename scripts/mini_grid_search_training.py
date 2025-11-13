#!/usr/bin/env python3
"""
Mini Grid Search Training - Two-Stage Approach
Stage 1: Visual Quality (Pixel, Adv, Perc) - 27 configs
Stage 2: Recognition Impact (CTC, RecFeat) - 9 configs

KONFIGURASI:
- Epochs: 3 (quick convergence direction)
- Steps per epoch: 20 (subset data)
- Batch size: 4 (faster iteration)
- Est time per config: ~5-7 min
- Total: 36 configs × 6 min = ~3.6 hours
"""

import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import itertools

class MiniGridSearchTraining:
    def __init__(self):
        self.base_config_path = Path("configs/production_v3_academic_split_70_15_15.json")
        self.results_dir = Path("dual_modal_gan/docs/grid_search_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.base_config_path) as f:
            self.base_config = json.load(f)
        
        # Mini training settings
        self.mini_settings = {
            "epochs": 3,
            "steps_per_epoch": 20,
            "batch_size": 4,
            "save_interval": 10,
            "eval_interval": 1,
            "early_stopping": {"enabled": False},
            # Disable curriculum learning
            "curriculum_learning": False,
            "use_curriculum": False
        }
        
        # Parallel training on 2 GPUs (not distributed)
        self.num_parallel_jobs = 2
        self.gpu_devices = [0, 1]
        
        # OPSI C: Strategic Sampling - All 5 components tested simultaneously
        # Using Latin Hypercube-like distribution for better coverage
        # Total: 27 configs covering full loss weight space efficiently
        self.search_space = {
            'pixel_loss_weight': [25.0, 50.0, 100.0],           # Low, baseline, high
            'adv_loss_weight': [1.5, 3.0, 6.0],                 # 0.5×, 1×, 2× baseline
            'perceptual_loss_weight': [0.5, 1.0, 2.0],          # 0.5×, 1×, 2× baseline
            'ctc_loss_weight': [0.075, 0.15, 0.3],              # 0.5×, 1×, 2× baseline
            'rec_feat_loss_weight': [0.0, 8.0, 16.0]            # Disable, baseline, double
        }
        
        # Strategic combinations using orthogonal array principles
        # Instead of 3^5=243 configs, we use 27 strategically sampled configs
        self.strategic_configs = self._generate_strategic_combinations()
        self.results = []
    
    def _generate_strategic_combinations(self):
        """Generate 27 strategically sampled configs using balanced coverage"""
        import numpy as np
        
        # Strategy: Ensure each level appears equally often for each component
        # This gives better coverage than random or full factorial
        configs = []
        
        # Core strategy: 3×3×3 base grid on most impactful components (Pixel, Perc, CTC)
        # Then distribute Adv and RecFeat to maximize diversity
        
        base_grid = list(itertools.product(
            [0, 1, 2],  # Pixel indices
            [0, 1, 2],  # Perc indices
            [0, 1, 2]   # CTC indices
        ))
        
        # For each base config, assign Adv and RecFeat cyclically for balance
        for idx, (px_idx, perc_idx, ctc_idx) in enumerate(base_grid):
            # Cyclic assignment for Adv (pattern: 0,1,2,0,1,2,...)
            adv_idx = idx % 3
            # Cyclic assignment for RecFeat with offset (pattern: 0,1,2,1,2,0,2,0,1,...)
            recfeat_idx = (idx + idx // 3) % 3
            
            configs.append({
                'pixel_loss_weight': self.search_space['pixel_loss_weight'][px_idx],
                'adv_loss_weight': self.search_space['adv_loss_weight'][adv_idx],
                'perceptual_loss_weight': self.search_space['perceptual_loss_weight'][perc_idx],
                'ctc_loss_weight': self.search_space['ctc_loss_weight'][ctc_idx],
                'rec_feat_loss_weight': self.search_space['rec_feat_loss_weight'][recfeat_idx]
            })
        
        return configs
    
    def generate_configs(self):
        """Generate all 27 strategically sampled configs"""
        configs = []
        for idx, weights in enumerate(self.strategic_configs, 1):
            configs.append({
                'id': idx,
                'weights': weights
            })
        return configs
    
    def create_config_file(self, config_id, weights):
        """Create config file for training"""
        config = self.base_config.copy()
        config.update(self.mini_settings)
        config.update(weights)
        
        config['experiment_name'] = f"mini_grid_search_{config_id:02d}"
        config['checkpoint_dir'] = f"dual_modal_gan/checkpoints/mini_grid_search_{config_id:02d}"
        config['sample_dir'] = f"dual_modal_gan/outputs/mini_grid_search_{config_id:02d}"
        
        config['use_lr_schedule'] = False
        config['warmup_epochs'] = 0
        config['annealing_epochs'] = 0
        
        config_path = self.results_dir / f"config_{config_id:02d}.json"
        with open(config_path, 'w') as f:
            json.dump(config, indent=2, fp=f)
        
        return config_path
    
    def run_training(self, config_id, config_path, gpu_id):
        """Run training for one config on specific GPU"""
        log_path = self.results_dir / f"training_log_{config_id:02d}.txt"
        
        # Command with specific GPU
        cmd = [
            "poetry", "run",
            "python", "dual_modal_gan/scripts/train_enhanced.py",
            "--config", str(config_path)
        ]
        
        # Set environment for specific GPU
        import os
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        start_time = time.time()
        
        try:
            with open(log_path, 'w') as log_file:
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            return process, log_path, start_time
                
        except Exception as e:
            print(f"❌ Error starting config {config_id}: {e}")
            return None, log_path, start_time
    
    def wait_for_training(self, process, config_id, log_path, start_time, gpu_id):
        """Wait for training process to complete"""
        try:
            returncode = process.wait(timeout=600)
            elapsed = time.time() - start_time
            
            if returncode == 0:
                print(f"✅ Config {config_id} (GPU {gpu_id}) completed in {elapsed/60:.1f} min")
                return True, elapsed
            else:
                print(f"❌ Config {config_id} (GPU {gpu_id}) failed with code {returncode}")
                return False, elapsed
                
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"⏰ Config {config_id} (GPU {gpu_id}) timeout after 10 min")
            return False, 600
        except Exception as e:
            print(f"❌ Config {config_id} (GPU {gpu_id}) error: {e}")
            return False, time.time() - start_time
    
    def parse_results(self, config_id, log_path, weights):
        """Parse training results from log"""
        try:
            with open(log_path, 'r') as f:
                log_content = f.read()
            
            lines = log_content.split('\n')
            psnr, ssim, cer, wer = None, None, None, None
            
            for line in reversed(lines):
                if 'PSNR:' in line and psnr is None:
                    import re
                    match = re.search(r'PSNR:\s+([\d.]+)', line)
                    if match:
                        psnr = float(match.group(1))
                
                if 'CER:' in line and cer is None:
                    match = re.search(r'CER:\s+([\d.]+)', line)
                    if match:
                        cer = float(match.group(1))
                
                if 'SSIM:' in line and ssim is None:
                    match = re.search(r'SSIM:\s+([\d.]+)', line)
                    if match:
                        ssim = float(match.group(1))
                
                if 'WER:' in line and wer is None:
                    match = re.search(r'WER:\s+([\d.]+)', line)
                    if match:
                        wer = float(match.group(1))
                
                if all([psnr, ssim, cer, wer]):
                    break
            
            score = psnr - 0.2 * cer * 100 if psnr and cer else None
            
            return {
                'config_id': config_id,
                'pixel_weight': weights['pixel_loss_weight'],
                'adv_weight': weights['adv_loss_weight'],
                'perc_weight': weights['perceptual_loss_weight'],
                'ctc_weight': weights['ctc_loss_weight'],
                'recfeat_weight': weights['rec_feat_loss_weight'],
                'psnr': psnr,
                'ssim': ssim,
                'cer': cer,
                'wer': wer,
                'score': score,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"⚠️  Failed to parse: {e}")
            return {
                'config_id': config_id,
                'pixel_weight': weights['pixel_loss_weight'],
                'adv_weight': weights['adv_loss_weight'],
                'perc_weight': weights['perceptual_loss_weight'],
                'ctc_weight': weights['ctc_loss_weight'],
                'recfeat_weight': weights['rec_feat_loss_weight'],
                'psnr': None, 'ssim': None, 'cer': None, 'wer': None, 'score': None,
                'timestamp': datetime.now().isoformat()
            }
    
    def save_results(self):
        """Save and analyze results"""
        df = pd.DataFrame(self.results)
        
        csv_path = self.results_dir / 'grid_search_results_raw.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Raw results: {csv_path}")
        
        df_valid = df[df['score'].notna()].copy()
        if len(df_valid) == 0:
            print("⚠️  No valid results")
            return
        
        df_valid = df_valid.sort_values('score', ascending=False)
        sorted_path = self.results_dir / 'grid_search_results_sorted.csv'
        df_valid.to_csv(sorted_path, index=False)
        print(f"💾 Sorted results: {sorted_path}")
        
        # Print top 5
        print(f"\n{'='*70}")
        print(f"🏆 TOP 5 CONFIGURATIONS")
        print(f"{'='*70}")
        
        for idx, row in df_valid.head(5).iterrows():
            rank = df_valid.index.get_loc(idx) + 1
            print(f"\n#{rank}. Config {row['config_id']:02d} - Score: {row['score']:.2f}")
            print(f"   Weights: Px={row['pixel_weight']:.0f}, Adv={row['adv_weight']:.1f}, "
                  f"Perc={row['perc_weight']:.1f}, CTC={row['ctc_weight']:.2f}, RecF={row['recfeat_weight']:.1f}")
            print(f"   Metrics: PSNR={row['psnr']:.2f}, CER={row['cer']*100:.1f}%, SSIM={row['ssim']:.4f}")
        
        # Find baseline
        baseline = df_valid[
            (df_valid['pixel_weight'] == 50.0) &
            (df_valid['adv_weight'] == 3.0) &
            (df_valid['perc_weight'] == 1.0) &
            (df_valid['ctc_weight'] == 0.15) &
            (df_valid['recfeat_weight'] == 8.0)
        ]
        
        if not baseline.empty:
            baseline_row = baseline.iloc[0]
            baseline_rank = df_valid.index.get_loc(baseline.index[0]) + 1
            
            print(f"\n📊 BASELINE (Current Production)")
            print(f"   Rank: #{baseline_rank}/{len(df_valid)}")
            print(f"   Score: {baseline_row['score']:.2f}")
            print(f"   PSNR={baseline_row['psnr']:.2f}, CER={baseline_row['cer']*100:.1f}%")
            
            if baseline_rank == 1:
                print(f"   ✅ BASELINE IS OPTIMAL!")
            else:
                best = df_valid.iloc[0]
                improvement = ((best['score'] - baseline_row['score']) / abs(baseline_row['score'])) * 100
                print(f"   ⚠️  Potential improvement: {improvement:.1f}%")
                print(f"      Best: Px={best['pixel_weight']:.0f}, Adv={best['adv_weight']:.1f}, "
                      f"Perc={best['perc_weight']:.1f}, CTC={best['ctc_weight']:.2f}, RecF={best['recfeat_weight']:.1f}")
        
        # Analyze CTC impact specifically
        print(f"\n{'='*70}")
        print("🔍 CTC IMPACT ANALYSIS: Does reducing CTC improve visual quality?")
        print(f"{'='*70}")
        
        for ctc_val in [0.075, 0.15, 0.3]:
            ctc_configs = df_valid[df_valid['ctc_weight'] == ctc_val]
            if not ctc_configs.empty:
                best_ctc = ctc_configs.iloc[0]
                avg_psnr = ctc_configs['psnr'].mean()
                avg_cer = ctc_configs['cer'].mean()
                print(f"\nCTC = {ctc_val:.3f} ({len(ctc_configs)} configs):")
                print(f"   Best Score: {best_ctc['score']:.2f}")
                print(f"   Avg PSNR: {avg_psnr:.2f}, Avg CER: {avg_cer*100:.1f}%")
        
        # Analyze RecFeat impact
        print(f"\n{'='*70}")
        print("🔍 RecFeat IMPACT ANALYSIS")
        print(f"{'='*70}")
        
        for rf_val in [0.0, 8.0, 16.0]:
            rf_configs = df_valid[df_valid['recfeat_weight'] == rf_val]
            if not rf_configs.empty:
                best_rf = rf_configs.iloc[0]
                avg_psnr = rf_configs['psnr'].mean()
                avg_cer = rf_configs['cer'].mean()
                rf_label = "Disabled" if rf_val == 0 else f"{rf_val:.1f}"
                print(f"\nRecFeat = {rf_label} ({len(rf_configs)} configs):")
                print(f"   Best Score: {best_rf['score']:.2f}")
                print(f"   Avg PSNR: {avg_psnr:.2f}, Avg CER: {avg_cer*100:.1f}%")
        
        print(f"\n{'='*70}\n")
        
        self.generate_latex_table(df_valid)
    
    def generate_latex_table(self, df_valid):
        """Generate LaTeX table"""
        latex_path = self.results_dir / 'grid_search_top_results.tex'
        
        with open(latex_path, 'w') as f:
            f.write("% Mini Grid Search Results - Two-Stage Approach\n\n")
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\caption{Mini Grid Search: Top 10 Konfigurasi Loss Weights}\n")
            f.write("\\label{tab:mini-grid-search}\n")
            f.write("\\footnotesize\n")
            f.write("\\begin{tabular}{cccccccccc}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Rank} & \\textbf{Px} & \\textbf{Adv} & \\textbf{Perc} & "
                   "\\textbf{CTC} & \\textbf{RecF} & \\textbf{PSNR} & \\textbf{CER} & "
                   "\\textbf{Score} & \\textbf{Status} \\\\\n")
            f.write("\\hline\n")
            
            baseline_id = None
            for idx, row in df_valid.iterrows():
                if (row['pixel_weight'] == 50.0 and row['adv_weight'] == 3.0 and 
                    row['perc_weight'] == 1.0 and row['ctc_weight'] == 0.15 and 
                    row['recfeat_weight'] == 8.0):
                    baseline_id = row['config_id']
                    break
            
            for rank, (idx, row) in enumerate(df_valid.head(10).iterrows(), 1):
                status = ""
                if row['config_id'] == baseline_id:
                    status = "\\textbf{Baseline}"
                if rank == 1 and row['config_id'] != baseline_id:
                    status = "\\textbf{Best}"
                elif rank == 1:
                    status = "\\textbf{Best=Baseline}"
                
                f.write(f"{rank} & {row['pixel_weight']:.0f} & {row['adv_weight']:.1f} & "
                       f"{row['perc_weight']:.1f} & {row['ctc_weight']:.2f} & {row['recfeat_weight']:.1f} & "
                       f"{row['psnr']:.2f} & {row['cer']*100:.1f}\\% & {row['score']:.2f} & {status} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\multicolumn{10}{l}{\\footnotesize Strategic sampling: 27 configs covering all 5 components (Orthogonal Array)} \\\\\n")
            f.write("\\multicolumn{10}{l}{\\footnotesize Score = PSNR - 0.2×CER×100 (higher better)} \\\\\n")
    def run(self):
        """Main execution - Parallel training on 2 GPUs"""
        print("="*70)
        print("MINI GRID SEARCH - STRATEGIC SAMPLING (Opsi C)")
        print("="*70)
        print("\n✓ All 5 loss components tested simultaneously")
        print("✓ 27 strategically sampled configs (Orthogonal Array)")
        print("✓ 2 trainings running in PARALLEL (GPU 0 & GPU 1)")
        print("✓ Curriculum learning DISABLED")
        print("\nKey Question: Does reducing CTC improve visual quality?")
        print("="*70)
        
        configs = self.generate_configs()
        print(f"\nTotal configs: {len(configs)}")
        print(f"\nSearch space for each component:")
        print(f"  Pixel: {self.search_space['pixel_loss_weight']}")
        print(f"  Adversarial: {self.search_space['adv_loss_weight']}")
        print(f"  Perceptual: {self.search_space['perceptual_loss_weight']}")
        print(f"  CTC: {self.search_space['ctc_loss_weight']}")
        print(f"  RecFeat: {self.search_space['rec_feat_loss_weight']}")
        print(f"\n⏱️  Estimated time: {len(configs)/2} × 6 min = ~{len(configs)/2*6/60:.1f} hours")
        print(f"    (2× faster than sequential)")
        print()
        
        # Show first 5 configs as preview
        print("Preview (first 5 configs):")
        for i, cfg in enumerate(configs[:5], 1):
            w = cfg['weights']
            print(f"  {i}. Px={w['pixel_loss_weight']:.0f}, Adv={w['adv_loss_weight']:.1f}, "
                  f"Perc={w['perceptual_loss_weight']:.1f}, CTC={w['ctc_loss_weight']:.3f}, RecF={w['rec_feat_loss_weight']:.1f}")
        print()
        print("🚀 Starting parallel training...")
        print()
        
        start_time = time.time()
        config_idx = 0
        active_jobs = {}  # {gpu_id: (process, config_id, log_path, start_time)}
        
        while config_idx < len(configs) or active_jobs:
            # Start new jobs on available GPUs
            for gpu_id in self.gpu_devices:
                if gpu_id not in active_jobs and config_idx < len(configs):
                    config = configs[config_idx]
                    config_id = config['id']
                    weights = config['weights']
                    
                    print(f"\n🚀 Starting Config {config_id}/{len(configs)} on GPU {gpu_id}")
                    print(f"   Px={weights['pixel_loss_weight']:.0f}, Adv={weights['adv_loss_weight']:.1f}, "
                          f"Perc={weights['perceptual_loss_weight']:.1f}, CTC={weights['ctc_loss_weight']:.3f}, RecF={weights['rec_feat_loss_weight']:.1f}")
                    
                    config_path = self.create_config_file(config_id, weights)
                    process, log_path, train_start = self.run_training(config_id, config_path, gpu_id)
                    
                    if process:
                        active_jobs[gpu_id] = (process, config_id, log_path, train_start, weights)
                        config_idx += 1
            
            # Check completed jobs
            completed_gpus = []
            for gpu_id, (process, config_id, log_path, train_start, weights) in active_jobs.items():
                if process.poll() is not None:  # Process finished
                    success, elapsed = self.wait_for_training(process, config_id, log_path, train_start, gpu_id)
                    
                    result = self.parse_results(config_id, log_path, weights)
                    result['training_time'] = elapsed
                    result['success'] = success
                    result['gpu_id'] = gpu_id
                    self.results.append(result)
                    
                    completed_gpus.append(gpu_id)
                    
                    # Save intermediate results
                    if len(self.results) % 5 == 0:
                        self.save_results()
                    
                    print(f"Progress: {len(self.results)}/{len(configs)} ({len(self.results)/len(configs)*100:.1f}%)")
            
            # Remove completed jobs
            for gpu_id in completed_gpus:
                del active_jobs[gpu_id]
            
            # Small delay before checking again
            if active_jobs:
                time.sleep(2)
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"✅ ALL TRAININGS COMPLETED")
        print(f"   Total time: {total_time/3600:.2f} hours")
        print(f"   Average per config: {total_time/len(configs)/60:.1f} minutes")
        print(f"   Speedup from parallel: ~{len(configs)*6/(total_time/60):.1f}×")
        print(f"{'='*70}")
        
        self.save_results()


def main():
    searcher = MiniGridSearchTraining()
    searcher.run()


if __name__ == "__main__":
    main()
