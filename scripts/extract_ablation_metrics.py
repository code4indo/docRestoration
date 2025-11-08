#!/usr/bin/env python3
"""
ABLATION METRICS EXTRACTOR
Automatically extracts best metrics from each ablation experiment log
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

class AblationMetricsExtractor:
    def __init__(self, project_root: str = "/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration"):
        self.project_root = Path(project_root)
        self.logs_dir = self.project_root / "logs"
        
    def extract_best_metrics(self, log_file: Path) -> Dict:
        """Extract best PSNR, SSIM, CER from a training log"""
        
        best_metrics = {
            'experiment': log_file.stem.replace('ablation_', '').replace('_training', ''),
            'best_psnr': 0.0,
            'best_ssim': 0.0,
            'best_cer': 100.0,
            'best_psnr_epoch': 0,
            'best_ssim_epoch': 0,
            'best_cer_epoch': 0,
            'final_epoch': 0
        }
        
        if not log_file.exists():
            print(f"⚠️  Log file not found: {log_file}")
            return best_metrics
        
        # Regex patterns
        epoch_pattern = r"Epoch (\d+)/\d+"
        psnr_pattern = r"Val PSNR:\s*([\d.]+)"
        ssim_pattern = r"Val SSIM:\s*([\d.]+)"
        cer_pattern = r"Val CER:\s*([\d.]+)%"
        
        current_epoch = 0
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # Track current epoch
                    epoch_match = re.search(epoch_pattern, line)
                    if epoch_match:
                        current_epoch = int(epoch_match.group(1))
                        best_metrics['final_epoch'] = current_epoch
                    
                    # Extract PSNR
                    psnr_match = re.search(psnr_pattern, line)
                    if psnr_match:
                        psnr = float(psnr_match.group(1))
                        if psnr > best_metrics['best_psnr']:
                            best_metrics['best_psnr'] = psnr
                            best_metrics['best_psnr_epoch'] = current_epoch
                    
                    # Extract SSIM
                    ssim_match = re.search(ssim_pattern, line)
                    if ssim_match:
                        ssim = float(ssim_match.group(1))
                        if ssim > best_metrics['best_ssim']:
                            best_metrics['best_ssim'] = ssim
                            best_metrics['best_ssim_epoch'] = current_epoch
                    
                    # Extract CER
                    cer_match = re.search(cer_pattern, line)
                    if cer_match:
                        cer = float(cer_match.group(1))
                        if cer < best_metrics['best_cer']:
                            best_metrics['best_cer'] = cer
                            best_metrics['best_cer_epoch'] = current_epoch
                            
        except Exception as e:
            print(f"❌ Error reading {log_file}: {e}")
        
        return best_metrics
    
    def get_loss_config(self, experiment_num: int) -> Dict:
        """Get loss configuration for each experiment"""
        configs = {
            1: {
                'name': 'Pixel Only',
                'components': ['L1'],
                'pixel': 50.0, 'adv': 0.0, 'rec_feat': 0.0, 'ctc': 0.0, 'perc': 0.0
            },
            2: {
                'name': 'Pixel + Adversarial',
                'components': ['L1', 'Adv'],
                'pixel': 50.0, 'adv': 3.0, 'rec_feat': 0.0, 'ctc': 0.0, 'perc': 0.0
            },
            3: {
                'name': 'Pixel + Adv + Perceptual',
                'components': ['L1', 'Adv', 'VGG'],
                'pixel': 50.0, 'adv': 3.0, 'rec_feat': 0.0, 'ctc': 0.0, 'perc': 1.0
            },
            4: {
                'name': 'Pixel + Adv + Perc + CTC',
                'components': ['L1', 'Adv', 'VGG', 'CTC'],
                'pixel': 50.0, 'adv': 3.0, 'rec_feat': 0.0, 'ctc': 0.15, 'perc': 1.0
            },
            5: {
                'name': 'Full (+ RecFeat)',
                'components': ['L1', 'Adv', 'VGG', 'CTC', 'RecFeat'],
                'pixel': 50.0, 'adv': 3.0, 'rec_feat': 8.0, 'ctc': 0.15, 'perc': 1.0
            }
        }
        return configs.get(experiment_num, {})
    
    def extract_all_experiments(self) -> List[Dict]:
        """Extract metrics from all 5 ablation experiments"""
        results = []
        
        for i in range(1, 6):
            log_file = self.logs_dir / f"ablation_{i:02d}_training.log"
            print(f"\n📊 Processing Experiment {i}/5: {log_file.name}")
            
            metrics = self.extract_best_metrics(log_file)
            config = self.get_loss_config(i)
            
            # Merge metrics and config
            result = {**metrics, **config}
            results.append(result)
            
            # Print summary
            if metrics['best_psnr'] > 0:
                print(f"   ✅ Best PSNR: {metrics['best_psnr']:.2f} dB (epoch {metrics['best_psnr_epoch']})")
                print(f"   ✅ Best SSIM: {metrics['best_ssim']:.4f} (epoch {metrics['best_ssim_epoch']})")
                print(f"   ✅ Best CER: {metrics['best_cer']:.2f}% (epoch {metrics['best_cer_epoch']})")
            else:
                print(f"   ⏳ Training in progress or not started (epoch {metrics['final_epoch']}/15)")
        
        return results
    
    def generate_latex_table(self, results: List[Dict]) -> str:
        """Generate LaTeX table for paper"""
        
        latex = r"""
\begin{table}[htbp]
\centering
\caption{Studi Ablasi Incremental: Kontribusi Setiap Komponen Loss}
\label{tab:ablation_incremental}
\begin{tabular}{clccc}
\toprule
\textbf{Exp} & \textbf{Komponen Loss} & \textbf{PSNR (dB)} & \textbf{SSIM} & \textbf{CER (\%)} \\
\midrule
"""
        
        for i, r in enumerate(results, 1):
            components = ' + '.join(r.get('components', []))
            psnr = r['best_psnr']
            ssim = r['best_ssim']
            cer = r['best_cer']
            
            # Format values
            psnr_str = f"{psnr:.2f}" if psnr > 0 else "---"
            ssim_str = f"{ssim:.4f}" if ssim > 0 else "---"
            cer_str = f"{cer:.2f}" if cer < 100 else "---"
            
            # Highlight best (exp 5)
            if i == 5 and psnr > 0:
                latex += f"{i} & \\textbf{{{components}}} & \\textbf{{{psnr_str}}} & \\textbf{{{ssim_str}}} & \\textbf{{{cer_str}}} \\\\\n"
            else:
                latex += f"{i} & {components} & {psnr_str} & {ssim_str} & {cer_str} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textbf{L1}: Pixel reconstruction loss (MAE)
\item \textbf{Adv}: Adversarial loss untuk realisme visual
\item \textbf{VGG}: Perceptual loss (VGG19 features)
\item \textbf{CTC}: Connectionist Temporal Classification loss (text-aware)
\item \textbf{RecFeat}: Recognition feature loss dari diskriminator
\end{tablenotes}
\end{table}
"""
        return latex
    
    def generate_markdown_report(self, results: List[Dict]) -> str:
        """Generate markdown report"""
        
        md = "# ABLATION STUDY RESULTS\n\n"
        md += "## Incremental Loss Component Analysis\n\n"
        md += "| Exp | Loss Components | PSNR (dB) | SSIM | CER (%) | Config |\n"
        md += "|-----|----------------|-----------|------|---------|--------|\n"
        
        for i, r in enumerate(results, 1):
            components = ' + '.join(r.get('components', []))
            psnr = f"{r['best_psnr']:.2f}" if r['best_psnr'] > 0 else "---"
            ssim = f"{r['best_ssim']:.4f}" if r['best_ssim'] > 0 else "---"
            cer = f"{r['best_cer']:.2f}" if r['best_cer'] < 100 else "---"
            
            # Loss weights
            weights = f"L1:{r['pixel']}, Adv:{r['adv']}, VGG:{r['perc']}, CTC:{r['ctc']}, RecFeat:{r['rec_feat']}"
            
            md += f"| {i} | {components} | {psnr} | {ssim} | {cer} | {weights} |\n"
        
        md += "\n## Analysis\n\n"
        
        # Calculate improvements
        if len(results) >= 5 and results[0]['best_psnr'] > 0 and results[4]['best_psnr'] > 0:
            baseline_psnr = results[0]['best_psnr']
            full_psnr = results[4]['best_psnr']
            improvement = full_psnr - baseline_psnr
            
            md += f"### Overall Improvement\n"
            md += f"- **Baseline (Pixel Only)**: PSNR {baseline_psnr:.2f} dB, CER {results[0]['best_cer']:.2f}%\n"
            md += f"- **Full Model**: PSNR {full_psnr:.2f} dB, CER {results[4]['best_cer']:.2f}%\n"
            md += f"- **Total Improvement**: +{improvement:.2f} dB PSNR, -{results[0]['best_cer']-results[4]['best_cer']:.2f}% CER\n\n"
            
            md += "### Incremental Contributions\n"
            for i in range(1, len(results)):
                if results[i]['best_psnr'] > 0 and results[i-1]['best_psnr'] > 0:
                    delta_psnr = results[i]['best_psnr'] - results[i-1]['best_psnr']
                    delta_cer = results[i-1]['best_cer'] - results[i]['best_cer']
                    added_component = list(set(results[i]['components']) - set(results[i-1]['components']))
                    
                    md += f"- **Adding {added_component[0] if added_component else 'component'}**: "
                    md += f"PSNR {delta_psnr:+.2f} dB, CER {delta_cer:+.2f}%\n"
        
        return md
    
    def save_results(self, results: List[Dict], output_dir: Path = None):
        """Save results to JSON, LaTeX, and Markdown"""
        
        if output_dir is None:
            output_dir = self.project_root / "results" / "ablation_study"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_file = output_dir / "ablation_metrics.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Saved JSON: {json_file}")
        
        # Save LaTeX table
        latex_file = output_dir / "ablation_table.tex"
        with open(latex_file, 'w') as f:
            f.write(self.generate_latex_table(results))
        print(f"💾 Saved LaTeX: {latex_file}")
        
        # Save Markdown report
        md_file = output_dir / "ablation_report.md"
        with open(md_file, 'w') as f:
            f.write(self.generate_markdown_report(results))
        print(f"💾 Saved Markdown: {md_file}")

def main():
    print("=" * 60)
    print("  ABLATION STUDY METRICS EXTRACTOR")
    print("=" * 60)
    
    extractor = AblationMetricsExtractor()
    results = extractor.extract_all_experiments()
    extractor.save_results(results)
    
    print("\n" + "=" * 60)
    print("  ✅ EXTRACTION COMPLETE")
    print("=" * 60)
    print("\n📊 View results in: results/ablation_study/")
    print("   - ablation_metrics.json (raw data)")
    print("   - ablation_table.tex (for paper)")
    print("   - ablation_report.md (analysis)")

if __name__ == "__main__":
    main()
