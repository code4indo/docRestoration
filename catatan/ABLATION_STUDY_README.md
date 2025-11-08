# ABLATION STUDY - INCREMENTAL LOSS COMPONENT ANALYSIS

## 📊 Overview

Systematic ablation study untuk mengukur kontribusi individual setiap komponen loss terhadap performa model GAN-HTR.

**Status**: 🔄 **RUNNING** (Started: Nov 2, 2025 14:50 WIB)
**ETA Completion**: Nov 2, 2025 ~21:20 WIB (~6.5 hours)

## 🎯 Objectives

1. **Quantify individual contributions** dari setiap loss component
2. **Validate architecture design** dengan empirical evidence
3. **Support Q1 journal submission** dengan rigorous experimental data

## 🧪 Experimental Design

### Sequential Incremental Approach

| Exp | Loss Components | Weights | Expected Impact |
|-----|----------------|---------|-----------------|
| 1 | Pixel (L1) | 50 | Baseline reconstruction (25-28 dB) |
| 2 | + Adversarial | +3 | Texture realism (+1-2 dB) |
| 3 | + Perceptual (VGG) | +1 | Edge preservation (+0.5-1.5 dB) |
| 4 | + CTC | +0.15 | Text-awareness direct (+0.3-1 dB) |
| 5 | + RecFeat | +8 | Text-awareness indirect (+0.5-1.5 dB) |

### Training Configuration

```json
{
  "epochs": 15,
  "steps_per_epoch": 200,
  "batch_size": 2,
  "lr_g": 0.0002,
  "lr_d": 0.0002,
  "gpu_id": "1",
  "seed": 42
}
```

**Rationale for 15 epochs × 200 steps:**
- 200 steps = 400 images/epoch (batch_size=2)
- Total dataset ~4300 training images
- Coverage ~9.3% per epoch, ~1.4× full dataset over 15 epochs
- Sufficient untuk **trend isolation** dan **component contribution analysis**
- Efficient: ~75 min/experiment vs 4+ hours untuk full epochs

## 📁 File Structure

```
configs/
├── ablation_01_pixel_only.json          # Exp 1: L1 only
├── ablation_02_pixel_adv.json           # Exp 2: + Adversarial
├── ablation_03_pixel_adv_perc.json      # Exp 3: + Perceptual
├── ablation_04_pixel_adv_perc_ctc.json  # Exp 4: + CTC
└── ablation_05_full.json                # Exp 5: + RecFeat (FULL)

logs/
├── ablation_01_training.log
├── ablation_02_training.log
├── ablation_03_training.log
├── ablation_04_training.log
├── ablation_05_training.log
└── ablation_master.log                  # Sequence orchestration log

scripts/
├── run_ablation_sequence.sh             # Main launcher (sequential)
├── ablation_status.sh                   # Quick status check
├── monitor_ablation.sh                  # Live monitor (auto-extract when done)
└── extract_ablation_metrics.py          # Metrics extraction to JSON/LaTeX/MD

results/ablation_study/
├── ablation_metrics.json                # Raw results (auto-generated)
├── ablation_table.tex                   # LaTeX table for paper
└── ablation_report.md                   # Analysis report
```

## 🚀 Usage

### 1. Monitor Progress

```bash
# Quick status
./scripts/ablation_status.sh

# Live monitoring (auto-extracts when complete)
./scripts/monitor_ablation.sh

# View specific experiment log
tail -f logs/ablation_01_training.log
```

### 2. Extract Results (Manual)

```bash
cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration
poetry run python scripts/extract_ablation_metrics.py
```

**Output:**
- `results/ablation_study/ablation_metrics.json` - Raw data
- `results/ablation_study/ablation_table.tex` - Ready for paper
- `results/ablation_study/ablation_report.md` - Analysis with delta calculations

### 3. Update Paper

After extraction, copy LaTeX table from `results/ablation_study/ablation_table.tex` to paper:

```latex
% In Paper/main/jatniko_id.tex
% Replace placeholder table with actual results
\input{../../results/ablation_study/ablation_table.tex}
```

## 📊 Expected Results

### Hypothesis

```
Exp 1 (Pixel):           PSNR ~25-28 dB, CER ~35-45%
Exp 2 (+ Adv):          PSNR ~27-29 dB, CER ~32-42%
Exp 3 (+ Perc):         PSNR ~28-30 dB, CER ~30-38%
Exp 4 (+ CTC):          PSNR ~29-30.5 dB, CER ~28-35%
Exp 5 (+ RecFeat FULL): PSNR ~30-31 dB, CER ~26-29%
```

**Reference (Production V3 @ epoch 44):**
- PSNR: 30.91 dB
- SSIM: 0.9869
- CER: 27.11%

If ablation Exp 5 significantly underperforms, indicates need for extended training (50 epochs) for full convergence.

## 🔬 Scientific Justification

### Why Incremental Ablation?

1. **Isolate Contributions**: Each experiment adds ONE component → clear attribution
2. **Understand Synergies**: Measure interaction effects between components
3. **Validate Claims**: Empirical evidence > theoretical arguments
4. **Q1 Journal Standard**: Reviewers expect thorough ablation for novel architectures

### Key Research Questions

- **Q1**: Can pixel loss alone achieve acceptable restoration? (Exp 1)
- **Q2**: How much does adversarial training improve perceptual quality? (Exp 2 - Exp 1)
- **Q3**: Is perceptual loss redundant with adversarial? (Exp 3 - Exp 2)
- **Q4**: Does direct CTC gradient help or harm? (Exp 4 - Exp 3)
- **Q5**: Is RecFeat complementary or redundant with CTC? (Exp 5 - Exp 4)

## ⚠️ Important Notes

### CTC Loss Correction

**Previous Incorrect Claim**: "CTC loss hanya monitoring, tidak backprop"

**ACTUAL IMPLEMENTATION** (verified from `train_enhanced.py` lines 1259-1276):

```python
total_gen_loss = (
    (args.adv_loss_weight * adversarial_loss) + 
    (args.pixel_loss_weight * pixel_loss) + 
    (rec_feat_weight * rec_feat_loss) +
    (percep_weight * perceptual_loss) +
    (ctc_weight * ctc_loss)  # ← CTC IS BACKPROPAGATED!
)
generator_gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
generator_optimizer.apply_gradients(...)
```

✅ **CTC loss (weight=0.15) IS backpropagated** to generator
✅ Provides **direct text-awareness gradient signal**
✅ Complements **indirect signal** from RecFeat loss (weight=8.0)
✅ **Dual-modal text guidance** = CTC (direct) + RecFeat (indirect)

### Loss Function Formula

```
Total Generator Loss = 
    50.0 × L_pixel +       # Reconstruction fidelity
    3.0  × L_adv +         # Adversarial realism
    8.0  × L_recfeat +     # Recognition feature alignment (indirect HTR)
    1.0  × L_perc +        # VGG perceptual quality
    0.15 × L_ctc           # CTC text loss (direct HTR)
```

**Weight Rationale:**
- Pixel (50) dominates for stability
- RecFeat (8) provides strong text-awareness
- Adversarial (3) balanced for realism without artifacts
- Perceptual (1) supplements high-level features
- CTC (0.15) small but effective, gradient clipped at 400.0

## 📝 Paper Integration

### Ablation Section Draft

See `Paper/main/jatniko_id.tex` section:
- **Line ~2301**: `\subsubsection{Studi Ablasi Incremental}`
- **Table \ref{table_ablation_loss}**: Incremental results table
- **Analysis**: Component contribution breakdown
- **Justification**: Dual-modal text guidance strategy

### Key Points for Discussion

1. **Multi-component necessity**: Pixel alone insufficient (Exp 1 vs Exp 5)
2. **Adversarial value**: Texture quality vs potential artifacts
3. **Perceptual contribution**: VGG features for edge/structure
4. **Dual HTR guidance**: CTC (direct) + RecFeat (indirect) complementarity
5. **Weight optimization**: Empirical tuning based on ablation insights

## 🎯 Next Steps

1. ✅ **Monitor experiments** (~6.5 hours runtime)
2. ⏳ **Extract metrics** (auto or manual after completion)
3. 📊 **Update paper table** with actual results
4. 📝 **Write analysis section** interpreting incremental contributions
5. 🔬 **Compare with baselines** (Souibgui et al., other SOTA methods)
6. 📈 **Visualize results** (bar charts, contribution breakdown)

## 🆘 Troubleshooting

### If Training Fails

```bash
# Check specific experiment log
tail -100 logs/ablation_0X_training.log

# Check master sequence log
tail -100 logs/ablation_master.log

# Restart specific experiment
cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration
nohup ./scripts/universal_train_from_json.sh configs/ablation_0X_....json > logs/ablation_0X_training.log 2>&1 &
```

### If Results Below Expected

**Possible causes:**
1. **Limited epochs** (15 vs 50 in production) → Run extended version
2. **Limited steps** (200 vs ~600 full) → Increase steps_per_epoch
3. **Random seed variance** → Re-run with different seeds
4. **Hyperparameter mismatch** → Verify config matches production

**Solutions:**
- Run extended ablation: 25 epochs, 400 steps/epoch
- Run full production configs for comparison
- Ensemble multiple random seeds

## 📚 References

- Production V3: `configs/production_v3_academic_split_70_15_15.json`
- Training script: `dual_modal_gan/scripts/train_enhanced.py`
- Paper section: `Paper/main/jatniko_id.tex` (lines 2301+)
- Baseline paper: Souibgui et al. (2022) - Enhance to Read Better

---

**Last Updated**: November 2, 2025 14:55 WIB
**Status**: Experiment 1/5 running (Epoch 2/15)
**Maintained by**: ML Engineering Team
