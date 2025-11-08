# Fine-tuning Guide: Thin Stroke Preservation Model

## 📋 Overview

Fine-tuning `thin_stroke_preservation_v1_academic` model dengan dokumen ANRI real untuk meningkatkan performa pada degradasi dokumen kuno spesifik.

## 🎯 Strategi Fine-tuning

### Dataset Creation
- **Source**: 1 gambar full-page dari ANRI (ID-ANRI_K66b_082_0064)
- **Dimension**: 4013 x 2735 px
- **Strips**: 31 horizontal strips @ 128px height × 1024px width
- **Augmentation**: 5x (original + flip + brightness variations + contrast)
- **Total samples**: ~155 samples (31 strips × 5 augmentations)

### Split Strategy (ACADEMIC PROTOCOL)
- **Sequential split** (NO random!) untuk mencegah data leakage:
  - **Train (70%)**: Top strips (0-70%) → ~108 samples
  - **Val (15%)**: Middle strips (70-85%) → ~23 samples  
  - **Test (15%)**: Bottom strips (85-100%) → ~24 samples

**⚠️ CRITICAL**: Sequential split memastikan tidak ada korelasi antara train/val/test karena strips yang bersebelahan memiliki karakteristik visual yang mirip.

### Training Configuration

**Base Model**:
- Checkpoint: `thin_stroke_preservation_v1_academic/best_model/ckpt-99`
- Performance: TBD (check training logs)

**Fine-tuning Parameters**:
```json
{
  "lr_g": 0.00001,              // 10x reduction (2e-4 → 1e-5)
  "lr_d": 0.00001,              // Gentle adaptation
  "epochs": 15,                 // Minimal to prevent catastrophic forgetting
  "warmup_epochs": 0,           // Skip (model already converged)
  "annealing_epochs": 0,        // Skip (model already converged)
  "early_stopping_patience": 8, // Reduced (prevent overfitting)
  "early_stopping_metric": "psnr_only" // Focus on visual quality
}
```

**Loss Weights** (inherited from base model):
```json
{
  "pixel_loss_weight": 200.0,      // STRONG preservation
  "adv_loss_weight": 1.5,          // Balanced realism
  "perceptual_loss_weight": 10.0,  // Stroke topology
  "rec_feat_loss_weight": 5.0,     // HTR-aware
  "ctc_loss_weight": 0.15          // Text readability
}
```

## 🚀 Usage

### Step 1: Prepare Images

Letakkan pasangan gambar di:
```
DokumenRusak/manual_restoration/
├── gt/                          # Ground truth (clean)
│   └── ID-ANRI_K66b_082_0064.png
└── deg/                         # Degraded
    └── ID-ANRI_K66b_082_0064.jpg
```

### Step 2: Create Dataset & Launch Training

```bash
# All-in-one launcher
./scripts/launch_finetuning.sh
```

Script akan:
1. ✅ Create TFRecord dataset dari strips
2. ✅ Verify pretrained checkpoint
3. ✅ Launch training in background
4. ✅ Save logs to `logs/finetuning/`

**Manual Alternative**:
```bash
# Create dataset only
poetry run python scripts/create_finetuning_strips.py \
    --gt_dir DokumenRusak/manual_restoration/gt \
    --deg_dir DokumenRusak/manual_restoration/deg \
    --output_dir dual_modal_gan/data/finetuning \
    --augment

# Launch training
nohup ./scripts/universal_train_from_json.sh \
    configs/thin_stroke_preservation_v1_finetuning.json \
    > logs/finetuning/finetuning_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Step 3: Monitor Training

```bash
# View live logs
tail -f logs/finetuning/finetuning_*.log

# Check checkpoint progress
ls -lh dual_modal_gan/checkpoints/thin_stroke_preservation_v1_finetuning/
```

## 📊 Expected Behavior

### Training Dynamics
- **Early epochs (1-3)**: Rapid PSNR improvement as model adapts to real degradation
- **Middle epochs (4-8)**: PSNR plateau, fine-grained adaptation
- **Late epochs (9-15)**: Risk of overfitting (early stopping should trigger)

### Success Metrics
- **PSNR**: Should improve by 1-3 dB on validation set
- **Visual quality**: Better handling of real ink bleed-through patterns
- **Thin strokes**: Maintain >85% preservation rate from base model

### Failure Signals
⚠️ **Overfitting**:
- Val PSNR decreases while train PSNR increases
- Generated images show "memorization" artifacts
- **Action**: Restore best weights, reduce epochs

⚠️ **Catastrophic Forgetting**:
- Val PSNR much worse than base model
- Model "forgets" how to handle synthetic degradation
- **Action**: Reduce LR further, add synthetic data mixing

## 🎓 Academic Considerations

### Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Overfitting** | HIGH | Sequential split + Early stopping (patience=8) + Augmentation |
| **Data Leakage** | NONE | Sequential split (not random) |
| **Catastrophic Forgetting** | MEDIUM | Low LR (1e-5) + Minimal epochs (15) |
| **Poor Generalization** | HIGH | Only 1 document - expect domain-specific adaptation |

### Evaluation Protocol

1. **During Training**: Monitor val_psnr (PSNR-only metric)
2. **After Training**: 
   - Test on **test split** (strips 85-100%, NEVER seen during training)
   - Test on **original synthetic test set** (check catastrophic forgetting)
   - Visual inspection of generated samples

### Publication Notes

If using for research paper:
- ✅ **Report sequential split strategy** (prevents data leakage accusation)
- ✅ **Acknowledge limited dataset** (1 document, domain-specific)
- ✅ **Compare base model vs fine-tuned** (show improvement on real data)
- ⚠️ **Do NOT claim generalization** (only 1 document, not representative)

## 📁 Output Structure

```
dual_modal_gan/
├── data/finetuning/
│   ├── finetuning_train.tfrecord
│   ├── finetuning_val.tfrecord
│   ├── finetuning_test.tfrecord
│   └── finetuning_metadata.json
├── checkpoints/thin_stroke_preservation_v1_finetuning/
│   ├── ckpt-*                  # Checkpoints every epoch
│   ├── best_model/             # Best model (separate)
│   ├── epoch_info.json         # Resume info
│   └── metrics/                # JSON metrics
└── outputs/samples_thin_stroke_preservation_v1_finetuning/
    └── epoch_*.png             # Sample outputs every 3 epochs
```

## 🔧 Troubleshooting

### "Pretrained checkpoint not found"
```bash
# Check checkpoint exists
ls dual_modal_gan/checkpoints/thin_stroke_preservation_v1_academic/best_model/

# Update config if checkpoint name differs
# Edit: configs/thin_stroke_preservation_v1_finetuning.json
# Field: "pretrained_checkpoint"
```

### "No image pairs found"
```bash
# Check images exist
ls DokumenRusak/manual_restoration/gt/
ls DokumenRusak/manual_restoration/deg/

# Ensure filenames match (same stem)
# GT:  ID-ANRI_K66b_082_0064.png
# DEG: ID-ANRI_K66b_082_0064.jpg  (OK, extension dapat berbeda)
```

### "Training crashes immediately"
```bash
# Check GPU memory
nvidia-smi

# Reduce batch size if OOM
# Edit config: "batch_size": 1

# Check dataset created correctly
ls -lh dual_modal_gan/data/finetuning/
```

## 🎯 Next Steps

After fine-tuning completes:

1. **Evaluate on test set** (strips 85-100%)
2. **Compare with base model** (quantitative + visual)
3. **Test on other ANRI documents** (generalization check)
4. **If successful**: Repeat with more documents for robust fine-tuning
5. **If overfitting**: Reduce epochs, increase regularization, collect more data

---

**Created**: 2025-10-30  
**Base Model**: thin_stroke_preservation_v1_academic (ckpt-99)  
**Dataset**: 1 ANRI document (ID-ANRI_K66b_082_0064)  
**Strategy**: Sequential split + Low LR + Minimal epochs
