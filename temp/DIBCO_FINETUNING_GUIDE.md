# DIBCO Fine-tuning Quick Start Guide

## 📋 Overview

Fine-tuning GAN model yang sudah trained pada synthetic dataset menggunakan real DIBCO documents untuk meningkatkan generalization pada degradasi dokumen historical yang sebenarnya.

**Status:** ✅ Ready to launch
**Dataset:** 152 DIBCO samples (exclude 2012 for test)
**Base Model:** thin_stroke_preservation_v1_academic (ckpt-99)
**Training Mode:** Visual-only (CTC loss = 0, no text labels)

---

## 🎯 What's Been Prepared

### 1. ✅ DIBCO Dataset Converted to TFRecord
- **Location:** `dual_modal_gan/data/dibco_finetuning.tfrecord`
- **Samples:** 152 image pairs (degraded + clean)
- **Excluded:** DIBCO 2012 (reserved for final test)
- **Distribution:**
  - 2009: 10 samples
  - 2010: 10 samples
  - 2011: 16 samples
  - 2013: 16 samples
  - 2014: 10 samples
  - 2016: 10 samples
  - 2017: 20 samples
  - 2018: 10 samples
  - PALM: 50 samples

### 2. ✅ Fine-tuning Config Created
- **Location:** `configs/dibco_finetuning_visual_only.json`
- **Key Settings:**
  - Pretrained checkpoint: `thin_stroke_preservation_v1_academic/best_model/ckpt-99`
  - CTC loss: 0.0 (disabled - no text labels)
  - Learning rate: 0.00005 (40x lower for fine-tuning)
  - Epochs: 20
  - Batch size: 4
  - Early stopping: PSNR-only, patience=8

### 3. ✅ Training Script Updated
- **Modified:** `dual_modal_gan/scripts/train_enhanced.py`
- **New Feature:** `--pretrained_checkpoint` argument
- **Behavior:** Loads weights from pretrained model but starts training from epoch 0 with fresh optimizer

---

## 🚀 How to Launch Fine-tuning

### Option 1: Using Universal Launcher (Recommended)

```bash
# Launch fine-tuning in background
nohup ./scripts/universal_train_from_json.sh \
    configs/dibco_finetuning_visual_only.json &

# Monitor progress
tail -f nohup.out

# Or monitor specific log (check nohup.out for path)
```

### Option 2: Direct Python Command

```bash
# Activate environment
cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration

# Launch training
nohup poetry run python dual_modal_gan/scripts/train_enhanced.py \
    --generator_version enhanced \
    --discriminator_version enhanced_v2_fixed \
    --tfrecord_path dual_modal_gan/data/dibco_finetuning.tfrecord \
    --charset_path real_data_preparation/real_data_charlist.txt \
    --recognizer_weights /home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5 \
    --pretrained_checkpoint dual_modal_gan/checkpoints/thin_stroke_preservation_v1_academic/best_model/ckpt-99 \
    --checkpoint_dir dual_modal_gan/checkpoints/dibco_finetuning_visual_only \
    --sample_dir dual_modal_gan/outputs/samples_dibco_finetuning \
    --gpu_id 0 \
    --epochs 20 \
    --batch_size 4 \
    --lr_g 0.00005 \
    --lr_d 0.00005 \
    --pixel_loss_weight 200.0 \
    --adv_loss_weight 1.5 \
    --rec_feat_loss_weight 0.0 \
    --ctc_loss_weight 0.0 \
    --perceptual_loss_weight 10.0 \
    --train_split 0.8 \
    --val_split 0.1 \
    --early_stopping \
    --early_stopping_metric psnr_only \
    --patience 8 \
    --save_best_model_separately \
    > /tmp/dibco_finetuning.log 2>&1 &

# Monitor
tail -f /tmp/dibco_finetuning.log
```

---

## 📊 What to Expect

### Training Duration
- **Epochs:** 20 (with early stopping, may finish earlier)
- **Steps per epoch:** ~30 steps (152 samples / batch_size 4 / train_split 0.8)
- **Estimated time:** ~1-2 hours (depending on GPU)

### Expected Metrics
- **Baseline (synthetic):** PSNR ~30 dB
- **Target (DIBCO):** PSNR >32 dB
- **Improvement:** Better handling of real degradation (bleed-through, fading, stains)

### Checkpoints Saved
- **Main:** `dual_modal_gan/checkpoints/dibco_finetuning_visual_only/ckpt-N`
- **Best:** `dual_modal_gan/checkpoints/dibco_finetuning_visual_only/best_model/ckpt-N`
- **Samples:** `dual_modal_gan/outputs/samples_dibco_finetuning/`

---

## 🔍 Monitoring Progress

### Check Training Log
```bash
# Real-time monitoring
tail -f /tmp/dibco_finetuning.log

# Search for PSNR improvements
grep "New best model" /tmp/dibco_finetuning.log

# Check epoch progress
grep "Epoch" /tmp/dibco_finetuning.log | tail -20
```

### Check Samples
```bash
# View generated samples
ls -lth dual_modal_gan/outputs/samples_dibco_finetuning/

# Latest samples
ls -t dual_modal_gan/outputs/samples_dibco_finetuning/ | head -10
```

### MLflow Dashboard
```bash
# Start MLflow UI
poetry run mlflow ui

# Open browser: http://localhost:5000
# Look for experiment: "dibco_finetuning_visual_only"
```

---

## ⚠️ Important Notes

### 1. Visual-Only Training
- **CTC loss = 0:** No text recognition training (DIBCO has no labels)
- **Focus:** Pixel-level restoration quality only
- **Metrics:** PSNR, SSIM (no CER/WER)

### 2. Fine-tuning Strategy
- **Low LR:** 0.00005 (40x lower than base training)
- **Rationale:** Prevent catastrophic forgetting of synthetic knowledge
- **Goal:** Gentle adaptation to real degradation patterns

### 3. Early Stopping
- **Metric:** PSNR only
- **Patience:** 8 epochs
- **Reason:** Real data has more variability, need patience

### 4. Test Set
- **DIBCO 2012:** Reserved for final evaluation (never seen during training)
- **Evaluation:** After training completes, evaluate on 2012 to measure generalization

---

## 🧪 After Training: Evaluation

### 1. Evaluate on Validation Set
```bash
poetry run python scripts/evaluate_psnr_comprehensive.py \
    --checkpoint_path dual_modal_gan/checkpoints/dibco_finetuning_visual_only/best_model/ckpt-N \
    --tfrecord_path dual_modal_gan/data/dibco_finetuning.tfrecord
```

### 2. Evaluate on Test Set (DIBCO 2012)
```bash
# First, convert DIBCO 2012 to TFRecord
poetry run python scripts/convert_dibco_to_tfrecord.py \
    --dibco_root real_data_preparation/dataset_dibco \
    --output_tfrecord dual_modal_gan/data/dibco_2012_test.tfrecord \
    --exclude_years 2009 2010 2011 2013 2014 2016 2017 2018 PALM \
    --verify

# Evaluate
poetry run python scripts/evaluate_psnr_comprehensive.py \
    --checkpoint_path dual_modal_gan/checkpoints/dibco_finetuning_visual_only/best_model/ckpt-N \
    --tfrecord_path dual_modal_gan/data/dibco_2012_test.tfrecord
```

---

## 📈 Success Criteria

✅ **Training Successful If:**
- PSNR on validation > 30 dB (matches or exceeds synthetic baseline)
- No overfitting (val PSNR doesn't degrade over epochs)
- Visual samples show improved handling of real degradation

✅ **Generalization Successful If:**
- PSNR on DIBCO 2012 test set > 30 dB
- Visual quality on test samples better than baseline
- Model handles unseen degradation patterns

---

## 🛠️ Troubleshooting

### Issue: OOM (Out of Memory)
**Solution:** Reduce batch size in config
```json
"batch_size": 2
```

### Issue: Training too slow
**Solution:** Reduce steps_per_epoch for faster iteration
```json
"steps_per_epoch": 20
```

### Issue: Model not improving
**Solution:** Increase learning rate (but be careful)
```json
"lr_g": 0.0001,
"lr_d": 0.0001
```

### Issue: Overfitting (val PSNR dropping)
**Solution:** Enable early stopping (already enabled)
- Check patience setting
- Reduce epochs

---

## 📝 Files Created

1. **Converter Script:** `scripts/convert_dibco_to_tfrecord.py`
2. **Dataset:** `dual_modal_gan/data/dibco_finetuning.tfrecord`
3. **Metadata:** `dual_modal_gan/data/dibco_finetuning.json`
4. **Config:** `configs/dibco_finetuning_visual_only.json`
5. **This Guide:** `DIBCO_FINETUNING_GUIDE.md`

---

## 🎓 Next Steps After Fine-tuning

1. **Evaluate on test set** (DIBCO 2012)
2. **Compare with baseline** (synthetic-only model)
3. **Inference on real documents** from Arsip Nasional
4. **Document findings** for research paper
5. **Iterate if needed** (adjust hyperparameters, try different checkpoints)

---

**Good luck with the fine-tuning! 🚀**

**Questions?** Check the logs, review config, or adjust hyperparameters as needed.
