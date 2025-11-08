# ANRI Pseudo-Labeling Fine-Tuning Pipeline

## 📋 Overview

Pipeline lengkap untuk **fine-tuning model** menggunakan **33 halaman dokumen ANRI** melalui teknik **pseudo-labeling** (self-supervised learning).

**Konsep**: Model yang sudah trained pada data sintetis digunakan untuk generate "pseudo ground-truth" pada dokumen real ANRI, kemudian model di-fine-tune menggunakan pseudo-GT tersebut untuk domain adaptation.

---

## 🎯 Tujuan

1. **Adapt model** dari synthetic paleographic data ke real ANRI documents
2. **Improve restoration quality** pada dokumen asli dari Arsip Nasional
3. **No manual labeling required** - fully automated pseudo-labeling
4. **Prove feasibility** dengan 33 pages, scale ke 1000+ pages nanti

---

## 📊 Dataset

- **Total pages**: 33 full-page ANRI documents (K66a, K66b collections)
- **Resolution**: ~3000×4700 px per page
- **Degradation**: HEAVY (mean intensity ~165, high noise)
- **Format**: JPEG
- **Split**: 28 pages training / 5 pages validation

### Patch Extraction
- **Patch size**: 256×256 px
- **Overlap**: 64 px (prevent edge artifacts)
- **Estimated patches**: ~11,385 total
- **After filtering**: ~7,969 high-quality patches (70%)

---

## 🔧 Pipeline Workflow

### Phase 1: Patch Extraction
```bash
poetry run python scripts/extract_patches_from_full_pages.py \
    --input_dir DokumenRusak/full_pages_ANRI \
    --output_dir DokumenRusak/anri_patches \
    --patch_size 256 \
    --overlap 64
```

**Output**:
- `DokumenRusak/anri_patches/train/degraded/` - Training degraded patches
- `DokumenRusak/anri_patches/val/degraded/` - Validation degraded patches
- `extraction_metadata.json` - Patch metadata

**Duration**: ~1-2 hours

---

### Phase 2: Pseudo-Label Generation
```bash
poetry run python scripts/generate_pseudo_labels_patches.py \
    --input_dir DokumenRusak/anri_patches \
    --checkpoint_dir dual_modal_gan/checkpoints/thin_stroke_preservation_v1_academic/best_model \
    --checkpoint_name ckpt-99 \
    --gpu_id 0
```

**Output**:
- `DokumenRusak/anri_patches/train/pseudo_gt/` - Pseudo ground-truth patches
- `DokumenRusak/anri_patches/val/pseudo_gt/` - Validation pseudo-GT
- `pseudo_label_metrics.json` - Quality metrics untuk filtering

**Duration**: ~6-8 hours (GPU dependent)

---

### Phase 3: Quality Filtering
```bash
poetry run python scripts/filter_pseudo_patches.py \
    --input_dir DokumenRusak/anri_patches \
    --output_dir DokumenRusak/anri_patches_filtered \
    --ssim_min 0.6 \
    --ssim_max 0.95 \
    --text_preservation_min 0.7 \
    --top_percentile 0.7
```

**Filtering Criteria**:
- SSIM range: [0.6, 0.95] (reasonable similarity)
- Text preservation ≥ 0.7 (avoid over-cleaning)
- Restored std ≥ 20 (not flat/over-cleaned)
- Keep top 70% by quality

**Output**:
- `DokumenRusak/anri_patches_filtered/train/` - Filtered training pairs
- `DokumenRusak/anri_patches_filtered/val/` - Filtered validation pairs
- `visual_inspection/` - Sample images for manual review
- `filtered_metrics.json` - Filtering statistics

**Duration**: ~2-3 hours (includes manual inspection)

---

### Phase 4: TFRecord Creation
```bash
poetry run python scripts/create_pseudo_tfrecord.py \
    --input_dir DokumenRusak/anri_patches_filtered \
    --output_dir DokumenRusak/anri_tfrecords
```

**Output**:
- `train_anri_pseudo.tfrecord` - Training dataset
- `val_anri_pseudo.tfrecord` - Validation dataset

**Duration**: ~1 hour

---

### Phase 5: Fine-Tuning (Visual-Only Mode)
```bash
nohup ./scripts/universal_train_from_json.sh \
    configs/finetune_anri_pseudo_visual_only.json \
    > logs/finetune_anri_pseudo_v1.log 2>&1 &
```

**Key Configuration** (`finetune_anri_pseudo_visual_only.json`):
```json
{
  "resume_from": "ckpt-99",
  "epochs": 15,
  "batch_size": 8,
  "lr_g": 1e-6,
  "lr_d": 1e-6,
  
  "ctc_loss_weight": 0.0,        // ❌ Disabled (no text labels)
  "rec_feat_loss_weight": 0.0,   // ❌ Disabled (no text labels)
  "pixel_loss_weight": 50.0,     // ✅ Main loss
  "perceptual_loss_weight": 1.0, // ✅ VGG loss
  "adv_loss_weight": 2.0,        // ✅ Adversarial
  
  "discriminator_mode": "predicted" // Use recognizer output
}
```

**Why Visual-Only?**
- ❌ No text transcription labels available
- ✅ Can still train using visual losses (pixel + perceptual + adversarial)
- ✅ Discriminator uses predicted text from frozen recognizer

**Duration**: ~8-12 hours (depends on GPU, early stopping)

---

### Phase 6: Evaluation
```bash
poetry run python scripts/evaluate_finetuned_on_anri.py \
    --val_pages_dir DokumenRusak/full_pages_ANRI \
    --baseline_checkpoint_dir dual_modal_gan/checkpoints/thin_stroke_preservation_v1_academic/best_model \
    --finetuned_checkpoint_dir dual_modal_gan/checkpoints/finetune_anri_pseudo_v1/best_model \
    --output_dir results/anri_finetuning_evaluation
```

**Metrics**:
- **No-reference quality**:
  - Contrast (std)
  - Sharpness (Laplacian variance)
  - Entropy
  - Text ratio
- **Comparison**: Baseline vs Fine-tuned
- **Visual**: Side-by-side comparisons

**Output**:
- `evaluation_results.json` - Quantitative metrics
- `comparisons/` - Visual comparison images (degraded | baseline | fine-tuned)

**Duration**: ~1-2 hours

---

## 🚀 Quick Start

### Option 1: Run Full Pipeline (Automated)
```bash
./scripts/run_anri_pseudo_labeling_pipeline.sh
```

**Notes**:
- Will run all 6 phases sequentially
- Pauses for manual inspection after filtering
- Training runs in background
- Total time: ~1.5-2 days

### Option 2: Run Step-by-Step (Manual)
```bash
# 1. Extract patches
poetry run python scripts/extract_patches_from_full_pages.py

# 2. Generate pseudo-labels
poetry run python scripts/generate_pseudo_labels_patches.py \
    --checkpoint_dir dual_modal_gan/checkpoints/thin_stroke_preservation_v1_academic/best_model \
    --checkpoint_name ckpt-99 \
    --gpu_id 0

# 3. Filter patches
poetry run python scripts/filter_pseudo_patches.py

# 4. Check visual inspection samples
ls DokumenRusak/anri_patches_filtered/visual_inspection/train/

# 5. Create TFRecord
poetry run python scripts/create_pseudo_tfrecord.py

# 6. Fine-tune
nohup ./scripts/universal_train_from_json.sh \
    configs/finetune_anri_pseudo_visual_only.json &

# 7. Monitor training
tail -f logs/finetune_anri_pseudo_v1.log

# 8. After training completes, evaluate
poetry run python scripts/evaluate_finetuned_on_anri.py
```

---

## 📈 Expected Results

### Conservative Estimates
- **Iteration 1**: 10-20% improvement in visual quality
- **Contrast**: Better adaptation to ANRI degradation
- **Over-cleaning**: Reduced (model learns real degradation patterns)

### Metrics to Track
| Metric | Baseline | Fine-tuned | Target |
|--------|----------|------------|--------|
| Contrast (std) | ~40 | ~45+ | +10% |
| Sharpness | ~800 | ~900+ | +10% |
| Text preservation | ~0.75 | ~0.85+ | +10% |

---

## ⚠️ Risk Mitigation

### 1. Overfitting (Small Dataset)
**Risk**: 33 pages → 8k patches masih bisa overfit  
**Mitigation**:
- ✅ Very low LR (1e-6)
- ✅ Few epochs (15 max)
- ✅ Early stopping (patience=5)
- ✅ Held-out validation (5 pages)

### 2. Catastrophic Forgetting
**Risk**: Model forgets synthetic knowledge  
**Mitigation**:
- ✅ Resume from ckpt-99 (transfer learning)
- ✅ Very low LR
- ✅ Short training duration

### 3. Pseudo-GT Errors
**Risk**: Model belajar dari pseudo-GT yang salah  
**Mitigation**:
- ✅ Quality filtering (SSIM, text preservation)
- ✅ Manual visual inspection (20 samples)
- ✅ Keep only top 70% quality

### 4. Domain Collapse
**Risk**: Model only works on ANRI style  
**Mitigation**:
- ✅ Validate on held-out pages
- ✅ Can mix synthetic data (optional)
- ✅ Test on other datasets later

---

## 📁 Output Structure

```
DokumenRusak/
├── full_pages_ANRI/                    # Input: 33 ANRI pages
├── anri_patches/                       # Phase 1 output
│   ├── train/degraded/                 # Training degraded patches
│   ├── train/pseudo_gt/                # Phase 2 output
│   ├── val/degraded/
│   ├── val/pseudo_gt/
│   ├── extraction_metadata.json
│   └── pseudo_label_metrics.json
├── anri_patches_filtered/              # Phase 3 output
│   ├── train/degraded/
│   ├── train/clean/                    # Filtered pseudo-GT
│   ├── val/degraded/
│   ├── val/clean/
│   ├── visual_inspection/              # Manual review samples
│   └── filtered_metrics.json
└── anri_tfrecords/                     # Phase 4 output
    ├── train_anri_pseudo.tfrecord
    └── val_anri_pseudo.tfrecord

dual_modal_gan/checkpoints/
└── finetune_anri_pseudo_v1/            # Phase 5 output
    ├── best_model/
    │   └── ckpt-best
    └── checkpoints/

results/
└── anri_finetuning_evaluation/         # Phase 6 output
    ├── comparisons/                    # Visual comparisons
    └── evaluation_results.json         # Metrics
```

---

## 🔍 Monitoring & Debugging

### Monitor Training
```bash
# Real-time logs
tail -f logs/finetune_anri_pseudo_v1.log

# Check GPU usage
nvidia-smi

# TensorBoard (if enabled)
tensorboard --logdir dual_modal_gan/checkpoints/finetune_anri_pseudo_v1
```

### Check Intermediate Results
```bash
# Patch extraction stats
cat DokumenRusak/anri_patches/extraction_metadata.json | jq '.statistics'

# Pseudo-label quality
cat DokumenRusak/anri_patches/pseudo_label_metrics.json | jq '.train_statistics'

# Filtering results
cat DokumenRusak/anri_patches_filtered/filtered_metrics.json | jq '.statistics'
```

### Troubleshooting

**Problem**: Too few patches after filtering  
**Solution**: Relax filtering criteria (lower `top_percentile` to 0.5)

**Problem**: Training loss not decreasing  
**Solution**: Check LR (might be too low), increase to 5e-6

**Problem**: Validation quality worse than baseline  
**Solution**: Overfitting! Stop training earlier, use ckpt with best val metric

---

## 🎓 Academic Contribution

### Publication Angle
**Title**: *Self-Supervised Domain Adaptation for Indonesian Paleographic Document Restoration via Pseudo-Labeling*

**Key Contributions**:
1. ✅ **No manual labeling required** - scalable approach
2. ✅ **First application** to Indonesian ANRI paleographic documents
3. ✅ **Patch-based amplification** strategy for small datasets
4. ✅ **Visual-only training** without text transcription

**Experiments**:
- Baseline: Model on synthetic only (ckpt-99)
- Ablation 1: Fine-tune on ANRI pseudo-labels (this pipeline)
- Ablation 2: Iterative refinement (iteration 2-3)
- Analysis: Quantitative metrics + visual quality assessment

**Expected Q1 Journal Acceptance**: HIGH (novel application, scalable methodology, real-world impact)

---

## 📝 Notes

### Current Status (33 Pages)
- ⚠️ **Borderline** dataset size
- ✅ **Proof of concept** - demonstrates feasibility
- 🎯 **Target**: Scale to 100-1000 pages for robust fine-tuning

### Future Work (1000 Pages)
When you provide 1000 pages:
- 🚀 Re-run pipeline with 700 train / 200 val / 100 test
- 🚀 Longer training (30-50 epochs)
- 🚀 Iterative refinement (3-5 iterations)
- 🚀 Multi-scale training (different patch sizes)
- 🚀 Ensemble models

### Iteration 2-3 (Optional)
```bash
# Use fine-tuned model to re-generate pseudo-GT
poetry run python scripts/generate_pseudo_labels_patches.py \
    --checkpoint_dir dual_modal_gan/checkpoints/finetune_anri_pseudo_v1/best_model \
    --checkpoint_name ckpt-best

# Re-filter and re-train
# Expected: Progressive improvement each iteration
```

---

## 🤝 Contact & Support

**Questions?** Check `catatan/` directory for detailed analysis notes.

**Issues?** Review `logs/` for error messages.

**Success?** Document results in `results/anri_finetuning_evaluation/` and prepare for publication! 🎉
