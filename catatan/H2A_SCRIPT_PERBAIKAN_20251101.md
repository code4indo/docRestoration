# H2A Experiment - Perbaikan Script & Implementation

**Tanggal**: 2025-11-01  
**Status**: ✅ Script Diperbaiki, Siap Eksekusi  
**Estimasi Waktu**: ~13 jam (vs 24-48 jam versi lama)

---

## 🚨 Masalah yang Ditemukan di Script Original

### 1. **INEFFICIENCY KRITIS** - Training Ulang Model yang Sudah Ada
- ❌ Script original akan train dual-modal FROM SCRATCH
- ❌ Membuang model production_v3 yang sudah 96 epochs
- ❌ Total waktu: 24-48 jam untuk train both models
- 💰 Biaya komputasi: SANGAT TINGGI

### 2. **ANALISIS STATISTIK TIDAK AKAN BEKERJA**
- ❌ Script `h2a_statistical_analysis.py` mengharapkan per-sample CER
- ❌ Tapi training hanya simpan mean CER per epoch
- ❌ Tidak ada data untuk paired t-test yang proper

### 3. **MISSING COMPONENTS**
- ❌ Tidak ada script untuk extract per-sample CER dari model
- ❌ Import `Concatenate` hilang di `discriminator_single_modal.py`
- ❌ Tidak ada poetry run prefix (critical untuk venv)

### 4. **TIMING ESTIMATE SALAH**
- ❌ Script claim: ~6 hours per model
- ✅ Reality: 12-24 hours per model
- ❌ Total underestimate: ~50%

---

## ✅ Solusi yang Diimplementasikan

### 1. **EFFICIENT WORKFLOW** - Gunakan Model yang Ada
```bash
# OLD: Train both from scratch (24-48h)
./scripts/launch_h2a_experiment.sh

# NEW: Use existing dual-modal, train only single-modal (13h)
./scripts/h2a_experiment_efficient.sh
```

**Time Saved**: 12-24 hours! 🎉

### 2. **NEW EVALUATION SCRIPT** - Per-Sample CER Extraction
```bash
# Extract individual CER values (n=710) untuk statistical analysis
poetry run python scripts/h2a_evaluate_model.py \
    --checkpoint <path> \
    --discriminator_type <dual_modal|single_modal> \
    --output <cer_results.json>
```

**Features**:
- ✅ Extract CER untuk setiap sample di validation set
- ✅ Save sebagai JSON dengan metadata lengkap
- ✅ Support both dual-modal & single-modal discriminators
- ✅ Progress tracking every 50 samples

### 3. **IMPROVED STATISTICAL ANALYSIS** - V2 Script
```bash
# Analisis statistik dari pre-extracted CER JSON
poetry run python scripts/h2a_statistical_analysis_v2.py \
    --dual_modal_cer dual_modal_cer.json \
    --single_modal_cer single_modal_cer.json \
    --output_dir h2a_results/
```

**Improvements**:
- ✅ Load dari JSON files (bukan dari metrics di checkpoint)
- ✅ One-tailed paired t-test (dual < single)
- ✅ Proper Cohen's d calculation
- ✅ Statistical power estimation
- ✅ Comprehensive 6-panel visualization
- ✅ Publication-ready output

### 4. **BUG FIXES**
- ✅ Added `Concatenate` import to `discriminator_single_modal.py`
- ✅ Added `poetry run` prefix to all commands
- ✅ Fixed timing estimates
- ✅ Added progress monitoring commands

---

## 📊 New Workflow (4 Phases)

### Phase 1: Evaluate Dual-Modal (~30 min)
- Use existing `production_v3_academic_split_70_15_15`
- Extract per-sample CER (n=710)
- Output: `dual_modal_cer.json`

### Phase 2: Train Single-Modal (~12 hours)
- Train from scratch dengan same hyperparameters
- CNN-only discriminator (no LSTM, no text processing)
- Fair comparison: same params (~19M), same data

### Phase 3: Evaluate Single-Modal (~30 min)
- Extract per-sample CER dari trained model
- Output: `single_modal_cer.json`

### Phase 4: Statistical Analysis (~10 min)
- Load CER data dari JSON files
- Paired t-test & effect size
- Generate visualization & report
- Hypothesis validation

**Total Time: ~13 hours** (vs 24-48h original)

---

## 🎯 Expected Hypothesis Validation

### Criteria untuk H2A SUPPORTED:
1. ✅ p < 0.05 (statistical significance)
2. ✅ |Cohen's d| > 0.5 (medium effect size)
3. ✅ d < 0 (dual-modal has LOWER CER than single-modal)

### Output yang Akan Dihasilkan:
```
🏁 HYPOTHESIS CONCLUSION
═══════════════════════════════════════════
✅ H2A: FULLY SUPPORTED

✓ Statistical significance: p = 0.00XX < 0.05
✓ Effect size: medium (d = -0.XX)
✓ Direction: Dual-modal has LOWER CER ✅
✓ Practical impact: XX% CER reduction

CONCLUSION:
  Dual-Modal discriminator (CNN+LSTM) significantly
  outperforms Single-Modal (CNN-only) for HTR tasks.
```

---

## 📁 Files Created/Modified

### New Files:
1. `scripts/h2a_experiment_efficient.sh` - Main launcher (EFFICIENT)
2. `scripts/h2a_evaluate_model.py` - Per-sample CER extraction
3. `scripts/h2a_statistical_analysis_v2.py` - Statistical analysis improved
4. `catatan/H2A_EXPERIMENT_IMPLEMENTATION_GUIDE.md` - Complete guide

### Modified Files:
1. `dual_modal_gan/src/models/discriminator_single_modal.py` - Added Concatenate import

### Deprecated (DO NOT USE):
1. `scripts/launch_h2a_experiment.sh` - Original inefficient version
2. `scripts/h2a_statistical_analysis.py` - V1 (won't work without per-sample CER)
3. `configs/h2a_dual_modal_experiment.json` - Not needed (use existing model)

---

## 🚀 How to Execute

### Step 1: Verify Prerequisites
```bash
# Check existing dual-modal checkpoint
ls -lh dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/

# Check config
cat configs/h2a_single_modal_experiment.json | grep -E "experiment_name|discriminator_version|epochs"

# Verify GPU availability
nvidia-smi
```

### Step 2: Launch Experiment
```bash
# Activate environment (if not already)
cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration

# Run efficient workflow
./scripts/h2a_experiment_efficient.sh
```

### Step 3: Monitor Progress
```bash
# Watch experiment directory creation
watch -n 30 ls -lh h2a_experiment_*/

# Monitor single-modal training
tail -f h2a_experiment_*/logs/phase2_single_modal_training.log

# Check GPU usage
watch -n 5 nvidia-smi
```

### Step 4: Review Results
```bash
# After completion (~13 hours)
cd h2a_experiment_<timestamp>/results/

# View report
cat h2a_analysis_report_*.json | jq '.results.hypothesis_conclusion'

# View visualization
xdg-open h2a_statistical_analysis.png
```

---

## 📈 Cost-Benefit Analysis

### Old Approach (launch_h2a_experiment.sh):
- Time: 24-48 hours
- GPU Hours: 48-96 hours
- Resource Waste: Train model yang sudah ada ulang
- Complexity: Same

### New Approach (h2a_experiment_efficient.sh):
- Time: ~13 hours
- GPU Hours: ~13 hours
- Resource Optimization: Reuse existing model
- Complexity: Same + better data extraction

**Savings**: 
- ⏱️ **11-35 hours** of time
- 💰 **35-83 GPU hours** of cost
- 🌍 **Lower carbon footprint**
- 📊 **Better statistical validity** (per-sample CER)

---

## ✅ Validation Checks

Sebelum dianggap selesai, pastikan:

- [x] Script efficient created dan executable
- [x] Evaluation script tested (syntax)
- [x] Statistical analysis V2 ready
- [x] Import errors fixed (Concatenate)
- [x] Documentation complete
- [x] Old H2A checkpoints cleaned up
- [ ] **EXECUTION** - Jalankan experiment (~13 hours)
- [ ] **VERIFICATION** - Review hypothesis results
- [ ] **PUBLICATION** - Update paper dengan findings

---

## 🎓 Lessons Learned

1. **ALWAYS check if models already exist** before planning re-training
2. **Per-sample metrics > Aggregate metrics** untuk statistical analysis
3. **One-tailed tests** when hypothesis has specific direction
4. **Effect size matters** as much as p-value
5. **Time is money** - optimize before executing

---

## 🔜 Next Steps

1. **Execute H2A experiment** dengan efficient script
2. **Analyze results** - apakah hypothesis supported?
3. **Update paper** dengan empirical evidence
4. **Prepare publication figures** dari visualization
5. **Plan H2B & H2C** experiments jika H2A berhasil

---

**Decision**: ✅ READY TO EXECUTE  
**Command**: `./scripts/h2a_experiment_efficient.sh`  
**Estimated Completion**: ~13 hours from launch  
**Expected Result**: Empirical proof bahwa dual-modal > single-modal untuk HTR tasks
