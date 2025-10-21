# 🎯 STRATEGI DATASET SPLIT & TESTING - REKOMENDASI AKADEMIS

**Date**: 2025-10-21  
**Current Status**: ❌ NO TEST SET (only train/val)  
**Required**: ✅ Train/Val/Test split for academic publication

---

## 🚨 MASALAH SAAT INI

### Current Dataset Split:
```
Total: 4739 samples
├── Train: 4265 samples (90%)
├── Val:   474 samples (10%)
└── Test:  0 samples (0%) ❌ MISSING!
```

**Masalah**:
1. ❌ **No independent test set** - validation digunakan untuk tuning hyperparameters
2. ❌ **Data leakage risk** - model selection based on validation
3. ❌ **Cannot report unbiased performance** - reviewers akan reject
4. ❌ **Not following ML best practices** - train/val/test adalah standar

---

## ✅ REKOMENDASI AKADEMIS: 3-WAY SPLIT

### Recommended Split (70/15/15):
```
Total: 4739 samples
├── Train: 3317 samples (70%) - untuk training model
├── Val:   711 samples (15%)  - untuk hyperparameter tuning & early stopping
└── Test:  711 samples (15%)  - untuk final evaluation (NEVER TOUCHED during training)
```

### Justifikasi Academic:

#### 1. **Train Set (70%)**:
- **Purpose**: Model learning
- **Usage**: Gradient updates, loss optimization
- **Size**: 3317 samples (cukup untuk learning)

#### 2. **Validation Set (15%)**:
- **Purpose**: Hyperparameter tuning, early stopping, model selection
- **Usage**: Evaluated setiap epoch untuk monitoring
- **Size**: 711 samples (2x lipat dari current, lebih robust)
- **Key**: Boleh dilihat berkali-kali selama training

#### 3. **Test Set (15%)**: ⭐ **CRITICAL**
- **Purpose**: **FINAL unbiased evaluation**
- **Usage**: **HANYA SEKALI** setelah training selesai
- **Size**: 711 samples (sama dengan val untuk fair comparison)
- **Key**: **TIDAK BOLEH dilihat sampai model final**

---

## 📊 ALTERNATIF SPLIT OPTIONS

### Option A: 70/15/15 (Recommended for your dataset size)
```
Train: 3317 (70%) - Adequate for learning
Val:   711 (15%)  - Strong statistical power
Test:  711 (15%)  - Strong statistical power
```
**Pros**: Balanced, strong statistical power on val & test  
**Best for**: Dataset 4000-10000 samples

### Option B: 80/10/10 (Standard ML)
```
Train: 3791 (80%) - More training data
Val:   474 (10%)  - Adequate for monitoring
Test:  474 (10%)  - Adequate for final eval
```
**Pros**: More training data, standard ratio  
**Best for**: When training data is critical

### Option C: 60/20/20 (High confidence)
```
Train: 2843 (60%) - Less but sufficient
Val:   948 (20%)  - Very strong statistical power
Test:  948 (20%)  - Very strong statistical power
```
**Pros**: Highest confidence in val/test metrics  
**Best for**: When evaluation confidence is critical for publication

---

## 🎓 ACADEMIC BEST PRACTICES

### 1. **When to Split**:
- ✅ **Before any training** - split once at the beginning
- ✅ **Stratified by difficulty** - if possible (optional)
- ✅ **Random but reproducible** - with fixed seed

### 2. **How to Use Each Set**:

**Training Set**:
- Use for: Model weight updates
- Frequency: Every batch
- Purpose: Learning patterns

**Validation Set**:
- Use for: Hyperparameter tuning, early stopping, checkpoint selection
- Frequency: Every epoch
- Purpose: Guide training process
- ⚠️ **CAN introduce bias** if used for model selection

**Test Set**: ⭐ **MOST IMPORTANT**
- Use for: **FINAL evaluation ONLY**
- Frequency: **ONCE** after training complete
- Purpose: Report unbiased performance
- 🚫 **NEVER use for**:
  - Hyperparameter tuning
  - Early stopping decisions
  - Model selection
  - Any training decisions

### 3. **Reporting in Paper**:

**Good Practice** ✅:
```
"We split the dataset into train (70%, n=3317), validation (15%, n=711), 
and test (15%, n=711) sets using stratified random sampling. The validation 
set was used for early stopping and hyperparameter tuning. The test set was 
held out and used only once for final evaluation after model selection was 
complete. All reported results are on the test set."
```

**Bad Practice** ❌:
```
"We split into train (90%) and validation (10%). We report validation 
set performance."
```
→ Reviewers akan tanya: "Where is test set? This is validation performance, not test!"

---

## 🔬 WHEN TO EVALUATE ON TEST SET

### Training Timeline:

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Development (Use Train + Val)                          │
├─────────────────────────────────────────────────────────────────┤
│ 1. Try different architectures                                   │
│ 2. Tune hyperparameters                                          │
│ 3. Experiment with loss weights                                  │
│ 4. Multiple training runs                                        │
│ 5. Select best model based on VALIDATION performance            │
│                                                                   │
│ ⚠️ Test set: LOCKED - DO NOT TOUCH                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Final Evaluation (UNLOCK Test Set)                     │
├─────────────────────────────────────────────────────────────────┤
│ 1. Training complete, best model selected                        │
│ 2. Load best checkpoint (based on validation)                   │
│ 3. Evaluate on TEST SET - ONLY ONCE                             │
│ 4. Report test set metrics in paper                             │
│ 5. Compare with baseline on same test set                       │
│                                                                   │
│ ✅ This is your OFFICIAL result                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Critical Rule**: 
> "Test set should be touched EXACTLY ONCE - after all development is done."

---

## 🛠️ IMPLEMENTATION PLAN

### Step 1: Create Test Split Function
```python
def create_dataset_with_test(tfrecord_path, batch_size, train_split=0.7, val_split=0.15):
    """
    Create train/val/test split: 70/15/15
    Test set is NEVER seen during training.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    total_size = sum(1 for _ in dataset)
    
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    dataset = dataset.map(_parse_tfrecord_fn)
    
    # Split: first 70% train, next 15% val, last 15% test
    train_dataset = dataset.take(train_size)
    remaining = dataset.skip(train_size)
    val_dataset = remaining.take(val_size)
    test_dataset = remaining.skip(val_size)
    
    # Only train is shuffled and repeated
    train_dataset = train_dataset.shuffle(1024).repeat().batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)  # No shuffle
    test_dataset = test_dataset.batch(batch_size)  # No shuffle
    
    return train_dataset, val_dataset, test_dataset, train_size, val_size, test_size
```

### Step 2: Modify Training Script
- Use `train_dataset` and `val_dataset` during training
- Save `test_dataset` info but DON'T evaluate
- Add flag: `--evaluate_test_set` (disabled by default)

### Step 3: Create Separate Test Evaluation Script
```bash
scripts/evaluate_test_set.py
```
- Loads best checkpoint
- Evaluates on test set ONCE
- Generates comprehensive report with CI
- Saves to separate results file

### Step 4: Training Protocol
```bash
# 1. Train model (val set for monitoring)
./scripts/launch_production_v2_20251021.sh

# 2. Training completes, best model selected via validation

# 3. ONLY AFTER training done, evaluate test set
poetry run python scripts/evaluate_test_set.py \
  --checkpoint dual_modal_gan/checkpoints/production_v2/best_model \
  --tfrecord dual_modal_gan/data/dataset_gan.tfrecord \
  --output test_set_results.json
```

---

## 📊 REPORTING IN PAPER

### Methods Section:
```
Dataset: We collected 4739 document image pairs... 

Split Strategy: The dataset was partitioned into training (70%, n=3317), 
validation (15%, n=711), and test (15%, n=711) sets using deterministic 
split with fixed random seed (42) to ensure reproducibility. The splits 
were created by taking sequential samples to preserve temporal ordering 
if present.

Training Protocol: The model was trained using the training set with 
hyperparameters tuned using the validation set. Early stopping was applied 
based on validation set performance with patience of 25 epochs. The test 
set was held out and not accessed during model development or hyperparameter 
tuning.

Evaluation: Final performance was evaluated on the test set after training 
completion. We report mean ± standard deviation and 95% confidence intervals 
for all metrics computed across all test samples (n=711).
```

### Results Section:
```
Table 1: Performance on Test Set (n=711)

Method          | PSNR (dB)      | SSIM          | CER (%)
----------------|----------------|---------------|----------
Baseline        | 25.32 ± 2.15   | 0.8234 ± 0.05 | 12.34 ± 3.2
Our Method      | 31.45 ± 2.08   | 0.9542 ± 0.02 | 4.23 ± 1.8

All metrics computed on held-out test set. 95% CI: PSNR [31.29, 31.61], 
SSIM [0.9527, 0.9557].
```

---

## ⚠️ COMMON MISTAKES TO AVOID

### ❌ Mistake 1: Using validation as test
```
"We report validation set performance as final results."
→ WRONG: Validation was used for model selection (biased)
```

### ❌ Mistake 2: Multiple looks at test
```
"We evaluated on test set every 10 epochs and selected best."
→ WRONG: Test set should be seen ONCE
```

### ❌ Mistake 3: No test set at all
```
"We split into train/val only."
→ WRONG: No unbiased evaluation possible
```

### ❌ Mistake 4: Tuning on test
```
"We tried different hyperparameters and picked best based on test performance."
→ WRONG: Test set now biased
```

### ✅ Correct Approach:
```
1. Split: train/val/test
2. Develop: Use train + val (look at val many times)
3. Select: Best model based on val
4. Evaluate: Load best model, evaluate test ONCE
5. Report: Test set metrics in paper
```

---

## 🚀 IMPLEMENTATION PRIORITY

### Must Do (Before Paper Submission):
1. ✅ Create 70/15/15 split
2. ✅ Modify training to use train+val only
3. ✅ Create test evaluation script
4. ✅ Run final test evaluation ONCE
5. ✅ Report test metrics with CI in paper

### Nice to Have:
- K-fold cross-validation on train+val (extra confidence)
- Bootstrap CI on test set
- Significance testing vs baseline on same test set

---

## 📋 CHECKLIST FOR YOUR RESEARCH

- [ ] **Split created**: 70% train, 15% val, 15% test
- [ ] **Test set locked**: Never touched during training
- [ ] **Validation used**: For early stopping & hyperparameter tuning
- [ ] **Best model selected**: Based on validation performance
- [ ] **Test evaluated**: ONCE after training complete
- [ ] **Statistics reported**: Mean ± std, 95% CI on test set
- [ ] **Sample sizes stated**: n=3317/711/711 clearly documented
- [ ] **Protocol documented**: Clear description in paper
- [ ] **Reproducible**: Seed fixed, splits deterministic

---

## 💡 RECOMMENDATION FOR YOUR CASE

**For 4739 samples, I recommend**:

### **Option A: 70/15/15** (BEST CHOICE)
- Train: 3317 samples
- Val:   711 samples  
- Test:  711 samples

**Justification**:
1. Training set 3317 cukup untuk learning (>3000 adalah good)
2. Val set 711 memberikan statistical power kuat (n>>30)
3. Test set 711 sama dengan val - fair comparison
4. Balanced approach untuk publikasi akademis

**Implementation Priority**:
1. **IMMEDIATE**: Create split function & test evaluation script
2. **BEFORE TRAINING**: Implement 70/15/15 split
3. **AFTER TRAINING**: Evaluate test set ONCE
4. **FOR PAPER**: Report test set metrics with proper statistics

---

## 🎓 ACADEMIC ACCEPTABILITY

**Current Approach** (90/10/0):
- Academic Acceptability: ❌ **WEAK** (no test set)
- Reviewer Response: "Where is test set? Validation is biased."

**Recommended Approach** (70/15/15):
- Academic Acceptability: ✅ **STRONG** (proper 3-way split)
- Reviewer Response: "Good methodology, results are unbiased."

**Key Message**: 
> "Without test set, your results cannot be accepted in high-quality venues (Q1 journals, top conferences). Test set is NOT optional for ML research."

---

**CONCLUSION**: Implementasikan 70/15/15 split SEBELUM training production. Evaluasi test set HANYA SEKALI setelah model final. Report test metrics dalam paper dengan 95% CI dan sample size.
