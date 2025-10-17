# Checkpoint Resume Feature Documentation

## 🎯 Overview

Automatic checkpoint resume capability untuk melindungi training dari kegagalan ditengah jalan. Training dapat secara otomatis melanjutkan dari epoch terakhir yang berhasil diselesaikan.

## ✨ Features

1. **Auto-detect latest checkpoint** - Sistem otomatis mencari checkpoint terakhir
2. **Resume from last completed epoch** - Training continue dari epoch yang tepat
3. **Epoch tracking** - Menyimpan informasi epoch di `epoch_info.json`
4. **Best model preservation** - Best model tetap di-track dan di-save
5. **Zero configuration** - Cukup tambahkan `"resume_from_checkpoint": true` di config JSON

## 📋 Usage

### Option 1: Using JSON Config (Recommended)

**File**: `configs/experiment_config.json`

```json
{
  "training": {
    "resume_from_checkpoint": true,
    ...
  }
}
```

**Launch**:
```bash
bash scripts/universal_train_from_json.sh configs/experiment_config.json
```

### Option 2: Direct Python Call

```bash
poetry run python dual_modal_gan/scripts/train_enhanced.py \
  --resume \
  --checkpoint_dir dual_modal_gan/checkpoints/your_experiment \
  --epochs 15 \
  ...
```

## 🔧 How It Works

### 1. Checkpoint Structure
```
dual_modal_gan/checkpoints/experiment_name/
├── checkpoint                     # TensorFlow checkpoint index
├── ckpt-1.data-00000-of-00001    # Checkpoint weights (epoch 1)
├── ckpt-1.index                  # Checkpoint metadata
├── ckpt-2.data-00000-of-00001    # Checkpoint weights (epoch 2)
├── ckpt-2.index                  # Checkpoint metadata
└── epoch_info.json               # Epoch tracking info
```

### 2. Epoch Info Format
```json
{
  "last_completed_epoch": 2,
  "best_epoch": 1,
  "best_score": 25.34,
  "timestamp": "2025-10-17T14:30:00"
}
```

### 3. Resume Logic

**Without `--resume` flag**:
- Training starts from epoch 0 (fresh training)
- Creates new checkpoints

**With `--resume` flag**:
- Check if `epoch_info.json` exists
- Load latest checkpoint (e.g., ckpt-2)
- Continue from `last_completed_epoch + 1` (e.g., epoch 3)
- Preserve all previous checkpoint data

## 📊 Example Scenario

### Initial Training
```bash
# Train 5 epochs
bash scripts/universal_train_from_json.sh configs/experiment.json

# Training completes:
# - Epoch 1/5 ✅
# - Epoch 2/5 ✅
# - Epoch 3/5 ✅ (CRASH HERE!)
```

**Result**: 
- Checkpoint saved: ckpt-2 (after epoch 2)
- epoch_info.json: `{"last_completed_epoch": 2}`
- Lost: Epoch 3 progress (but checkpoint at epoch 2 preserved)

### Resume Training
```bash
# Add resume flag to config:
{
  "training": {
    "resume_from_checkpoint": true,
    "epochs": 5
  }
}

# Re-launch same command
bash scripts/universal_train_from_json.sh configs/experiment.json
```

**Output**:
```
🔄 RESUME MODE DETECTED!
   Checkpoint dir: dual_modal_gan/checkpoints/experiment
   Last checkpoint: ckpt-2
   Last completed epoch: 2
   Continuing from epoch: 3/5
   
Restored from dual_modal_gan/checkpoints/experiment/ckpt-2

Epoch 3/5 (continuing...)
Epoch 4/5 (continuing...)
Epoch 5/5 (continuing...)

✅ Training completed!
```

## 🛡️ Safety Features

### 1. Checkpoint Every Epoch
```python
# Checkpoint saved after EVERY epoch completes
# Not just when best model improves
ckpt_manager.save()
```

### 2. Epoch Info Persistence
```python
# epoch_info.json updated after every epoch
{
  "last_completed_epoch": current_epoch,
  "best_epoch": best_epoch,
  "best_score": best_score
}
```

### 3. Validation Before Continue
```python
if not os.path.exists(epoch_info_path):
    print("⚠️  No epoch_info.json found")
    print("   Starting from scratch")
```

## ⚙️ Configuration Options

### JSON Config
```json
{
  "training": {
    "resume_from_checkpoint": true,  // Enable resume
    "no_restore": false              // Force fresh training (ignores checkpoints)
  },
  "checkpoints": {
    "max_checkpoints": 1,            // How many checkpoints to keep
    "save_best_only": false          // Save every epoch (recommended for resume)
  }
}
```

### Command Line Flags
- `--resume` - Enable checkpoint resume
- `--no_restore` - Force fresh training (ignore existing checkpoints)
- `--checkpoint_dir` - Checkpoint directory path

## 🔍 Troubleshooting

### Issue: "No checkpoint found"
**Cause**: First time training or checkpoint directory empty
**Solution**: Normal behavior, training starts from epoch 0

### Issue: "epoch_info.json not found"
**Cause**: Training with old version (before resume feature)
**Solution**: Manually create epoch_info.json:
```json
{
  "last_completed_epoch": 2,
  "best_epoch": 1,
  "best_score": 25.0,
  "timestamp": "2025-10-17T14:00:00"
}
```

### Issue: Resume from wrong epoch
**Cause**: epoch_info.json outdated or corrupted
**Solution**: 
1. Check checkpoint directory: `ls -la dual_modal_gan/checkpoints/experiment/`
2. Verify epoch_info.json: `cat dual_modal_gan/checkpoints/experiment/epoch_info.json`
3. Manually edit if needed

## 📝 Best Practices

### 1. Always Enable Resume for Long Training
```json
{
  "training": {
    "epochs": 15,                    // Long training (6+ hours)
    "resume_from_checkpoint": true   // ✅ ENABLE THIS
  }
}
```

### 2. Keep Reasonable Checkpoint Count
```json
{
  "checkpoints": {
    "max_checkpoints": 1   // Saves disk space, keeps latest checkpoint
  }
}
```

### 3. Save Every Epoch (Not Just Best)
```json
{
  "checkpoints": {
    "save_best_only": false  // ✅ Save every epoch for resume
  }
}
```

### 4. Monitor epoch_info.json
```bash
# During training, check progress:
watch -n 10 "cat dual_modal_gan/checkpoints/experiment/epoch_info.json"
```

## 🎯 Implementation Details

### Files Modified
1. `dual_modal_gan/scripts/train_enhanced.py` - Core resume logic
2. `scripts/universal_train_from_json.sh` - JSON config support
3. `configs/EXAMPLE_resume_training.json` - Example config

### Key Changes
```python
# 1. Parse --resume flag
parser.add_argument('--resume', action='store_true', 
                   help='Resume from latest checkpoint')

# 2. Load epoch info
if args.resume:
    epoch_info = load_epoch_info(checkpoint_dir)
    start_epoch = epoch_info['last_completed_epoch'] + 1

# 3. Save checkpoint every epoch
ckpt_manager.save()
save_epoch_info(checkpoint_dir, epoch, best_epoch, best_score)

# 4. Adjust training loop
for epoch in range(start_epoch, args.epochs):
    # Training continues from correct epoch
```

## ✅ Testing

### Smoke Test
```bash
bash scripts/test_checkpoint_resume.sh
```

**Test Plan**:
1. Train 2 epochs (creates checkpoint)
2. Verify epoch_info.json created
3. Resume with --resume flag
4. Verify continues from epoch 3

**Expected Output**:
```
✅ SUCCESS: Checkpoint resume working correctly!
   - Phase 1 trained epochs 0-1
   - Phase 2 resumed from epoch 2 and completed epochs 2-3
   - Final last_completed_epoch: 3
```

## 📊 Impact

**Before Resume Feature**:
- Training crash at epoch 10/15 = lose 6 hours progress ❌
- Must restart from epoch 0
- High risk for long training

**After Resume Feature**:
- Training crash at epoch 10/15 = lose ~25 min (1 epoch) ✅
- Resume from epoch 10
- Low risk, production ready

## 🚀 Future Enhancements

1. **Optimizer state saving** - Save Adam momentum, etc
2. **Multi-checkpoint strategy** - Keep top-3 checkpoints
3. **Auto-resume on crash** - Watchdog process
4. **Cloud sync** - Auto-upload checkpoints to S3/GCS

---

**Status**: ✅ Production Ready  
**Version**: 1.0  
**Date**: 2025-10-17  
**Author**: AI Agent (Copilot + Human Collaboration)
