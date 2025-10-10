#!/bin/bash
# Verify dataset provenance and check for potential overlap

echo "=========================================="
echo "Dataset Provenance Verification"
echo "=========================================="
echo ""

# Check dataset files
echo "1. DATASET FILES:"
echo "---"
ls -lh dual_modal_gan/data/*.tfrecord 2>/dev/null && echo "" || echo "  No GAN dataset found"
ls -lh real_data_preparation/*.tfrecord 2>/dev/null && echo "" || echo "  No HTR dataset found"

# Count samples
echo "2. DATASET SIZES:"
echo "---"

if [ -f "dual_modal_gan/data/dataset_gan.tfrecord" ]; then
    GAN_COUNT=$(python3 -c "
import tensorflow as tf
count = sum(1 for _ in tf.data.TFRecordDataset('dual_modal_gan/data/dataset_gan.tfrecord'))
print(count)
" 2>/dev/null)
    echo "  GAN dataset: $GAN_COUNT samples"
else
    echo "  GAN dataset: NOT FOUND"
fi

if [ -f "real_data_preparation/real_data.tfrecord" ]; then
    HTR_COUNT=$(python3 -c "
import tensorflow as tf
count = sum(1 for _ in tf.data.TFRecordDataset('real_data_preparation/real_data.tfrecord'))
print(count)
" 2>/dev/null)
    echo "  HTR dataset: $HTR_COUNT samples"
else
    echo "  HTR dataset: NOT FOUND"
fi

echo ""

# Check creation timestamps
echo "3. CREATION TIMESTAMPS:"
echo "---"
stat -c "%y %n" dual_modal_gan/data/*.tfrecord 2>/dev/null || echo "  GAN dataset: N/A"
stat -c "%y %n" real_data_preparation/*.tfrecord 2>/dev/null || echo "  HTR dataset: N/A"
echo ""

# Check for creation scripts
echo "4. DATASET CREATION SCRIPTS:"
echo "---"
echo "  GAN dataset scripts:"
find dual_modal_gan -name "*create*" -o -name "*prepare*" -o -name "*generate*" 2>/dev/null | grep -v __pycache__ | head -10
echo ""
echo "  HTR dataset scripts:"
find real_data_preparation -name "*create*" -o -name "*prepare*" -o -name "*generate*" 2>/dev/null | grep -v __pycache__ | head -10
echo ""

# Extract sample hashes for comparison
echo "5. SAMPLE FINGERPRINTING (first 5 samples):"
echo "---"

echo "  Hashing GAN dataset samples..."
python3 << 'PYEOF'
import tensorflow as tf
import hashlib
import sys

try:
    dataset = tf.data.TFRecordDataset('dual_modal_gan/data/dataset_gan.tfrecord')
    gan_hashes = []
    
    for i, record in enumerate(dataset.take(5)):
        hash_obj = hashlib.md5(record.numpy()).hexdigest()
        gan_hashes.append(hash_obj)
        print(f"    GAN sample {i+1}: {hash_obj}")
    
    # Save to temp file
    with open('/tmp/gan_hashes.txt', 'w') as f:
        for h in gan_hashes:
            f.write(h + '\n')
    
except Exception as e:
    print(f"    Error: {e}")
PYEOF

echo ""
echo "  Hashing HTR dataset samples..."
python3 << 'PYEOF'
import tensorflow as tf
import hashlib
import sys

try:
    dataset = tf.data.TFRecordDataset('real_data_preparation/real_data.tfrecord')
    htr_hashes = []
    
    for i, record in enumerate(dataset.take(5)):
        hash_obj = hashlib.md5(record.numpy()).hexdigest()
        htr_hashes.append(hash_obj)
        print(f"    HTR sample {i+1}: {hash_obj}")
    
    # Save to temp file
    with open('/tmp/htr_hashes.txt', 'w') as f:
        for h in htr_hashes:
            f.write(h + '\n')
    
except Exception as e:
    print(f"    Error: {e}")
PYEOF

echo ""

# Check for overlap
if [ -f "/tmp/gan_hashes.txt" ] && [ -f "/tmp/htr_hashes.txt" ]; then
    echo "6. OVERLAP CHECK (first 5 samples):"
    echo "---"
    
    OVERLAP=$(comm -12 <(sort /tmp/gan_hashes.txt) <(sort /tmp/htr_hashes.txt) | wc -l)
    
    if [ "$OVERLAP" -eq 0 ]; then
        echo "  ✅ NO OVERLAP detected in first 5 samples"
        echo "  (This is a good sign, but not conclusive for entire dataset)"
    else
        echo "  ⚠️  OVERLAP DETECTED: $OVERLAP matching samples"
        echo "  (This indicates datasets may share sources)"
    fi
    
    # Cleanup
    rm /tmp/gan_hashes.txt /tmp/htr_hashes.txt 2>/dev/null
else
    echo "6. OVERLAP CHECK: Skipped (hash files not created)"
fi

echo ""
echo "=========================================="
echo "7. ASSESSMENT SUMMARY"
echo "=========================================="
echo ""

# Generate summary
cat << 'EOF'
INTERPRETATION GUIDE:

✅ SAFE: No overlap detected + Different creation dates + Different scripts
⚠️  UNCERTAIN: Cannot determine overlap (need full dataset scan)
❌ RISK: Overlap detected or same source files

RECOMMENDED ACTIONS:
1. If no overlap: Document in paper methodology
2. If overlap detected: Verify if it's intentional (e.g., different degradation)
3. If uncertain: Run full dataset comparison (may take time)

For full dataset hash comparison:
  python3 scripts/compare_datasets_full.py
EOF

echo ""
echo "=========================================="
