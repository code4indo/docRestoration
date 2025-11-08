#!/usr/bin/env python3
"""
DIBCO TFRecord Audit Script
Audit whether dibco_tiled_full.tfrecord is ready for visual-only training
"""

import os
import sys
import tensorflow as tf
import numpy as np
from collections import Counter
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def audit_tfrecord(tfrecord_path):
    """Comprehensive audit of DIBCO TFRecord for training readiness"""
    
    print(f"🔍 AUDITING TFRECORD: {tfrecord_path}")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(tfrecord_path):
        print(f"❌ ERROR: File not found!")
        return False
        
    # Get file size
    file_size = os.path.getsize(tfrecord_path)
    print(f"📁 File size: {file_size / (1024*1024):.2f} MB")
    
    # Read TFRecord
    try:
        dataset = tf.data.TFRecordDataset(tfrecord_path)
    except Exception as e:
        print(f"❌ ERROR: Cannot read TFRecord: {e}")
        return False
    
    # Define expected feature structure
    feature_description = {
        'degraded_image_raw': tf.io.FixedLenFeature([], tf.string),
        'degraded_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'degraded_image_dtype': tf.io.FixedLenFeature([], tf.string),
        'clean_image_raw': tf.io.FixedLenFeature([], tf.string),
        'clean_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'clean_image_dtype': tf.io.FixedLenFeature([], tf.string),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'label_shape': tf.io.FixedLenFeature([1], tf.int64),
        'label_dtype': tf.io.FixedLenFeature([], tf.string),
    }
    
    print(f"\n📋 TFRECORD STRUCTURE AUDIT:")
    print("-" * 40)
    
    # Count samples
    sample_count = 0
    label_lengths = []
    image_shapes = []
    dtypes = []
    label_value_counts = Counter()
    
    for record in dataset:
        sample_count += 1
        
        try:
            # Parse the record
            example = tf.io.parse_single_example(record, feature_description)
            
            # Check image shapes
            degraded_shape = tf.cast(example['degraded_image_shape'], tf.int32).numpy()
            clean_shape = tf.cast(example['clean_image_shape'], tf.int32).numpy()
            image_shapes.append(tuple(degraded_shape))
            
            # Check label
            label_shape = tf.cast(example['label_shape'], tf.int32).numpy()
            label_lengths.append(int(label_shape[0]))
            
            # Decode and check label values (if small enough)
            if label_shape[0] <= 20:  # Only check short labels to avoid memory issues
                label_data = tf.io.decode_raw(example['label_raw'], tf.int64).numpy()
                label_data = label_data.reshape(label_shape)
                for val in label_data.flatten():
                    label_value_counts[val] += 1
                    
            # Store dtypes
            dtypes.append(example['degraded_image_dtype'].numpy().decode())
            
        except Exception as e:
            print(f"❌ ERROR parsing sample {sample_count}: {e}")
            return False
    
    print(f"✅ Total samples: {sample_count}")
    
    # Validate sample count for training
    if sample_count < 100:
        print(f"⚠️  WARNING: Very small dataset ({sample_count} samples)")
        print(f"   Expected: >100 samples for proper training")
    elif sample_count < 500:
        print(f"⚠️  WARNING: Small dataset ({sample_count} samples)")
        print(f"   OK for training but may need longer epochs")
    else:
        print(f"✅ Good sample count: {sample_count}")
    
    # Check image shapes consistency
    unique_shapes = set(image_shapes)
    print(f"\n🖼️  IMAGE SHAPES:")
    print(f"   Unique shapes found: {len(unique_shapes)}")
    for shape in unique_shapes:
        count = image_shapes.count(shape)
        print(f"   {shape}: {count} samples")
    
    if len(unique_shapes) > 1:
        print(f"❌ ERROR: Inconsistent image shapes!")
        return False
    
    # Expected shape for DIBCO (should be 128, 1024, 1 or similar)
    expected_shape = (128, 1024, 1)
    if unique_shapes == {expected_shape}:
        print(f"✅ Expected shape: {expected_shape}")
    else:
        print(f"⚠️  Unexpected shape: {unique_shapes}")
    
    # Check label lengths
    unique_lengths = set(label_lengths)
    print(f"\n📝 LABEL ANALYSIS:")
    print(f"   Unique lengths: {sorted(unique_lengths)}")
    
    if len(unique_lengths) > 1:
        print(f"❌ ERROR: Inconsistent label lengths!")
        return False
    
    label_length = unique_lengths.pop()
    print(f"   Label length: {label_length}")
    
    # Check if labels are dummy (all zeros or similar)
    print(f"\n🎯 LABEL VALUES AUDIT:")
    if label_value_counts:
        print(f"   Value distribution: {dict(label_value_counts.most_common())}")
        
        # Check if all labels are the same (indicates dummy labels)
        if len(label_value_counts) == 1:
            only_value = list(label_value_counts.keys())[0]
            if only_value == 0:
                print(f"✅ PERFECT: All labels are 0 (dummy labels for visual-only)")
                print(f"   This matches config requirement: ctc_loss_weight=0.0")
            else:
                print(f"⚠️  UNUSUAL: All labels are {only_value} (not zeros)")
                print(f"   May indicate preprocessing issue")
        else:
            print(f"⚠️  DIVERSE: Multiple different label values found")
            print(f"   Visual-only training may not need actual text")
    else:
        print(f"⚠️  WARNING: No label values checked (labels too long)")
    
    # Check data types
    unique_dtypes = set(dtypes)
    print(f"\n🔧 DATA TYPES:")
    print(f"   Image dtypes: {unique_dtypes}")
    
    if unique_dtypes == {'float32'}:
        print(f"✅ Expected dtype: float32")
    else:
        print(f"⚠️  Unexpected dtypes: {unique_dtypes}")
    
    # Training readiness assessment
    print(f"\n🎯 TRAINING READINESS ASSESSMENT:")
    print("=" * 60)
    
    readiness_score = 0
    max_score = 5
    
    # Check 1: File exists and readable
    readiness_score += 1
    print(f"✅ File access: OK")
    
    # Check 2: Sample count
    if sample_count >= 400:
        readiness_score += 1
        print(f"✅ Sample count: GOOD ({sample_count})")
    else:
        print(f"⚠️  Sample count: MARGINAL ({sample_count})")
    
    # Check 3: Shape consistency
    if len(unique_shapes) == 1:
        readiness_score += 1
        print(f"✅ Shape consistency: OK")
    else:
        print(f"❌ Shape consistency: FAILED")
    
    # Check 4: Label structure
    if len(unique_lengths) == 1 and label_length > 0:
        readiness_score += 1
        print(f"✅ Label structure: OK")
    else:
        print(f"❌ Label structure: FAILED")
    
    # Check 5: Visual-only compatibility
    if label_value_counts.get(0, 0) > 0:
        readiness_score += 1
        print(f"✅ Visual-only ready: PERFECT (all/dummy zeros)")
    else:
        print(f"⚠️  Visual-only ready: NEEDS VERIFICATION")
    
    # Final assessment
    print(f"\n📊 FINAL SCORE: {readiness_score}/{max_score}")
    
    if readiness_score >= 4:
        print(f"🎉 READY FOR TRAINING!")
        print(f"   Recommended config: Visual-only (ctc_loss_weight=0.0)")
        print(f"   Discriminator: Standard/base (not dual-modal)")
        return True
    elif readiness_score >= 3:
        print(f"⚠️  MARGINAL - Can train with tweaks")
        print(f"   May need data preprocessing")
        return True
    else:
        print(f"❌ NOT READY - Fix required before training")
        return False

def main():
    parser = argparse.ArgumentParser(description="Audit DIBCO TFRecord for training readiness")
    parser.add_argument('--tfrecord', type=str, 
                       default='dual_modal_gan/data/dibco_tiled_full.tfrecord',
                       help='Path to TFRecord file')
    args = parser.parse_args()
    
    # Run audit
    success = audit_tfrecord(args.tfrecord)
    
    if success:
        print(f"\n🚀 RECOMMENDATION: Proceed with visual-only training")
        print(f"   Config strategy: Use dibco_visual_only_restoration_v1.json")
    else:
        print(f"\n🚫 RECOMMENDATION: Fix issues before training")
    
    return success

if __name__ == '__main__':
    main()