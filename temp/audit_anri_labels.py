#!/usr/bin/env python3
"""
ANRI Dataset Audit Script
Verify if ANRI dataset has real text labels or dummy labels like DIBCO
"""

import os
import sys
import tensorflow as tf
import numpy as np
from collections import Counter
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def audit_anri_tfrecord(tfrecord_path, dataset_name="ANRI"):
    """Comprehensive audit of ANRI TFRecord for label analysis"""
    
    print(f"🔍 AUDITING {dataset_name} TFRECORD: {tfrecord_path}")
    print("=" * 70)
    
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
    
    print(f"\n📋 {dataset_name} STRUCTURE AUDIT:")
    print("-" * 50)
    
    # Count samples
    sample_count = 0
    label_lengths = []
    image_shapes = []
    dtypes = []
    label_value_counts = Counter()
    unique_labels = []
    
    # Sample labels for analysis
    sample_labels = []
    
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
            
            # Decode and check label values
            label_data = tf.io.decode_raw(example['label_raw'], tf.int64).numpy()
            label_data = label_data.reshape(label_shape)
            
            # Store sample for analysis
            if sample_count <= 10:  # Store first 10 samples for manual inspection
                sample_labels.append(label_data.flatten())
            
            # Count all label values
            for val in label_data.flatten():
                label_value_counts[val] += 1
                
            # Store dtypes
            dtypes.append(example['degraded_image_dtype'].numpy().decode())
            
        except Exception as e:
            print(f"❌ ERROR parsing sample {sample_count}: {e}")
            return False
    
    print(f"✅ Total samples: {sample_count}")
    
    # Check image shapes consistency
    unique_shapes = set(image_shapes)
    print(f"\n🖼️  IMAGE SHAPES:")
    print(f"   Unique shapes found: {len(unique_shapes)}")
    for shape in unique_shapes:
        count = image_shapes.count(shape)
        print(f"   {shape}: {count} samples")
    
    if len(unique_shapes) == 1:
        print(f"✅ Shape consistency: GOOD")
    else:
        print(f"⚠️  Shape consistency: Multiple shapes found")
    
    # Check label lengths
    unique_lengths = set(label_lengths)
    print(f"\n📝 LABEL ANALYSIS:")
    print(f"   Unique lengths: {sorted(unique_lengths)}")
    
    if len(unique_lengths) == 1:
        label_length = unique_lengths.pop()
        print(f"   Label length: {label_length}")
    else:
        print(f"   Multiple label lengths: {unique_lengths}")
        label_length = max(unique_lengths)
    
    # CRITICAL: Label values analysis
    print(f"\n🎯 LABEL VALUES ANALYSIS:")
    print(f"   Total label positions: {sample_count * label_length}")
    print(f"   Unique values: {len(label_value_counts)}")
    print(f"   Value distribution: {dict(label_value_counts.most_common(10))}")
    
    # Show sample labels
    print(f"\n📋 SAMPLE LABELS (first 10 samples):")
    for i, label in enumerate(sample_labels):
        print(f"   Sample {i+1}: {label}")
    
    # Analyze label patterns
    print(f"\n🔍 LABEL PATTERN ANALYSIS:")
    
    # Check if all labels are identical (dummy)
    all_zeros = all(val == 0 for val in label_value_counts.keys())
    all_same = len(label_value_counts) == 1
    has_nonzero = any(val != 0 for val in label_value_counts.keys())
    
    print(f"   All zeros: {all_zeros}")
    print(f"   All identical: {all_same}")
    print(f"   Has non-zero: {has_nonzero}")
    
    if all_zeros:
        print(f"   🚨 CONCLUSION: DUMMY LABELS (all zeros)")
        print(f"   📊 This is IDENTICAL to DIBCO dataset!")
        return "dummy"
    elif all_same:
        other_value = list(label_value_counts.keys())[0]
        print(f"   🚨 CONCLUSION: SINGLE DUMMY VALUE ({other_value})")
        print(f"   📊 Similar pattern to DIBCO (but different value)")
        return "dummy"
    elif has_nonzero:
        print(f"   ✅ CONCLUSION: REAL TEXT LABELS")
        print(f"   📊 Contains meaningful text information")
        return "real"
    else:
        print(f"   ❓ CONCLUSION: UNCLEAR")
        return "unknown"
    
    # Check data types
    unique_dtypes = set(dtypes)
    print(f"\n🔧 DATA TYPES:")
    print(f"   Image dtypes: {unique_dtypes}")
    
    return "analyzed"

def compare_anri_dibco():
    """Compare ANRI vs DIBCO label characteristics"""
    
    print(f"\n" + "="*80)
    print(f"📊 ANRI vs DIBCO LABEL COMPARISON")
    print(f"="*80)
    
    # Find all relevant TFRecords
    tfrecords = {
        "DIBCO": "dual_modal_gan/data/dibco_tiled_full.tfrecord",
        "ANRI Mixed": "dual_modal_gan/data/mixed_stage1_70base_30anri.tfrecord", 
        "ANRI Pure": "data/anri_pseudo_labeled_train.tfrecord"
    }
    
    results = {}
    
    for name, path in tfrecords.items():
        if os.path.exists(path):
            print(f"\n{'='*60}")
            result = audit_anri_tfrecord(path, name)
            results[name] = result
        else:
            print(f"\n❌ File not found: {path}")
            results[name] = "not_found"
    
    # Final comparison
    print(f"\n" + "="*80)
    print(f"🎯 FINAL COMPARISON SUMMARY")
    print(f"="*80)
    
    for name, result in results.items():
        if result == "dummy":
            status = "❌ DUMMY LABELS (like DIBCO)"
        elif result == "real":
            status = "✅ REAL TEXT LABELS"
        elif result == "not_found":
            status = "❓ FILE NOT FOUND"
        else:
            status = "❓ UNKNOWN"
        
        print(f"   {name:15}: {status}")
    
    # Critical insight
    print(f"\n💡 CRITICAL INSIGHT:")
    if "ANRI Mixed" in results and results["ANRI Mixed"] == "dummy":
        print(f"   🚨 ANRI juga menggunakan DUMMY LABELS!")
        print(f"   📊 FAKTA: ANRI success TIDAK dari real text labels")
        print(f"   🧠 ANRI success factor: TRANSFER LEARNING + MIXED DATASET")
        print(f"   🎯 Bukan karena dual-modal discriminator + real labels")
        
        print(f"\n📋 REVISED ANALYSIS:")
        print(f"   ✅ ANRI: Dummy labels + Transfer learning + Mixed dataset = SUCCESS")
        print(f"   ❌ DIBCO: Dummy labels + From scratch + Pure dataset = FAIL")
        print(f"   🔑 KEY DIFFERENCE: Transfer learning, bukan text labels!")
        
    elif "ANRI Pure" in results and results["ANRI Pure"] == "real":
        print(f"   ✅ ANRI memiliki REAL text labels")
        print(f"   📊 DIBCO tetap bermasalah karena from scratch training")
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Audit ANRI dataset for label analysis")
    parser.add_argument('--tfrecord', type=str, 
                       help='Path to ANRI TFRecord file')
    parser.add_argument('--compare', action='store_true',
                       help='Compare ANRI vs DIBCO label characteristics')
    args = parser.parse_args()
    
    if args.compare:
        results = compare_anri_dibco()
        return results
    elif args.tfrecord:
        result = audit_anri_tfrecord(args.tfrecord)
        return result
    else:
        # Default: compare all datasets
        results = compare_anri_dibco()
        return results

if __name__ == '__main__':
    main()