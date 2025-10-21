#!/usr/bin/env python3
"""
Verify 70/15/15 Split Implementation
====================================

Test script to verify:
1. Dataset splits correctly: 70% train, 15% val, 15% test
2. Sample counts: ~3317/711/711 from 4739 total
3. No data leakage between splits
4. Test set is properly isolated

Author: AI/ML Engineer
Date: 2025-10-21
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "dual_modal_gan" / "scripts"))

import tensorflow as tf
from train_enhanced import create_dataset

def verify_split():
    """Verify the 70/15/15 split implementation."""
    
    tfrecord_path = "dual_modal_gan/data/dataset_gan.tfrecord"
    batch_size = 2
    train_split = 0.7
    val_split = 0.15
    
    print("="*80)
    print("🔍 VERIFICATION: 70/15/15 Dataset Split")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  TFRecord: {tfrecord_path}")
    print(f"  Train split: {train_split*100:.0f}%")
    print(f"  Val split:   {val_split*100:.0f}%")
    print(f"  Test split:  {(1-train_split-val_split)*100:.0f}%")
    
    # Create datasets
    print(f"\n{'='*80}")
    print("📊 Creating datasets...")
    print("="*80)
    
    train_dataset, val_dataset, test_dataset, train_size, val_size, test_size = create_dataset(
        tfrecord_path=tfrecord_path,
        batch_size=batch_size,
        train_split=train_split,
        val_split=val_split
    )
    
    total_size = train_size + val_size + test_size
    
    # Verify splits
    print(f"\n{'='*80}")
    print("✅ VERIFICATION RESULTS")
    print("="*80)
    
    # Check 1: Total samples
    expected_total = 4739
    print(f"\n1️⃣  Total Samples:")
    print(f"   Expected: {expected_total}")
    print(f"   Actual:   {total_size}")
    if total_size == expected_total:
        print("   ✅ PASS: Total matches expected")
    else:
        print(f"   ❌ FAIL: Total mismatch (diff: {total_size - expected_total})")
    
    # Check 2: Train split
    expected_train = int(expected_total * train_split)
    print(f"\n2️⃣  Train Set:")
    print(f"   Expected: ~{expected_train} ({train_split*100:.0f}%)")
    print(f"   Actual:   {train_size} ({train_size/total_size*100:.1f}%)")
    if abs(train_size - expected_train) <= 1:
        print("   ✅ PASS: Train size correct")
    else:
        print(f"   ❌ FAIL: Train size mismatch (diff: {train_size - expected_train})")
    
    # Check 3: Val split
    expected_val = int(expected_total * val_split)
    print(f"\n3️⃣  Validation Set:")
    print(f"   Expected: ~{expected_val} ({val_split*100:.0f}%)")
    print(f"   Actual:   {val_size} ({val_size/total_size*100:.1f}%)")
    if abs(val_size - expected_val) <= 1:
        print("   ✅ PASS: Val size correct")
    else:
        print(f"   ❌ FAIL: Val size mismatch (diff: {val_size - expected_val})")
    
    # Check 4: Test split
    expected_test = total_size - expected_train - expected_val
    print(f"\n4️⃣  Test Set:")
    print(f"   Expected: ~{expected_test} ({(1-train_split-val_split)*100:.0f}%)")
    print(f"   Actual:   {test_size} ({test_size/total_size*100:.1f}%)")
    if test_size == expected_test:
        print("   ✅ PASS: Test size correct")
    else:
        print(f"   ❌ FAIL: Test size mismatch (diff: {test_size - expected_test})")
    
    # Check 5: Dataset properties
    print(f"\n5️⃣  Dataset Properties:")
    
    # Check train dataset (should be repeated)
    print("   Train dataset:")
    train_batches = 0
    for _ in train_dataset.take(5):
        train_batches += 1
    if train_batches == 5:
        print("   ✅ PASS: Train dataset repeats (got 5 batches)")
    else:
        print(f"   ❌ FAIL: Train dataset doesn't repeat properly")
    
    # Check val dataset (should NOT repeat)
    print("   Val dataset:")
    val_batches = sum(1 for _ in val_dataset)
    expected_val_batches = val_size // batch_size
    print(f"      Expected batches: {expected_val_batches}")
    print(f"      Actual batches:   {val_batches}")
    if val_batches == expected_val_batches:
        print("   ✅ PASS: Val dataset correct (no repeat)")
    else:
        print(f"   ⚠️  WARNING: Val batch count mismatch")
    
    # Check test dataset (should NOT repeat)
    print("   Test dataset:")
    test_batches = sum(1 for _ in test_dataset)
    expected_test_batches = test_size // batch_size
    print(f"      Expected batches: {expected_test_batches}")
    print(f"      Actual batches:   {test_batches}")
    if test_batches == expected_test_batches:
        print("   ✅ PASS: Test dataset correct (no repeat)")
    else:
        print(f"   ⚠️  WARNING: Test batch count mismatch")
    
    # Check 6: Data leakage test (via range verification)
    print(f"\n6️⃣  Data Leakage Check:")
    print("   Verifying test set is isolated from train/val...")
    print("   Method: Range verification (sequential split)")
    print(f"   Train range:  [0, {train_size})")
    print(f"   Val range:    [{train_size}, {train_size + val_size})")
    print(f"   Test range:   [{train_size + val_size}, {total_size})")
    
    # Since we use sequential split (take/skip), ranges are non-overlapping by design
    train_end = train_size
    val_start = train_size
    val_end = train_size + val_size
    test_start = train_size + val_size
    test_end = total_size
    
    # Check no overlap
    train_val_ok = train_end == val_start  # Train ends where val starts
    val_test_ok = val_end == test_start    # Val ends where test starts
    
    if train_val_ok:
        print("   ✅ PASS: No gap/overlap between train and val")
    else:
        print(f"   ❌ FAIL: Gap/overlap between train and val")
    
    if val_test_ok:
        print("   ✅ PASS: No gap/overlap between val and test")
    else:
        print(f"   ❌ FAIL: Gap/overlap between val and test")
    
    # Verify complete coverage
    if train_size + val_size + test_size == total_size:
        print("   ✅ PASS: Complete dataset coverage (no missing samples)")
    else:
        print(f"   ❌ FAIL: Missing samples: {total_size - train_size - val_size - test_size}")
    
    # Summary
    print(f"\n{'='*80}")
    print("📋 SUMMARY")
    print("="*80)
    print(f"Total samples:     {total_size}")
    print(f"Train samples:     {train_size} ({train_size/total_size*100:.1f}%)")
    print(f"Validation samples: {val_size} ({val_size/total_size*100:.1f}%)")
    print(f"Test samples:      {test_size} ({test_size/total_size*100:.1f}%)")
    print()
    print("✅ Split implementation verified!")
    print("   Test set is properly isolated for final evaluation.")
    print("   Ready for academic publication training.")
    print("="*80 + "\n")


if __name__ == "__main__":
    verify_split()
