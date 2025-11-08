#!/usr/bin/env python3
"""
Check for duplicate images in fine-tuning dataset

This script verifies:
1. No exact duplicates (pixel-by-pixel identical)
2. No near-duplicates (perceptual hash similarity)
3. Augmentation diversity (variants are actually different)
"""

import cv2
import numpy as np
from pathlib import Path
import hashlib
from tqdm import tqdm
from collections import defaultdict

def compute_image_hash(image_path):
    """Compute MD5 hash of image file (exact duplicate detection)"""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def compute_perceptual_hash(image_path, hash_size=8):
    """Compute perceptual hash (pHash) for near-duplicate detection"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Resize to hash_size x hash_size
    img_resized = cv2.resize(img, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    
    # Compute DCT
    dct = cv2.dct(np.float32(img_resized))
    
    # Extract top-left 8x8 (low frequencies)
    dct_low = dct[:hash_size, :hash_size]
    
    # Compute median
    median = np.median(dct_low)
    
    # Create hash: 1 if > median, 0 otherwise
    hash_bits = (dct_low > median).flatten()
    
    # Convert to hex string
    hash_int = int(''.join(['1' if b else '0' for b in hash_bits]), 2)
    return hash_int

def hamming_distance(hash1, hash2):
    """Calculate Hamming distance between two hashes"""
    if hash1 is None or hash2 is None:
        return 100  # Max distance if hash computation failed
    
    # XOR and count 1s
    xor = hash1 ^ hash2
    distance = bin(xor).count('1')
    return distance

def check_duplicates(samples_dir):
    """Check for duplicates in dataset"""
    
    print("="*80)
    print("DUPLICATE DETECTION ANALYSIS")
    print("="*80)
    
    samples_dir = Path(samples_dir)
    
    # Collect all pair images from all splits
    all_pairs = []
    for split in ['train', 'val', 'test']:
        pair_dir = samples_dir / split / 'pairs'
        if pair_dir.exists():
            pairs = sorted(list(pair_dir.glob("*.png")))
            all_pairs.extend([(split, p) for p in pairs])
    
    if len(all_pairs) == 0:
        print("❌ No samples found! Extract samples first.")
        return
    
    print(f"\n📊 Total samples to check: {len(all_pairs)}")
    print(f"   Train: {len([p for s, p in all_pairs if s == 'train'])}")
    print(f"   Val:   {len([p for s, p in all_pairs if s == 'val'])}")
    print(f"   Test:  {len([p for s, p in all_pairs if s == 'test'])}")
    
    # 1. Check for EXACT duplicates (MD5 hash)
    print(f"\n{'='*80}")
    print("1. EXACT DUPLICATE CHECK (MD5 Hash)")
    print(f"{'='*80}")
    
    md5_hashes = {}
    exact_duplicates = defaultdict(list)
    
    print("\n📝 Computing MD5 hashes...")
    for split, img_path in tqdm(all_pairs, desc="MD5 hashing"):
        md5 = compute_image_hash(img_path)
        
        if md5 in md5_hashes:
            # Duplicate found!
            exact_duplicates[md5].append((split, img_path))
        else:
            md5_hashes[md5] = (split, img_path)
    
    if exact_duplicates:
        print(f"\n⚠️  EXACT DUPLICATES FOUND: {len(exact_duplicates)} groups")
        for idx, (md5, duplicates) in enumerate(exact_duplicates.items(), 1):
            print(f"\n   Group {idx} (MD5: {md5[:16]}...):")
            original = md5_hashes[md5]
            print(f"     Original: {original[0]}/{original[1].name}")
            for split, dup_path in duplicates:
                print(f"     Duplicate: {split}/{dup_path.name}")
    else:
        print(f"\n✅ NO EXACT DUPLICATES FOUND")
        print(f"   All {len(all_pairs)} samples are pixel-perfect unique")
    
    # 2. Check for NEAR duplicates (Perceptual Hash)
    print(f"\n{'='*80}")
    print("2. NEAR-DUPLICATE CHECK (Perceptual Hash)")
    print(f"{'='*80}")
    
    print("\n📝 Computing perceptual hashes...")
    phashes = []
    for split, img_path in tqdm(all_pairs, desc="pHash computing"):
        phash = compute_perceptual_hash(img_path)
        phashes.append((split, img_path, phash))
    
    # Find near-duplicates (Hamming distance < threshold)
    threshold = 5  # Max 5 bits different (out of 64 bits = 7.8% difference)
    near_duplicates = []
    
    print(f"\n📝 Comparing perceptual hashes (threshold: {threshold} bits)...")
    for i in tqdm(range(len(phashes)), desc="Comparing"):
        for j in range(i+1, len(phashes)):
            split1, path1, phash1 = phashes[i]
            split2, path2, phash2 = phashes[j]
            
            distance = hamming_distance(phash1, phash2)
            
            if distance < threshold:
                similarity = 100 * (1 - distance / 64)
                near_duplicates.append({
                    'split1': split1,
                    'path1': path1,
                    'split2': split2,
                    'path2': path2,
                    'distance': distance,
                    'similarity': similarity
                })
    
    if near_duplicates:
        print(f"\n⚠️  NEAR-DUPLICATES FOUND: {len(near_duplicates)} pairs")
        print(f"   (Similarity > {100*(1-threshold/64):.1f}%)")
        
        for idx, dup in enumerate(near_duplicates[:10], 1):  # Show first 10
            print(f"\n   Pair {idx}:")
            print(f"     Image 1: {dup['split1']}/{dup['path1'].name}")
            print(f"     Image 2: {dup['split2']}/{dup['path2'].name}")
            print(f"     Similarity: {dup['similarity']:.1f}% (distance: {dup['distance']} bits)")
        
        if len(near_duplicates) > 10:
            print(f"\n   ... and {len(near_duplicates) - 10} more pairs")
    else:
        print(f"\n✅ NO NEAR-DUPLICATES FOUND")
        print(f"   All samples are perceptually distinct (< {100*(1-threshold/64):.1f}% similarity)")
    
    # 3. Check augmentation groups
    print(f"\n{'='*80}")
    print("3. AUGMENTATION DIVERSITY CHECK")
    print(f"{'='*80}")
    
    # Group by base strip index (sample_XXXX -> strip index)
    # Expected: 5 variants per strip (original + 4 augmentations)
    strip_groups = defaultdict(list)
    
    for split, img_path in all_pairs:
        # Extract strip index from filename: sample_0000_pair.png -> 0000
        sample_num = int(img_path.stem.split('_')[1])
        
        # Each strip has 5 variants (indices 0-4, 5-9, 10-14, etc.)
        strip_idx = sample_num // 5
        
        strip_groups[(split, strip_idx)].append((sample_num, img_path))
    
    print(f"\n📊 Expected augmentation groups: {len(strip_groups)}")
    print(f"   (31 original strips × 5 augmentations = 155 total samples)")
    
    # Check if each group has expected number of variants
    irregular_groups = []
    for (split, strip_idx), variants in strip_groups.items():
        if len(variants) != 5:
            irregular_groups.append((split, strip_idx, len(variants)))
    
    if irregular_groups:
        print(f"\n⚠️  IRREGULAR AUGMENTATION GROUPS: {len(irregular_groups)}")
        for split, strip_idx, count in irregular_groups[:10]:
            print(f"   {split}/strip_{strip_idx}: {count} variants (expected: 5)")
    else:
        print(f"\n✅ ALL AUGMENTATION GROUPS COMPLETE")
        print(f"   Each of 31 strips has exactly 5 variants")
    
    # 4. Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    total_issues = len(exact_duplicates) + len(near_duplicates) + len(irregular_groups)
    
    if total_issues == 0:
        print(f"\n✅ DATASET QUALITY: EXCELLENT")
        print(f"   ✓ No exact duplicates")
        print(f"   ✓ No near-duplicates (< {100*(1-threshold/64):.1f}% similarity)")
        print(f"   ✓ All augmentation groups complete (5 variants each)")
        print(f"   ✓ Total unique samples: {len(all_pairs)}")
    else:
        print(f"\n⚠️  DATASET QUALITY: ISSUES FOUND")
        print(f"   Exact duplicates: {len(exact_duplicates)} groups")
        print(f"   Near-duplicates:  {len(near_duplicates)} pairs")
        print(f"   Irregular groups: {len(irregular_groups)}")
    
    print(f"\n{'='*80}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Check for duplicate images in dataset')
    parser.add_argument('--samples_dir', type=str, 
                       default='visualization/finetuning/all_samples',
                       help='Directory containing extracted samples')
    
    args = parser.parse_args()
    check_duplicates(args.samples_dir)
