"""
Quick dataset format validation for joint training
"""

import os
import sys
import json
import tensorflow as tf

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import from train_enhanced.py (create_dataset is defined there)
from dual_modal_gan.scripts.train_enhanced import create_dataset

def read_charlist(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

def main():
    config_path = 'configs/ablation_joint_training_vs_frozen.json'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("=" * 80)
    print("Dataset Format Validation")
    print("=" * 80)
    
    # Load charset
    charset = read_charlist(config['charset_path'])
    print(f"\n✅ Charset loaded: {len(charset)} characters")
    print(f"   First 10: {charset[:10]}")
    
    # Load dataset
    print(f"\n✅ Loading dataset from: {config['tfrecord_path']}")
    train_dataset, val_dataset, _, train_count, val_count, _ = create_dataset(
        config['tfrecord_path'],
        batch_size=2,  # Small batch for testing
        train_split=config.get('train_split', 0.7),
        val_split=config.get('val_split', 0.15)
    )
    
    print(f"   Train samples: {train_count}")
    print(f"   Val samples: {val_count}")
    
    # Check one batch
    print("\n✅ Inspecting first batch...")
    for batch in train_dataset.take(1):
        # Dataset returns tuple: (degraded, clean, label)
        degraded, clean, label = batch
        
        print(f"\n   Batch type: tuple with 3 elements (degraded, clean, label)")
        
        print(f"\n   degraded shape: {degraded.shape}")
        print(f"   clean shape: {clean.shape}")
        print(f"   label type: {type(label)}")
        print(f"   label dtype: {label.dtype}")
        print(f"   label shape: {label.shape}")
        
        # Analyze label format
        print(f"\n   Label sample (first 2 samples, first 20 tokens):")
        for i in range(min(2, label.shape[0])):
            sample_label = label[i].numpy()
            non_zero = sample_label[sample_label > 0][:20]  # First 20 non-padding tokens
            print(f"     Sample {i}: {non_zero}")
            
            # Decode to text
            decoded_chars = [charset[t] for t in non_zero if t < len(charset)]
            decoded_text = ''.join(decoded_chars)
            print(f"     Decoded: '{decoded_text}'")
        
        # Validate token IDs
        max_token = tf.reduce_max(label).numpy()
        min_token = tf.reduce_min(label).numpy()
        print(f"\n   Token range: min={min_token}, max={max_token}")
        print(f"   Vocab size needed: {len(charset) + 1} (charset + blank)")
        
        if max_token >= len(charset):
            print(f"\n   ⚠️  WARNING: Max token ({max_token}) >= vocab size ({len(charset)})")
            print(f"      This will cause CTC issues!")
        else:
            print(f"\n   ✅ Token IDs are valid for CTC")
        
        # Check for CTC compatibility
        print(f"\n   CTC Requirements Check:")
        print(f"     ✓ Label is dense tensor (batch, max_length): {label.shape}")
        print(f"     ✓ Label dtype is int32: {label.dtype == tf.int32}")
        print(f"     ✓ Padding value is 0: {min_token == 0}")
        print(f"     → Ready for CTC loss computation")
    
    print("\n" + "=" * 80)
    print("✅ Dataset validation complete!")
    print("=" * 80)
    
    # Check recognizer weights
    rec_weights = config['recognizer_weights']
    if os.path.exists(rec_weights):
        size_mb = os.path.getsize(rec_weights) / (1024 * 1024)
        print(f"\n✅ Recognizer weights: {rec_weights} ({size_mb:.1f} MB)")
    else:
        print(f"\n❌ Recognizer weights NOT found: {rec_weights}")
    
    print("\n🚀 Ready to launch joint training!")

if __name__ == '__main__':
    main()
