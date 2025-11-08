"""
Export 5 best test samples (degraded + clean) for DE-GAN comparison
"""
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import editdistance
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed as load_frozen_recognizer

def _parse_tfrecord_fn(example_proto):
    """Parse TFRecord function - matches production dataset format"""
    feature_description = {
        'degraded_image_raw': tf.io.FixedLenFeature([], tf.string),
        'degraded_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'clean_image_raw': tf.io.FixedLenFeature([], tf.string),
        'clean_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'label_shape': tf.io.FixedLenFeature([1], tf.int64),
    }
    
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    # Degraded image
    degraded_shape = tf.cast(example['degraded_image_shape'], tf.int32)
    degraded_image = tf.io.decode_raw(example['degraded_image_raw'], tf.float32)
    degraded_image = tf.reshape(degraded_image, degraded_shape)
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])
    degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])

    # Clean image
    clean_shape = tf.cast(example['clean_image_shape'], tf.int32)
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
    clean_image = tf.reshape(clean_image, clean_shape)
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])
    clean_image = tf.ensure_shape(clean_image, [1024, 128, 1])

    # Label
    label_shape = tf.cast(example['label_shape'], tf.int32)
    label = tf.io.decode_raw(example['label_raw'], tf.int64)
    label = tf.reshape(label, label_shape)
    label = tf.cast(label, tf.int32)
    padding = [[0, 128 - tf.shape(label)[0]]]
    label = tf.pad(label, padding, "CONSTANT", constant_values=0)
    label.set_shape([128])

    return degraded_image, clean_image, label

def create_test_dataset(tfrecord_path, batch_size=1, train_split=0.7, val_split=0.15):
    """Create test dataset"""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    total_size = sum(1 for _ in dataset)
    
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = dataset.skip(train_size + val_size)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    return test_dataset

def load_production_model_inference():
    """Load production generator model for inference"""
    # Create generator
    generator = unet_enhanced(input_size=(1024, 128, 1))
    
    checkpoint_dir = 'dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model'
    checkpoint = tf.train.Checkpoint(generator=generator)
    
    # Restore
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    status.expect_partial()  # Ignore discriminator/recognizer weights
    
    return generator

def read_charlist(filepath):
    """Read charset from file - matches visualize_method_comparison.py"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

def decode_label(label, charset):
    """Decode label indices to text - matches visualize_method_comparison.py"""
    decoded_chars = []
    prev_id = -1
    for label_id in label:
        label_id = int(label_id)
        if label_id > 0 and label_id != prev_id:
            if label_id - 1 < len(charset):
                decoded_chars.append(charset[label_id - 1])
        prev_id = label_id
    return ''.join(decoded_chars)

def decode_ctc_predictions(logits, charset):
    """Manual CTC decode - matches visualize_method_comparison.py"""
    charset_size = len(charset)
    batch_size = logits.shape[0]
    results = []
    
    for b in range(batch_size):
        raw_preds = np.argmax(logits[b], axis=-1)
        
        # CTC decode with deduplication
        deduped = []
        prev = -1
        for token in raw_preds:
            if token != prev:
                deduped.append(token)
                prev = token
        
        # Remove blank tokens
        result = []
        for token in deduped:
            if token != charset_size:
                result.append(token)
        
        # Convert to string
        decoded = ''.join([charset[i] if 0 <= i < len(charset) else '<?>' for i in result])
        results.append(decoded)
    
    return results

def calculate_cer(ground_truth, prediction):
    """Calculate Character Error Rate"""
    if len(ground_truth) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    distance = editdistance.eval(ground_truth, prediction)
    return distance / len(ground_truth)

def export_samples():
    """Export 5 best samples for DE-GAN comparison"""
    
    print("="*80)
    print("EXPORT TEST SAMPLES FOR DE-GAN COMPARISON")
    print("="*80)
    
    # Configuration
    TFRECORD_PATH = 'dual_modal_gan/data/dataset_gan.tfrecord'
    CHARSET_PATH = 'real_data_preparation/real_data_charlist.txt'
    RECOGNIZER_WEIGHTS = '/home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5'
    OUTPUT_DIR = 'DataUjiKuantitatifSintetis'
    
    # Create output directories
    os.makedirs(os.path.join(OUTPUT_DIR, 'degraded'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'ground_truth'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'metadata'), exist_ok=True)
    
    # Load charset
    charset = read_charlist(CHARSET_PATH)
    vocab_size = len(charset) + 1
    print(f"\nCharset: {vocab_size} characters")
    
    # Load generator to evaluate samples
    print("\n🔧 Loading generator...")
    generator = load_production_model_inference()
    print("  ✅ Generator loaded")
    
    # Load recognizer for CER evaluation
    print("  Loading frozen recognizer...")
    recognizer = load_frozen_recognizer(
        weights_path=RECOGNIZER_WEIGHTS,
        charset_size=vocab_size - 1,
        return_feature_map=False
    )
    print("  ✅ Recognizer loaded")
    
    # Load test dataset
    print(f"\n📂 Loading test dataset...")
    test_dataset = create_test_dataset(TFRECORD_PATH, batch_size=1)
    
    # FIXED INDICES - Match with visualisasi yang sudah dibuat sebelumnya
    # Diverifikasi dengan membandingkan pixel values dari file yang sudah diexport
    FIXED_INDICES = [8, 9, 48, 4, 18]  # Indices dari visualisasi
    
    print(f"\n🎨 Loading 5 FIXED samples (indices: {FIXED_INDICES})...")
    all_samples = []
    
    for idx, (degraded, clean, label) in enumerate(test_dataset.take(100)):
        # Skip jika bukan index yang diinginkan
        if idx not in FIXED_INDICES:
            continue
            
        degraded_np = degraded[0].numpy().squeeze()
        clean_np = clean[0].numpy().squeeze()
        
        # Transpose to horizontal for processing
        degraded_np = degraded_np.T  # (128, 1024)
        clean_np = clean_np.T  # (128, 1024)
        
        # Generate with proposed method
        degraded_vertical = degraded_np.T  # Back to (1024, 128)
        degraded_vertical_input = np.expand_dims(np.expand_dims(degraded_vertical, 0), -1)
        proposed_vertical = generator(tf.convert_to_tensor(degraded_vertical_input, dtype=tf.float32), training=False)[0].numpy().squeeze()
        
        # Denormalize if needed
        if proposed_vertical.min() < 0 or proposed_vertical.max() > 1:
            proposed_vertical = (proposed_vertical + 1.0) / 2.0
            proposed_vertical = np.clip(proposed_vertical, 0, 1)
        
        proposed_result = proposed_vertical.T  # To horizontal
        
        # Calculate PSNR for proposed method
        clean_v = np.expand_dims(np.expand_dims(clean_np.T, 0), -1)
        proposed_v = np.expand_dims(np.expand_dims(proposed_result.T, 0), -1)
        
        psnr_proposed = tf.image.psnr(clean_v, proposed_v, max_val=1.0).numpy()[0]
        
        # Get ground truth text
        gt_text = decode_label(label[0].numpy(), charset)
        
        # Get prediction for CER calculation
        logits = recognizer(tf.convert_to_tensor(proposed_v, dtype=tf.float32), training=False)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        
        pred_text = decode_ctc_predictions(logits.numpy(), charset)[0]
        cer = calculate_cer(gt_text, pred_text)
        
        all_samples.append({
            'idx': idx,
            'degraded': degraded_np,
            'clean': clean_np,
            'proposed': proposed_result,
            'gt_text': gt_text,
            'pred_text': pred_text,
            'psnr': psnr_proposed,
            'cer': cer
        })
        
        # Stop jika sudah dapat semua sample yang diinginkan
        if len(all_samples) == len(FIXED_INDICES):
            break
    
    # Sort by original FIXED_INDICES order
    index_order = {idx: i for i, idx in enumerate(FIXED_INDICES)}
    all_samples.sort(key=lambda x: index_order[x['idx']])
    best_samples = all_samples
    
    print(f"\n✅ Loaded {len(best_samples)} FIXED samples (SAME as visualization):")
    
    # Export samples
    metadata_lines = []
    for i, sample in enumerate(best_samples, 1):
        print(f"  Sample {i}: CER: {sample['cer']*100:.1f}%, PSNR: {sample['psnr']:.2f} dB, Text: {sample['gt_text'][:40]}...")
        
        # Save degraded image (vertical orientation for DE-GAN)
        degraded_img = (sample['degraded'].T * 255).astype(np.uint8)  # Transpose to vertical (1024, 128)
        degraded_pil = Image.fromarray(degraded_img, mode='L')
        degraded_path = os.path.join(OUTPUT_DIR, 'degraded', f'sample_{i:02d}.png')
        degraded_pil.save(degraded_path)
        
        # Save ground truth image (vertical orientation)
        clean_img = (sample['clean'].T * 255).astype(np.uint8)  # Transpose to vertical (1024, 128)
        clean_pil = Image.fromarray(clean_img, mode='L')
        clean_path = os.path.join(OUTPUT_DIR, 'ground_truth', f'sample_{i:02d}.png')
        clean_pil.save(clean_path)
        
        # Save our proposed result for reference
        proposed_img = (sample['proposed'].T * 255).astype(np.uint8)
        proposed_pil = Image.fromarray(proposed_img, mode='L')
        proposed_path = os.path.join(OUTPUT_DIR, 'metadata', f'sample_{i:02d}_proposed.png')
        proposed_pil.save(proposed_path)
        
        # Save metadata
        metadata_lines.append(f"Sample {i:02d}:")
        metadata_lines.append(f"  Original index: {sample['idx']}")
        metadata_lines.append(f"  CER (Proposed): {sample['cer']*100:.1f}%")
        metadata_lines.append(f"  PSNR (Proposed): {sample['psnr']:.2f} dB")
        metadata_lines.append(f"  Ground truth text: {sample['gt_text']}")
        metadata_lines.append(f"  Predicted text: {sample['pred_text']}")
        metadata_lines.append(f"  Image shape: (1024, 128) - vertical orientation")
        metadata_lines.append("")
    
    # Save metadata file
    metadata_path = os.path.join(OUTPUT_DIR, 'metadata', 'samples_info.txt')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(metadata_lines))
    
    # Create README
    readme_content = f"""# Data Uji Kuantitatif Sintetis

Dataset ini berisi 5 sampel yang SAMA PERSIS dengan visualisasi perbandingan metode.
**Kriteria: FIXED indices {FIXED_INDICES} yang diverifikasi berdasarkan pixel matching.**

## Struktur Direktori:
- `degraded/` - Gambar dokumen terdegradasi (input untuk DE-GAN)
- `ground_truth/` - Gambar dokumen bersih (ground truth untuk evaluasi)
- `metadata/` - Informasi sampel dan hasil metode usulan untuk referensi

## Format Gambar:
- Dimensi: 1024 × 128 pixels (vertical orientation)
- Format: Grayscale PNG
- Range nilai: [0, 255]

## Penggunaan:
1. Gunakan gambar di folder `degraded/` sebagai input untuk DE-GAN
2. Bandingkan hasil restorasi DE-GAN dengan `ground_truth/` untuk menghitung metrik (PSNR, SSIM, CER)
3. Lihat `metadata/samples_info.txt` untuk informasi detail setiap sampel
4. Hasil metode usulan tersedia di `metadata/*_proposed.png` untuk perbandingan visual

## Metrik Baseline (Metode Usulan):
Lihat file `metadata/samples_info.txt` untuk nilai PSNR setiap sampel dari metode usulan.

Generated: {tf.timestamp().numpy()}
"""
    
    readme_path = os.path.join(OUTPUT_DIR, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n📁 Files saved to: {OUTPUT_DIR}/")
    print(f"   - {len(best_samples)} degraded images")
    print(f"   - {len(best_samples)} ground truth images")
    print(f"   - {len(best_samples)} proposed results (reference)")
    print(f"   - metadata/samples_info.txt")
    print(f"   - README.md")
    
    print(f"\n{'='*80}")
    print("✅ Export completed!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    export_samples()
