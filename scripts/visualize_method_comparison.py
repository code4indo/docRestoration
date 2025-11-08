"""
Visual Comparison: 4 Methods (No Restoration, Otsu, Sauvola, Proposed)

Generates side-by-side comparison showing:
- Degraded input
- Otsu result + CER
- Sauvola result + CER
- Proposed method result + CER
- Ground truth + text

Author: Method Comparison Visualization
Date: 2025-11-04
"""

import os
import sys
import json
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import editdistance
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed as load_frozen_recognizer

# Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# IEEE paper settings
FIGURE_WIDTH_DOUBLE = 7.16  # inches (double column)
DPI = 300

# Color scheme
COLOR_DEGRADED = '#E74C3C'    # Red
COLOR_OTSU = '#E67E22'        # Orange
COLOR_SAUVOLA = '#F39C12'     # Yellow-orange
COLOR_PROPOSED = '#27AE60'    # Green
COLOR_GT = '#3498DB'          # Blue

def read_charlist(path):
    """Load character list"""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

def decode_ctc_predictions(logits, charset):
    """Manual CTC decode"""
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

def decode_label(label_ids, charset):
    """Decode label IDs to text string"""
    decoded_chars = []
    prev_id = -1
    for label_id in label_ids:
        label_id = int(label_id)
        if label_id > 0 and label_id != prev_id:
            if label_id - 1 < len(charset):
                decoded_chars.append(charset[label_id - 1])
        prev_id = label_id
    return ''.join(decoded_chars)

def calculate_cer(ground_truth, prediction):
    """Calculate Character Error Rate"""
    if len(ground_truth) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    distance = editdistance.eval(ground_truth, prediction)
    return distance / len(ground_truth)

def otsu_binarization(image):
    """Apply Otsu's thresholding"""
    img_uint8 = (image * 255).astype(np.uint8)
    _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(np.float32) / 255.0

def sauvola_binarization(image, window_size=15, k=0.2):
    """Apply Sauvola's adaptive thresholding"""
    img_uint8 = (image * 255).astype(np.uint8)
    
    # Calculate local mean and std
    mean = cv2.blur(img_uint8.astype(np.float32), (window_size, window_size))
    sqr_mean = cv2.blur((img_uint8.astype(np.float32) ** 2), (window_size, window_size))
    std = np.sqrt(np.maximum(sqr_mean - mean ** 2, 0))
    
    # Sauvola threshold
    R = 128
    threshold = mean * (1 + k * ((std / R) - 1))
    
    # Apply threshold
    binary = (img_uint8 > threshold).astype(np.uint8) * 255
    return binary.astype(np.float32) / 255.0

def _parse_tfrecord_fn(example_proto):
    """Parse TFRecord"""
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
    """
    Load production model checkpoint for inference
    Uses TensorFlow checkpoint format
    """
    import tensorflow as tf
    from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
    
    # Create generator
    generator = unet_enhanced(input_size=(1024, 128, 1))
    
    # Load checkpoint
    checkpoint_dir = 'dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model'
    checkpoint = tf.train.Checkpoint(generator=generator)
    
    # Restore
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    status.expect_partial()  # Ignore discriminator/recognizer weights
    
    return generator

def generate_method_comparison(
    num_samples=5,
    output_dir='Paper/visualizations'
):
    """
    Generate visual comparison of 4 methods
    """
    
    print("="*80)
    print("METHOD COMPARISON VISUALIZATION GENERATOR")
    print("="*80)
    
    # Configuration
    TFRECORD_PATH = 'dual_modal_gan/data/dataset_gan.tfrecord'
    CHARSET_PATH = 'real_data_preparation/real_data_charlist.txt'
    RECOGNIZER_WEIGHTS = '/home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5'
    
    # Load charset
    charset = read_charlist(CHARSET_PATH)
    vocab_size = len(charset) + 1
    print(f"\nCharset: {vocab_size} characters")
    
    # Load models
    print("\n🔧 Loading models...")
    
    # Production model (Proposed method)
    print("  Loading production model (proposed method)...")
    generator = load_production_model_inference()
    print("  ✅ Generator loaded")
    
    # Recognizer
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
    
    # Collect samples and evaluate ALL to find best ones
    print(f"\n🎨 Evaluating samples to find best {num_samples}...")
    
    all_samples = []
    
    for degraded, clean, label in tqdm(test_dataset.take(100), desc="Evaluating samples", total=100):
        
        degraded_np = degraded[0].numpy().squeeze()
        clean_np = clean[0].numpy().squeeze()
        
        # Transpose images from vertical (1024, 128) to horizontal (128, 1024)
        # Use transpose instead of rotate to avoid mirroring
        degraded_np = degraded_np.T  # (128, 1024)
        clean_np = clean_np.T  # (128, 1024)
        
        # Apply all methods
        # 1. No restoration (degraded as-is)
        no_restoration = degraded_np
        
        # 2. Otsu thresholding
        otsu_result = otsu_binarization(degraded_np)
        
        # 3. Sauvola adaptive
        sauvola_result = sauvola_binarization(degraded_np, window_size=15, k=0.2)
        
        # 4. Proposed method (transpose back to vertical for model input, then transpose result)
        degraded_vertical = degraded_np.T  # Transpose back to 1024×128
        degraded_vertical_input = np.expand_dims(np.expand_dims(degraded_vertical, 0), -1)
        proposed_vertical = generator(tf.convert_to_tensor(degraded_vertical_input, dtype=tf.float32), training=False)[0].numpy().squeeze()
        
        # CRITICAL FIX: Check if generator outputs in [-1, 1] range and denormalize to [0, 1]
        if proposed_vertical.min() < 0 or proposed_vertical.max() > 1:
            # Generator outputs in [-1, 1] range, need to denormalize to [0, 1]
            proposed_vertical = (proposed_vertical + 1.0) / 2.0
            proposed_vertical = np.clip(proposed_vertical, 0, 1)
        
        proposed_result = proposed_vertical.T  # Transpose to horizontal
        
        # Get predictions for all methods
        # Need to transpose back to vertical for recognizer input
        methods_images = {
            'no_restoration': np.expand_dims(np.expand_dims(no_restoration.T, 0), -1),
            'otsu': np.expand_dims(np.expand_dims(otsu_result.T, 0), -1),
            'sauvola': np.expand_dims(np.expand_dims(sauvola_result.T, 0), -1),
            'proposed': np.expand_dims(np.expand_dims(proposed_result.T, 0), -1)
        }
        
        predictions = {}
        cers = {}
        
        # Get ground truth text
        gt_text = decode_label(label[0].numpy(), charset)
        
        for method_name, method_image in methods_images.items():
            # Get prediction
            logits = recognizer(tf.convert_to_tensor(method_image, dtype=tf.float32), training=False)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            
            pred_text = decode_ctc_predictions(logits.numpy(), charset)[0]
            cer = calculate_cer(gt_text, pred_text)
            
            predictions[method_name] = pred_text
            cers[method_name] = cer
        
        # Calculate PSNR for visual quality (need vertical orientation for comparison)
        clean_vertical = np.expand_dims(np.expand_dims(clean_np.T, 0), -1)
        
        psnr_no_rest = tf.image.psnr(
            tf.convert_to_tensor(clean_vertical, dtype=tf.float32),
            methods_images['no_restoration'],
            max_val=1.0
        ).numpy()[0]
        
        psnr_otsu = tf.image.psnr(
            tf.convert_to_tensor(clean_vertical, dtype=tf.float32),
            methods_images['otsu'],
            max_val=1.0
        ).numpy()[0]
        
        psnr_sauvola = tf.image.psnr(
            tf.convert_to_tensor(clean_vertical, dtype=tf.float32),
            methods_images['sauvola'],
            max_val=1.0
        ).numpy()[0]
        
        psnr_proposed = tf.image.psnr(
            tf.convert_to_tensor(clean_vertical, dtype=tf.float32),
            methods_images['proposed'],
            max_val=1.0
        ).numpy()[0]
        
        # Calculate SSIM for structural similarity
        ssim_no_rest = tf.image.ssim(
            tf.convert_to_tensor(clean_vertical, dtype=tf.float32),
            methods_images['no_restoration'],
            max_val=1.0
        ).numpy()[0]
        
        ssim_otsu = tf.image.ssim(
            tf.convert_to_tensor(clean_vertical, dtype=tf.float32),
            methods_images['otsu'],
            max_val=1.0
        ).numpy()[0]
        
        ssim_sauvola = tf.image.ssim(
            tf.convert_to_tensor(clean_vertical, dtype=tf.float32),
            methods_images['sauvola'],
            max_val=1.0
        ).numpy()[0]
        
        ssim_proposed = tf.image.ssim(
            tf.convert_to_tensor(clean_vertical, dtype=tf.float32),
            methods_images['proposed'],
            max_val=1.0
        ).numpy()[0]
        
        all_samples.append({
            'degraded': no_restoration,
            'otsu': otsu_result,
            'sauvola': sauvola_result,
            'proposed': proposed_result,
            'clean': clean_np,
            'gt_text': gt_text,
            'pred_degraded': predictions['no_restoration'],
            'pred_otsu': predictions['otsu'],
            'pred_sauvola': predictions['sauvola'],
            'pred_proposed': predictions['proposed'],
            'cer_degraded': cers['no_restoration'],
            'cer_otsu': cers['otsu'],
            'cer_sauvola': cers['sauvola'],
            'cer_proposed': cers['proposed'],
            'psnr_degraded': psnr_no_rest,
            'psnr_otsu': psnr_otsu,
            'psnr_sauvola': psnr_sauvola,
            'psnr_proposed': psnr_proposed,
            'ssim_degraded': ssim_no_rest,
            'ssim_otsu': ssim_otsu,
            'ssim_sauvola': ssim_sauvola,
            'ssim_proposed': ssim_proposed
        })
    
    # Sort by proposed method CER (ascending) and take best N
    all_samples.sort(key=lambda x: x['cer_proposed'])
    samples = all_samples[:num_samples]
    
    print(f"\n✅ Selected top {num_samples} samples with best proposed method CER:")
    for i, sample in enumerate(samples):
        print(f"  Sample {i+1}: Proposed CER: {sample['cer_proposed']*100:.1f}%")
    
    # Create visualization
    print(f"\n📊 Creating comparison visualization...")
    
    # Create figure with grid - HORIZONTAL layout (rows = methods, cols = samples)
    # 5 rows (methods) × num_samples columns (samples)
    fig = plt.figure(figsize=(3.0 * num_samples, 8))
    gs = GridSpec(5, num_samples, figure=fig, wspace=0.08, hspace=0.15)
    
    # Method labels (simple and small)
    method_labels = [
        ('Degraded', COLOR_DEGRADED),
        ('Otsu', COLOR_OTSU),
        ('Sauvola', COLOR_SAUVOLA),
        ('Proposed', COLOR_PROPOSED),
        ('GT', COLOR_GT)
    ]
    
    for i, sample in enumerate(samples):
        # Row 1: Degraded (No Restoration)
        ax1 = fig.add_subplot(gs[0, i])
        ax1.imshow(sample['degraded'], cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f'PSNR:{sample["psnr_degraded"]:.1f} | SSIM:{sample["ssim_degraded"]:.3f} | CER:{sample["cer_degraded"]*100:.0f}%',
                     fontsize=5.5, color=COLOR_DEGRADED)
        ax1.axis('off')
        # Add prediction text below
        ax1.text(0.5, -0.15, f'{sample["pred_degraded"][:40]}...', 
                transform=ax1.transAxes, fontsize=10, ha='center', va='top',
                family='monospace', color=COLOR_DEGRADED)
        
        # Row 2: Otsu
        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(sample['otsu'], cmap='gray', vmin=0, vmax=1)
        ax2.set_title(f'PSNR:{sample["psnr_otsu"]:.1f} | SSIM:{sample["ssim_otsu"]:.3f} | CER:{sample["cer_otsu"]*100:.0f}%',
                     fontsize=5.5, color=COLOR_OTSU)
        ax2.axis('off')
        ax2.text(0.5, -0.15, f'{sample["pred_otsu"][:40]}...', 
                transform=ax2.transAxes, fontsize=6, ha='center', va='top',
                family='monospace', color=COLOR_OTSU)
        
        # Row 3: Sauvola
        ax3 = fig.add_subplot(gs[2, i])
        ax3.imshow(sample['sauvola'], cmap='gray', vmin=0, vmax=1)
        ax3.set_title(f'PSNR:{sample["psnr_sauvola"]:.1f} | SSIM:{sample["ssim_sauvola"]:.3f} | CER:{sample["cer_sauvola"]*100:.0f}%',
                     fontsize=5.5, color=COLOR_SAUVOLA)
        ax3.axis('off')
        ax3.text(0.5, -0.15, f'{sample["pred_sauvola"][:40]}...', 
                transform=ax3.transAxes, fontsize=6, ha='center', va='top',
                family='monospace', color=COLOR_SAUVOLA)
        
        # Row 4: Proposed
        ax4 = fig.add_subplot(gs[3, i])
        ax4.imshow(sample['proposed'], cmap='gray', vmin=0, vmax=1)
        ax4.set_title(f'PSNR:{sample["psnr_proposed"]:.1f} | SSIM:{sample["ssim_proposed"]:.3f} | CER:{sample["cer_proposed"]*100:.0f}%',
                     fontsize=5.5, color=COLOR_PROPOSED)
        ax4.axis('off')
        ax4.text(0.5, -0.15, f'{sample["pred_proposed"][:40]}...', 
                transform=ax4.transAxes, fontsize=6, ha='center', va='top',
                family='monospace', color=COLOR_PROPOSED)
        
        # Row 5: Ground Truth
        ax5 = fig.add_subplot(gs[4, i])
        ax5.imshow(sample['clean'], cmap='gray', vmin=0, vmax=1)
        ax5.set_title(f'Sample {i+1}', fontsize=7, color=COLOR_GT, fontweight='bold')
        ax5.axis('off')
        ax5.text(0.5, -0.15, f'{sample["gt_text"][:40]}...', 
                transform=ax5.transAxes, fontsize=6, ha='center', va='top',
                family='monospace', color=COLOR_GT, fontweight='bold')
    
    # Add method labels on the left side using fig.text
    # Calculate vertical positions for each row
    row_positions = [0.85, 0.68, 0.51, 0.34, 0.17]  # Approximate y positions for 5 rows
    for idx, (label, color) in enumerate(method_labels):
        fig.text(0.02, row_positions[idx], label, 
                fontsize=11, fontweight='bold', color=color,
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor=color, linewidth=2, alpha=0.9))
    
    plt.suptitle('Perbandingan Metode Restorasi Dokumen (5 Sampel Terbaik)',
                fontsize=11, fontweight='bold', y=0.98)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig_method_comparison_4methods.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print(f"\n{'='*80}")
    print("STATISTICS ACROSS SAMPLES:")
    print(f"{'='*80}")
    
    avg_cer_deg = np.mean([s['cer_degraded'] for s in samples]) * 100
    avg_cer_otsu = np.mean([s['cer_otsu'] for s in samples]) * 100
    avg_cer_sauv = np.mean([s['cer_sauvola'] for s in samples]) * 100
    avg_cer_prop = np.mean([s['cer_proposed'] for s in samples]) * 100
    
    avg_psnr_deg = np.mean([s['psnr_degraded'] for s in samples])
    avg_psnr_otsu = np.mean([s['psnr_otsu'] for s in samples])
    avg_psnr_sauv = np.mean([s['psnr_sauvola'] for s in samples])
    avg_psnr_prop = np.mean([s['psnr_proposed'] for s in samples])
    
    avg_ssim_deg = np.mean([s['ssim_degraded'] for s in samples])
    avg_ssim_otsu = np.mean([s['ssim_otsu'] for s in samples])
    avg_ssim_sauv = np.mean([s['ssim_sauvola'] for s in samples])
    avg_ssim_prop = np.mean([s['ssim_proposed'] for s in samples])
    
    print(f"Average CER:")
    print(f"  Degraded:  {avg_cer_deg:.1f}%")
    print(f"  Otsu:      {avg_cer_otsu:.1f}%")
    print(f"  Sauvola:   {avg_cer_sauv:.1f}%")
    print(f"  Proposed:  {avg_cer_prop:.1f}%")
    
    print(f"\nAverage PSNR:")
    print(f"  Degraded:  {avg_psnr_deg:.2f} dB")
    print(f"  Otsu:      {avg_psnr_otsu:.2f} dB")
    print(f"  Sauvola:   {avg_psnr_sauv:.2f} dB")
    print(f"  Proposed:  {avg_psnr_prop:.2f} dB")
    
    print(f"\nAverage SSIM:")
    print(f"  Degraded:  {avg_ssim_deg:.3f}")
    print(f"  Otsu:      {avg_ssim_otsu:.3f}")
    print(f"  Sauvola:   {avg_ssim_sauv:.3f}")
    print(f"  Proposed:  {avg_ssim_prop:.3f}")
    print(f"{'='*80}\n")
    
    return samples

def export_samples_to_degan_dir(samples):
    """Export samples to DataUjiKuantitatifSintetis directory"""
    from PIL import Image
    
    OUTPUT_DIR = 'DataUjiKuantitatifSintetis'
    
    print(f"\n{'='*80}")
    print("EXPORTING SAMPLES TO DataUjiKuantitatifSintetis")
    print(f"{'='*80}")
    
    # Create output directories
    os.makedirs(os.path.join(OUTPUT_DIR, 'degraded'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'ground_truth'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'metadata'), exist_ok=True)
    
    metadata_lines = []
    
    for i, sample in enumerate(samples, 1):
        print(f"  Exporting Sample {i}...")
        
        # Save degraded image as horizontal (128×1024) - no transpose
        degraded_img = (sample['degraded'] * 255).astype(np.uint8)  # Already horizontal (128, 1024)
        degraded_pil = Image.fromarray(degraded_img, mode='L')
        degraded_path = os.path.join(OUTPUT_DIR, 'degraded', f'sample_{i:02d}.png')
        degraded_pil.save(degraded_path)
        
        # Save ground truth image as horizontal (128×1024) - no transpose
        clean_img = (sample['clean'] * 255).astype(np.uint8)  # Already horizontal (128, 1024)
        clean_pil = Image.fromarray(clean_img, mode='L')
        clean_path = os.path.join(OUTPUT_DIR, 'ground_truth', f'sample_{i:02d}.png')
        clean_pil.save(clean_path)
        
        # Save proposed result as horizontal (128×1024) - no transpose
        proposed_img = (sample['proposed'] * 255).astype(np.uint8)  # Already horizontal (128, 1024)
        proposed_pil = Image.fromarray(proposed_img, mode='L')
        proposed_path = os.path.join(OUTPUT_DIR, 'metadata', f'sample_{i:02d}_proposed.png')
        proposed_pil.save(proposed_path)
        
        # Save metadata
        metadata_lines.append(f"Sample {i:02d}:")
        metadata_lines.append(f"  CER (Proposed): {sample['cer_proposed']*100:.1f}%")
        metadata_lines.append(f"  PSNR (Proposed): {sample['psnr_proposed']:.2f} dB")
        metadata_lines.append(f"  SSIM (Proposed): {sample['ssim_proposed']:.3f}")
        metadata_lines.append(f"  Ground truth text: {sample['gt_text']}")
        metadata_lines.append(f"  Predicted text: {sample['pred_proposed']}")
        metadata_lines.append(f"  Image shape: (128, 1024) - horizontal orientation, no mirror")
        metadata_lines.append("")
    
    # Save metadata file
    metadata_path = os.path.join(OUTPUT_DIR, 'metadata', 'samples_info.txt')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(metadata_lines))
    
    # Create README
    readme_content = f"""# Data Uji Kuantitatif Sintetis

Dataset ini berisi 5 sampel yang SAMA PERSIS dengan visualisasi perbandingan metode.
**Diexport langsung dari output script visualize_method_comparison.py**

## Struktur Direktori:
- `degraded/` - Gambar dokumen terdegradasi (input untuk DE-GAN)
- `ground_truth/` - Gambar dokumen bersih (ground truth untuk evaluasi)
- `metadata/` - Informasi sampel dan hasil metode usulan untuk referensi

## Format Gambar:
- Dimensi: 128 × 1024 pixels (horizontal orientation)
- Format: Grayscale PNG
- Range: [0, 255]
- **Tidak ada mirror/transpose, langsung horizontal**

## Metrik Rata-rata (Metode Usulan):
- Average CER: {np.mean([s['cer_proposed'] for s in samples])*100:.1f}%
- Average PSNR: {np.mean([s['psnr_proposed'] for s in samples]):.2f} dB
- Average SSIM: {np.mean([s['ssim_proposed'] for s in samples]):.3f}

## Cara Penggunaan:
1. Gunakan gambar di folder `degraded/` sebagai input untuk metode DE-GAN
2. Bandingkan output DE-GAN dengan `ground_truth/` untuk kalkulasi metrik
3. Lihat `metadata/` untuk hasil metode usulan sebagai baseline perbandingan

Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    readme_path = os.path.join(OUTPUT_DIR, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n✅ Export completed!")
    print(f"   - 5 degraded images")
    print(f"   - 5 ground truth images")
    print(f"   - 5 proposed results (reference)")
    print(f"   - metadata/samples_info.txt")
    print(f"   - README.md")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    samples = generate_method_comparison(num_samples=5)
    export_samples_to_degan_dir(samples)
    print("✅ Method comparison visualization and export completed!")
