#!/usr/bin/env python3
"""
🚀 RELIABLE V3: Baseline + Proven Improvements - Target CER 25-28%

PRAGMATIC APPROACH: Focus on HIGH-IMPACT improvements within GPU constraints
==============================================================================
Baseline (V2) - CER 33.72%:
1. ✅ 6 transformer layers (proven)
2. ✅ Batch size 32
3. ✅ Post-LayerNorm
4. ✅ Basic augmentation
5. ✅ Cosine LR schedule
6. ✅ Label smoothing 0.1

V3 IMPROVEMENTS (GPU memory constrained):
7. ✅ Pre-LayerNorm architecture (proven better gradient flow)
8. ✅ OneCycle LR schedule (proven superior to cosine)
9. ✅ Enhanced augmentation (wider brightness/contrast, gamma correction)
10. ✅ More training (200 → 300 epochs)
11. ✅ Higher label smoothing (0.1 → 0.12)
12. ✅ Peak LR increased (3e-4 → 5e-4 OneCycle)

NOTE: Deeper networks (7-8 layers) caused OOM on available GPU (1120 MB).
      Focus on quality improvements rather than capacity increases.

REALISTIC TARGET: CER 25-28% (~25-35% improvement from baseline 33.72%)
"""
import os
import sys
import argparse
import subprocess
import json
import numpy as np
import tensorflow as tf
import jiwer
from pyctcdecode import build_ctcdecoder
from tensorflow.keras import layers, Model, mixed_precision
from tensorflow.keras.optimizers import AdamW
import warnings
import logging
import random
import time

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# CRITICAL FIX: Force use GPU 1 only (GPU 0 is occupied by other process with 13GB usage)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Only GPU 1 (14GB free) visible to TF
tf.get_logger().setLevel('ERROR')

# ===================== UTILS (Inline) =====================
# Paths
CHARSET_PATH = '/home/lambda_one/tesis/GAN-HTR-ORI/real_data_preparation/real_data_charlist.txt'
DEFAULT_TFRECORD_PATH = '/home/lambda_one/tesis/GAN-HTR-ORI/real_data_final_fixed_v2.tfrecord'

# Parameters
IMG_WIDTH = 1024
IMG_HEIGHT = 128
MAX_LABEL_LENGTH = 128

def setup_gpu():
    """Setup GPU memory growth - USING GPU 1 ONLY"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Found {len(gpus)} GPU(s), memory growth enabled")
            print(f"   Using GPU 1 only (CUDA_VISIBLE_DEVICES=1) to avoid OOM on GPU 0")
        else:
            print("⚠️  No GPUs found, using CPU")
    except Exception as e:
        print(f"⚠️  GPU setup error: {e}")

def read_charlist(file_path):
    """Read character list from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        charset = []
        for line in f:
            content = line.rstrip('\n')
            if content == ' ':
                charset.append(' ')
            elif content:
                charset.append(content)
    return charset

def parse_tfrecord(example, char_to_num):
    """Parse TFRecord with support for both old and new formats"""
    feature_description = {
        'image_degraded': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'text': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }
    parsed = tf.io.parse_single_example(example, feature_description)
    
    # Parse image
    def _parse_serialized():
        t = tf.io.parse_tensor(parsed['image'], out_type=tf.float32)
        t = tf.cond(tf.equal(tf.rank(t), 2), lambda: tf.expand_dims(t, -1), lambda: t)
        t.set_shape([IMG_HEIGHT, IMG_WIDTH, 1])
        return t
    
    def _decode_legacy():
        img_legacy = tf.io.decode_png(parsed['image_degraded'], channels=1)
        img_legacy = tf.image.convert_image_dtype(img_legacy, tf.float32)
        return img_legacy
    
    img = tf.cond(tf.not_equal(parsed['image'], tf.constant(b'')), _parse_serialized, _decode_legacy)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.transpose(img, perm=[1, 0, 2])  # (W, H, C)
    
    # Parse label
    def parse_numeric_label():
        raw = parsed['label']
        seq = tf.io.parse_tensor(raw, out_type=tf.int64)
        seq = tf.cast(seq, tf.int32)
        return seq
    
    def parse_text_label():
        text = parsed['text']
        tokens = tf.strings.split(text, sep=' ')
        tokens = tokens[tokens != '']
        seq = char_to_num.lookup(tokens)
        seq = tf.cast(seq, tf.int32)
        return seq
    
    label_seq = tf.cond(tf.not_equal(parsed['label'], b''), parse_numeric_label, parse_text_label)
    current_len = tf.shape(label_seq)[0]
    pad_len = MAX_LABEL_LENGTH - current_len
    label = tf.pad(label_seq, [[0, pad_len]], constant_values=0)
    label = tf.ensure_shape(label, [MAX_LABEL_LENGTH])
    
    return img, label

# V3 RELIABLE CONFIGURATION - PROVEN BASELINE + INCREMENTAL IMPROVEMENTS
EPOCHS = 300  # Increased from 200 (more training for deeper network)
LEARNING_RATE = 3e-4  # Same as baseline
MAX_LR = 5e-4  # For OneCycle schedule
BATCH_SIZE = 32  # Back to baseline (GPU 1 has 14GB free)
NUM_LAYERS = 6  # Same as proven baseline (7-8 layers OOM on available GPU memory)
NUM_HEADS = 8  # Same as baseline (proven)
FF_DIM = 2048  # Same as baseline (proven)
PROJ_DIM = 512  # Same as baseline (proven)
WEIGHT_DECAY = 2e-4  # Same as baseline
GRADIENT_CLIP_NORM = 1.0  # ADD gradient clipping
DROPOUT_RATE = 0.20  # Same as proven baseline
LABEL_SMOOTHING = 0.12  # Slightly higher than 0.1

# Learning rate schedule (OneCycle - proven better than cosine)
WARMUP_EPOCHS = 30  # 10% of total epochs
LR_FACTOR = 0.5
LR_PATIENCE = 10  # More patience for deeper network
MIN_LR = 1e-7

def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

# ===================== ENHANCED DATA AUGMENTATION (RandAugment-inspired) =====================
def augment_image(image):
    """Apply RandAugment-inspired augmentation to HTR images."""
    # Random brightness (WIDER range for better generalization)
    image = tf.image.random_brightness(image, max_delta=0.30)  # Increased from 0.20
    
    # Random contrast (WIDER range)
    image = tf.image.random_contrast(image, lower=0.70, upper=1.30)  # Wider from 0.80-1.20
    
    # Gaussian noise (stronger variance for robustness)
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.10, dtype=image.dtype)  # Increased from 0.08
    image = image + noise
    
    # Random gamma correction (simulate lighting variations)
    gamma = tf.random.uniform((), minval=0.8, maxval=1.2)
    image = tf.pow(tf.clip_by_value(image, 0.0, 1.0), gamma)
    
    # Clip to valid range [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image

def parse_and_augment(serialized_example, char_to_num, augment=True):
    """Parse TFRecord with optional augmentation"""
    image, label = parse_tfrecord(serialized_example, char_to_num)
    
    if augment:
        image = augment_image(image)
    
    return image, label

# ===================== IMPROVED MODEL =====================
def create_improved_model(charset_size, proj_dim=512, target_time_steps=128, 
                         num_transformer_layers=6, dropout_rate=0.20):
    """
    IMPROVED Model Architecture:
    - Deeper transformer (6 layers default)
    - Proper dropout usage
    - Better CNN backbone
    - Layer scaling for stability
    """
    inputs = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1), name='image_input')
    x = inputs
    
    # ========== CNN BACKBONE (Improved) ==========
    def conv_block(inp, filters, k=3, s=(1,1), name_prefix='cb', dropout=0.0):
        """Proper conv block with BatchNorm and correct dropout"""
        y = layers.Conv2D(filters, k, strides=s, padding='same', 
                         use_bias=False, name=f'{name_prefix}_conv')(inp)
        y = layers.BatchNormalization(name=f'{name_prefix}_bn')(y)
        y = layers.Activation('gelu', name=f'{name_prefix}_gelu')(y)
        if dropout > 0:
            # FIXED: Remove training=False, let Keras handle it automatically
            y = layers.Dropout(dropout, name=f'{name_prefix}_drop')(y)
        return y
    
    # Progressive feature extraction
    x = conv_block(x, 64, k=7, s=(1,2), name_prefix='s1_1', dropout=dropout_rate*0.5)
    x = conv_block(x, 64, k=3, s=(1,1), name_prefix='s1_2', dropout=dropout_rate*0.5)
    x = layers.MaxPooling2D(pool_size=(2,2), name='pool1')(x)
    
    x = conv_block(x, 128, k=3, s=(1,1), name_prefix='s2_1', dropout=dropout_rate*0.7)
    x = conv_block(x, 128, k=3, s=(1,1), name_prefix='s2_2', dropout=dropout_rate*0.7)
    x = layers.MaxPooling2D(pool_size=(2,2), name='pool2')(x)
    
    x = conv_block(x, 256, k=3, s=(1,1), name_prefix='s3_1', dropout=dropout_rate)
    x = conv_block(x, 256, k=3, s=(1,1), name_prefix='s3_2', dropout=dropout_rate)
    x = layers.MaxPooling2D(pool_size=(2,1), name='pool3')(x)
    
    x = conv_block(x, 512, k=3, s=(1,1), name_prefix='s4_1', dropout=dropout_rate)
    x = conv_block(x, 512, k=3, s=(1,1), name_prefix='s4_2', dropout=dropout_rate)
    
    # ========== SEQUENCE PROJECTION ==========
    x = layers.Lambda(
        lambda t: tf.reshape(t, (tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2]*tf.shape(t)[3])), 
        name='flatten_height'
    )(x)
    
    x = layers.Dense(proj_dim, name='proj_dense')(x)
    x = layers.LayerNormalization(name='proj_ln')(x)
    x = layers.Dropout(dropout_rate, name='proj_drop')(x)
    
    # ========== POSITIONAL ENCODING ==========
    seq_len = target_time_steps
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_embedding_layer = layers.Embedding(
        input_dim=seq_len, 
        output_dim=proj_dim, 
        name='positional_embedding'
    )
    x = x + pos_embedding_layer(positions)
    
    # ========== TRANSFORMER ENCODER (PRE-LAYERNORM FOR DEEP NETWORKS) ==========
    # Pre-LN proven better for deep networks (8 layers)
    # Ref: "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
    for i in range(num_transformer_layers):
        # Pre-LN Multi-head attention
        attn_norm = layers.LayerNormalization(name=f'trn_ln1_{i}')(x)
        attn = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, 
            key_dim=proj_dim // NUM_HEADS, 
            dropout=dropout_rate,
            name=f'trn_attn_{i}'
        )(attn_norm, attn_norm)
        attn = layers.Dropout(dropout_rate, name=f'trn_attn_drop_{i}')(attn)
        x = x + attn  # Residual connection
        
        # Pre-LN Feed-forward network
        ffn_norm = layers.LayerNormalization(name=f'trn_ln2_{i}')(x)
        ffn = layers.Dense(FF_DIM, activation='gelu', name=f'trn_ffn1_{i}')(ffn_norm)
        ffn = layers.Dropout(dropout_rate, name=f'trn_ffn_drop_{i}')(ffn)
        ffn = layers.Dense(proj_dim, name=f'trn_ffn2_{i}')(ffn)
        ffn = layers.Dropout(dropout_rate, name=f'trn_ffn2_drop_{i}')(ffn)
        x = x + ffn  # Residual connection
    
    # ========== CTC OUTPUT ==========
    outputs = layers.Dense(charset_size + 1, activation=None, name='logits')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='HTR_Transformer_Improved_V2')
    
    print(f"✅ Model created: {num_transformer_layers} transformer layers, "
          f"{NUM_HEADS} heads, FFN dim={FF_DIM}")
    print(f"   Output units: {charset_size + 1} (charset={charset_size}, blank={charset_size})")
    
    return model

# ===================== CTC LOSS WITH LABEL SMOOTHING =====================
def ctc_loss_with_smoothing(charset_size, label_smoothing=0.1):
    """CTC loss with label smoothing for better generalization"""
    def ctc_loss_fn(y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]
        seq_length = tf.shape(y_pred)[1]
        
        # Input and label lengths
        input_length = tf.fill([batch_size], seq_length)
        label_mask = tf.cast(y_true > 0, tf.int32)
        label_length = tf.reduce_sum(label_mask, axis=-1)
        label_length = tf.maximum(label_length, 1)
        label_length = tf.minimum(label_length, seq_length)
        
        # Process labels for CTC (1-indexed to 0-indexed)
        y_true_processed = tf.cast(tf.where(y_true > 0, y_true - 1, -1), tf.int32)
        
        # Standard CTC loss
        raw_loss = tf.nn.ctc_loss(
            labels=y_true_processed,
            logits=y_pred,
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=False,
            blank_index=charset_size
        )
        
        loss = tf.reduce_mean(raw_loss)
        
        # Optional: Add label smoothing via entropy regularization
        if label_smoothing > 0:
            log_probs = tf.nn.log_softmax(y_pred)
            entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(log_probs) * log_probs, axis=-1))
            loss = loss - label_smoothing * entropy
        
        return loss
    
    return ctc_loss_fn

# ===================== LEARNING RATE SCHEDULE (ONECYCLE - PROVEN BETTER) =====================
class OneCycleLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    OneCycle Learning Rate Schedule (Smith, 2019)
    Proven superior to cosine decay for deep networks.
    
    Phase 1 (30%): Linear warmup from base_lr to max_lr
    Phase 2 (70%): Cosine decay from max_lr to min_lr
    """
    def __init__(self, max_lr, total_steps, pct_warmup=0.30, base_lr=None, min_lr=1e-7):
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_warmup = pct_warmup
        self.base_lr = base_lr if base_lr else max_lr / 25.0  # default: max_lr / 25
        self.min_lr = min_lr
        self.warmup_steps = int(total_steps * pct_warmup)
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        
        # Phase 1: Linear warmup (0% → 30%)
        warmup_lr = self.base_lr + (self.max_lr - self.base_lr) * (step / warmup_steps)
        
        # Phase 2: Cosine decay (30% → 100%)
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * progress))
        decayed_lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
        
        return tf.where(step < warmup_steps, warmup_lr, decayed_lr)
    
    def get_config(self):
        return {
            'max_lr': self.max_lr,
            'total_steps': self.total_steps,
            'pct_warmup': self.pct_warmup,
            'base_lr': self.base_lr,
            'min_lr': self.min_lr
        }

# Fallback: WarmupCosineDecay (legacy - kept for compatibility)
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Warmup + Cosine Annealing (legacy)"""
    def __init__(self, initial_learning_rate, warmup_steps, total_steps, min_lr=1e-7):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        
        # Warmup phase
        warmup_lr = self.initial_learning_rate * (step / warmup_steps)
        
        # Cosine decay phase
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * progress))
        decayed_lr = self.min_lr + (self.initial_learning_rate - self.min_lr) * cosine_decay
        
        return tf.where(step < warmup_steps, warmup_lr, decayed_lr)

# ===================== EVALUATION =====================
def calculate_cer(truth, prediction):
    """Calculate Character Error Rate"""
    if not truth and not prediction:
        return 0.0
    if not truth or not prediction:
        return 1.0
    try:
        return jiwer.cer(truth, prediction)
    except Exception:
        # Fallback: Levenshtein distance
        m, n = len(truth), len(prediction)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if truth[i-1] == prediction[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[m][n] / max(len(truth), 1)

def safe_ctc_decode(logits, charset):
    """Safe CTC decoding with manual fallback"""
    charset_size = len(charset)
    raw_preds = np.argmax(logits, axis=-1)[0]
    
    # Manual CTC decode
    deduped = []
    prev = -1
    for token in raw_preds:
        if token != prev:
            deduped.append(token)
            prev = token
    
    result = []
    for token in deduped:
        if token != charset_size:  # Skip blank
            result.append(token)
    
    text = ''.join([charset[i] if 0 <= i < len(charset) else '<?>' for i in result])
    return text

def evaluate_model(model, dataset, charset, quiet=False):
    """Evaluate model on dataset"""
    if not quiet:
        print("\n🔬 Evaluating model...")
    
    try:
        all_logits = model.predict(dataset, verbose=0 if quiet else 1)
        all_labels = np.concatenate([labels.numpy() for _, labels in dataset], axis=0)
        
        results = []
        for i in range(len(all_logits)):
            logits = all_logits[i:i+1]
            label = all_labels[i]
            
            # Ground truth
            label_indices = [int(idx) - 1 for idx in label if idx > 0]
            ground_truth = ''.join([charset[idx] for idx in label_indices 
                                   if 0 <= idx < len(charset)])
            
            # Prediction
            prediction = safe_ctc_decode(logits, charset)
            
            # Calculate metrics
            cer = calculate_cer(ground_truth, prediction)
            wer = jiwer.wer(ground_truth, prediction) if ground_truth and prediction else 1.0
            
            results.append({
                'ground_truth': ground_truth,
                'prediction': prediction,
                'cer': cer,
                'wer': wer
            })
        
        avg_cer = sum(r['cer'] for r in results) / len(results)
        avg_wer = sum(r['wer'] for r in results) / len(results)
        
        if not quiet:
            print(f"   Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
            print(f"   Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
        
        return avg_cer, avg_wer, results
    
    except Exception as e:
        if not quiet:
            print(f"❌ Evaluation failed: {e}")
        return -1.0, -1.0, []

# ===================== MAIN TRAINING =====================
def main():
    parser = argparse.ArgumentParser(description='Improved Transformer HTR Training')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--num_transformer_layers', type=int, default=NUM_LAYERS, help='Number of transformer layers (default from NUM_LAYERS global)')
    parser.add_argument('--warmup_epochs', type=int, default=WARMUP_EPOCHS, help='Warmup epochs (set to 0 for smoke tests)')
    parser.add_argument('--use_mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--smoke_test', action='store_true', help='Quick smoke test with 100 samples')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint path')
    parser.add_argument('--auto_resume', action='store_true', help='Auto-resume from latest checkpoint if exists')
    parser.add_argument('--checkpoint_interval', type=int, default=20, help='Save periodic checkpoint every N epochs')
    args = parser.parse_args()
    
    # Set seed
    set_global_seed(42)
    
    # Setup GPU
    setup_gpu()
    
    # Mixed precision
    if args.use_mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("✅ Mixed precision training enabled (FP16)")
    
    # Create output directory with resume support in HTR folder
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_htr_dir = '/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/HTR'
    output_dir = os.path.join(base_htr_dir, f"v3_reliable_{timestamp}")
    initial_epoch = 0
    
    # Auto-resume from latest checkpoint if requested
    if args.auto_resume and not args.resume_from:
        # Find latest checkpoint directory in HTR folder
        if os.path.exists(base_htr_dir):
            checkpoint_dirs = [d for d in os.listdir(base_htr_dir) if d.startswith('v3_reliable_') and os.path.isdir(os.path.join(base_htr_dir, d))]
            if checkpoint_dirs:
                latest_dir = os.path.join(base_htr_dir, sorted(checkpoint_dirs)[-1])
                best_checkpoint = os.path.join(latest_dir, 'best_model.weights.h5')
                if os.path.exists(best_checkpoint):
                    args.resume_from = best_checkpoint
                    output_dir = latest_dir  # Reuse same directory
                    print(f"🔄 Auto-resuming from: {args.resume_from}")
                    
                    # Try to get last epoch from CSV log
                    csv_log = os.path.join(latest_dir, 'training_log.csv')
                    if os.path.exists(csv_log):
                        import pandas as pd
                        try:
                            df = pd.read_csv(csv_log)
                            initial_epoch = len(df)
                            print(f"   Resuming from epoch {initial_epoch}")
                        except:
                            pass
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Load charset
    charset = read_charlist(CHARSET_PATH)
    charset_size = len(charset)
    print(f"📚 Charset size: {charset_size}")
    
    # Create lookup table
    keys = tf.constant(charset, dtype=tf.string)
    values = tf.range(1, len(charset) + 1, dtype=tf.int64)
    char_to_num = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values), 
        default_value=tf.constant(0, dtype=tf.int64)
    )
    
    # Load dataset
    print("📊 Loading dataset...")
    dataset = tf.data.TFRecordDataset(DEFAULT_TFRECORD_PATH, buffer_size=10000)
    
    if args.smoke_test:
        dataset = dataset.take(100)
        print("🔥 Smoke test mode: using 100 samples")
    
    dataset_size = sum(1 for _ in dataset)
    train_size = max(1, int(dataset_size * 0.8))
    val_size = max(1, dataset_size - train_size)
    
    print(f"   Total: {dataset_size}, Train: {train_size}, Val: {val_size}")
    
    # Create datasets with augmentation
    dataset = tf.data.TFRecordDataset(DEFAULT_TFRECORD_PATH, buffer_size=10000)
    if args.smoke_test:
        dataset = dataset.take(100)
    
    # Training dataset WITH augmentation
    train_dataset = dataset.take(train_size) \
        .shuffle(buffer_size=min(train_size, 5000), reshuffle_each_iteration=True) \
        .map(lambda x: parse_and_augment(x, char_to_num, augment=True), 
             num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(args.batch_size, drop_remainder=True) \
        .prefetch(tf.data.AUTOTUNE)
    
    # Validation dataset WITHOUT augmentation
    val_dataset = dataset.skip(train_size).take(val_size) \
        .map(lambda x: parse_and_augment(x, char_to_num, augment=False), 
             num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(args.batch_size, drop_remainder=True) \
        .prefetch(tf.data.AUTOTUNE)
    
    # Create model
    print("🏗️  Building improved model...")
    model = create_improved_model(
        charset_size=charset_size,
        proj_dim=PROJ_DIM,
        num_transformer_layers=NUM_LAYERS,  # Use global NUM_LAYERS (reduced to 2)
        dropout_rate=DROPOUT_RATE
    )
    
    if args.resume_from:
        print(f"📥 Loading weights from {args.resume_from}")
        model.load_weights(args.resume_from)
    
    # Calculate steps
    steps_per_epoch = train_size // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs  # Use args.warmup_epochs instead of global
    
    # ✨ V3 IMPROVEMENT: OneCycle LR (proven superior to cosine)
    lr_schedule = OneCycleLR(
        max_lr=MAX_LR,  # Peak LR (5e-4)
        total_steps=total_steps,
        pct_warmup=0.30,  # 30% warmup phase
        base_lr=LEARNING_RATE,  # Starting LR (3e-4)
        min_lr=MIN_LR  # Final LR (1e-7)
    )
    
    print(f"📊 OneCycle LR: base={LEARNING_RATE:.0e} → max={MAX_LR:.0e} → min={MIN_LR:.0e}")
    print(f"   Warmup: {warmup_steps} steps (30%), Total: {total_steps} steps")
    
    # Compile model
    optimizer = AdamW(
        learning_rate=lr_schedule,
        weight_decay=WEIGHT_DECAY,
        clipnorm=GRADIENT_CLIP_NORM
    )
    loss_fn = ctc_loss_with_smoothing(charset_size, label_smoothing=LABEL_SMOOTHING)
    model.compile(optimizer=optimizer, loss=loss_fn)
    
    print(f"✅ Model compiled:")
    print(f"   Optimizer: AdamW (LR={LEARNING_RATE}, WD={WEIGHT_DECAY})")
    print(f"   Loss: CTC with label smoothing={LABEL_SMOOTHING}")
    print(f"   LR Schedule: Warmup({WARMUP_EPOCHS} epochs) + Cosine Annealing")
    
    # Callbacks - OPTIMIZED FOR DISK SPACE
    callbacks = [
        # Save ONLY the best model (saves disk space)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.weights.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        # CSV logger with append mode for resume support
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(output_dir, 'training_log.csv'),
            append=True  # Append if resuming
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,  # Increased patience for 300 epochs
            restore_best_weights=True,
            verbose=1
        )
        # Note: Only 1 weight file saved (best_model) - saves disk space
        # Periodic checkpoint removed to save disk space (rely on best_model only)
    ]
    
    # Train
    remaining_epochs = args.epochs - initial_epoch
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    if initial_epoch > 0:
        print(f"   Resuming from epoch {initial_epoch}, {remaining_epochs} epochs remaining")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        initial_epoch=initial_epoch,  # Resume from this epoch
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    print("\n📊 Final evaluation on validation set...")
    avg_cer, avg_wer, results = evaluate_model(model, val_dataset, charset)
    
    # Save results
    summary = {
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": LEARNING_RATE,
            "num_heads": NUM_HEADS,
            "ff_dim": FF_DIM,
            "proj_dim": PROJ_DIM,
            "num_transformer_layers": args.num_transformer_layers,
            "dropout_rate": DROPOUT_RATE,
            "label_smoothing": LABEL_SMOOTHING,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
        },
        "dataset": {
            "total_samples": dataset_size,
            "training_samples": train_size,
            "validation_samples": val_size,
            "tfrecord_path": DEFAULT_TFRECORD_PATH,
        },
        "results": {
            "final_cer": float(avg_cer),
            "final_wer": float(avg_wer),
        },
        "training_history": {
            "loss": [float(x) for x in history.history.get('loss', [])],
            "val_loss": [float(x) for x in history.history.get('val_loss', [])],
        },
        "environment": {
            "git_hash": get_git_hash(),
            "tensorflow_version": tf.__version__,
            "mixed_precision": args.use_mixed_precision,
        }
    }
    
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"✅ Training summary saved to {summary_path}")
    
    # Save sample predictions
    if results:
        import pandas as pd
        results_df = pd.DataFrame(results[:50])  # Top 50 samples
        results_df.to_csv(os.path.join(output_dir, 'sample_predictions.csv'), index=False)
        print(f"✅ Sample predictions saved")
    
    print(f"\n🎉 Training complete!")
    print(f"   Final CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"   Final WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"   Best model: {output_dir}/best_model.weights.h5")

if __name__ == "__main__":
    main()
