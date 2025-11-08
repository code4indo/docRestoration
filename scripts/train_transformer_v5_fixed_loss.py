#!/usr/bin/env python3
"""
HTR Training V5 - FIXED LOSS FUNCTION
CRITICAL FIX: Remove buggy entropy regularization that caused all-blank predictions

Root Cause Analysis V4 Failure:
- Model predicted ONLY blank tokens (100% CER)
- Val loss went negative (-9.93) which is impossible for standard CTC
- Bug: loss = loss - label_smoothing * entropy (WRONG SIGN!)
- This incentivized model to maximize entropy = predict all blanks

Solution V5:
- Use STANDARD CTC loss without any modifications
- No label smoothing entropy tricks
- Proven, reliable, no surprises
- Same architecture as V4 (proven transformer design)

Target: CER 28-30% (baseline 33.72%)
"""
import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
import jiwer
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import warnings
import logging
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.get_logger().setLevel('ERROR')

# ===================== CONFIGURATION =====================
CHARSET_PATH = '/home/lambda_one/tesis/GAN-HTR-ORI/real_data_preparation/real_data_charlist.txt'
DEFAULT_TFRECORD_PATH = '/home/lambda_one/tesis/GAN-HTR-ORI/real_data_final_fixed_v2.tfrecord'

# Image parameters
IMG_WIDTH = 1024
IMG_HEIGHT = 128
MAX_LABEL_LENGTH = 128

# Training hyperparameters
EPOCHS = 150
BATCH_SIZE = 8  # Reduced from 16 to 8 - V2 architecture very memory intensive
LEARNING_RATE = 0.0005
WARMUP_STEPS = 1000

# Model architecture (same as V4 - proven)
NUM_TRANSFORMER_LAYERS = 6
NUM_HEADS = 8
FF_DIM = 2048
PROJ_DIM = 512

# Regularization (same as V4 but WITHOUT buggy label smoothing)
DROPOUT_RATE = 0.15
WEIGHT_DECAY = 0.0002
GRADIENT_CLIP_NORM = 1.0

# Early stopping
EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_MIN_DELTA = 0.001

# ===================== UTILS =====================
def setup_gpu():
    """Setup GPU memory growth"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Found {len(gpus)} GPU(s), memory growth enabled")
            return True
        else:
            print("⚠️  No GPUs found, using CPU")
            return False
    except Exception as e:
        print(f"⚠️  GPU setup error: {e}")
        return False

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
    """Parse TFRecord (same as V4)"""
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
    
    # Parse label - EXACT COPY FROM V2 BASELINE (PROVEN)
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

# ===================== MODEL ARCHITECTURE =====================
# ===================== V2 EXACT MODEL ARCHITECTURE =====================
def build_model(num_chars, num_transformer_layers=6):
    """
    EXACT V2 MODEL ARCHITECTURE (PROVEN CER 33.72%)
    - Proper CNN backbone with BatchNorm
    - Progressive dropout
    - LayerNormalization  
    - Deeper CNN (8 layers)
    """
    inputs = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1), name='image_input')
    x = inputs
    
    # ========== CNN BACKBONE (V2 EXACT) ==========
    def conv_block(inp, filters, k=3, s=(1,1), name_prefix='cb', dropout=0.0):
        """Proper conv block with BatchNorm and correct dropout"""
        y = layers.Conv2D(filters, k, strides=s, padding='same', 
                         use_bias=False, name=f'{name_prefix}_conv')(inp)
        y = layers.BatchNormalization(name=f'{name_prefix}_bn')(y)
        y = layers.Activation('gelu', name=f'{name_prefix}_gelu')(y)
        if dropout > 0:
            y = layers.Dropout(dropout, name=f'{name_prefix}_drop')(y)
        return y
    
    # Progressive feature extraction (V2 exact)
    x = conv_block(x, 64, k=7, s=(1,2), name_prefix='s1_1', dropout=DROPOUT_RATE*0.5)
    x = conv_block(x, 64, k=3, s=(1,1), name_prefix='s1_2', dropout=DROPOUT_RATE*0.5)
    x = layers.MaxPooling2D(pool_size=(2,2), name='pool1')(x)
    
    x = conv_block(x, 128, k=3, s=(1,1), name_prefix='s2_1', dropout=DROPOUT_RATE*0.7)
    x = conv_block(x, 128, k=3, s=(1,1), name_prefix='s2_2', dropout=DROPOUT_RATE*0.7)
    x = layers.MaxPooling2D(pool_size=(2,2), name='pool2')(x)
    
    x = conv_block(x, 256, k=3, s=(1,1), name_prefix='s3_1', dropout=DROPOUT_RATE)
    x = conv_block(x, 256, k=3, s=(1,1), name_prefix='s3_2', dropout=DROPOUT_RATE)
    x = layers.MaxPooling2D(pool_size=(2,1), name='pool3')(x)
    
    x = conv_block(x, 512, k=3, s=(1,1), name_prefix='s4_1', dropout=DROPOUT_RATE)
    x = conv_block(x, 512, k=3, s=(1,1), name_prefix='s4_2', dropout=DROPOUT_RATE)
    
    # ========== SEQUENCE PROJECTION (V2 exact) ==========
    x = layers.Lambda(
        lambda t: tf.reshape(t, (tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2]*tf.shape(t)[3])), 
        name='flatten_height'
    )(x)
    
    x = layers.Dense(PROJ_DIM, name='proj_dense')(x)
    x = layers.LayerNormalization(name='proj_ln')(x)
    x = layers.Dropout(DROPOUT_RATE, name='proj_drop')(x)
    
    # ========== POSITIONAL ENCODING (V2 exact) ==========
    seq_len = 128  # target_time_steps
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_embedding_layer = layers.Embedding(
        input_dim=seq_len, 
        output_dim=PROJ_DIM, 
        name='positional_embedding'
    )
    x = x + pos_embedding_layer(positions)
    
    # ========== TRANSFORMER ENCODER (V2 exact) ==========
    for i in range(num_transformer_layers):
        # Multi-head attention
        attn = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, 
            key_dim=PROJ_DIM // NUM_HEADS, 
            dropout=DROPOUT_RATE,
            name=f'trn_attn_{i}'
        )(x, x)
        x = layers.LayerNormalization(name=f'trn_ln1_{i}')(x + attn)
        
        # Feed-forward network
        ffn = layers.Dense(FF_DIM, activation='gelu', name=f'trn_ffn1_{i}')(x)
        ffn = layers.Dropout(DROPOUT_RATE, name=f'trn_ffn_drop_{i}')(ffn)
        ffn = layers.Dense(PROJ_DIM, name=f'trn_ffn2_{i}')(ffn)
        x = layers.LayerNormalization(name=f'trn_ln2_{i}')(x + ffn)
        x = layers.Dropout(DROPOUT_RATE, name=f'trn_drop_{i}')(x)
    
    # ========== CTC OUTPUT ==========
    outputs = layers.Dense(num_chars + 1, activation=None, name='logits')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='HTR_V5_V2_EXACT_ARCHITECTURE')
    
    return model

# ===================== CTC LOSS - EXACT COPY FROM V2 BASELINE =====================
def standard_ctc_loss(charset_size):
    """
    EXACT COPY from train_transformer_improved_v2.py (PROVEN CER 33.72%)
    Using same loss implementation to ensure compatibility.
    """
    def ctc_loss_fn(y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]
        seq_length = tf.shape(y_pred)[1]
        
        # Input and label lengths - EXACT FROM V2
        input_length = tf.fill([batch_size], seq_length)
        label_mask = tf.cast(y_true > 0, tf.int32)
        label_length = tf.reduce_sum(label_mask, axis=-1)
        label_length = tf.maximum(label_length, 1)
        label_length = tf.minimum(label_length, seq_length)
        
        # Process labels for CTC (1-indexed to 0-indexed) - EXACT FROM V2
        y_true_processed = tf.cast(tf.where(y_true > 0, y_true - 1, -1), tf.int32)
        
        # Standard CTC loss - EXACT FROM V2
        raw_loss = tf.nn.ctc_loss(
            labels=y_true_processed,
            logits=y_pred,
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=False,
            blank_index=charset_size
        )
        
        loss = tf.reduce_mean(raw_loss)
        
        # NO LABEL SMOOTHING - V2 had it but we skip for now to isolate issues
        
        return loss
    
    return ctc_loss_fn

# ===================== CTC METRICS - V2 EXACT =====================
def safe_ctc_decode(logits, charset):
    """
    Safe CTC decoding with manual fallback (V2 EXACT)
    This is PROVEN to work in V2 baseline
    """
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

def ctc_decode_predictions(y_pred, charset):
    """Decode CTC predictions to text - V2 MANUAL DECODE"""
    output_texts = []
    for i in range(len(y_pred)):
        logits = y_pred[i:i+1]
        prediction = safe_ctc_decode(logits, charset)
        output_texts.append(prediction)
    
    return output_texts

def compute_cer_wer(y_true, y_pred, charset):
    """Compute CER and WER metrics"""
    # Decode ground truth
    gt_texts = []
    for label in y_true:
        chars = []
        for idx in label:
            if idx > 0 and idx <= len(charset):
                chars.append(charset[idx - 1])  # 1-indexed to 0-indexed
        gt_texts.append(''.join(chars))
    
    # Decode predictions
    pred_texts = ctc_decode_predictions(y_pred, charset)
    
    # Compute metrics
    cer = jiwer.cer(gt_texts, pred_texts)
    wer = jiwer.wer(gt_texts, pred_texts)
    
    return cer, wer, gt_texts, pred_texts

# ===================== CUSTOM CALLBACK =====================
class CERCallback(tf.keras.callbacks.Callback):
    """Callback to compute CER on validation set"""
    def __init__(self, val_data, charset, log_file):
        super().__init__()
        self.val_data = val_data
        self.charset = charset
        self.log_file = log_file
        self.best_cer = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        # Sample validation batch
        for imgs, labels in self.val_data.take(1):
            preds = self.model.predict(imgs, verbose=0)
            cer, wer, gt_texts, pred_texts = compute_cer_wer(labels.numpy(), preds, self.charset)
            
            logs['val_cer'] = cer
            logs['val_wer'] = wer
            
            # Log to file
            with open(self.log_file, 'a') as f:
                f.write(f"\nEpoch {epoch + 1} - CER: {cer:.4f}, WER: {wer:.4f}\n")
                f.write(f"Sample predictions:\n")
                for i in range(min(3, len(gt_texts))):
                    f.write(f"  GT: {gt_texts[i][:80]}\n")
                    f.write(f"  PD: {pred_texts[i][:80]}\n")
            
            if cer < self.best_cer:
                self.best_cer = cer
                print(f"\n🎯 New best CER: {cer:.4f} (WER: {wer:.4f})")
            
            break

# ===================== MAIN =====================
def main():
    print("\n" + "="*80)
    print("🚀 HTR Training V5 - FIXED LOSS FUNCTION")
    print("="*80)
    print(f"🐛 BUG FIX: Remove entropy regularization that caused all-blank predictions")
    print(f"📊 Strategy: Standard CTC loss + proven architecture")
    print(f"🎯 Target CER: 28-30% (baseline 33.72%, V4 failed 100%)")
    print("="*80 + "\n")
    
    # Setup
    setup_gpu()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"HTR/v5_fixed_loss_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    log_file = f"logs/v5_fixed_loss_{timestamp}.log"
    os.makedirs("logs", exist_ok=True)
    
    # Load charset
    charset = read_charlist(CHARSET_PATH)
    print(f"✅ Loaded charset: {len(charset)} characters")
    
    # Create lookup table (int32 for values)
    char_to_num = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=list(charset),
            values=tf.constant(list(range(1, len(charset) + 1)), dtype=tf.int32)
        ),
        default_value=0
    )
    
    # Load dataset
    print(f"\n📂 Loading dataset: {DEFAULT_TFRECORD_PATH}")
    dataset = tf.data.TFRecordDataset(DEFAULT_TFRECORD_PATH)
    dataset = dataset.map(lambda x: parse_tfrecord(x, char_to_num), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Count samples
    total_samples = sum(1 for _ in dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    print(f"✅ Total samples: {total_samples} (Train: {train_size}, Val: {val_size})")
    
    # Split dataset
    train_data = dataset.take(train_size).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_data = dataset.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Build model
    print(f"\n🏗️  Building model...")
    model = build_model(len(charset), num_transformer_layers=NUM_TRANSFORMER_LAYERS)
    print(f"✅ Model built: {model.count_params():,} parameters")
    
    # Compile with STANDARD CTC loss (no tricks!)
    optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=GRADIENT_CLIP_NORM)
    model.compile(
        optimizer=optimizer,
        loss=standard_ctc_loss(len(charset))  # FIXED: Standard loss only!
    )
    
    print(f"\n✅ Model compiled with STANDARD CTC loss (no entropy tricks)")
    print(f"   - Learning rate: {LEARNING_RATE}")
    print(f"   - Gradient clip norm: {GRADIENT_CLIP_NORM}")
    print(f"   - Dropout rate: {DROPOUT_RATE}")
    
    # Callbacks
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.weights.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        CERCallback(val_data, charset, log_file)
    ]
    
    # Training
    print(f"\n🎓 Starting training...")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Output dir: {output_dir}")
    print(f"   - Log file: {log_file}\n")
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*80)
    print("✅ Training complete!")
    print("="*80)
    
    # Save summary
    summary = {
        'version': 'v5_fixed_loss',
        'timestamp': timestamp,
        'bug_fix': 'Removed entropy regularization from loss function',
        'architecture': {
            'transformer_layers': NUM_TRANSFORMER_LAYERS,
            'num_heads': NUM_HEADS,
            'ff_dim': FF_DIM,
            'proj_dim': PROJ_DIM,
            'dropout': DROPOUT_RATE,
        },
        'training': {
            'epochs_completed': len(history.history['loss']),
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'gradient_clip': GRADIENT_CLIP_NORM,
        },
        'dataset': {
            'total_samples': total_samples,
            'train_samples': train_size,
            'val_samples': val_size,
        },
        'history': {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
        }
    }
    
    summary_file = os.path.join(output_dir, 'training_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Training summary saved: {summary_file}")
    print(f"📈 Final val_loss: {history.history['val_loss'][-1]:.4f}")
    print(f"🎯 Check CER in log file: {log_file}\n")

if __name__ == '__main__':
    main()
