#!/usr/bin/env python3
"""
Conservative HTR Improvement - V4
Strategy: Minimal changes from proven baseline (v2)
Target: CER 28-30% (down from 33.72%)

Changes from v2:
1. Epochs: 100 → 150 (more training time)
2. Dropout: 0.1 → 0.15 (better regularization) 
3. Label smoothing: 0.1 → 0.12 (slight increase)
4. Gradient clipping: norm=1.0 (stability)
5. Early stopping: patience 25 (prevent overfit)
6. Keep: Post-LayerNorm, Adam, same architecture

RATIONALE: v3 failed because too many changes at once.
This takes proven v2 baseline and adds ONLY regularization improvements.
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

# Training parameters (CONSERVATIVE changes only)
EPOCHS = 150  # Increased from 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0005  # Increased from 0.0003 - escape local minima faster
WARMUP_STEPS = 1000

# Model architecture (SAME as v2 - proven)
NUM_TRANSFORMER_LAYERS = 6
NUM_HEADS = 8
FF_DIM = 2048
PROJ_DIM = 512

# Regularization (TARGETED improvements)
DROPOUT_RATE = 0.15  # Increased from 0.1
LABEL_SMOOTHING = 0.12  # Increased from 0.1
WEIGHT_DECAY = 0.0002  # Same as v2
GRADIENT_CLIP_NORM = 1.0  # NEW: prevent gradient explosion

# Early stopping (NEW: prevent overfitting like v3)
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
    """Parse TFRecord (same as v2 - proven)"""
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
        return seq[:MAX_LABEL_LENGTH]
    
    def parse_string_label():
        text = tf.cond(tf.not_equal(parsed['text'], tf.constant(b'')), 
                      lambda: parsed['text'], lambda: parsed['label'])
        chars = tf.strings.unicode_split(text, 'UTF-8')
        chars = chars[:MAX_LABEL_LENGTH]
        indices = tf.map_fn(lambda c: char_to_num.lookup(c), chars, dtype=tf.int64)
        return tf.cast(indices, tf.int32)
    
    label = tf.cond(tf.strings.regex_full_match(parsed['label'], b'^\\d+$'),
                   parse_numeric_label, parse_string_label)
    
    # Pad label
    label_len = tf.shape(label)[0]
    label = tf.pad(label, [[0, MAX_LABEL_LENGTH - label_len]])
    
    return img, label

# ===================== MODEL ARCHITECTURE (SAME AS V2) =====================
class TransformerBlock(layers.Layer):
    """Transformer block with Post-LayerNorm (proven architecture)"""
    def __init__(self, proj_dim, num_heads, ff_dim, dropout_rate=0.15):
        super().__init__()
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.proj_dim // self.num_heads,
            dropout=self.dropout_rate
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(self.ff_dim, activation="relu"),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.proj_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        
    def call(self, inputs, training=False):
        # Post-LayerNorm (proven)
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_model(num_chars, num_transformer_layers=6):
    """Build transformer HTR model (same as v2)"""
    # Input
    input_img = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1), name='image')
    
    # CNN Feature Extraction (same as v2)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    
    # Reshape for transformer
    new_shape = ((IMG_WIDTH // 8), (IMG_HEIGHT // 8) * 512)
    x = layers.Reshape(target_shape=new_shape)(x)
    x = layers.Dense(PROJ_DIM)(x)
    
    # Positional encoding (same as v2)
    positions = tf.range(start=0, limit=IMG_WIDTH // 8, delta=1)
    position_embedding = layers.Embedding(input_dim=IMG_WIDTH // 8, output_dim=PROJ_DIM)(positions)
    x = x + position_embedding
    
    # Transformer layers (same count as v2)
    for _ in range(num_transformer_layers):
        x = TransformerBlock(PROJ_DIM, NUM_HEADS, FF_DIM, dropout_rate=DROPOUT_RATE)(x)
    
    # Output layer - CRITICAL: No activation! CTC needs logits, not probabilities
    x = layers.Dense(num_chars + 1, activation=None, name='logits')(x)
    
    return Model(inputs=input_img, outputs=x, name='transformer_htr_v4_conservative')

# ===================== CTC LOSS =====================
def ctc_loss_with_smoothing(charset_size, label_smoothing=0.12):
    """CTC loss with label smoothing - EXACT copy from V2 baseline"""
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

# ===================== MAIN =====================
def main():
    print("\n" + "="*80)
    print("🚀 HTR Training V4 - Conservative Improvement")
    print("="*80)
    print(f"📊 Strategy: Minimal changes from proven baseline")
    print(f"🎯 Target CER: 28-30% (down from 33.72%)")
    print(f"📈 Changes: Epochs↑, Dropout↑, Label Smoothing↑, Gradient Clipping✓")
    print("="*80 + "\n")
    
    # Setup
    setup_gpu()
    
    # Load charset
    print("Loading character set...")
    charset = read_charlist(CHARSET_PATH)
    char_to_num = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=list(charset), values=list(range(len(charset))), key_dtype=tf.string, value_dtype=tf.int64
        ), default_value=0
    )
    num_to_char = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=list(range(len(charset))), values=list(charset), key_dtype=tf.int64, value_dtype=tf.string
        ), default_value=""
    )
    print(f"✓ Loaded {len(charset)} characters")
    
    # Load dataset
    print(f"\nLoading dataset from: {DEFAULT_TFRECORD_PATH}")
    raw_dataset = tf.data.TFRecordDataset(DEFAULT_TFRECORD_PATH)
    
    # Count samples
    total_samples = sum(1 for _ in raw_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    print(f"✓ Total samples: {total_samples}")
    print(f"✓ Training: {train_size}, Validation: {val_size}")
    
    # Create train/val splits
    train_dataset = raw_dataset.take(train_size)
    val_dataset = raw_dataset.skip(train_size)
    
    # Parse and batch
    train_dataset = (train_dataset
                     .map(lambda x: parse_tfrecord(x, char_to_num), num_parallel_calls=tf.data.AUTOTUNE)
                     .batch(BATCH_SIZE)
                     .prefetch(tf.data.AUTOTUNE))
    
    val_dataset = (val_dataset
                   .map(lambda x: parse_tfrecord(x, char_to_num), num_parallel_calls=tf.data.AUTOTUNE)
                   .batch(BATCH_SIZE)
                   .prefetch(tf.data.AUTOTUNE))
    
    # Build model
    print("\nBuilding model...")
    model = build_model(len(charset), num_transformer_layers=NUM_TRANSFORMER_LAYERS)
    
    # Compile with gradient clipping and proper CTC loss (EXACT from V2)
    optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=GRADIENT_CLIP_NORM)
    loss_fn = ctc_loss_with_smoothing(charset_size=len(charset), label_smoothing=LABEL_SMOOTHING)
    model.compile(optimizer=optimizer, loss=loss_fn)
    
    print(f"✓ Model built: {model.count_params():,} parameters")
    print(f"✓ Architecture: {NUM_TRANSFORMER_LAYERS} transformer layers")
    print(f"✓ Gradient clipping: norm={GRADIENT_CLIP_NORM}")
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"HTR/v4_conservative_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, 
                      min_delta=EARLY_STOPPING_MIN_DELTA, restore_best_weights=True, verbose=1),
        ModelCheckpoint(f"{output_dir}/best_model.weights.h5", save_best_only=True, 
                        save_weights_only=True, monitor='val_loss', verbose=1)
    ]
    
    print(f"\n✓ Callbacks configured:")
    print(f"  - ReduceLROnPlateau: patience=10, factor=0.5")
    print(f"  - EarlyStopping: patience={EARLY_STOPPING_PATIENCE}, min_delta={EARLY_STOPPING_MIN_DELTA}")
    print(f"  - ModelCheckpoint: {output_dir}/best_model.weights.h5")
    
    # Training
    print(f"\n{'='*80}")
    print(f"🏋️  Starting Training - {EPOCHS} epochs")
    print(f"{'='*80}\n")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save results
    print(f"\n✓ Training complete! Saving results to {output_dir}/")
    
    # Save training history
    with open(f"{output_dir}/training_log.csv", 'w') as f:
        f.write("epoch,loss,val_loss\n")
        for i in range(len(history.history['loss'])):
            f.write(f"{i},{history.history['loss'][i]},{history.history['val_loss'][i]}\n")
    
    # Save summary
    summary = {
        "hyperparameters": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_heads": NUM_HEADS,
            "ff_dim": FF_DIM,
            "proj_dim": PROJ_DIM,
            "num_transformer_layers": NUM_TRANSFORMER_LAYERS,
            "dropout_rate": DROPOUT_RATE,
            "label_smoothing": LABEL_SMOOTHING,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE
        },
        "dataset": {
            "total_samples": total_samples,
            "training_samples": train_size,
            "validation_samples": val_size,
            "tfrecord_path": DEFAULT_TFRECORD_PATH
        },
        "training_history": {
            "loss": history.history['loss'],
            "val_loss": history.history['val_loss']
        }
    }
    
    with open(f"{output_dir}/training_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"🎉 Training complete!")
    print(f"{'='*80}")
    print(f"📁 Output directory: {output_dir}/")
    print(f"📊 Best model: {output_dir}/best_model.weights.h5")
    print(f"📈 Training log: {output_dir}/training_log.csv")
    print(f"📋 Summary: {output_dir}/training_summary.json")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
