"""
EXPERIMENT: train.py dengan FP32 (tanpa mixed precision)

Tujuan: Membuktikan bahwa FP16 adalah penyebab utama degradasi performa
Modifikasi: Disable mixed precision, keep semua fitur lain (CER/WER, discriminator_mode)
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import tensorflow as tf
import cv2
import mlflow
import editdistance
from datetime import datetime

# Disable XLA optimization to avoid layout errors - MUST be before TF import
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import tensorflow as tf
import numpy as np
import cv2

# Additional XLA disable at TF config level (runtime)
tf.config.optimizer.set_jit(False)

# ❌ DISABLED Mixed Precision for this experiment
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')
print("🔬 EXPERIMENT: FP32 Mode (Mixed Precision DISABLED)")

from tqdm import tqdm
import editdistance  # For CER/WER calculation

# Fix for ModuleNotFoundError: Add project root to the Python path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import models and utils
from dual_modal_gan.src.models.generator import unet
from dual_modal_gan.src.models.recognizer import load_frozen_recognizer
from dual_modal_gan.src.models.discriminator import build_dual_modal_discriminator

# --- Utility Functions ---
def read_charlist(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    print(f"--- Random seeds set to {seed} for reproducibility ---")

def decode_label(label_ids, charset):
    """Decode label IDs to text string, removing blank tokens (0) and padding."""
    decoded_chars = []
    prev_id = -1
    for label_id in label_ids:
        label_id = int(label_id)
        # Skip blank token (0) and padding (0), and CTC duplicates
        if label_id > 0 and label_id != prev_id:
            if label_id - 1 < len(charset):
                decoded_chars.append(charset[label_id - 1])
        prev_id = label_id
    return ''.join(decoded_chars)

def calculate_cer(ground_truth, prediction):
    """Calculate Character Error Rate using edit distance."""
    if len(ground_truth) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    distance = editdistance.eval(ground_truth, prediction)
    return distance / len(ground_truth)

def calculate_wer(ground_truth, prediction):
    """Calculate Word Error Rate using edit distance on word level."""
    gt_words = ground_truth.split()
    pred_words = prediction.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    distance = editdistance.eval(gt_words, pred_words)
    return distance / len(gt_words)


# --- Dataset Pipeline ---
def _parse_tfrecord_fn(example_proto):
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
    
    degraded_image_shape = tf.cast(example['degraded_image_shape'], tf.int32)
    degraded_image = tf.io.decode_raw(example['degraded_image_raw'], tf.float32)
    degraded_image = tf.reshape(degraded_image, degraded_image_shape)
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])
    degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])

    clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
    clean_image = tf.reshape(clean_image, clean_image_shape)
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])
    clean_image = tf.ensure_shape(clean_image, [1024, 128, 1])

    label_shape = tf.cast(example['label_shape'], tf.int32)
    label = tf.io.decode_raw(example['label_raw'], tf.int64)
    label = tf.reshape(label, label_shape)
    label = tf.cast(label, tf.int32)
    padding = [[0, 128 - tf.shape(label)[0]]]
    label = tf.pad(label, padding, "CONSTANT", constant_values=0)
    label.set_shape([128])

    return degraded_image, clean_image, label

def create_dataset(tfrecord_path, batch_size, val_split=0.1):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    total_size = sum(1 for _ in dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    train_dataset = train_dataset.shuffle(buffer_size=1024).repeat().batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, train_size, val_size

# --- Validation (with CER/WER) ---
def run_validation_step(val_dataset, generator, recognizer, charset, psnr_metric, ssim_metric, cer_metric, wer_metric):
    """Note: Cannot use @tf.function due to Python-based CER/WER calculation."""
    for degraded_images, clean_images, labels in val_dataset:
        generated_images = generator(degraded_images, training=False)
        
        # No FP16 casting needed - already FP32
        psnr = tf.image.psnr(clean_images, generated_images, max_val=1.0)
        ssim = tf.image.ssim(clean_images, generated_images, max_val=1.0)
        psnr_metric.update_state(psnr)
        ssim_metric.update_state(ssim)
        
        clean_logits = recognizer(clean_images, training=False)
        generated_logits = recognizer(generated_images, training=False)
        
        clean_predictions = tf.argmax(clean_logits, axis=-1, output_type=tf.int32)
        generated_predictions = tf.argmax(generated_logits, axis=-1, output_type=tf.int32)
        
        labels_np = labels.numpy()
        clean_pred_np = clean_predictions.numpy()
        generated_pred_np = generated_predictions.numpy()
        
        batch_cer = []
        batch_wer = []
        for i in range(labels_np.shape[0]):
            gt_text = decode_label(labels_np[i], charset)
            clean_text = decode_label(clean_pred_np[i], charset)
            generated_text = decode_label(generated_pred_np[i], charset)
            
            cer = calculate_cer(clean_text, generated_text)
            wer = calculate_wer(clean_text, generated_text)
            
            batch_cer.append(cer)
            batch_wer.append(wer)
        
        cer_metric.update_state(batch_cer)
        wer_metric.update_state(batch_wer)

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print(f"--- Configuring to use GPU: {args.gpu_id} ---")
    print("--- 🔬 EXPERIMENT: Dual-Modal GAN-HTR (FP32 Mode) ---")

    set_seeds(args.seed)

    strategy = tf.distribute.get_strategy()
    print(f"Running with strategy: {strategy}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    metrics_dir = os.path.join(args.checkpoint_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    print("\n[Phase 1/6] Loading Datasets and Charset...")
    charset = read_charlist(args.charset_path)
    vocab_size = len(charset) + 1
    print(f"Charset loaded: {vocab_size} characters (including blank token)")

    train_dataset, val_dataset, train_count, val_count = create_dataset(args.tfrecord_path, args.batch_size)
    print(f"Dataset created: {train_count} training samples, {val_count} validation samples.")

    steps_per_epoch = args.steps_per_epoch or (train_count // args.batch_size)

    with strategy.scope():
        print("\n[Phase 2/6] Building Models...")
        generator = unet(input_size=(1024, 128, 1))
        recognizer = load_frozen_recognizer(weights_path=args.recognizer_weights, charset_size=vocab_size - 1)
        discriminator = build_dual_modal_discriminator(img_shape=(1024, 128, 1), vocab_size=vocab_size, max_text_len=128)
        print("All models built.")

    with strategy.scope():
        print("\n[Phase 3/6] Setting up Optimizers and Checkpoints...")
        # ✅ NO LossScaleOptimizer wrapping - pure FP32
        generator_optimizer = tf.keras.optimizers.Adam(args.lr_g, beta_1=0.5, clipnorm=args.gradient_clip_norm)
        discriminator_optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr_d, momentum=0.9, clipnorm=args.gradient_clip_norm)
        
        print(f"  Generator optimizer: Adam(lr={args.lr_g}, clipnorm={args.gradient_clip_norm}) [FP32]")
        print(f"  Discriminator optimizer: SGD(lr={args.lr_d}, momentum=0.9, clipnorm={args.gradient_clip_norm}) [FP32]")
        
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)
        ckpt_manager = tf.train.CheckpointManager(checkpoint, args.checkpoint_dir, max_to_keep=5)
        
        if ckpt_manager.latest_checkpoint and not args.no_restore:
            checkpoint.restore(ckpt_manager.latest_checkpoint)
            print(f"Restored from {ckpt_manager.latest_checkpoint}")
        else:
            print("Initializing from scratch.")

    with strategy.scope():
        bce_loss_fn = tf.keras.losses.BinaryCrossentropy()
        mae_loss_fn = tf.keras.losses.MeanAbsoluteError()
        val_psnr_metric = tf.keras.metrics.Mean(name='val_psnr')
        val_ssim_metric = tf.keras.metrics.Mean(name='val_ssim')
        val_cer_metric = tf.keras.metrics.Mean(name='val_cer')
        val_wer_metric = tf.keras.metrics.Mean(name='val_wer')

        @tf.function
        def train_step(degraded_images, clean_images, ground_truth_text):
            real_labels_disc = tf.ones([args.batch_size, 1]) * 0.9
            fake_labels_disc = tf.zeros([args.batch_size, 1])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(degraded_images, training=True)
                clean_logits = recognizer(clean_images, training=False)
                generated_logits = recognizer(generated_images, training=False)

                clean_text_pred = tf.argmax(clean_logits, axis=-1, output_type=tf.int32)
                generated_text_pred = tf.argmax(generated_logits, axis=-1, output_type=tf.int32)

                if args.discriminator_mode == 'ground_truth':
                    real_output = discriminator([clean_images, ground_truth_text], training=True)
                else:
                    real_output = discriminator([clean_images, clean_text_pred], training=True)
                fake_output = discriminator([generated_images, generated_text_pred], training=True)

                disc_loss_real = bce_loss_fn(real_labels_disc, real_output)
                disc_loss_fake = bce_loss_fn(fake_labels_disc, fake_output)
                
                # ✅ NO casting needed - already FP32
                total_disc_loss = disc_loss_real + disc_loss_fake

                adversarial_loss = bce_loss_fn(real_labels_disc, fake_output)
                pixel_loss = mae_loss_fn(clean_images, generated_images)
                
                if args.ctc_loss_weight > 0:
                    label_len = tf.math.count_nonzero(ground_truth_text, axis=1, keepdims=True, dtype=tf.int32)
                    label_len = tf.reshape(label_len, [-1])
                    logit_len = tf.fill([args.batch_size], generated_logits.shape[1])
                    
                    ctc_loss_raw = tf.reduce_mean(tf.nn.ctc_loss(labels=tf.cast(ground_truth_text, tf.int32), logits=generated_logits, label_length=label_len, logit_length=logit_len, logits_time_major=False, blank_index=0))
                    ctc_loss = tf.clip_by_value(ctc_loss_raw, 0.0, args.ctc_loss_clip_max)
                else:
                    ctc_loss = tf.constant(0.0, dtype=tf.float32)
                    ctc_loss_raw = tf.constant(0.0, dtype=tf.float32)

                # ✅ NO casting needed - already FP32
                total_gen_loss = (args.adv_loss_weight * adversarial_loss) + (args.pixel_loss_weight * pixel_loss) + (args.ctc_loss_weight * ctc_loss)

            generator_gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)
            
            generator_gradients, gen_grad_norm = tf.clip_by_global_norm(generator_gradients, args.gradient_clip_norm)
            discriminator_gradients, disc_grad_norm = tf.clip_by_global_norm(discriminator_gradients, args.gradient_clip_norm)
            
            generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

            return total_gen_loss, total_disc_loss, adversarial_loss, pixel_loss, ctc_loss, ctc_loss_raw, gen_grad_norm, disc_grad_norm

    print("\n[Phase 4/6] Starting Training Loop...")
    dataset_iterator = iter(train_dataset)
    val_iterator = iter(val_dataset)
    sample_batch = next(val_iterator)

    best_val_psnr = -1.0
    
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = f"GAN_HTR_FP32_EXPERIMENT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)
    
    print(f"\n📊 MLflow tracking: {experiment_name}")
    
    training_history = {
        "experiment": "FP32_MODE",
        "start_time": datetime.now().isoformat(),
        "hyperparameters": {
            "precision": "FP32 (no mixed precision)",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "steps_per_epoch": steps_per_epoch,
            "lr_generator": args.lr_g,
            "lr_discriminator": args.lr_d,
            "pixel_loss_weight": args.pixel_loss_weight,
            "ctc_loss_weight": args.ctc_loss_weight,
            "discriminator_mode": args.discriminator_mode,
            "adv_loss_weight": args.adv_loss_weight,
        },
        "epochs": []
    }

    with mlflow.start_run():
        mlflow.log_params({
            "experiment_type": "FP32_vs_FP16",
            "precision": "FP32",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "pixel_loss_weight": args.pixel_loss_weight,
            "ctc_loss_weight": args.ctc_loss_weight,
        })
    
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            
            epoch_metrics = {
                "epoch": epoch + 1,
                "losses": {
                    "g_loss": [],
                    "d_loss": [],
                    "ctc_loss": [],
                    "ctc_loss_raw": [],
                    "l1_loss": [],
                }
            }
            
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}")
            for step in pbar:
                degraded_batch, clean_batch, text_batch = next(dataset_iterator)
                g_loss, d_loss, adv_loss, pix_loss, ctc_loss, ctc_loss_raw, g_grad, d_grad = train_step(degraded_batch, clean_batch, text_batch)
                
                epoch_metrics["losses"]["g_loss"].append(float(g_loss.numpy()))
                epoch_metrics["losses"]["d_loss"].append(float(d_loss.numpy()))
                epoch_metrics["losses"]["ctc_loss"].append(float(ctc_loss.numpy()))
                epoch_metrics["losses"]["ctc_loss_raw"].append(float(ctc_loss_raw.numpy()))
                epoch_metrics["losses"]["l1_loss"].append(float(pix_loss.numpy()))

                if (step + 1) % 50 == 0:
                    pbar.set_postfix({
                        'G': f'{g_loss:.4f}', 
                        'D': f'{d_loss:.4f}', 
                        'CTC': f'{ctc_loss:.2f}',
                        'L1': f'{pix_loss:.4f}'
                    })

            # Validation
            if (epoch + 1) % args.eval_interval == 0:
                print("  Running validation...")
                run_validation_step(val_dataset, generator, recognizer, charset, val_psnr_metric, val_ssim_metric, val_cer_metric, val_wer_metric)
                psnr_result = val_psnr_metric.result()
                ssim_result = val_ssim_metric.result()
                cer_result = val_cer_metric.result()
                wer_result = val_wer_metric.result()
                
                print(f"  📊 PSNR: {psnr_result:.2f}, SSIM: {ssim_result:.4f}, CER: {cer_result:.4f}, WER: {wer_result:.4f}")
                
                epoch_metrics["validation"] = {
                    "psnr": float(psnr_result.numpy()),
                    "ssim": float(ssim_result.numpy()),
                    "cer": float(cer_result.numpy()),
                    "wer": float(wer_result.numpy())
                }
                
                mlflow.log_metrics({
                    "val_psnr": float(psnr_result.numpy()),
                    "val_ssim": float(ssim_result.numpy()),
                    "val_cer": float(cer_result.numpy()),
                    "val_wer": float(wer_result.numpy())
                }, step=epoch)
                
                if psnr_result > best_val_psnr:
                    best_val_psnr = psnr_result
                    best_ckpt_save_path = os.path.join(args.checkpoint_dir, "best_model_fp32")
                    checkpoint.save(file_prefix=best_ckpt_save_path)
                    print(f"  ✅ Best FP32 model: PSNR {best_val_psnr:.2f}")
                
                val_psnr_metric.reset_state()
                val_ssim_metric.reset_state()
                val_cer_metric.reset_state()
                val_wer_metric.reset_state()

            # Save samples
            if (epoch + 1) % args.save_interval == 0:
                generated_samples = generator(sample_batch[0], training=False)
                img_to_save = (generated_samples * 255).numpy().astype(np.uint8)
                for i in range(min(args.batch_size, 4)):
                    img = img_to_save[i]
                    if img.shape[-1] == 1:
                        img = np.squeeze(img, axis=-1)
                    
                    # FIX: Transpose image to correct orientation for horizontal text
                    # Model outputs (1024, 128) but text lines should be (128, 1024) horizontal
                    if img.shape[0] > img.shape[1]:  # If height > width, transpose
                        img = np.transpose(img)
                    
                    cv2.imwrite(os.path.join(args.sample_dir, f'fp32_epoch_{epoch+1:04d}_sample_{i}.png'), img)

            epoch_time = time.time() - epoch_start_time
            print(f"⏱️  Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            
            training_history["epochs"].append(epoch_metrics)

        training_history["end_time"] = datetime.now().isoformat()
        training_history["best_val_psnr"] = float(best_val_psnr)
        
        final_metrics_file = os.path.join(metrics_dir, "training_metrics_fp32_experiment.json")
        with open(final_metrics_file, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        mlflow.log_artifact(final_metrics_file)
        print(f"\n✅ FP32 Experiment completed. Best PSNR: {best_val_psnr:.2f}")

    print("\n[Phase 6/6] Training Finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EXPERIMENT: Train with FP32 (no mixed precision)')
    parser.add_argument('--tfrecord_path', type=str, default='dual_modal_gan/data/dataset_gan.tfrecord')
    parser.add_argument('--charset_path', type=str, default='real_data_preparation/real_data_charlist.txt')
    parser.add_argument('--recognizer_weights', type=str, default='models/best_htr_recognizer/best_model.weights.h5')
    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--no_restore', action='store_true')
    parser.add_argument('--checkpoint_dir', type=str, default='dual_modal_gan/outputs/checkpoints_fp32_experiment')
    parser.add_argument('--sample_dir', type=str, default='dual_modal_gan/outputs/samples_fp32_experiment')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--pixel_loss_weight', type=float, default=100.0)
    parser.add_argument('--ctc_loss_weight', type=float, default=1.0)
    parser.add_argument('--adv_loss_weight', type=float, default=2.0)
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0)
    parser.add_argument('--ctc_loss_clip_max', type=float, default=1000.0)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--discriminator_mode', type=str, default='predicted', choices=['predicted', 'ground_truth'])
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)
