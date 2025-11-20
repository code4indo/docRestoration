"""Quick diagnostic test for joint training setup"""
import tensorflow as tf
import time

print("=== DIAGNOSTIC TEST START ===\n")

# Time tracking
times = {}

# 1. Import and GPU setup
start = time.time()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices([gpus[0]], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
times['gpu_setup'] = time.time() - start
print(f"✅ GPU setup: {times['gpu_setup']:.2f}s\n")

# 2. Import models
start = time.time()
from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed
from dual_modal_gan.src.models.discriminator_enhanced_v2_fixed import build_dual_modal_discriminator_enhanced_v2_fixed

def read_charlist(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

times['imports'] = time.time() - start
print(f"✅ Model imports: {times['imports']:.2f}s\n")

# 3. Load charset
start = time.time()
charset = read_charlist("real_data_preparation/real_data_charlist.txt")
vocab_size = len(charset) + 1
times['charset'] = time.time() - start
print(f"✅ Charset loaded ({len(charset)} chars): {times['charset']:.2f}s\n")

# 4. Build generator
start = time.time()
generator = unet_enhanced()
times['generator'] = time.time() - start
print(f"✅ Generator built: {times['generator']:.2f}s\n")

# 5. Load recognizer
start = time.time()
recognizer = load_frozen_recognizer_fixed(
    weights_path="/home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5",
    charset_size=len(charset),
    return_feature_map=True
)
recognizer.trainable = True
times['recognizer'] = time.time() - start
print(f"✅ Recognizer loaded: {times['recognizer']:.2f}s\n")

# 6. Build discriminator
start = time.time()
discriminator = build_dual_modal_discriminator_enhanced_v2_fixed(
    img_shape=(1024, 128, 1),  # Note: (W, H, C) format
    max_text_len=128,
    vocab_size=vocab_size
)
times['discriminator'] = time.time() - start
print(f"✅ Discriminator built: {times['discriminator']:.2f}s\n")

# 7. Load dataset (just first batch)
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

start = time.time()
dataset = tf.data.TFRecordDataset("dual_modal_gan/data/dataset_gan.tfrecord")
dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.take(1).batch(1).prefetch(1)
batch = next(iter(dataset))
times['dataset'] = time.time() - start
print(f"✅ Dataset loaded (1 batch): {times['dataset']:.2f}s\n")

# 8. Setup optimizers
start = time.time()
optimizer_g = tf.keras.optimizers.Adam(0.0001)
optimizer_d = tf.keras.optimizers.Adam(0.0001)
optimizer_r = tf.keras.optimizers.RMSprop(0.0003)
times['optimizers'] = time.time() - start
print(f"✅ Optimizers created: {times['optimizers']:.2f}s\n")

# 9. Run ONE training step
@tf.function
def train_step_test(degraded, clean, labels):
    with tf.GradientTape(persistent=True) as tape:
        # Forward pass
        restored = generator(degraded, training=True)
        
        # Discriminator
        real_score = discriminator([clean, labels], training=True)
        fake_score = discriminator([restored, labels], training=True)
        
        # Recognizer
        rec_logits, _ = recognizer(restored, training=True)
        
        # Losses
        loss_g = tf.reduce_mean(tf.abs(restored - clean))
        loss_d = -tf.reduce_mean(tf.math.log(real_score + 1e-10)) - tf.reduce_mean(tf.math.log(1 - fake_score + 1e-10))
        loss_r = tf.reduce_mean(tf.abs(rec_logits))  # Dummy loss
        
    # Gradients
    grads_g = tape.gradient(loss_g, generator.trainable_variables)
    grads_d = tape.gradient(loss_d, discriminator.trainable_variables)
    grads_r = tape.gradient(loss_r, recognizer.trainable_variables)
    
    # Apply
    optimizer_g.apply_gradients(zip(grads_g, generator.trainable_variables))
    optimizer_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))
    optimizer_r.apply_gradients(zip(grads_r, recognizer.trainable_variables))
    
    del tape
    return loss_g, loss_d, loss_r

start = time.time()
print("🔄 Running test training step...")
degraded_batch, clean_batch, label_batch = batch
loss_g, loss_d, loss_r = train_step_test(degraded_batch, clean_batch, label_batch)
times['first_step'] = time.time() - start
print(f"✅ First training step (WITH COMPILATION): {times['first_step']:.2f}s")
print(f"   Losses: G={loss_g:.4f}, D={loss_d:.4f}, R={loss_r:.4f}\n")

# 10. Run second step (already compiled)
start = time.time()
loss_g, loss_d, loss_r = train_step_test(degraded_batch, clean_batch, label_batch)
times['second_step'] = time.time() - start
print(f"✅ Second training step (NO COMPILATION): {times['second_step']:.2f}s")
print(f"   Losses: G={loss_g:.4f}, D={loss_d:.4f}, R={loss_r:.4f}\n")

# Summary
print("="*60)
print("TIMING SUMMARY:")
print("="*60)
total = 0
for name, duration in times.items():
    print(f"{name:20s}: {duration:8.2f}s")
    total += duration
print(f"{'TOTAL':20s}: {total:8.2f}s")
print("="*60)

print("\n✅ DIAGNOSTIC TEST COMPLETE - All components working!")
