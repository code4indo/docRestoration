"""Test dataset split to verify TFRecord loading works correctly"""
import tensorflow as tf

def _parse_tfrecord_fn(example_proto):
    """Parse TFRecord"""
    feature_description = {
        'degraded': tf.io.FixedLenFeature([], tf.string),
        'clean': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    height = tf.cast(parsed['height'], tf.int32)
    width = tf.cast(parsed['width'], tf.int32)
    
    degraded = tf.io.decode_raw(parsed['degraded'], tf.uint8)
    degraded = tf.reshape(degraded, [height, width, 3])
    degraded = tf.cast(degraded, tf.float32) / 255.0
    
    clean = tf.io.decode_raw(parsed['clean'], tf.uint8)
    clean = tf.reshape(clean, [height, width, 3])
    clean = tf.cast(clean, tf.float32) / 255.0
    
    label = parsed['label']
    
    return degraded, clean, label

print("Testing TFRecord dataset loading...")
tfrecord_path = "dual_modal_gan/data/dataset_gan.tfrecord"

print("\n1. Testing single dataset load and parse:")
dataset1 = tf.data.TFRecordDataset(tfrecord_path).map(_parse_tfrecord_fn)
sample1 = next(iter(dataset1))
print(f"   ✓ Sample shape: {sample1[0].shape}, {sample1[1].shape}")
print(f"   ✓ Label: {sample1[2].numpy().decode('utf-8')[:50]}...")

print("\n2. Testing dataset split with separate instances:")
total_size = 4739
train_size = 3317
val_size = 710

dataset_train = tf.data.TFRecordDataset(tfrecord_path).map(_parse_tfrecord_fn).take(train_size)
dataset_val = tf.data.TFRecordDataset(tfrecord_path).map(_parse_tfrecord_fn).skip(train_size).take(val_size)
dataset_test = tf.data.TFRecordDataset(tfrecord_path).map(_parse_tfrecord_fn).skip(train_size + val_size)

print("   Loading train sample...")
train_sample = next(iter(dataset_train))
print(f"   ✓ Train shape: {train_sample[0].shape}")

print("   Loading val sample...")
val_sample = next(iter(dataset_val))
print(f"   ✓ Val shape: {val_sample[0].shape}")

print("   Loading test sample...")
test_sample = next(iter(dataset_test))
print(f"   ✓ Test shape: {test_sample[0].shape}")

print("\n3. Testing iteration (5 samples each):")
print("   Train samples:")
for i, (deg, clean, label) in enumerate(dataset_train.take(5)):
    print(f"      {i+1}. Shape: {deg.shape}, Label: {label.numpy().decode('utf-8')[:30]}...")

print("   Val samples:")
for i, (deg, clean, label) in enumerate(dataset_val.take(5)):
    print(f"      {i+1}. Shape: {deg.shape}, Label: {label.numpy().decode('utf-8')[:30]}...")

print("\n✅ All tests passed! Dataset split approach works.")
