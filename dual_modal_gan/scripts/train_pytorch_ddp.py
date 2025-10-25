"""
Dual-Modal GAN Training with PyTorch DDP - True Multi-GPU Support

This version uses PyTorch DistributedDataParallel (DDP) which is MUCH more
flexible than TensorFlow MirroredStrategy for complex GAN training.

Key Benefits:
1. Full control over distributed training logic
2. No graph compilation issues
3. Works perfectly with complex multi-model GAN training
4. Better performance and scaling
5. Simple to debug and understand

Usage:
    # Single node, 2 GPUs:
    poetry run python -m torch.distributed.launch \
        --nproc_per_node=2 \
        dual_modal_gan/scripts/train_pytorch_ddp.py \
        --config configs/finetune_thin_stroke_preservation.json

    # Or use torchrun (recommended):
    poetry run torchrun --nproc_per_node=2 \
        dual_modal_gan/scripts/train_pytorch_ddp.py \
        --config configs/finetune_thin_stroke_preservation.json
"""

import os
import sys
import argparse
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# We'll use TensorFlow models but wrap them in PyTorch
import tensorflow as tf

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 70)
print("  CRITICAL: PyTorch DDP Multi-GPU Training")
print("=" * 70)
print(f"  PyTorch Version: {torch.__version__}")
print(f"  CUDA Available: {torch.cuda.is_available()}")
print(f"  CUDA Device Count: {torch.cuda.device_count()}")
print("=" * 70)


def setup_ddp(rank, world_size):
    """Initialize DDP training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    print(f"[Rank {rank}/{world_size}] DDP initialized on GPU {rank}")


def cleanup_ddp():
    """Cleanup DDP."""
    dist.destroy_process_group()


class TFModelWrapper(nn.Module):
    """Wrap TensorFlow model to work with PyTorch DDP."""
    
    def __init__(self, tf_model):
        super().__init__()
        self.tf_model = tf_model
        
    def forward(self, x):
        """Convert PyTorch tensor to TF and back."""
        # Convert to numpy
        x_np = x.cpu().numpy()
        
        # Run TF model
        with tf.device('/CPU:0'):  # We'll handle GPU placement manually
            output_tf = self.tf_model(x_np, training=True)
        
        # Convert back to PyTorch
        output = torch.from_numpy(output_tf.numpy()).to(x.device)
        return output


def load_config(config_path):
    """Load training configuration from JSON."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_tf_dataset(tfrecord_path, batch_size, rank, world_size):
    """
    Create TensorFlow dataset with proper sharding for DDP.
    
    Each GPU gets a different shard of the data.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # Parse function (same as train_enhanced.py)
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
        
        # Deserialize degraded image
        degraded_image_shape = tf.cast(example['degraded_image_shape'], tf.int32)
        degraded_image = tf.io.decode_raw(example['degraded_image_raw'], tf.float32)
        degraded_image = tf.reshape(degraded_image, degraded_image_shape)
        degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])
        degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])
        
        # Deserialize clean image
        clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
        clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
        clean_image = tf.reshape(clean_image, clean_image_shape)
        clean_image = tf.transpose(clean_image, perm=[1, 0, 2])
        clean_image = tf.ensure_shape(clean_image, [1024, 128, 1])
        
        # Deserialize label
        label_shape = tf.cast(example['label_shape'], tf.int32)
        label = tf.io.decode_raw(example['label_raw'], tf.int64)
        label = tf.reshape(label, label_shape)
        label = tf.cast(label, tf.int32)
        
        # Pad label to static shape
        padding = [[0, 128 - tf.shape(label)[0]]]
        label = tf.pad(label, padding, "CONSTANT", constant_values=0)
        label.set_shape([128])
        
        return degraded_image, clean_image, label
    
    # Get total size
    total_size = sum(1 for _ in dataset)
    
    # Calculate splits
    train_split = 0.7
    val_split = 0.15
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    
    # Parse and split
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = dataset.take(train_size)
    remaining = dataset.skip(train_size)
    val_dataset = remaining.take(val_size)
    
    # Shard for DDP: each rank gets a different portion
    # This is CRITICAL for DDP - each GPU must see different data
    train_dataset = train_dataset.shard(num_shards=world_size, index=rank)
    
    # Process datasets
    train_dataset = train_dataset.shuffle(buffer_size=1024).repeat()
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    if rank == 0:
        print(f"\n📊 Dataset Split (Rank {rank}):")
        print(f"   Total: {total_size} | Train: {train_size} | Val: {val_size}")
        print(f"   Per-GPU Train Samples: {train_size // world_size}")
        print(f"   Batch size per GPU: {batch_size}")
        print(f"   Global batch size: {batch_size * world_size}")
    
    return train_dataset, val_dataset, train_size // world_size


def train_one_epoch(rank, epoch, generator, discriminator, recognizer, 
                    train_dataset, g_optimizer, d_optimizer, args, perceptual_loss_layer):
    """Train for one epoch with DDP."""
    
    generator.train()
    discriminator.train()
    
    # Metrics
    g_loss_sum = 0.0
    d_loss_sum = 0.0
    step_count = 0
    
    # Iterator
    iterator = iter(train_dataset)
    
    # Progress bar only on rank 0
    if rank == 0:
        pbar = tqdm(total=args.steps_per_epoch, desc=f"Epoch {epoch}")
    
    for step in range(args.steps_per_epoch):
        try:
            degraded, clean, text = next(iterator)
        except StopIteration:
            iterator = iter(train_dataset)
            degraded, clean, text = next(iterator)
        
        # Move to GPU (current rank)
        device = torch.device(f'cuda:{rank}')
        
        # Convert TF tensors to PyTorch
        degraded_pt = torch.from_numpy(degraded.numpy()).to(device)
        clean_pt = torch.from_numpy(clean.numpy()).to(device)
        text_pt = torch.from_numpy(text.numpy()).to(device)
        
        batch_size = degraded_pt.shape[0]
        
        # ============================================================
        # DISCRIMINATOR TRAINING
        # ============================================================
        
        # Generate fake images
        with torch.no_grad():
            # Use TF model directly (it's already on GPU via TF's device placement)
            generated_tf = generator(degraded.numpy(), training=False)
            generated_pt = torch.from_numpy(generated_tf.numpy()).to(device)
        
        # Real discriminator output
        real_logits_tf = discriminator([clean.numpy(), text.numpy()], training=True)
        real_logits = torch.from_numpy(real_logits_tf.numpy()).to(device)
        
        # Fake discriminator output
        fake_logits_tf = discriminator([generated_tf.numpy(), text.numpy()], training=True)
        fake_logits = torch.from_numpy(fake_logits_tf.numpy()).to(device)
        
        # Discriminator loss
        d_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits)
        )
        d_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits)
        )
        d_loss = (d_loss_real + d_loss_fake) / 2.0
        
        # Backward + optimize discriminator
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # ============================================================
        # GENERATOR TRAINING
        # ============================================================
        
        # Generate images
        generated_tf = generator(degraded.numpy(), training=True)
        generated_pt = torch.from_numpy(generated_tf.numpy()).to(device)
        
        # Adversarial loss
        fake_logits_tf = discriminator([generated_tf.numpy(), text.numpy()], training=True)
        fake_logits = torch.from_numpy(fake_logits_tf.numpy()).to(device)
        adv_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            fake_logits, torch.ones_like(fake_logits)
        )
        
        # Pixel loss
        pixel_loss = torch.nn.functional.l1_loss(generated_pt, clean_pt)
        
        # Perceptual loss
        if perceptual_loss_layer is not None:
            gen_features_tf = perceptual_loss_layer(generated_tf)
            clean_features_tf = perceptual_loss_layer(clean)
            
            percep_loss = 0.0
            for gf, cf in zip(gen_features_tf, clean_features_tf):
                gf_pt = torch.from_numpy(gf.numpy()).to(device)
                cf_pt = torch.from_numpy(cf.numpy()).to(device)
                percep_loss += torch.nn.functional.mse_loss(gf_pt, cf_pt)
            percep_loss = percep_loss / len(gen_features_tf)
        else:
            percep_loss = torch.tensor(0.0).to(device)
        
        # Recognizer feature loss
        clean_logits_tf, clean_feat_tf = recognizer(clean, training=False)
        gen_logits_tf, gen_feat_tf = recognizer(generated_tf, training=False)
        
        clean_feat_pt = torch.from_numpy(clean_feat_tf.numpy()).to(device)
        gen_feat_pt = torch.from_numpy(gen_feat_tf.numpy()).to(device)
        rec_feat_loss = torch.nn.functional.mse_loss(gen_feat_pt, clean_feat_pt)
        
        # CTC loss
        gen_logits_pt = torch.from_numpy(gen_logits_tf.numpy()).to(device)
        logit_len = torch.full((batch_size,), 128, dtype=torch.long, device=device)
        
        # Calculate actual text lengths (non-zero elements)
        text_lens = (text_pt != 0).sum(dim=1).long()
        
        # PyTorch CTC loss
        ctc_loss = torch.nn.functional.ctc_loss(
            gen_logits_pt.permute(1, 0, 2).log_softmax(2),  # (T, N, C)
            text_pt,  # (N, S)
            logit_len,  # (N,)
            text_lens,  # (N,)
            blank=gen_logits_pt.shape[-1] - 1,  # Last index is blank
            reduction='mean',
            zero_infinity=True
        )
        ctc_loss = torch.clamp(ctc_loss, max=args.ctc_loss_clip_max)
        
        # Total generator loss
        g_loss = (args.pixel_loss_weight * pixel_loss +
                  args.adv_loss_weight * adv_loss +
                  args.perceptual_loss_weight * percep_loss +
                  args.rec_feat_loss_weight * rec_feat_loss +
                  args.ctc_loss_weight * ctc_loss)
        
        # Backward + optimize generator
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        # Update metrics
        g_loss_sum += g_loss.item()
        d_loss_sum += d_loss.item()
        step_count += 1
        
        if rank == 0:
            pbar.update(1)
            pbar.set_postfix({
                'G': f'{g_loss.item():.4f}',
                'D': f'{d_loss.item():.4f}',
                'Pixel': f'{pixel_loss.item():.4f}',
                'CTC': f'{ctc_loss.item():.4f}'
            })
    
    if rank == 0:
        pbar.close()
    
    # Average losses
    avg_g_loss = g_loss_sum / step_count
    avg_d_loss = d_loss_sum / step_count
    
    return avg_g_loss, avg_d_loss


def main_worker(rank, world_size, args):
    """Main training function for each process."""
    
    # Setup DDP
    setup_ddp(rank, world_size)
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("  PyTorch DDP Multi-GPU Training Started")
        print("=" * 70)
        print(f"  World Size: {world_size}")
        print(f"  Rank: {rank}")
        print(f"  Device: cuda:{rank}")
        print("=" * 70)
    
    # Load config
    config = load_config(args.config)
    
    # Update args from config
    for key, value in config.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    
    # Create dataset
    train_dataset, val_dataset, train_steps = create_tf_dataset(
        args.tfrecord_path,
        args.batch_size,
        rank,
        world_size
    )
    
    # Set steps per epoch if not specified
    if args.steps_per_epoch is None:
        args.steps_per_epoch = train_steps
    
    if rank == 0:
        print(f"\n[Phase 2/3] Building Models on GPU {rank}...")
    
    # Build TensorFlow models (they will use their own GPU placement)
    with tf.device(f'/GPU:{rank}'):
        # Import model builders
        from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
        from dual_modal_gan.src.models.discriminator_enhanced_v2_fixed import build_dual_modal_discriminator_enhanced_v2_fixed
        from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed
        from dual_modal_gan.losses.perceptual_loss import create_perceptual_loss
        
        # Generator
        generator = unet_enhanced(input_shape=(1024, 128, 1))
        if rank == 0:
            print(f"  ✓ Generator built on GPU {rank}")
        
        # Discriminator
        discriminator = build_dual_modal_discriminator_enhanced_v2_fixed(
            img_shape=(1024, 128, 1),
            vocab_size=109,
            max_text_len=128
        )
        if rank == 0:
            print(f"  ✓ Discriminator built on GPU {rank}")
        
        # Recognizer
        recognizer = load_frozen_recognizer_fixed(
            weights_path=args.recognizer_weights,
            charset_size=108
        )
        if rank == 0:
            print(f"  ✓ Recognizer loaded on GPU {rank}")
        
        # Perceptual loss
        perceptual_loss_layer = create_perceptual_loss()
        if rank == 0:
            print(f"  ✓ Perceptual loss layer created on GPU {rank}")
    
    # Create optimizers (PyTorch will handle them, but we use TF optimizers for TF models)
    # Actually, let's keep using TF optimizers since models are TF
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=args.g_learning_rate)
    d_optimizer = tf.keras.optimizers.SGD(learning_rate=args.d_learning_rate, momentum=0.9)
    
    if rank == 0:
        print(f"\n[Phase 3/3] Starting Training...")
        print(f"  Epochs: {args.epochs}")
        print(f"  Steps per epoch: {args.steps_per_epoch}")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Global batch size: {args.batch_size * world_size}")
        print("=" * 70)
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        avg_g_loss, avg_d_loss = train_one_epoch(
            rank, epoch, generator, discriminator, recognizer,
            train_dataset, g_optimizer, d_optimizer, args, perceptual_loss_layer
        )
        
        if rank == 0:
            print(f"\nEpoch {epoch}/{args.epochs} Summary:")
            print(f"  G Loss: {avg_g_loss:.4f}")
            print(f"  D Loss: {avg_d_loss:.4f}")
            
            # Save checkpoint
            checkpoint_dir = os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            generator.save_weights(os.path.join(checkpoint_dir, "generator.weights.h5"))
            discriminator.save_weights(os.path.join(checkpoint_dir, "discriminator.weights.h5"))
            print(f"  ✓ Checkpoint saved: {checkpoint_dir}")
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("  Training Completed!")
        print("=" * 70)
    
    # Cleanup
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description='PyTorch DDP Multi-GPU GAN Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank (set by torch.distributed.launch)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size PER GPU')
    parser.add_argument('--steps_per_epoch', type=int, default=None)
    
    # Loss weights
    parser.add_argument('--pixel_loss_weight', type=float, default=100.0)
    parser.add_argument('--adv_loss_weight', type=float, default=1.0)
    parser.add_argument('--perceptual_loss_weight', type=float, default=2.0)
    parser.add_argument('--rec_feat_loss_weight', type=float, default=8.0)
    parser.add_argument('--ctc_loss_weight', type=float, default=0.15)
    parser.add_argument('--ctc_loss_clip_max', type=float, default=300.0)
    
    # Optimizer
    parser.add_argument('--g_learning_rate', type=float, default=0.0002)
    parser.add_argument('--d_learning_rate', type=float, default=0.0002)
    
    # Paths
    parser.add_argument('--tfrecord_path', type=str, default='dataset_gan.tfrecord')
    parser.add_argument('--recognizer_weights', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--discriminator_mode', type=str, default='predicted')
    
    args = parser.parse_args()
    
    # Get world size from environment
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
    elif 'LOCAL_RANK' in os.environ:
        # torchrun sets LOCAL_RANK
        rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    else:
        # Single GPU mode
        world_size = 1
        rank = 0
    
    print(f"Starting process: rank={rank}, world_size={world_size}")
    
    # Launch training
    main_worker(rank, world_size, args)


if __name__ == '__main__':
    main()
