#!/usr/bin/env python3
"""
Diagnostic Script: Analyze Cross-Modal Attention Weights
Purpose: Verify if discriminator actually uses text information
Author: Senior ML Engineer
Date: 2025-11-06
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dual_modal_gan.src.models.discriminator_enhanced_v2_fixed import build_dual_modal_discriminator_enhanced_v2_fixed

def load_checkpoint(checkpoint_dir):
    """Load discriminator from checkpoint"""
    print(f"\n{'='*80}")
    print(f"Loading checkpoint from: {checkpoint_dir}")
    print(f"{'='*80}\n")
    
    # Load epoch info
    epoch_info_path = Path(checkpoint_dir) / "epoch_info.json"
    if epoch_info_path.exists():
        with open(epoch_info_path) as f:
            epoch_info = json.load(f)
        print(f"Checkpoint Info:")
        print(f"  Best Epoch:  {epoch_info['best_epoch']}")
        print(f"  Best PSNR:   {epoch_info['best_psnr']:.2f} dB")
        print(f"  Best CER:    {epoch_info['best_cer']*100:.1f}%")
    
    # Find latest checkpoint
    ckpt_path = Path(checkpoint_dir) / "best_model"
    if not ckpt_path.exists():
        ckpt_path = Path(checkpoint_dir)
    
    latest = tf.train.latest_checkpoint(str(ckpt_path))
    if latest is None:
        print(f"❌ No checkpoint found in {ckpt_path}")
        return None
    
    print(f"\n✅ Found checkpoint: {latest}\n")
    return latest, epoch_info


def create_sample_data(batch_size=4):
    """Create sample data for testing"""
    # Sample images (H, W, 1)
    images = tf.random.normal([batch_size, 128, 1024, 1])
    
    # Sample text (batch, max_len)
    # Use realistic text length distribution
    texts = tf.random.uniform([batch_size, 128], minval=1, maxval=100, dtype=tf.int32)
    
    return images, texts


def analyze_attention_layer(discriminator, images, texts, layer_name='cross_modal_attention'):
    """Analyze attention weights from cross-modal attention layer"""
    
    print(f"\n{'='*80}")
    print(f"ANALYZING CROSS-MODAL ATTENTION LAYER: {layer_name}")
    print(f"{'='*80}\n")
    
    # Try to get the layer
    try:
        attention_layer = discriminator.get_layer(layer_name)
        print(f"✅ Found attention layer: {attention_layer.name}")
        print(f"   Type: {type(attention_layer)}")
    except:
        print(f"⚠️  Layer '{layer_name}' not found. Searching for attention layers...")
        
        # List all layers
        attention_layers = [l for l in discriminator.layers if 'attention' in l.name.lower()]
        if attention_layers:
            print(f"\nFound {len(attention_layers)} attention-related layers:")
            for layer in attention_layers:
                print(f"  - {layer.name} ({type(layer).__name__})")
            
            # Use first attention layer
            attention_layer = attention_layers[0]
            print(f"\n✅ Using: {attention_layer.name}")
        else:
            print(f"❌ No attention layers found in discriminator!")
            return None
    
    # Create a model that outputs attention weights
    # We need to modify this based on actual architecture
    print(f"\n📊 Attempting to extract attention weights...\n")
    
    # Method 1: Direct layer output
    try:
        # Create intermediate model
        if hasattr(attention_layer, 'output'):
            attention_model = tf.keras.Model(
                inputs=discriminator.input,
                outputs=attention_layer.output
            )
            
            # Forward pass
            if isinstance(discriminator.input, list):
                attention_output = attention_model([images, texts])
            else:
                attention_output = attention_model(images)
            
            print(f"✅ Attention output shape: {attention_output.shape}")
            return attention_output
        else:
            print(f"⚠️  Layer doesn't have direct output")
    except Exception as e:
        print(f"⚠️  Method 1 failed: {e}")
    
    # Method 2: Check layer weights
    try:
        weights = attention_layer.get_weights()
        if weights:
            print(f"✅ Found {len(weights)} weight matrices in attention layer:")
            for i, w in enumerate(weights):
                print(f"   Weight {i}: shape {w.shape}, mean={w.mean():.6f}, std={w.std():.6f}")
            return weights
        else:
            print(f"⚠️  No weights found in layer")
    except Exception as e:
        print(f"⚠️  Method 2 failed: {e}")
    
    return None


def analyze_discriminator_architecture(discriminator):
    """Deep dive into discriminator architecture"""
    
    print(f"\n{'='*80}")
    print(f"DISCRIMINATOR ARCHITECTURE ANALYSIS")
    print(f"{'='*80}\n")
    
    print(f"Model name: {discriminator.name}")
    print(f"Total layers: {len(discriminator.layers)}")
    print(f"Total params: {discriminator.count_params():,}")
    
    # Analyze inputs
    print(f"\n📥 INPUTS:")
    if isinstance(discriminator.input, list):
        print(f"   Multi-input model ({len(discriminator.input)} inputs):")
        for i, inp in enumerate(discriminator.input):
            print(f"   Input {i}: {inp.name}, shape={inp.shape}")
    else:
        print(f"   Single input: {discriminator.input.name}, shape={discriminator.input.shape}")
    
    # Analyze outputs
    print(f"\n📤 OUTPUTS:")
    if isinstance(discriminator.output, list):
        print(f"   Multi-output model ({len(discriminator.output)} outputs):")
        for i, out in enumerate(discriminator.output):
            print(f"   Output {i}: {out.name}, shape={out.shape}")
    else:
        print(f"   Single output: {discriminator.output.name}, shape={discriminator.output.shape}")
    
    # Find key layers
    print(f"\n🔍 KEY LAYERS:")
    
    key_patterns = ['attention', 'lstm', 'bidir', 'cross', 'modal', 'text', 'image']
    for pattern in key_patterns:
        matching = [l for l in discriminator.layers if pattern in l.name.lower()]
        if matching:
            print(f"\n   {pattern.upper()} layers ({len(matching)}):")
            for layer in matching[:5]:  # Show first 5
                print(f"   - {layer.name}")
                print(f"     Type: {type(layer).__name__}")
                if hasattr(layer, 'output_shape'):
                    print(f"     Output shape: {layer.output_shape}")


def test_discriminator_forward_pass(discriminator, images, texts):
    """Test discriminator with different text inputs"""
    
    print(f"\n{'='*80}")
    print(f"TESTING DISCRIMINATOR FORWARD PASS")
    print(f"{'='*80}\n")
    
    batch_size = images.shape[0]
    
    # Test 1: Original texts
    print(f"Test 1: Original random texts")
    try:
        if isinstance(discriminator.input, list):
            output1 = discriminator([images, texts], training=False)
        else:
            output1 = discriminator(images, training=False)
        print(f"✅ Output shape: {output1.shape}")
        print(f"   Mean: {tf.reduce_mean(output1).numpy():.6f}")
        print(f"   Std:  {tf.reduce_std(output1).numpy():.6f}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        output1 = None
    
    # Test 2: All zeros text (no text information)
    print(f"\nTest 2: Zero texts (no information)")
    zeros_text = tf.zeros_like(texts)
    try:
        if isinstance(discriminator.input, list):
            output2 = discriminator([images, zeros_text], training=False)
        else:
            output2 = discriminator(images, training=False)
        print(f"✅ Output shape: {output2.shape}")
        print(f"   Mean: {tf.reduce_mean(output2).numpy():.6f}")
        print(f"   Std:  {tf.reduce_std(output2).numpy():.6f}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        output2 = None
    
    # Test 3: All ones text (uniform information)
    print(f"\nTest 3: Ones texts (uniform)")
    ones_text = tf.ones_like(texts)
    try:
        if isinstance(discriminator.input, list):
            output3 = discriminator([images, ones_text], training=False)
        else:
            output3 = discriminator(images, training=False)
        print(f"✅ Output shape: {output3.shape}")
        print(f"   Mean: {tf.reduce_mean(output3).numpy():.6f}")
        print(f"   Std:  {tf.reduce_std(output3).numpy():.6f}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        output3 = None
    
    # Compare outputs
    print(f"\n📊 COMPARISON:")
    if output1 is not None and output2 is not None and output3 is not None:
        diff_1_2 = tf.reduce_mean(tf.abs(output1 - output2)).numpy()
        diff_1_3 = tf.reduce_mean(tf.abs(output1 - output3)).numpy()
        diff_2_3 = tf.reduce_mean(tf.abs(output2 - output3)).numpy()
        
        print(f"   |Original - Zero|:    {diff_1_2:.6f}")
        print(f"   |Original - Ones|:    {diff_1_3:.6f}")
        print(f"   |Zero - Ones|:        {diff_2_3:.6f}")
        
        print(f"\n🔍 DIAGNOSIS:")
        threshold = 0.01  # If difference < 0.01, texts don't matter
        
        if diff_1_2 < threshold and diff_1_3 < threshold:
            print(f"   ❌ CRITICAL BUG: Text input has NO EFFECT on discriminator!")
            print(f"      Different text inputs produce nearly identical outputs")
            print(f"      → Cross-modal attention is NOT working!")
            return False
        else:
            print(f"   ✅ Text input DOES affect discriminator output")
            print(f"      Different texts produce different outputs")
            print(f"      → Cross-modal attention appears to be working")
            return True
    
    return None


def main():
    """Main diagnostic routine"""
    
    print(f"\n{'#'*80}")
    print(f"# CROSS-MODAL ATTENTION DIAGNOSTIC TOOL")
    print(f"# Purpose: Verify if dual-modal discriminator uses text information")
    print(f"{'#'*80}\n")
    
    # Checkpoints to analyze
    checkpoints = [
        "dual_modal_gan/checkpoints/exp_proof_ground_truth",
        "dual_modal_gan/checkpoints/exp_proof_predicted",
    ]
    
    results = {}
    
    for ckpt_dir in checkpoints:
        print(f"\n{'#'*80}")
        print(f"# ANALYZING: {ckpt_dir}")
        print(f"{'#'*80}")
        
        if not Path(ckpt_dir).exists():
            print(f"⚠️  Checkpoint directory not found, skipping...")
            continue
        
        # Load checkpoint
        ckpt_info = load_checkpoint(ckpt_dir)
        if ckpt_info is None:
            continue
        
        ckpt_path, epoch_info = ckpt_info
        
        # Create discriminator architecture
        print(f"\n📦 Creating discriminator architecture...")
        
        # Use same config as training
        discriminator_config = {
            'spatial_attention_kernel': 3,
            'cross_modal_common_dim': 128,
            'batchnorm_momentum': 0.9,
            'dropout_rate': 0.1,
        }
        
        discriminator = build_dual_modal_discriminator_enhanced_v2_fixed(
            img_shape=(128, 1024, 1),
            vocab_size=100,  # Approximate
            max_text_len=128,
            text_embed_dim=128,
            lstm_units=256,
            config=discriminator_config
        )
        
        print(f"✅ Discriminator created")
        
        # Load weights
        print(f"\n📥 Loading weights from checkpoint...")
        try:
            discriminator.load_weights(ckpt_path)
            print(f"✅ Weights loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load weights: {e}")
            continue
        
        # Analyze architecture
        analyze_discriminator_architecture(discriminator)
        
        # Create sample data
        print(f"\n📊 Creating sample data...")
        images, texts = create_sample_data(batch_size=4)
        print(f"   Images shape: {images.shape}")
        print(f"   Texts shape:  {texts.shape}")
        
        # Test forward pass with different text inputs
        text_matters = test_discriminator_forward_pass(discriminator, images, texts)
        
        # Try to analyze attention layer
        attention_output = analyze_attention_layer(discriminator, images, texts)
        
        # Store results
        results[ckpt_dir] = {
            'text_matters': text_matters,
            'attention_found': attention_output is not None,
            'best_psnr': epoch_info['best_psnr'],
            'best_cer': epoch_info['best_cer']
        }
    
    # Final summary
    print(f"\n{'#'*80}")
    print(f"# DIAGNOSTIC SUMMARY")
    print(f"{'#'*80}\n")
    
    for ckpt_dir, result in results.items():
        mode = "GROUND TRUTH" if "ground_truth" in ckpt_dir else "PREDICTED"
        print(f"\n{mode} MODE:")
        print(f"  PSNR:           {result['best_psnr']:.2f} dB")
        print(f"  CER:            {result['best_cer']*100:.1f}%")
        print(f"  Text matters:   {'✅ YES' if result['text_matters'] else '❌ NO'}")
        print(f"  Attention found: {'✅ YES' if result['attention_found'] else '❌ NO'}")
    
    # Overall diagnosis
    print(f"\n{'='*80}")
    print(f"OVERALL DIAGNOSIS:")
    print(f"{'='*80}\n")
    
    text_effectiveness = [r['text_matters'] for r in results.values() if r['text_matters'] is not None]
    
    if not text_effectiveness:
        print(f"⚠️  Could not determine if text affects discriminator")
        print(f"   RECOMMENDATION: Manual inspection of model architecture required")
    elif not any(text_effectiveness):
        print(f"❌ CRITICAL BUG CONFIRMED!")
        print(f"   Text input does NOT affect discriminator output")
        print(f"   Cross-modal attention is NOT working!")
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Check attention layer implementation")
        print(f"   2. Increase text feature dimensions (128 → 512)")
        print(f"   3. Add attention regularization")
        print(f"   4. Balance image-text contribution")
    else:
        print(f"✅ Text input DOES affect discriminator")
        print(f"   Cross-modal attention appears to be working")
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Analyze WHY ground_truth ≈ predicted despite working attention")
        print(f"   2. Possible causes:")
        print(f"      - Text contribution too weak (5-10% vs 90-95% image)")
        print(f"      - Frozen recognizer bottleneck")
        print(f"      - Dataset degradation too simple")
    
    return results


if __name__ == "__main__":
    main()
