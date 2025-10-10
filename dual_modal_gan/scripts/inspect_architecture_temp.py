#!/usr/bin/env python3
"""
Inspect Architecture - Deep Dive Analysis
Tujuan: Menganalisis arsitektur secara detail dan mencari masalah potensial
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import tensorflow as tf
import numpy as np
from pathlib import Path
import json

from dual_modal_gan.src.models.generator import unet
from dual_modal_gan.src.models.discriminator import build_dual_modal_discriminator
from dual_modal_gan.src.models.recognizer import load_frozen_recognizer

def read_charlist(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

# ============================================================================
# CRITICAL ANALYSIS: GENERATOR
# ============================================================================
def analyze_generator():
    """Analisis mendalam arsitektur Generator"""
    print("\n" + "="*80)
    print("CRITICAL ANALYSIS: GENERATOR (U-Net)")
    print("="*80)
    
    generator = unet(input_size=(128, 1024, 1))
    
    print("\n🔍 1. LAYER STRUCTURE ANALYSIS")
    print("-" * 80)
    
    # Collect all layers info
    layer_info = []
    for i, layer in enumerate(generator.layers):
        info = {
            'index': i,
            'name': layer.name,
            'type': type(layer).__name__,
            'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A',
            'num_params': layer.count_params() if hasattr(layer, 'count_params') else 0
        }
        
        # Get activation if Conv2D
        if hasattr(layer, 'activation'):
            if layer.activation is not None:
                info['activation'] = layer.activation.__name__
            else:
                info['activation'] = 'linear'
        
        layer_info.append(info)
    
    # Analyze encoder path
    print("\n📊 ENCODER PATH (Downsampling):")
    encoder_layers = [l for l in layer_info if 'pool' in l['name'] or ('conv' in l['name'] and int(l['name'].split('_')[1]) <= 5)]
    for layer in encoder_layers[:15]:  # First 15 layers
        print(f"  {layer['index']:3d}. {layer['name']:20s} ({layer['type']:20s}) -> {layer['output_shape']}")
    
    # Analyze decoder path
    print("\n📊 DECODER PATH (Upsampling):")
    decoder_layers = [l for l in layer_info if 'up_sampling' in l['name'] or ('conv' in l['name'] and int(l['name'].split('_')[1]) >= 6)]
    for layer in decoder_layers[:15]:
        print(f"  {layer['index']:3d}. {layer['name']:20s} ({layer['type']:20s}) -> {layer['output_shape']}")
    
    # Find the output layer
    output_layer = generator.layers[-1]
    print(f"\n📊 OUTPUT LAYER:")
    print(f"  Name: {output_layer.name}")
    print(f"  Type: {type(output_layer).__name__}")
    print(f"  Config: {output_layer.get_config()}")
    
    activation = output_layer.get_config().get('activation', 'none')
    print(f"\n🎯 CRITICAL CHECK - Output Activation: {activation}")
    
    if activation == 'sigmoid':
        print("  ✅ CORRECT: Sigmoid activation ensures output in [0, 1]")
    else:
        print("  ❌ ERROR: Output activation bukan sigmoid!")
        print("  💡 PENYEBAB MASALAH: Output tidak bounded, bisa menghasilkan nilai negatif atau > 1")
    
    # Check for skip connections
    print(f"\n🔍 2. SKIP CONNECTIONS CHECK")
    print("-" * 80)
    concatenate_layers = [l for l in layer_info if l['type'] == 'Concatenate']
    print(f"  Found {len(concatenate_layers)} skip connections")
    for layer in concatenate_layers:
        print(f"    - {layer['name']} at index {layer['index']}")
    
    if len(concatenate_layers) == 4:
        print("  ✅ Standard U-Net with 4 skip connections")
    else:
        print(f"  ⚠️ WARNING: Expected 4 skip connections, found {len(concatenate_layers)}")
    
    # Check activations in main path
    print(f"\n🔍 3. ACTIVATION FUNCTIONS CHECK")
    print("-" * 80)
    
    activations = {}
    for layer in layer_info:
        if 'activation' in layer:
            act = layer['activation']
            if act not in activations:
                activations[act] = 0
            activations[act] += 1
    
    print("  Activation distribution:")
    for act, count in activations.items():
        print(f"    - {act}: {count} layers")
    
    # Check for potential issues
    issues = []
    
    # Issue 1: Linear activations in hidden layers
    linear_layers = [l for l in layer_info if l.get('activation') == 'linear' and l['name'] != output_layer.name]
    if len(linear_layers) > 0:
        issues.append(f"Found {len(linear_layers)} linear activations in hidden layers (might limit expressiveness)")
    
    # Issue 2: No batch normalization
    bn_layers = [l for l in layer_info if 'BatchNorm' in l['type']]
    if len(bn_layers) == 0:
        issues.append("No Batch Normalization layers (could affect training stability)")
    
    # Issue 3: Large network
    total_params = sum([l['num_params'] for l in layer_info])
    if total_params > 50_000_000:
        issues.append(f"Very large network ({total_params:,} params) - might be hard to train")
    
    print(f"\n🎯 POTENTIAL ISSUES FOUND: {len(issues)}")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    # Save architecture to file
    output_dir = Path('dual_modal_gan/outputs/debug_architecture')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'generator_layers.json', 'w') as f:
        json.dump(layer_info, f, indent=2)
    
    print(f"\n📝 Layer info saved to: {output_dir / 'generator_layers.json'}")
    
    return generator, issues

# ============================================================================
# CRITICAL ANALYSIS: DISCRIMINATOR
# ============================================================================
def analyze_discriminator():
    """Analisis mendalam arsitektur Discriminator"""
    print("\n" + "="*80)
    print("CRITICAL ANALYSIS: DISCRIMINATOR (Dual-Modal)")
    print("="*80)
    
    discriminator = build_dual_modal_discriminator(
        img_shape=(128, 1024, 1),
        vocab_size=100,
        max_text_len=128
    )
    
    print("\n🔍 DUAL-MODAL ARCHITECTURE")
    print("-" * 80)
    
    # Check inputs
    print(f"Number of inputs: {len(discriminator.inputs)}")
    for i, inp in enumerate(discriminator.inputs):
        print(f"  Input {i}: {inp.name} - shape: {inp.shape}")
    
    # Check output
    print(f"\nOutput: {discriminator.output.name} - shape: {discriminator.output.shape}")
    
    # Analyze layers
    layer_info = []
    for i, layer in enumerate(discriminator.layers):
        info = {
            'index': i,
            'name': layer.name,
            'type': type(layer).__name__,
            'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A',
        }
        layer_info.append(info)
    
    # Find image branch layers
    print("\n📊 IMAGE BRANCH:")
    img_branch = [l for l in layer_info if 'conv2d' in l['name'] or 'leaky' in l['name'] or 'batch' in l['name'] or 'flatten' in l['name']]
    for layer in img_branch[:15]:
        print(f"  {layer['name']:30s} -> {layer['output_shape']}")
    
    # Find text branch layers
    print("\n📊 TEXT BRANCH:")
    text_branch = [l for l in layer_info if 'embedding' in l['name'] or 'lstm' in l['name']]
    for layer in text_branch:
        print(f"  {layer['name']:30s} -> {layer['output_shape']}")
    
    # Find fusion layers
    print("\n📊 FUSION & CLASSIFICATION:")
    fusion_layers = [l for l in layer_info if 'concatenate' in l['name'] or 'dense' in l['name'] or 'validity' in l['name']]
    for layer in fusion_layers:
        print(f"  {layer['name']:30s} -> {layer['output_shape']}")
    
    # Critical checks
    issues = []
    
    # Check 1: Output activation
    output_layer = discriminator.layers[-1]
    output_config = output_layer.get_config()
    output_activation = output_config.get('activation', 'none')
    
    print(f"\n🎯 CRITICAL CHECK - Output Activation: {output_activation}")
    if output_activation == 'sigmoid':
        print("  ✅ CORRECT: Sigmoid for binary classification")
    else:
        print("  ❌ ERROR: Output should use sigmoid!")
        issues.append("Output activation bukan sigmoid")
    
    # Check 2: Text embedding vocab size
    embedding_layers = [l for l in discriminator.layers if 'Embedding' in type(l).__name__]
    if len(embedding_layers) > 0:
        for emb in embedding_layers:
            vocab_size = emb.input_dim
            embed_dim = emb.output_dim
            print(f"\n📊 Text Embedding: vocab_size={vocab_size}, embed_dim={embed_dim}")
    
    print(f"\n🎯 POTENTIAL ISSUES FOUND: {len(issues)}")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    return discriminator, issues

# ============================================================================
# CRITICAL ANALYSIS: RECOGNIZER
# ============================================================================
def analyze_recognizer():
    """Analisis mendalam arsitektur Recognizer"""
    print("\n" + "="*80)
    print("CRITICAL ANALYSIS: RECOGNIZER (Frozen Transformer)")
    print("="*80)
    
    charset_path = 'real_data_preparation/real_data_charlist.txt'
    weights_path = 'transformer_improved_results_fixed_20250930_061317_best/best_model_fixed.weights.h5'
    
    charset = read_charlist(charset_path)
    vocab_size = len(charset) + 1
    
    recognizer = load_frozen_recognizer(
        weights_path=weights_path,
        charset_size=vocab_size - 1
    )
    
    print(f"\n🔍 RECOGNIZER PROPERTIES")
    print("-" * 80)
    print(f"  Vocab size: {vocab_size}")
    print(f"  Trainable: {recognizer.trainable}")
    print(f"  Total trainable params: {len(recognizer.trainable_variables)}")
    print(f"  Total non-trainable params: {len(recognizer.non_trainable_variables)}")
    
    # Check critical layers
    print(f"\n📊 KEY LAYERS:")
    
    # Find transpose layer
    transpose_layers = [l for l in recognizer.layers if 'transpose' in l.name]
    if len(transpose_layers) > 0:
        print(f"  ✅ Found transpose layer: {transpose_layers[0].name}")
        print(f"     Purpose: Convert (H,W,C) -> (W,H,C) for recognizer")
    else:
        print(f"  ⚠️ No transpose layer found!")
    
    # Find CNN backbone layers
    conv_layers = [l for l in recognizer.layers if 'Conv2D' in type(l).__name__]
    print(f"  ✅ CNN backbone: {len(conv_layers)} Conv2D layers")
    
    # Find transformer layers
    mha_layers = [l for l in recognizer.layers if 'MultiHeadAttention' in type(l).__name__]
    print(f"  ✅ Transformer: {len(mha_layers)} attention layers")
    
    # Find output layer
    output_layer = recognizer.layers[-1]
    output_config = output_layer.get_config()
    output_activation = output_config.get('activation', 'none')
    print(f"  ✅ Output layer: {output_layer.name}")
    print(f"     Activation: {output_activation}")
    print(f"     Units: {output_config.get('units', 'N/A')}")
    
    if output_activation is None or output_activation == 'linear':
        print(f"  ✅ CORRECT: Linear activation for CTC loss")
    else:
        print(f"  ⚠️ WARNING: Output activation should be linear for CTC!")
    
    # Test input/output shapes
    print(f"\n🔍 INPUT/OUTPUT SHAPE TEST")
    print("-" * 80)
    
    test_input = tf.random.uniform((1, 128, 1024, 1))
    test_output = recognizer(test_input, training=False)
    
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {test_output.shape}")
    
    expected_timesteps = 128
    expected_vocab = vocab_size
    
    if test_output.shape[1] == expected_timesteps:
        print(f"  ✅ Timesteps: {test_output.shape[1]} (expected: {expected_timesteps})")
    else:
        print(f"  ❌ Timesteps: {test_output.shape[1]} (expected: {expected_timesteps})")
    
    if test_output.shape[2] == expected_vocab:
        print(f"  ✅ Vocab size: {test_output.shape[2]} (expected: {expected_vocab})")
    else:
        print(f"  ❌ Vocab size: {test_output.shape[2]} (expected: {expected_vocab})")
    
    issues = []
    
    if len(recognizer.trainable_variables) > 0:
        issues.append(f"Model not fully frozen ({len(recognizer.trainable_variables)} trainable vars)")
    
    return recognizer, issues

# ============================================================================
# DATA FLOW ANALYSIS
# ============================================================================
def analyze_data_flow():
    """Analisis aliran data dari input hingga output"""
    print("\n" + "="*80)
    print("DATA FLOW ANALYSIS")
    print("="*80)
    
    print("\n🔍 COMPLETE PIPELINE:")
    print("-" * 80)
    
    print("""
    INPUT (Degraded Image)
       |
       | shape: (batch, 128, 1024, 1)
       v
    ┌──────────────────┐
    │   GENERATOR      │  U-Net architecture
    │   (trainable)    │  Encoder-Decoder with skip connections
    └──────────────────┘
       |
       | shape: (batch, 128, 1024, 1)
       | range: [0, 1] (sigmoid output)
       v
    ┌──────────────────┐
    │  RECOGNIZER      │  Transformer-based HTR
    │  (frozen)        │  CNN + Transformer
    └──────────────────┘
       |
       | shape: (batch, 128, vocab_size)
       | logits (linear activation)
       v
    ┌──────────────────┐
    │ DISCRIMINATOR    │  Dual-modal (image + text)
    │ (trainable)      │  CNN + LSTM + Fusion
    └──────────────────┘
       |
       | shape: (batch, 1)
       | range: [0, 1] (sigmoid output)
       v
    OUTPUT (Real/Fake score)
    """)
    
    print("\n🔍 LOSS COMPUTATION:")
    print("-" * 80)
    
    print("""
    Generator Loss = w1 * L1_Loss + w2 * Adv_Loss + w3 * CTC_Loss
    
    Where:
      L1_Loss = MAE(generated_image, clean_image)
               -> Direct pixel-wise comparison
               -> Encourages visual similarity
      
      Adv_Loss = BCE(discriminator(generated), real_labels)
               -> Fool discriminator
               -> Encourages realistic appearance
      
      CTC_Loss = CTC(recognizer(generated), ground_truth_text)
               -> Readability constraint
               -> Encourages recognizable text
    
    Discriminator Loss = BCE(D(real), 1) + BCE(D(fake), 0)
               -> Distinguish real from fake
               -> Provides adversarial signal
    """)
    
    print("\n🎯 CRITICAL BOTTLENECKS:")
    print("-" * 80)
    
    bottlenecks = [
        {
            'name': 'Generator Initialization',
            'issue': 'Random weights menghasilkan noise output',
            'impact': 'Generator harus belajar dari scratch',
            'solution': 'Pre-training dengan L1 loss saja'
        },
        {
            'name': 'Loss Weight Balance',
            'issue': 'CTC loss ~50-100, L1 loss ~0.3, Adv loss ~0.5',
            'impact': 'CTC loss mendominasi, visual quality diabaikan',
            'solution': 'Adjust weight: L1=100, Adv=2, CTC=1'
        },
        {
            'name': 'Discriminator Training',
            'issue': 'Discriminator bisa terlalu kuat atau terlalu lemah',
            'impact': 'Generator tidak dapat learning signal yang baik',
            'solution': 'Balance D_lr vs G_lr, label smoothing'
        },
        {
            'name': 'Mode Collapse',
            'issue': 'Generator stuck pada output seragam',
            'impact': 'Tidak ada diversity, training berhenti',
            'solution': 'Minibatch discrimination, feature matching'
        }
    ]
    
    for i, bottleneck in enumerate(bottlenecks, 1):
        print(f"\n{i}. {bottleneck['name']}")
        print(f"   Issue: {bottleneck['issue']}")
        print(f"   Impact: {bottleneck['impact']}")
        print(f"   💡 Solution: {bottleneck['solution']}")

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================
def generate_recommendations(gen_issues, disc_issues, rec_issues):
    """Generate actionable recommendations"""
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    all_issues = gen_issues + disc_issues + rec_issues
    
    print(f"\n📊 TOTAL ISSUES FOUND: {len(all_issues)}")
    
    if len(all_issues) == 0:
        print("\n✅ No architectural issues found!")
        print("\nProblema bukan di arsitektur, kemungkinan di:")
    else:
        print("\n⚠️ Issues detected:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        print("\n💡 Fix these issues first, then check:")
    
    print("\n🎯 DIAGNOSIS CHECKLIST:")
    print("-" * 80)
    
    checks = [
        "✓ Data pipeline: Gambar terbaca dengan benar dari TFRecord",
        "✓ Generator architecture: Output activation = sigmoid",
        "✓ Discriminator architecture: Output activation = sigmoid",
        "✓ Recognizer: Properly frozen, tidak trainable",
        "? Loss weight balance: Perlu ditest dengan berbagai kombinasi",
        "? Gradient flow: Perlu diverifikasi tidak vanishing/exploding",
        "? Training dynamics: Perlu monitoring loss curves",
        "? Output diversity: Perlu check apakah generator collapse"
    ]
    
    for check in checks:
        print(f"  {check}")
    
    print("\n💡 RECOMMENDED NEXT STEPS:")
    print("-" * 80)
    
    steps = [
        "1. Run diagnostic_gan_debug_temp.py untuk test dasar",
        "2. Run test_loss_functions_temp.py untuk test loss components",
        "3. Run test_recognizer_frozen_temp.py untuk verify recognizer",
        "4. Jika semua test pass, masalah ada di training dynamics:",
        "   a. Start dengan pre-training: pixel_loss=100, adv=0, ctc=0",
        "   b. Monitor loss curve, pastikan turun stabil",
        "   c. Check output samples setiap epoch",
        "   d. Jika pre-training berhasil, tambahkan adversarial loss",
        "   e. Terakhir tambahkan CTC loss dengan weight kecil",
        "5. Document semua temuan di logbook/"
    ]
    
    for step in steps:
        print(f"  {step}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*80)
    print("DEEP ARCHITECTURE INSPECTION & ANALYSIS")
    print("="*80)
    print("\nTujuan: Menemukan root cause dari Generator collapse issue")
    print("Pendekatan: Critical thinking, verify setiap asumsi")
    print("="*80)
    
    try:
        generator, gen_issues = analyze_generator()
        discriminator, disc_issues = analyze_discriminator()
        recognizer, rec_issues = analyze_recognizer()
        analyze_data_flow()
        generate_recommendations(gen_issues, disc_issues, rec_issues)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print("\n📝 Check dual_modal_gan/outputs/debug_architecture/ for details")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
