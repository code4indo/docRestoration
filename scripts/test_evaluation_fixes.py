#!/usr/bin/env python3
"""
Quick Test Script untuk Validasi Evaluation Fixes
Test dengan 1 DIBCO image untuk confirm PSNR improvement
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf

# ✅ Use GPU 1 (GPU 0 is being used for training)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_evaluation_fixes():
    print("🧪 Testing Evaluation Fixes")
    print("=" * 60)
    
    # 1. Load model (best model dari full_training_production_v1)
    print("\n📂 Step 1: Loading BEST MODEL...")
    model_path = "dual_modal_gan/checkpoints/full_training_production_v1/best_model"
    
    try:
        from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
        generator = unet_enhanced(input_size=(1024, 128, 1))
        print("   ✅ Using enhanced generator architecture")
    except Exception as e:
        print(f"   ⚠️ Fallback to base architecture: {e}")
        from dual_modal_gan.src.models.generator import unet
        generator = unet(input_size=(1024, 128, 1))
    
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    
    if checkpoint_path:
        checkpoint.restore(checkpoint_path).expect_partial()
        print(f"   ✅ Checkpoint loaded: {checkpoint_path}")
    else:
        print(f"   ❌ No checkpoint found!")
        return
    
    # 2. Load test image
    print("\n📄 Step 2: Loading test image...")
    test_image_path = "dibco_datasets/DIPCO2016_dataset/1.bmp"
    gt_image_path = "dibco_datasets/DIPCO2016_Dataset_GT/1_gt.bmp"
    
    if not os.path.exists(test_image_path):
        print(f"   ❌ Test image not found: {test_image_path}")
        return
    
    original = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_image_path) else None
    
    print(f"   ✅ Original shape: {original.shape}")
    if ground_truth is not None:
        print(f"   ✅ Ground truth shape: {ground_truth.shape}")
    
    # Normalize to [0,1]
    original = original.astype(np.float32) / 255.0
    if ground_truth is not None:
        ground_truth = ground_truth.astype(np.float32) / 255.0
        ground_truth = (ground_truth > 0.5).astype(np.float32)
    
    # 3. Test OLD method (WITHOUT fixes)
    print("\n🔴 Step 3: Testing OLD method (WITHOUT normalization fix)...")
    
    # Resize - CRITICAL: Model expects (1024, 128) which is (W, H)
    # cv2.resize uses (width, height) format
    resized_old = cv2.resize(original, (128, 1024), interpolation=cv2.INTER_AREA)  # Result: (1024, 128) in (H, W)
    # Transpose to get (W, H) = (1024, 128) WAIT NO! cv2.resize returns (H, W) format
    # So resized_old is already (1024, 128) in numpy (H, W) format
    # Model needs (1024, 128, 1) where first dim is W... let me check training
    # From train_enhanced.py: degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])  # (128, 1024, 1) → (1024, 128, 1)
    # So training TRANSPOSES from (H, W, C) to (W, H, C)!
    
    # Step 1: Resize to (W=128, H=1024) -> gives us (H, W) = (1024, 128) numpy array
    resized_old = cv2.resize(original, (128, 1024), interpolation=cv2.INTER_AREA)
    # Step 2: Add channel dim -> (1024, 128, 1)
    resized_old = resized_old[..., np.newaxis]
    # Step 3: Transpose to (W, H, C) = (128, 1024, 1) NO WAIT!
    # Training transpose is [1, 0, 2] which means (H, W, C) -> (W, H, C)
    # So (1024, 128, 1) -> (128, 1024, 1) NO!
    # Let me reread: original shape in TFRecord is (128, 1024, 1) in (H, W, C)
    # After transpose with [1, 0, 2]: (1024, 128, 1) in (W, H, C)
    # So final model input should be (1, 1024, 128, 1)
    
    resized_old = np.transpose(resized_old, (1, 0, 2))  # (1024, 128, 1) -> (128, 1024, 1) NO!
    # I'm confused. Let me just match exactly what training does:
    # Training: (128, 1024, 1) -> transpose[1,0,2] -> (1024, 128, 1)
    # So if we have (1024, 128, 1) we need to transpose it to (128, 1024, 1) first? NO!
    
    # CORRECT UNDERSTANDING:
    # Training TFRecord shape: (H=128, W=1024, C=1)
    # Training transpose [1,0,2]: swap H and W -> (W=1024, H=128, C=1)
    # So model expects input shape: (W=1024, H=128, C=1)
    
    # Our resize: (128, 1024) means we want WIDTH=128, HEIGHT=1024
    # cv2.resize returns numpy array in (H, W) format
    # So result is (H=1024, W=128)
    # Add channel: (1024, 128, 1)
    # Transpose [1, 0, 2]: (128, 1024, 1) CORRECT!
    
    resized_old = cv2.resize(original, (128, 1024), interpolation=cv2.INTER_AREA)  # (H=1024, W=128)
    resized_old = resized_old[..., np.newaxis]  # (1024, 128, 1)
    resized_old = np.transpose(resized_old, (1, 0, 2))  # (128, 1024, 1) WRONG!
    
    # Wait, let me trace this more carefully:
    # We want final shape (1024, 128, 1) for model
    # cv2.resize((width, height)) = cv2.resize((1024, 128))
    # Returns numpy array in (height, width) = (128, 1024)
    
    resized_old = cv2.resize(original, (1024, 128), interpolation=cv2.INTER_AREA)  # Returns (H=128, W=1024)
    resized_old = resized_old[..., np.newaxis]  # (128, 1024, 1)
    resized_old = np.transpose(resized_old, (1, 0, 2))  # Swap H and W: (1024, 128, 1) CORRECT!
    
    model_input_old = resized_old[np.newaxis, ...]  # (1, 1024, 128, 1)
    
    print(f"   Model input shape: {model_input_old.shape}")
    
    # Generate
    enhanced_old = generator(model_input_old, training=False)
    enhanced_old = enhanced_old.numpy()[0, ..., 0]
    enhanced_old = (enhanced_old + 1.0) / 2.0
    enhanced_old = np.clip(enhanced_old, 0, 1)
    
    print(f"   Enhanced output range OLD: [{enhanced_old.min():.3f}, {enhanced_old.max():.3f}]")
    
    # Calculate PSNR (resize GT to match)
    if ground_truth is not None:
        gt_resized = cv2.resize(ground_truth, (enhanced_old.shape[1], enhanced_old.shape[0]))
        mse_old = np.mean((enhanced_old - gt_resized) ** 2)
        psnr_old = 20 * np.log10(1.0 / np.sqrt(mse_old)) if mse_old > 0 else float('inf')
        print(f"   ❌ PSNR OLD (without fix): {psnr_old:.2f} dB")
    
    # 4. Test NEW method (WITH fixes)
    print("\n✅ Step 4: Testing NEW method (WITH normalization fix)...")
    
    # Resize to (W=1024, H=128) -> returns (H=128, W=1024) in numpy
    resized_new = cv2.resize(original, (1024, 128), interpolation=cv2.INTER_AREA)  # (128, 1024)
    resized_new = resized_new[..., np.newaxis]  # (128, 1024, 1)
    resized_new = np.transpose(resized_new, (1, 0, 2))  # (1024, 128, 1)
    model_input_new = resized_new[np.newaxis, ...]  # (1, 1024, 128, 1)
    
    # 🔥 FIX: Normalize to [-1,1]
    model_input_new = (model_input_new * 2.0) - 1.0
    
    print(f"   Model input shape: {model_input_new.shape}")
    print(f"   Model input range: [{model_input_new.min():.3f}, {model_input_new.max():.3f}]")
    
    # Generate
    enhanced_new = generator(model_input_new, training=False)
    enhanced_new = enhanced_new.numpy()[0, ..., 0]
    enhanced_new = (enhanced_new + 1.0) / 2.0
    enhanced_new = np.clip(enhanced_new, 0, 1)
    
    print(f"   Enhanced output range NEW: [{enhanced_new.min():.3f}, {enhanced_new.max():.3f}]")
    
    # Calculate PSNR (resize GT to match)
    if ground_truth is not None:
        gt_resized = cv2.resize(ground_truth, (enhanced_new.shape[1], enhanced_new.shape[0]))
        mse_new = np.mean((enhanced_new - gt_resized) ** 2)
        psnr_new = 20 * np.log10(1.0 / np.sqrt(mse_new)) if mse_new > 0 else float('inf')
        print(f"   ✅ PSNR NEW (with fix): {psnr_new:.2f} dB")
        
        # 5. Compare results
        print("\n📊 Step 5: Comparison Results")
        print("=" * 60)
        print(f"   PSNR OLD (no normalization): {psnr_old:.2f} dB")
        print(f"   PSNR NEW (with [-1,1] norm): {psnr_new:.2f} dB")
        print(f"   🎯 IMPROVEMENT: {psnr_new - psnr_old:+.2f} dB")
        
        if psnr_new > psnr_old + 5:
            print("\n   ✅ SUCCESS! Fix menghasilkan improvement signifikan!")
        elif psnr_new > psnr_old:
            print("\n   ⚠️ Fix menghasilkan improvement, tapi masih perlu optimasi")
        else:
            print("\n   ❌ WARNING! Fix tidak menghasilkan improvement")
    
    # 6. Save comparison images
    print("\n💾 Step 6: Saving comparison images...")
    
    # Squeeze channel dimension for saving
    enhanced_old_uint8 = (enhanced_old * 255).astype(np.uint8)
    enhanced_new_uint8 = (enhanced_new * 255).astype(np.uint8)
    
    # Get one of the resized images for comparison (they should be same size)
    resized_for_display = cv2.resize(original, (1024, 128), interpolation=cv2.INTER_AREA)
    resized_for_display = np.transpose(resized_for_display)  # Match model output orientation
    original_uint8 = (resized_for_display * 255).astype(np.uint8)
    
    cv2.imwrite("test_evaluation_original.png", original_uint8)
    cv2.imwrite("test_evaluation_enhanced_OLD.png", enhanced_old_uint8)
    cv2.imwrite("test_evaluation_enhanced_NEW.png", enhanced_new_uint8)
    
    # Create comparison
    h, w = enhanced_old.shape
    comparison = np.zeros((h * 3, w), dtype=np.uint8)
    comparison[:h, :] = original_uint8
    comparison[h:2*h, :] = enhanced_old_uint8
    comparison[2*h:, :] = enhanced_new_uint8
    
    cv2.imwrite("test_evaluation_comparison.png", comparison)
    
    print("   ✅ Images saved:")
    print("      - test_evaluation_original.png")
    print("      - test_evaluation_enhanced_OLD.png (no fix)")
    print("      - test_evaluation_enhanced_NEW.png (with fix)")
    print("      - test_evaluation_comparison.png")
    
    print("\n🎉 Test completed!")

if __name__ == '__main__':
    test_evaluation_fixes()
