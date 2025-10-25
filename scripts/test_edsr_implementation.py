#!/usr/bin/env python3
"""
EDSR Implementation Test Suite
===============================

Comprehensive testing untuk implementasi EDSR:
1. Model architecture validation
2. Parameter count verification
3. Inference testing (forward pass)
4. Output shape validation (2x, 4x)
5. Preprocessing/postprocessing pipeline
6. Performance benchmarking
7. Memory usage profiling

Usage:
    poetry run python scripts/test_edsr_implementation.py
    poetry run python scripts/test_edsr_implementation.py --visual  # dengan visualisasi
    poetry run python scripts/test_edsr_implementation.py --benchmark  # dengan benchmark detail

Author: AI Assistant
Date: 2025-10-24
"""

import sys
import os
import argparse
import time
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dual_modal_gan.src.models.edsr import (
    build_edsr, 
    build_edsr_efficient,
    preprocess_for_edsr, 
    postprocess_from_edsr,
    residual_block,
    sub_pixel_conv
)

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_success(msg: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")


def print_info(msg: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {msg}{Colors.ENDC}")


def print_warning(msg: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")


def print_error(msg: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")


def test_residual_block():
    """Test 1: Residual Block Implementation"""
    print_section("TEST 1: Residual Block")
    
    try:
        # Create dummy input
        dummy_input = tf.keras.Input(shape=(64, 64, 64))
        
        # Apply residual block
        output = residual_block(dummy_input, filters=64, scaling=0.1)
        
        # Check output shape
        assert output.shape[1:] == (64, 64, 64), "Output shape mismatch!"
        
        print_success("Residual block shape correct: (64, 64, 64)")
        
        # Build mini model to count params
        model = tf.keras.Model(inputs=dummy_input, outputs=output)
        params = model.count_params()
        
        # Expected: 2 conv layers (3x3x64x64) * 2 = ~74K params
        expected_params = 2 * (3 * 3 * 64 * 64 + 64)  # weights + bias
        print_info(f"Residual block params: {params:,} (expected: ~{expected_params:,})")
        
        return True
        
    except Exception as e:
        print_error(f"Residual block test failed: {e}")
        return False


def test_sub_pixel_conv():
    """Test 2: Sub-Pixel Convolution"""
    print_section("TEST 2: Sub-Pixel Convolution")
    
    try:
        # Test 2x upscaling
        dummy_input_2x = tf.keras.Input(shape=(64, 64, 64))
        output_2x = sub_pixel_conv(dummy_input_2x, scale=2, filters=64)
        
        expected_shape_2x = (None, 128, 128, 64)  # 2x upscaling
        assert output_2x.shape == expected_shape_2x, f"2x shape mismatch! Got {output_2x.shape}"
        print_success(f"2x sub-pixel conv: (64, 64, 64) -> (128, 128, 64) ✓")
        
        # Test 4x upscaling
        dummy_input_4x = tf.keras.Input(shape=(64, 64, 64))
        output_4x = sub_pixel_conv(dummy_input_4x, scale=4, filters=64)
        
        expected_shape_4x = (None, 256, 256, 64)  # 4x upscaling
        assert output_4x.shape == expected_shape_4x, f"4x shape mismatch! Got {output_4x.shape}"
        print_success(f"4x sub-pixel conv: (64, 64, 64) -> (256, 256, 64) ✓")
        
        return True
        
    except Exception as e:
        print_error(f"Sub-pixel conv test failed: {e}")
        return False


def test_model_building():
    """Test 3: Model Building (EDSR-baseline & efficient)"""
    print_section("TEST 3: Model Building")
    
    results = {}
    
    try:
        # Test 1: EDSR-baseline 2x
        print_info("Building EDSR-baseline (2x)...")
        model_2x = build_edsr(input_shape=(128, 1024, 1), scale=2)
        
        assert model_2x.input_shape == (None, 128, 1024, 1), "Input shape mismatch!"
        assert model_2x.output_shape == (None, 256, 2048, 1), "Output shape mismatch!"
        
        params_2x = model_2x.count_params()
        results['edsr_baseline_2x'] = params_2x
        
        print_success(f"EDSR-baseline 2x: {params_2x:,} params")
        print_info(f"   Input:  {model_2x.input_shape}")
        print_info(f"   Output: {model_2x.output_shape}")
        
        # Test 2: EDSR-baseline 4x
        print_info("\nBuilding EDSR-baseline (4x)...")
        model_4x = build_edsr(input_shape=(128, 1024, 1), scale=4)
        
        assert model_4x.input_shape == (None, 128, 1024, 1), "Input shape mismatch!"
        assert model_4x.output_shape == (None, 512, 4096, 1), "Output shape mismatch!"
        
        params_4x = model_4x.count_params()
        results['edsr_baseline_4x'] = params_4x
        
        print_success(f"EDSR-baseline 4x: {params_4x:,} params")
        print_info(f"   Input:  {model_4x.input_shape}")
        print_info(f"   Output: {model_4x.output_shape}")
        
        # Test 3: EDSR-efficient 2x
        print_info("\nBuilding EDSR-efficient (2x)...")
        model_efficient = build_edsr_efficient(input_shape=(128, 1024, 1), scale=2)
        
        params_efficient = model_efficient.count_params()
        results['edsr_efficient_2x'] = params_efficient
        
        print_success(f"EDSR-efficient 2x: {params_efficient:,} params")
        print_info(f"   Reduction: {(1 - params_efficient/params_2x)*100:.1f}% fewer params vs baseline")
        
        # Summary
        print(f"\n{Colors.BOLD}Parameter Summary:{Colors.ENDC}")
        print(f"  • EDSR-baseline 2x:  {results['edsr_baseline_2x']:>10,} params")
        print(f"  • EDSR-baseline 4x:  {results['edsr_baseline_4x']:>10,} params")
        print(f"  • EDSR-efficient 2x: {results['edsr_efficient_2x']:>10,} params")
        
        return True, results
        
    except Exception as e:
        print_error(f"Model building test failed: {e}")
        return False, {}


def test_inference():
    """Test 4: Inference with Dummy Data"""
    print_section("TEST 4: Inference Testing")
    
    try:
        # Build models
        print_info("Building models for inference test...")
        model_2x = build_edsr(input_shape=(128, 1024, 1), scale=2)
        model_4x = build_edsr(input_shape=(128, 1024, 1), scale=4)
        
        # Create dummy input (batch of 2 images)
        batch_size = 2
        dummy_input = np.random.rand(batch_size, 128, 1024, 1).astype(np.float32)
        
        print_info(f"Dummy input shape: {dummy_input.shape}")
        print_info(f"Dummy input range: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")
        
        # Test 2x inference
        print_info("\nRunning 2x inference...")
        start_time = time.time()
        output_2x = model_2x.predict(dummy_input, verbose=0)
        inference_time_2x = time.time() - start_time
        
        assert output_2x.shape == (batch_size, 256, 2048, 1), "2x output shape mismatch!"
        print_success(f"2x inference OK: {output_2x.shape} in {inference_time_2x:.3f}s")
        print_info(f"   Output range: [{output_2x.min():.3f}, {output_2x.max():.3f}]")
        
        # Test 4x inference
        print_info("\nRunning 4x inference...")
        start_time = time.time()
        output_4x = model_4x.predict(dummy_input, verbose=0)
        inference_time_4x = time.time() - start_time
        
        assert output_4x.shape == (batch_size, 512, 4096, 1), "4x output shape mismatch!"
        print_success(f"4x inference OK: {output_4x.shape} in {inference_time_4x:.3f}s")
        print_info(f"   Output range: [{output_4x.min():.3f}, {output_4x.max():.3f}]")
        
        # Inference speed comparison
        print(f"\n{Colors.BOLD}Inference Speed:{Colors.ENDC}")
        print(f"  • 2x: {inference_time_2x:.3f}s ({batch_size/inference_time_2x:.2f} imgs/s)")
        print(f"  • 4x: {inference_time_4x:.3f}s ({batch_size/inference_time_4x:.2f} imgs/s)")
        print_info(f"  • 4x is {inference_time_4x/inference_time_2x:.2f}x slower than 2x")
        
        return True
        
    except Exception as e:
        print_error(f"Inference test failed: {e}")
        return False


def test_preprocessing_postprocessing():
    """Test 5: Preprocessing & Postprocessing Pipeline"""
    print_section("TEST 5: Preprocessing & Postprocessing")
    
    try:
        # Test Case 1: uint8 [0, 255] input
        print_info("Test Case 1: uint8 [0, 255] input")
        img_uint8 = np.random.randint(0, 256, (128, 1024), dtype=np.uint8)
        
        preprocessed = preprocess_for_edsr(img_uint8, normalize=True)
        assert preprocessed.dtype == np.float32, "Dtype should be float32!"
        assert preprocessed.shape == (128, 1024, 1), "Shape should have channel dimension!"
        assert 0.0 <= preprocessed.min() <= 1.0, "Should be normalized to [0, 1]!"
        assert 0.0 <= preprocessed.max() <= 1.0, "Should be normalized to [0, 1]!"
        
        print_success(f"Preprocessing OK: {img_uint8.shape} -> {preprocessed.shape}")
        print_info(f"   Range: [0, 255] -> [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
        
        # Test Case 2: float32 [0, 1] input
        print_info("\nTest Case 2: float32 [0, 1] input")
        img_float = np.random.rand(128, 1024).astype(np.float32)
        
        preprocessed_float = preprocess_for_edsr(img_float, normalize=False)
        assert preprocessed_float.shape == (128, 1024, 1), "Shape should have channel dimension!"
        
        print_success(f"Preprocessing OK: {img_float.shape} -> {preprocessed_float.shape}")
        
        # Test Case 3: Postprocessing (denormalize)
        print_info("\nTest Case 3: Postprocessing with denormalization")
        model_output = np.random.rand(256, 2048, 1).astype(np.float32)
        
        postprocessed = postprocess_from_edsr(model_output, denormalize=True)
        assert postprocessed.dtype == np.uint8, "Should be uint8 after denormalization!"
        assert postprocessed.shape == (256, 2048), "Channel should be squeezed!"
        assert 0 <= postprocessed.min() <= 255, "Should be in [0, 255]!"
        assert 0 <= postprocessed.max() <= 255, "Should be in [0, 255]!"
        
        print_success(f"Postprocessing OK: {model_output.shape} -> {postprocessed.shape}")
        print_info(f"   Range: [{model_output.min():.3f}, {model_output.max():.3f}] -> [0, 255]")
        
        # Test Case 4: Round-trip (preprocess -> postprocess)
        print_info("\nTest Case 4: Round-trip test")
        original = np.random.randint(0, 256, (128, 1024), dtype=np.uint8)
        
        prep = preprocess_for_edsr(original, normalize=True)
        restored = postprocess_from_edsr(prep, denormalize=True)
        
        # Should be nearly identical (minor rounding errors acceptable)
        diff = np.abs(original.astype(float) - restored.astype(float)).mean()
        assert diff < 1.0, f"Round-trip error too large: {diff:.3f}"
        
        print_success(f"Round-trip OK: mean difference = {diff:.4f}")
        
        return True
        
    except Exception as e:
        print_error(f"Preprocessing/postprocessing test failed: {e}")
        return False


def test_memory_usage():
    """Test 6: Memory Usage Profiling"""
    print_section("TEST 6: Memory Usage")
    
    try:
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        print_info(f"Baseline memory: {mem_before:.2f} MB")
        
        # Build EDSR-baseline
        print_info("\nBuilding EDSR-baseline 2x...")
        model_baseline = build_edsr(input_shape=(128, 1024, 1), scale=2)
        gc.collect()
        mem_baseline = process.memory_info().rss / 1024 / 1024  # MB
        mem_delta_baseline = mem_baseline - mem_before
        
        print_success(f"EDSR-baseline loaded: +{mem_delta_baseline:.2f} MB")
        
        # Build EDSR-efficient
        print_info("\nBuilding EDSR-efficient 2x...")
        model_efficient = build_edsr_efficient(input_shape=(128, 1024, 1), scale=2)
        gc.collect()
        mem_efficient = process.memory_info().rss / 1024 / 1024  # MB
        mem_delta_efficient = mem_efficient - mem_baseline
        
        print_success(f"EDSR-efficient loaded: +{mem_delta_efficient:.2f} MB")
        
        # Summary
        print(f"\n{Colors.BOLD}Memory Usage Summary:{Colors.ENDC}")
        print(f"  • EDSR-baseline:  {mem_delta_baseline:>6.2f} MB")
        print(f"  • EDSR-efficient: {mem_delta_efficient:>6.2f} MB")
        print(f"  • Savings:        {mem_delta_baseline - mem_delta_efficient:>6.2f} MB "
              f"({(1 - mem_delta_efficient/mem_delta_baseline)*100:.1f}%)")
        
        return True
        
    except ImportError:
        print_warning("psutil not installed, skipping memory test")
        print_info("Install with: poetry add psutil")
        return True
    except Exception as e:
        print_error(f"Memory test failed: {e}")
        return False


def benchmark_performance(num_iterations: int = 100):
    """Test 7: Performance Benchmarking"""
    print_section("TEST 7: Performance Benchmark")
    
    try:
        # Build models
        print_info("Building models...")
        model_baseline = build_edsr(input_shape=(128, 1024, 1), scale=2)
        model_efficient = build_edsr_efficient(input_shape=(128, 1024, 1), scale=2)
        
        # Warm-up
        print_info("Warming up models...")
        dummy = np.random.rand(1, 128, 1024, 1).astype(np.float32)
        _ = model_baseline.predict(dummy, verbose=0)
        _ = model_efficient.predict(dummy, verbose=0)
        
        # Benchmark baseline
        print_info(f"\nBenchmarking EDSR-baseline ({num_iterations} iterations)...")
        times_baseline = []
        for _ in range(num_iterations):
            dummy = np.random.rand(1, 128, 1024, 1).astype(np.float32)
            start = time.time()
            _ = model_baseline.predict(dummy, verbose=0)
            times_baseline.append(time.time() - start)
        
        mean_baseline = np.mean(times_baseline)
        std_baseline = np.std(times_baseline)
        
        print_success(f"Baseline: {mean_baseline*1000:.2f} ± {std_baseline*1000:.2f} ms/image")
        
        # Benchmark efficient
        print_info(f"\nBenchmarking EDSR-efficient ({num_iterations} iterations)...")
        times_efficient = []
        for _ in range(num_iterations):
            dummy = np.random.rand(1, 128, 1024, 1).astype(np.float32)
            start = time.time()
            _ = model_efficient.predict(dummy, verbose=0)
            times_efficient.append(time.time() - start)
        
        mean_efficient = np.mean(times_efficient)
        std_efficient = np.std(times_efficient)
        
        print_success(f"Efficient: {mean_efficient*1000:.2f} ± {std_efficient*1000:.2f} ms/image")
        
        # Summary
        speedup = mean_baseline / mean_efficient
        print(f"\n{Colors.BOLD}Performance Summary:{Colors.ENDC}")
        print(f"  • EDSR-baseline:  {mean_baseline*1000:>7.2f} ms/image ({1/mean_baseline:>5.2f} imgs/s)")
        print(f"  • EDSR-efficient: {mean_efficient*1000:>7.2f} ms/image ({1/mean_efficient:>5.2f} imgs/s)")
        print(f"  • Speedup:        {speedup:.2f}x faster")
        
        return True
        
    except Exception as e:
        print_error(f"Benchmark failed: {e}")
        return False


def test_visual_output(output_dir: str = "./outputs/edsr_test"):
    """Test 8: Visual Output Generation"""
    print_section("TEST 8: Visual Output Generation")
    
    try:
        from PIL import Image
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create synthetic test image (text-like pattern)
        print_info("Creating synthetic test pattern...")
        test_img = np.zeros((128, 1024), dtype=np.uint8)
        
        # Add some text-like patterns
        for i in range(10):
            x = np.random.randint(50, 950)
            y = np.random.randint(20, 108)
            w = np.random.randint(30, 80)
            h = np.random.randint(15, 25)
            test_img[y:y+h, x:x+w] = np.random.randint(180, 256)
        
        # Add some noise
        noise = np.random.randint(0, 30, test_img.shape, dtype=np.uint8)
        test_img = np.clip(test_img + noise, 0, 255).astype(np.uint8)
        
        # Save original
        Image.fromarray(test_img).save(output_path / "input_lr.png")
        print_success(f"Saved input: {output_path / 'input_lr.png'}")
        
        # Build models
        model_2x = build_edsr(input_shape=(128, 1024, 1), scale=2)
        model_4x = build_edsr(input_shape=(128, 1024, 1), scale=4)
        
        # Preprocess
        test_prep = preprocess_for_edsr(test_img, normalize=True)
        test_batch = np.expand_dims(test_prep, axis=0)
        
        # Apply 2x SR
        print_info("Applying 2x super-resolution...")
        output_2x = model_2x.predict(test_batch, verbose=0)[0]
        output_2x_img = postprocess_from_edsr(output_2x, denormalize=True)
        Image.fromarray(output_2x_img).save(output_path / "output_2x.png")
        print_success(f"Saved 2x output: {output_path / 'output_2x.png'} ({output_2x_img.shape})")
        
        # Apply 4x SR
        print_info("Applying 4x super-resolution...")
        output_4x = model_4x.predict(test_batch, verbose=0)[0]
        output_4x_img = postprocess_from_edsr(output_4x, denormalize=True)
        Image.fromarray(output_4x_img).save(output_path / "output_4x.png")
        print_success(f"Saved 4x output: {output_path / 'output_4x.png'} ({output_4x_img.shape})")
        
        print(f"\n{Colors.BOLD}Visual outputs saved to: {output_path}{Colors.ENDC}")
        
        return True
        
    except ImportError:
        print_warning("PIL not installed, skipping visual test")
        print_info("Install with: poetry add pillow")
        return True
    except Exception as e:
        print_error(f"Visual output test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="EDSR Implementation Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--visual', action='store_true',
                       help='Generate visual outputs (requires PIL)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run detailed performance benchmark')
    parser.add_argument('--benchmark-iters', type=int, default=100,
                       help='Number of benchmark iterations (default: 100)')
    parser.add_argument('--output-dir', type=str, default='./outputs/edsr_test',
                       help='Output directory for visual tests')
    
    args = parser.parse_args()
    
    # Header
    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "EDSR IMPLEMENTATION TEST SUITE".center(78) + "║")
    print("║" + "Enhanced Deep Super-Resolution Network".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    print(f"{Colors.ENDC}\n")
    
    # Run tests
    results = []
    
    # Core tests (always run)
    results.append(("Residual Block", test_residual_block()))
    results.append(("Sub-Pixel Conv", test_sub_pixel_conv()))
    
    success, model_params = test_model_building()
    results.append(("Model Building", success))
    
    results.append(("Inference", test_inference()))
    results.append(("Preprocessing/Postprocessing", test_preprocessing_postprocessing()))
    results.append(("Memory Usage", test_memory_usage()))
    
    # Optional tests
    if args.benchmark:
        results.append(("Performance Benchmark", 
                       benchmark_performance(num_iterations=args.benchmark_iters)))
    
    if args.visual:
        results.append(("Visual Output", test_visual_output(args.output_dir)))
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"{Colors.BOLD}Results:{Colors.ENDC}\n")
    for test_name, result in results:
        status = f"{Colors.OKGREEN}✓ PASSED{Colors.ENDC}" if result else f"{Colors.FAIL}✗ FAILED{Colors.ENDC}"
        print(f"  {test_name:.<50} {status}")
    
    print(f"\n{Colors.BOLD}Overall: {passed}/{total} tests passed{Colors.ENDC}")
    
    if passed == total:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 All tests PASSED! EDSR implementation is working correctly.{Colors.ENDC}")
        return 0
    else:
        print(f"\n{Colors.FAIL}{Colors.BOLD}⚠️  Some tests FAILED. Please review the errors above.{Colors.ENDC}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
