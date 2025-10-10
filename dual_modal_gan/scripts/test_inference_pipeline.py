#!/usr/bin/env python3
"""
Test Script untuk GAN-HTR Inference Pipeline

Script ini untuk quick test pipeline dengan synthetic degraded document.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def create_test_document(output_path, num_lines=5):
    """
    Buat dokumen test dengan teks sederhana.
    Simulate degraded document dengan noise dan blur.
    """
    # Create blank document (A5 size at 150 DPI)
    width, height = 1240, 1754  # Half of A4 300 DPI
    document = np.ones((height, width), dtype=np.uint8) * 255
    
    # Add text lines
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    line_spacing = 150
    start_y = 200
    
    test_texts = [
        "This is the first line of text.",
        "Second line with different content here.",
        "Line number three contains more words.",
        "Fourth line for testing the pipeline.",
        "Final line to complete the test document."
    ]
    
    for i, text in enumerate(test_texts[:num_lines]):
        y_position = start_y + (i * line_spacing)
        cv2.putText(document, text, (100, y_position), font, 
                   font_scale, 0, thickness, cv2.LINE_AA)
    
    # Add degradation effects
    # 1. Gaussian noise
    noise = np.random.normal(0, 15, document.shape).astype(np.uint8)
    degraded = cv2.add(document, noise)
    
    # 2. Blur
    degraded = cv2.GaussianBlur(degraded, (3, 3), 0)
    
    # 3. Random spots (simulate paper aging)
    for _ in range(50):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        radius = np.random.randint(5, 20)
        color = np.random.randint(180, 240)
        cv2.circle(degraded, (x, y), radius, int(color), -1)
    
    # Save
    cv2.imwrite(str(output_path), degraded)
    print(f"✅ Test document created: {output_path}")
    print(f"   Size: {width}x{height}, Lines: {num_lines}")
    
    return output_path


def verify_output_structure(output_dir, doc_name):
    """Verify struktur output sesuai expected."""
    output_path = Path(output_dir) / doc_name
    
    checks = {
        'restored_image': output_path / f"{doc_name}_restored.png",
        'text_file': output_path / f"{doc_name}_text.txt",
        'metadata': output_path / f"{doc_name}_metadata.json",
        'lines_dir': output_path / "lines"
    }
    
    print("\n🔍 Verifying output structure...")
    all_ok = True
    
    for name, path in checks.items():
        if path.exists():
            if path.is_file():
                size = path.stat().st_size
                print(f"  ✅ {name}: {path.name} ({size} bytes)")
            else:
                num_files = len(list(path.glob('*')))
                print(f"  ✅ {name}: {path.name}/ ({num_files} files)")
        else:
            print(f"  ❌ {name}: NOT FOUND")
            all_ok = False
    
    return all_ok


def main():
    print("=" * 70)
    print("GAN-HTR Inference Pipeline - Test Script")
    print("=" * 70)
    
    # Setup paths
    test_dir = Path("/tmp/gan_htr_test")
    input_dir = test_dir / "input"
    output_dir = test_dir / "output"
    
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Test directories:")
    print(f"   Input : {input_dir}")
    print(f"   Output: {output_dir}")
    
    # Create test document
    print(f"\n[1/3] Creating synthetic test document...")
    test_doc_path = input_dir / "test_document.png"
    create_test_document(test_doc_path, num_lines=5)
    
    # Run inference pipeline
    print(f"\n[2/3] Running inference pipeline...")
    print(f"   (This will take ~10-30 seconds depending on GPU)")
    
    cmd = f"""
    poetry run python dual_modal_gan/scripts/inference_pipeline.py \
        --input_dir {input_dir} \
        --output_dir {output_dir} \
        --gan_checkpoint dual_modal_gan/outputs/checkpoints_fp32_smoke_test \
        --htr_weights models/best_htr_recognizer/best_model.weights.h5 \
        --charset_path real_data_preparation/real_data_charlist.txt \
        --gpu_id 1 \
        --batch_size 4 \
        --save_intermediates
    """
    
    print(f"\n💻 Command:")
    print(f"   {' '.join(cmd.split())}")
    print(f"\nExecuting...")
    
    ret = os.system(cmd)
    
    if ret != 0:
        print(f"\n❌ Pipeline failed with exit code: {ret}")
        return False
    
    # Verify output
    print(f"\n[3/3] Verifying output...")
    success = verify_output_structure(output_dir, "test_document")
    
    if success:
        print(f"\n✅ All checks passed!")
        
        # Display extracted text
        text_file = output_dir / "test_document" / "test_document_text.txt"
        if text_file.exists():
            print(f"\n📄 Extracted text:")
            print("-" * 70)
            with open(text_file, 'r') as f:
                print(f.read())
            print("-" * 70)
        
        print(f"\n📁 Full output available at: {output_dir}")
    else:
        print(f"\n⚠️  Some checks failed. See above for details.")
    
    print(f"\n🧹 Cleanup test directory:")
    print(f"   rm -rf {test_dir}")
    
    return success


if __name__ == '__main__':
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with exception:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
