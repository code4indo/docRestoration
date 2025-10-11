#!/bin/bash
# GPU Verification Script untuk RunPod
# Usage: docker exec gan-htr-training bash /workspace/docRestoration/scripts/test_gpu.sh

echo "================================================"
echo "🔍 GPU Detection Test - RunPod"
echo "================================================"
echo ""

echo "1️⃣ Check NVIDIA Driver (Host):"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "   ⚠️ nvidia-smi not found (run from host, not container)"
fi
echo ""

echo "2️⃣ Check TensorFlow GPU Support:"
python3 << 'EOF'
import tensorflow as tf
print(f"   TensorFlow Version: {tf.__version__}")
print(f"   Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"   GPU Available: {tf.test.is_gpu_available()}")
gpus = tf.config.list_physical_devices('GPU')
print(f"   GPU Count: {len(gpus)}")
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu.name}")
        details = tf.config.experimental.get_device_details(gpu)
        if 'device_name' in details:
            print(f"      Device: {details['device_name']}")
        if 'compute_capability' in details:
            print(f"      Compute Capability: {details['compute_capability']}")
else:
    print("   ❌ No GPU detected!")
    print("   Troubleshooting:")
    print("   - Verify container started with --gpus all")
    print("   - Check nvidia-smi on host")
    print("   - Restart Docker daemon")
EOF
echo ""

echo "3️⃣ Check CUDA Environment:"
python3 << 'EOF'
import os
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
tf_gpu_allow = os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'Not set')
print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
print(f"   TF_FORCE_GPU_ALLOW_GROWTH: {tf_gpu_allow}")
EOF
echo ""

echo "4️⃣ GPU Memory Test:"
python3 << 'EOF'
import tensorflow as tf
try:
    # Create small tensor on GPU
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
    print(f"   ✅ GPU computation successful!")
    print(f"   Result shape: {c.shape}")
    
    # Check memory usage
    gpu = tf.config.list_physical_devices('GPU')[0]
    print(f"   GPU Device: {gpu.name}")
except Exception as e:
    print(f"   ❌ GPU computation failed: {e}")
EOF
echo ""

echo "5️⃣ Check Training Files:"
echo "   Dataset: $([ -f /workspace/dual_modal_gan/data/dataset_gan.tfrecord ] && echo '✅' || echo '❌') /workspace/dual_modal_gan/data/dataset_gan.tfrecord"
echo "   Model: $([ -f /workspace/models/best_htr_recognizer/best_model.weights.h5 ] && echo '✅' || echo '❌') /workspace/models/best_htr_recognizer/best_model.weights.h5"
echo "   Charlist: $([ -f /workspace/real_data_preparation/real_data_charlist.txt ] && echo '✅' || echo '❌') /workspace/real_data_preparation/real_data_charlist.txt"
echo ""

echo "================================================"
echo "✅ GPU Test Complete"
echo "================================================"
echo ""
echo "Next Steps:"
echo "  • If GPU detected: Start training"
echo "  • If no GPU: Check container startup command (--gpus all)"
echo "  • Monitor: docker logs -f gan-htr-training | grep GPU"
echo ""
