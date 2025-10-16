#!/bin/bash

# Analisis Skala Loss Script
# Berdasarkan train32_smoke_test.sh
# Menjalankan training singkat dengan semua bobot loss = 1.0

echo "========================================="
echo "🔬 GAN-HTR Analisis Skala Loss"
echo "========================================="
echo ""
echo "Purpose: Menjalankan training singkat (3 epochs) dengan semua bobot loss diatur ke 1.0"
echo "  • Mengamati magnitudo alami dari setiap komponen loss"
echo "  • Mendapatkan titik awal yang ilmiah untuk pembobotan loss"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "📍 Project root: $PROJECT_ROOT"

# Menggunakan Poetry environment
PYTHON_CMD="poetry run python"
echo "🐍 Menggunakan lingkungan Poetry: $PYTHON_CMD"

# Set paths
DATASET_PATH="$PROJECT_ROOT/dual_modal_gan/data/dataset_gan.tfrecord"
CHARSET_PATH="$PROJECT_ROOT/real_data_preparation/real_data_charlist.txt"
RECOGNIZER_WEIGHTS="$PROJECT_ROOT/models/best_htr_recognizer/best_model.weights.h5"
CHECKPOINT_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/checkpoints_analisis_skala"
SAMPLE_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/samples_analisis_skala"

mkdir -p "$CHECKPOINT_DIR" "$SAMPLE_DIR"

cd "$PROJECT_ROOT" || {
    echo "❌ Error: Tidak bisa pindah ke direktori proyek: $PROJECT_ROOT"
    exit 1
}

echo "🚀 Memulai training untuk Analisis Skala Loss..."

# Menjalankan train32.py dengan parameter untuk analisis skala
$PYTHON_CMD dual_modal_gan/scripts/train32.py \
  --tfrecord_path "$DATASET_PATH" \
  --charset_path "$CHARSET_PATH" \
  --recognizer_weights "$RECOGNIZER_WEIGHTS" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --sample_dir "$SAMPLE_DIR" \
  --gpu_id 0 \
  --epochs 3 \
  --steps_per_epoch 100 \
  --batch_size 2 \
  --pixel_loss_weight 1.0 \
  --rec_feat_loss_weight 1.0 \
  --adv_loss_weight 1.0 \
  --no_restore

TRAIN_EXIT_CODE=$?

echo ""
echo "========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Analisis Skala Loss selesai."
else
    echo "❌ Analisis Skala Loss gagal dengan exit code: $TRAIN_EXIT_CODE"
fi
echo "========================================="