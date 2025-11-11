#!/bin/bash

# Script untuk menjalankan eksperimen perbandingan curriculum learning
# Eksekusi: bash run_curriculum_learning_comparison.sh

echo "🚀 EKSPERIMEN PERBANDINGAN CURRICULUM LEARNING"
echo "================================================"
echo ""
echo "Tujuan: Membandingkan efektivitas curriculum learning vs tanpa curriculum learning"
echo "Config 1: configs/experiment_curriculum_comparison_with_curriculum.json"
echo "Config 2: configs/experiment_curriculum_comparison_no_curriculum.json"
echo ""
echo "================================================"
echo ""

# Check if config files exist
if [ ! -f "configs/experiment_curriculum_comparison_with_curriculum.json" ]; then
    echo "❌ Error: Config file curriculum with tidak ditemukan!"
    exit 1
fi

if [ ! -f "configs/experiment_curriculum_comparison_no_curriculum.json" ]; then
    echo "❌ Error: Config file curriculum no tidak ditemukan!"
    exit 1
fi

echo "✅ Config files ditemukan!"
echo ""

# Choose execution mode
echo "Pilih mode eksekusi:"
echo "1) Sequential (berurutan: with curriculum dulu, lalu no curriculum)"
echo "2) Parallel (paralel: kedua eksperimen bersamaan)"
echo ""
read -p "Masukkan pilihan (1 atau 2): " mode

if [ "$mode" == "1" ]; then
    echo ""
    echo "🔄 MODE SEQUENTIAL DIPILIH"
    echo "=========================="
    echo ""
    
    echo "🚀 Memulai eksperimen 1: DENGAN CURRICULUM LEARNING"
    echo "Config: configs/experiment_curriculum_comparison_with_curriculum.json"
    echo "Log file: logs/experiment_curriculum_with.log"
    echo ""
    
    # Create log directory if not exists
    mkdir -p logs
    
    # Run with curriculum learning
    nohup poetry run ./scripts/universal_train_from_json.sh configs/experiment_curriculum_comparison_with_curriculum.json > logs/experiment_curriculum_with.log 2>&1 &
    
    echo "✅ Eksperimen dengan curriculum learning dimulai!"
    echo "📝 Log: logs/experiment_curriculum_with.log"
    echo "🖥️  Monitor: tail -f logs/experiment_curriculum_with.log"
    echo ""
    echo "⏳ Menunggu proses selesai..."
    
    # Wait for first experiment to finish
    wait %1
    
    echo ""
    echo "✅ Eksperimen 1 selesai!"
    echo ""
    
    echo "🚀 Memulai eksperimen 2: TANPA CURRICULUM LEARNING"
    echo "Config: configs/experiment_curriculum_comparison_no_curriculum.json"
    echo "Log file: logs/experiment_curriculum_no.log"
    echo ""
    
    # Run without curriculum learning
    nohup poetry run ./scripts/universal_train_from_json.sh configs/experiment_curriculum_comparison_no_curriculum.json > logs/experiment_curriculum_no.log 2>&1 &
    
    echo "✅ Eksperimen tanpa curriculum learning dimulai!"
    echo "📝 Log: logs/experiment_curriculum_no.log"
    echo "🖥️  Monitor: tail -f logs/experiment_curriculum_no.log"
    echo ""
    echo "⏳ Menunggu proses selesai..."
    
    # Wait for second experiment to finish
    wait %2
    
    echo ""
    echo "🎉 SEMUA EKSPERIMEN SELESAI!"
    echo "==========================="
    echo ""
    echo "📊 HASIL:"
    echo "  - Dengan curriculum: dual_modal_gan/checkpoints/experiment_curriculum_with/"
    echo "  - Tanpa curriculum: dual_modal_gan/checkpoints/experiment_curriculum_no/"
    echo ""
    echo "📝 LOG FILES:"
    echo "  - Dengan curriculum: logs/experiment_curriculum_with.log"
    echo "  - Tanpa curriculum: logs/experiment_curriculum_no.log"
    echo ""
    echo "🔍 LANJUTAN:"
    echo "  1. Analisis hasil training metrics"
    echo "  2. Bandingkan PSNR, CER, stabilitas loss"
    echo "  3. Update Tabel 4 di Chapter 5 dengan data aktual"
    
elif [ "$mode" == "2" ]; then
    echo ""
    echo "🔄 MODE PARALLEL DIPILIH"
    echo "========================"
    echo ""
    
    echo "🚀 Memulai kedua eksperimen secara paralel..."
    echo ""
    
    # Create log directory if not exists
    mkdir -p logs
    
    # Run both experiments in parallel
    echo "▶️  Starting experiment 1 (WITH curriculum) on GPU 0..."
    nohup poetry run ./scripts/universal_train_from_json.sh configs/experiment_curriculum_comparison_with_curriculum.json > logs/experiment_curriculum_with.log 2>&1 &
    pid1=$!
    
    echo "▶️  Starting experiment 2 (NO curriculum) on GPU 1..."
    nohup poetry run ./scripts/universal_train_from_json.sh configs/experiment_curriculum_comparison_no_curriculum.json > logs/experiment_curriculum_no.log 2>&1 &
    pid2=$!
    
    echo ""
    echo "✅ Kedua eksperimen dimulai!"
    echo ""
    echo "📝 LOG FILES:"
    echo "  - With curriculum:    logs/experiment_curriculum_with.log"
    echo "  - No curriculum:      logs/experiment_curriculum_no.log"
    echo ""
    echo "🖥️  MONITORING:"
    echo "  tail -f logs/experiment_curriculum_with.log"
    echo "  tail -f logs/experiment_curriculum_no.log"
    echo ""
    echo "⏳ Processes running:"
    echo "  PID 1 (with curriculum): $pid1"
    echo "  PID 2 (no curriculum):   $pid2"
    echo ""
    echo "🎯 TO CHECK STATUS:"
    echo "  ps aux | grep universal_train_from_json"
    echo ""
    
    # Wait for both processes
    echo "⏳ Menunggu kedua proses selesai..."
    wait $pid1 $pid2
    
    echo ""
    echo "🎉 SEMUA EKSPERIMEN SELESAI!"
    echo "==========================="
    echo ""
    echo "📊 HASIL:"
    echo "  - Dengan curriculum: dual_modal_gan/checkpoints/experiment_curriculum_with/"
    echo "  - Tanpa curriculum: dual_modal_gan/checkpoints/experiment_curriculum_no/"
    echo ""
    echo "🔍 LANJUTAN:"
    echo "  1. Analisis hasil training metrics"
    echo "  2. Bandingkan PSNR, CER, stabilitas loss"
    echo "  3. Update Tabel 4 di Chapter 5 dengan data aktual"

else
    echo "❌ Pilihan tidak valid! Harus 1 atau 2."
    exit 1
fi

echo ""
echo "✨ PERINGATAN:"
echo "==============="
echo "- Eksperimen ini akan berjalan ~3-4 jam (50 epochs, 100 steps/epoch)"
echo "- Pastikan GPU tersedia dan tidak ada proses lain yang berjalan"
echo "- Pantau log files untuk memastikan training berjalan normal"
echo "- Jika ada error, periksa log files dan config files"
echo ""
echo "📋 CHECKLIST SETELAH SELESAI:"
echo "- [ ] Cek training metrics di kedua checkpoint directories"
echo "- [ ] Analisis stabilitas loss (σ²) kedua eksperimen"
echo "- [ ] Bandingkan PSNR, CER, waktu konvergensi"
echo "- [ ] Buat tabel perbandingan faktual untuk Chapter 5"
echo "- [ ] Hapus data simulasi/hipotetis di Tabel 4"
echo ""
echo "🚀 SELESAI!"
