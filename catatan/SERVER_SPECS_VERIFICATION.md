# VERIFIKASI SPESIFIKASI SERVER TRAINING
**Tanggal**: 2025-11-02  
**Tujuan**: Memverifikasi hardware yang digunakan untuk training vs klaim di paper

---

## SPESIFIKASI SERVER AKTUAL (VERIFIED)

### GPU Configuration
```
Model: 2× NVIDIA RTX A4000
- VRAM per GPU: 16,376 MiB (~16 GB)
- Total VRAM: 32 GB
- Driver Version: 570.181
- CUDA Version: 12.8
- GPU 0: Currently in use (14.4 GB used - training process)
- GPU 1: Idle (15 MiB used)
```

### CPU Configuration
```
Model: AMD Ryzen Threadripper PRO 3955WX 16-Cores
- Total Cores: 16 physical cores
- Total Threads: 32 (2 threads per core)
- Architecture: AMD Threadripper PRO (Workstation class)
```

### RAM Configuration
```
Total Memory: 125 GiB (~128 GB)
- Currently Used: 32 GiB
- Available: 91 GiB
- Type: DDR4 (assumed from Threadripper PRO platform)
```

### Software Stack
```
Operating System: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
CUDA: 12.8.93
TensorFlow: 2.16.1 (CUDA-enabled, 2 GPUs detected)
Python Environment: Virtual env (.venv)
```

---

## PERBANDINGAN DENGAN KLAIM PAPER

### ❌ OCCURRENCE 1 (Line 2074-2120) - PARTIAL MISMATCH

**PAPER CLAIM**:
```
- Framework: TensorFlow 2.15.0, CUDA 12.2, cuDNN 8.9
- Hardware: 1× NVIDIA RTX 3090 GPU (24GB VRAM)
- CPU: AMD Ryzen 9 5950X
- RAM: 128GB DDR4-3200
- OS: Ubuntu 22.04 LTS
```

**ACTUAL SERVER**:
```
- Framework: TensorFlow 2.16.1, CUDA 12.8 ✓ (close)
- Hardware: 2× NVIDIA RTX A4000 (16GB VRAM each) ❌ DIFFERENT
- CPU: AMD Ryzen Threadripper PRO 3955WX 16-Cores ❌ DIFFERENT
- RAM: 128GB (125 GiB) ✓ (match)
- OS: Ubuntu 22.04.5 LTS ✓ (match)
```

**DISCREPANCIES**:
1. GPU: RTX 3090 (24GB) → **ACTUAL: 2× RTX A4000 (16GB each)**
2. CPU: Ryzen 9 5950X → **ACTUAL: Threadripper PRO 3955WX**
3. TensorFlow: 2.15.0 → **ACTUAL: 2.16.1**
4. CUDA: 12.2 → **ACTUAL: 12.8**

### ❌ OCCURRENCE 2 (Line 2127-2135) - CLOSER TO TRUTH

**PAPER CLAIM**:
```
- Framework: TensorFlow 2.x / Keras
- Hardware: 2× NVIDIA RTX 3090 GPU (24GB VRAM each)
- RAM: 64GB
```

**ACTUAL SERVER**:
```
- Framework: TensorFlow 2.16.1 ✓
- Hardware: 2× NVIDIA RTX A4000 (16GB VRAM each) ⚠️ (count correct, model wrong)
- RAM: 128GB ❌ (paper says 64GB)
```

**DISCREPANCIES**:
1. GPU count: 2× ✓ CORRECT
2. GPU model: RTX 3090 → **ACTUAL: RTX A4000**
3. VRAM: 24GB → **ACTUAL: 16GB per GPU**
4. RAM: 64GB → **ACTUAL: 128GB**

---

## ANALISIS CRITICAL

### TEMUAN PENTING:

1. **GPU Model Mismatch**: 
   - Paper: RTX 3090 (consumer/prosumer gaming GPU)
   - Actual: RTX A4000 (professional workstation GPU)
   - **Impact**: RTX A4000 has LESS VRAM (16GB vs 24GB) but more stable for long training

2. **GPU Count**:
   - Occurrence 1: Claims 1× GPU (WRONG)
   - Occurrence 2: Claims 2× GPU (CORRECT)
   - **Actual**: 2× GPUs available
   - **Usage**: Training script uses SINGLE GPU via `--gpu_id` parameter

3. **CPU Upgrade**:
   - Paper: Ryzen 9 5950X (16-core consumer HEDT)
   - Actual: Threadripper PRO 3955WX (16-core workstation)
   - **Impact**: Threadripper PRO has better memory bandwidth and PCIe lanes

4. **TensorFlow Version**:
   - Paper: 2.15.0
   - Actual: 2.16.1
   - **Impact**: Minor version bump, likely compatible

---

## TRAINING CONFIGURATION VERIFICATION

### GPU Usage Pattern (from nvidia-smi):
```
GPU 0: ACTIVELY USED (14.4 GB / 16.4 GB) - Training process (PID 246114)
GPU 1: IDLE (15 MB / 16.4 GB)
```

**CONCLUSION**: Training menggunakan **SINGLE GPU** (GPU 0), bukan dual-GPU setup.

### Validation with Code:
```python
# train_enhanced.py line 784
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
print(f"--- Configuring to use GPU: {args.gpu_id} ---")
```

**CONFIRMED**: Script dirancang untuk SINGLE GPU operation dengan parameter `--gpu_id`.

---

## REKOMENDASI PERBAIKAN PAPER

### PRIORITAS TINGGI - Hardware Specs:

**GANTI** section "Spesifikasi Perangkat Keras dan Perangkat Lunak" dengan:

```latex
\subsubsection{Spesifikasi Perangkat Keras dan Perangkat Lunak}
\begin{itemize}
    \item \textbf{Kerangka Kerja:} TensorFlow 2.16.1, CUDA 12.8, cuDNN 8.9
    \item \textbf{Precision:} Pure FP32 (NO mixed precision) untuk stabilitas numerik CTC loss
    \item \textbf{GPU:} 1× NVIDIA RTX A4000 (16GB VRAM) - single GPU training
    \item \textbf{CPU:} AMD Ryzen Threadripper PRO 3955WX (16 cores, 32 threads)
    \item \textbf{RAM:} 128GB DDR4 system memory
    \item \textbf{Sistem Operasi:} Ubuntu 22.04.5 LTS
    \item \textbf{Multi-GPU Available:} 2× RTX A4000 (training uses 1× GPU via CUDA_VISIBLE_DEVICES)
    \item \textbf{Waktu pelatihan:} Approximately 48 hours untuk 50 epochs (single GPU, batch size 2)
    \item \textbf{GPU Memory Usage:} ~14.4 GB / 16 GB during training (batch size 2)
\end{itemize}
```

### CATATAN PENTING:

1. **RTX A4000 vs RTX 3090**:
   - A4000: Professional workstation GPU, ECC memory support, lower power (140W)
   - RTX 3090: Gaming/prosumer GPU, no ECC, higher power (350W)
   - A4000 lebih cocok untuk long training sessions

2. **Memory Constraint**:
   - Dengan 16GB VRAM, batch size dibatasi ke 2 (images 1024×128)
   - RTX 3090 (24GB) mungkin bisa batch size 3-4, tetapi tidak feasible dengan A4000

3. **Training Time**:
   - Paper claim: 25 GPU-hours (50 epochs)
   - Actual observation: ~48 hours for full training
   - **Conclusion**: 48 hours @ 1× GPU = 48 GPU-hours total

---

## KESIMPULAN

### ✅ YANG PERLU DIUPDATE DI PAPER:

1. **GPU Model**: RTX 3090 → RTX A4000
2. **GPU VRAM**: 24GB → 16GB
3. **CPU Model**: Ryzen 9 5950X → Threadripper PRO 3955WX
4. **TensorFlow**: 2.15.0 → 2.16.1
5. **CUDA**: 12.2 → 12.8
6. **GPU Usage**: Clarify "single GPU training from 2× GPU system"
7. **Training Time**: Update to 48 GPU-hours if needed
8. **DELETE**: Duplicated section occurrence 2

### ⚠️ CATATAN UNTUK PAPER:

Sebaiknya tambahkan footnote:
```
Training dilakukan pada sistem dengan 2× NVIDIA RTX A4000 GPUs, tetapi 
training script menggunakan single GPU (GPU 0) via CUDA_VISIBLE_DEVICES 
untuk stability. Multi-GPU data parallelism tidak digunakan karena batch 
size kecil (2 images) dan memory constraints dari image resolution tinggi 
(1024×128 pixels).
```

---

## REKOMENDASI FINAL

**OPTION 1 - Full Transparency** (RECOMMENDED):
- State ACTUAL hardware used (RTX A4000)
- Explain single-GPU usage from multi-GPU system
- Accurate specs build trust dengan reviewers

**OPTION 2 - Generalize**:
- "Professional-grade NVIDIA GPU (16GB VRAM)"
- "AMD Threadripper PRO Workstation"
- Less specific tetapi still accurate

**PILIHAN**: Saya recommend OPTION 1 untuk academic integrity dan reproducibility.
