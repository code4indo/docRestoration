# Ablation Study Plan: Iterative Refinement vs Baseline

## 1. Tujuan Eksperimen

### 1.1. Hipotesis Penelitian
**Hipotesis Utama:** Implementasi "Iterative Refinement with Recognizer Attention" akan memberikan peningkatan performa signifikan dibandingkan dengan baseline GAN-HTR dual-modal biasa.

**Hipotesis Spesifik:**
1. **Kualitas Visual**: PSNR dan SSIM akan meningkat karena attention mechanism membantu generator fokus pada area teks
2. **Akurasi Teks**: CER dan WER akan menurun karena iterative refinement menghasilkan gambar yang lebih mudah dibaca oleh HTR
3. **Stabilitas Training**: Attention-guided refinement akan mengurangi fluktuasi loss selama training
4. **Convergence Speed**: Model akan konvergen lebih cepat dengan attention guidance

### 1.2. Pertanyaan Penelitian
1. Apakah iterative refinement memberikan peningkatan kualitas visual (PSNR/SSIM)?
2. Apakah iterative refinement meningkatkan akurasi pengenalan teks (CER/WER)?
3. Bagaimana pengaruh iterative refinement terhadap stabilitas training?
4. Apakah ada trade-off antara kompleksitas model dan performa?

## 2. Metodologi Eksperimen

### 2.1. Desain Eksperimen
**Ablation Study Design:**
- **Control Group**: Baseline GAN-HTR dual-modal (tanpa iterative refinement)
- **Treatment Group**: GAN-HTR dual-modal dengan iterative refinement
- **Variable**: Kehadiran attention-guided iterative refinement
- **Controlled Variables**: Semua parameter training, dataset, dan evaluasi

### 2.2. Parameter Eksperimen
**Parameter Training (Sama untuk kedua grup):**
```json
{
  "batch_size": 4,
  "epochs": 10,
  "steps_per_epoch": 100,
  "learning_rate_generator": 0.0002,
  "learning_rate_discriminator": 0.0002,
  "learning_rate_text_encoder": 0.0002,
  "pixel_loss_weight": 200.0,
  "recognition_feature_loss_weight": 5.0,
  "adversarial_loss_weight": 3.0,
  "contrastive_loss_weight": 1.0,
  "ctc_loss_weight": 1.0,
  "gradient_clip_norm": 1.0,
  "early_stopping_patience": 15,
  "discriminator_mode": "predicted"
}
```

**Parameter Dataset:**
- **Training Samples**: 4,266
- **Validation Samples**: 473
- **Image Dimensions**: 1024x128 pixels
- **Charset Size**: 109 characters

### 2.3. Perbedaan Implementasi

#### Baseline (Control Group)
```python
# Generator: 1-channel input
generated_images = generator(degraded_images, training=True)

# Loss calculation menggunakan generated_images langsung
# Tidak ada attention mechanism untuk refinement
```

#### Iterative Refinement (Treatment Group)
```python
# Generator: 2-channel input (degraded + attention)
# Step 1: Generate dengan dummy attention
input_v1 = tf.concat([degraded_images, dummy_attention], axis=-1)
generated_v1 = generator(input_v1, training=True)

# Step 2: Extract attention dari recognizer
_, _, _, attention_map = recognizer(generated_v1, training=False)
attention_map = tf.clip_by_value(attention_map, 0.0, 1.0)

# Step 3: Refine dengan real attention
input_v2 = tf.concat([degraded_images, attention_map], axis=-1)
generated_v2 = generator(input_v2, training=True)

# Loss calculation menggunakan generated_v2
```

## 3. Metrik Evaluasi

### 3.1. Metrik Kuantitatif
**Visual Quality Metrics:**
- **PSNR (Peak Signal-to-Noise Ratio)**: Mengukur kualitas rekonstruksi
- **SSIM (Structural Similarity Index)**: Mengukur kesamaan struktural

**Text Recognition Metrics:**
- **CER (Character Error Rate)**: Tingkat error karakter
- **WER (Word Error Rate)**: Tingkat error kata

**Training Stability Metrics:**
- **Loss Convergence**: Kecepatan dan stabilitas konvergensi
- **Gradient Norms**: Stabilitas gradien selama training

### 3.2. Metrik Kualitatif
**Visual Assessment:**
- Sample images per epoch untuk perbandingan visual
- Perhatian pada kualitas teks dan background
- Evaluasi artifact dan noise

### 3.3. Metrik Komparatif
**Primary Comparison Metrics:**
1. ΔPSNR = PSNR_iterative - PSNR_baseline
2. ΔSSIM = SSIM_iterative - SSIM_baseline
3. ΔCER = CER_iterative - CER_baseline (negatif = improvement)
4. ΔWER = WER_iterative - WER_baseline (negatif = improvement)

**Statistical Significance:**
- Best performance comparison dari masing-masing eksperimen
- Analysis dari training dynamics

## 4. Setup Eksperimen

### 4.1. Environment
- **Hardware**: NVIDIA RTX A4000 (14GB VRAM)
- **Software**: TensorFlow 2.16.1, Python 3.11
- **Precision**: Pure FP32 (untuk stabilitas numerik)
- **Reproducibility**: Random seed = 42

### 4.2. Output Directories
```
dual_modal_gan/outputs/
├── checkpoints_ablation/
│   ├── ablation_baseline/           # Control group
│   └── ablation_iterative_refinement/ # Treatment group
└── samples_ablation/
    ├── ablation_baseline/
    └── ablation_iterative_refinement/
```

### 4.3. Logging dan Monitoring
- **MLflow Tracking**: Untuk semua metrik training
- **Console Logging**: Progress monitoring
- **Model Checkpoints**: Auto-save best models
- **Sample Images**: Visual assessment per epoch

## 5. Timeline Eksperimen

### 5.1. Durasi Perkiraan
- **Setup**: 30 menit
- **Training per eksperimen**: ~2-3 jam (10 epochs × 100 steps)
- **Total eksperimen**: ~4-6 jam
- **Analysis**: 1-2 jam

### 5.2. Milestones
1. **T+30m**: Setup complete, training starts
2. **T+3h**: Baseline experiment complete
3. **T+6h**: Iterative refinement experiment complete
4. **T+8h**: Analysis complete, results ready

## 6. Risiko dan Mitigasi

### 6.1. Risiko Teknis
- **Memory Issues**: Batch size terlalu besar untuk VRAM
  - *Mitigasi*: Gunakan batch size 4, gradient checkpointing
  
- **Training Instability**: Loss explosion atau divergence
  - *Mitigasi*: Gradient clipping, learning rate scheduling
  
- **cuDNN Errors**: Compatibility issues dengan TensorFlow
  - *Mitigasi*: Use `use_cudnn=False` di RNN layers

### 6.2. Risiko Metodologis
- **Parameter Mismatch**: Perbedaan parameter antara eksperimen
  - *Mitigasi*: Use identical parameter sets
  
- **Reproducibility Issues**: Hasil tidak konsisten
  - *Mitigasi*: Fixed random seeds, deterministic ops
  
- **Overfitting**: Model terlalu fit ke training data
  - *Mitigasi*: Early stopping, proper validation

## 7. Expected Outcomes

### 7.1. Hasil Positif (Hipotesis Terbukti)
- **PSNR**: +1-2 dB improvement
- **SSIM**: +2-5% improvement  
- **CER**: -5-10% improvement
- **WER**: -3-8% improvement
- **Training**: More stable, faster convergence

### 7.2. Hasil Netral (Hipotesis Tidak Terbukti)
- **PSNR/SSIM**: Comparable performance (±0.5 dB, ±1%)
- **CER/WER**: Comparable performance (±2%)
- **Training**: Similar dynamics and convergence

### 7.3. Hasil Negatif (Hipotesis Ditolak)
- **PSNR/SSIM**: Decreased performance
- **CER/WER**: Increased error rates
- **Training**: Less stable, slower convergence

## 8. Next Steps Setelah Eksperimen

### 8.1. Jika Hasil Positif
1. **Production Deployment**: Integrate iterative refinement ke main pipeline
2. **Hyperparameter Tuning**: Optimasi attention mechanism parameters
3. **Architecture Enhancement**: Explore multi-scale attention
4. **Paper Writing**: Document findings untuk publication

### 8.2. Jika Hasil Netral/Negatif
1. **Root Cause Analysis**: Investigasi mengapa iterative refinement tidak efektif
2. **Architecture Modification**: Explore alternative attention mechanisms
3. **Simplification**: Return to simpler, more stable architecture
4. **Documentation**: Record negative findings untuk future research

---

*Dibuat pada: 2025-10-13*  
*Versi: 1.0*  
*Status: Ready for Execution*