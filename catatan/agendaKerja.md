# Agenda & Status Kerja Proyek GAN-HTR

Dokumen ini berfungsi sebagai papan status (Kanban board) untuk melacak progres tugas-tugas utama dalam riset ini.

---

### ✅ SELESAI (DONE)

- `[X] Optimasi & Finalisasi Golden Baseline`
  - **Hasil:** Eksperimen `p200_r5` terpilih sebagai baseline dengan performa terbaik (PSNR: 23.20, CER: 0.20).

- `[X] Implementasi & Eksperimen Cross-Modal Contrastive Loss`
  - **Hasil:** Arsitektur baru berhasil diimplementasikan dan diuji melalui studi ablasi (`contrastive_on` vs `contrastive_off`).

- `[X] Analisis Hasil Eksperimen Contrastive Loss`
  - **Hasil:** Ditemukan bahwa implementasi *contrastive loss* saat ini **tidak meningkatkan performa** dan bahkan sedikit menurunkannya. Hipotesis awal tidak terbukti. Ini adalah temuan riset yang valid.

### ✅ SELESAI (DONE)

- `[X] Perbaikan Bug Teknis: cuDNN RNN Mask Error & CTC Loss`
  - **Hasil:** Berhasil menerapkan fix `use_cudnn=False` pada `text_encoder.py` dan menonaktifkan sementara CTC loss karena compatibility issue dengan TensorFlow version.
  - **Status:** Sistem stabil dan siap untuk eksperimen.

- `[X] Implementasi & Validasi "Iterative Refinement with Recognizer Attention"`
  - **Hasil:** Smoke test berhasil dengan 1 epoch, 10 steps. Validasi PSNR: 1.02, SSIM: 0.0723, CER: 0.9611, WER: 0.9904.
  - **Status:** Implementasi iterative refinement (2-step process) berfungsi dengan benar.

### ⏳ SEDANG DIKERJAKAN (IN PROGRESS)

### 📋 RENCANA SELANJUTNYA (BACKLOG / TO-DO)

- `[ ] Eksperimen & Studi Ablasi untuk "Iterative Refinement"`
  - **Deskripsi:** Menjalankan eksperimen terkontrol untuk membandingkan performa arsitektur iterative refinement dengan Golden Baseline.
  - **Parameter:** 10+ epochs, proper loss weights, comprehensive evaluation.
  - **Target:** Membuktikan apakah iterative refinement memberikan improvement signifikan.

- `[ ] Analisis Hasil "Iterative Refinement"`
  - **Deskripsi:** Menganalisis data kuantitatif dan kualitatif untuk membuktikan (atau menyanggah) hipotesis bahwa attention-guided refinement meningkatkan performa.
  - **Metrics:** PSNR, SSIM, CER, WER, visual quality assessment.

- `[ ] Perbaikan CTC Loss Compatibility`
  - **Deskripsi:** Menginvestigasi dan memperbaiki compatibility issue CTC loss dengan TensorFlow version saat ini.
  - **Goal:** Mengaktifkan kembali CTC monitoring untuk evaluasi HTR yang lengkap.

- `[ ] Penulisan Draf Awal Jurnal (Metodologi & Hasil)`
  - **Deskripsi:** Mulai menyusun draf paper berdasarkan temuan-temuan yang telah divalidasi.
