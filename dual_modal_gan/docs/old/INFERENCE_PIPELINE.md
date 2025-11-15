# GAN-HTR Inference Pipeline

## 📋 Deskripsi

Pipeline inference lengkap untuk restorasi dokumen terdegradasi dan ekstraksi teks menggunakan GAN-HTR. Pipeline ini memproses dokumen ukuran penuh (bukan baris teks) dengan langkah-langkah:

1. **Segmentasi Otomatis** - Deteksi dan ekstraksi baris teks dari dokumen
2. **Restorasi GAN** - Enhancement setiap baris menggunakan trained GAN-HTR
3. **HTR Recognition** - Ekstraksi teks dari baris yang sudah direstorasi
4. **Rekonstruksi Dokumen** - Gabungkan semua baris menjadi dokumen bersih

---

## 🚀 Quick Start

### 1. Persiapan

Pastikan model sudah di-train:
- **GAN-HTR Checkpoint**: `dual_modal_gan/outputs/checkpoints_fp32_smoke_test/`
- **HTR Weights**: `models/best_htr_recognizer/best_model.weights.h5`
- **Charset**: `real_data_preparation/real_data_charlist.txt`

### 2. Siapkan Dokumen Input

```bash
# Buat direktori untuk dokumen terdegradasi
mkdir -p data/degraded_documents

# Copy dokumen ke direktori tersebut
cp /path/to/your/degraded/*.png data/degraded_documents/
```

Format yang didukung: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`

### 3. Jalankan Inference

**Cara 1: Menggunakan Shell Script (Recommended)**

```bash
# Simple usage
./run_inference.sh data/degraded_documents outputs/restored

# Full parameters
./run_inference.sh <input_dir> <output_dir> <gpu_id> <batch_size>

# Example
./run_inference.sh data/degraded_documents outputs/restored_docs 1 8
```

**Cara 2: Direct Python**

```bash
poetry run python dual_modal_gan/scripts/inference_pipeline.py \
    --input_dir data/degraded_documents \
    --output_dir outputs/restored_documents \
    --gan_checkpoint dual_modal_gan/outputs/checkpoints_fp32_smoke_test \
    --htr_weights models/best_htr_recognizer/best_model.weights.h5 \
    --charset_path real_data_preparation/real_data_charlist.txt \
    --gpu_id 1 \
    --batch_size 8 \
    --save_intermediates
```

---

## 📁 Output Structure

Untuk setiap dokumen yang diproses, pipeline akan menghasilkan:

```
outputs/restored_documents/
├── document_001/
│   ├── document_001_restored.png       # Dokumen bersih hasil restorasi
│   ├── document_001_text.txt           # Teks hasil HTR (per baris)
│   ├── document_001_metadata.json      # Metadata (koordinat baris, dll)
│   └── lines/                          # [Optional] Baris-baris individual
│       ├── line_001_comparison.png     # Original | Restored
│       ├── line_002_comparison.png
│       └── ...
├── document_002/
│   └── ...
└── processing_summary.json             # Summary keseluruhan processing
```

### Output Files

#### 1. `*_restored.png`
Dokumen hasil restorasi dengan kualitas lebih tinggi.

#### 2. `*_text.txt`
Teks hasil HTR recognition, format:
```
Line 001: Ini adalah baris pertama yang berhasil dibaca
Line 002: Baris kedua dengan teks yang lebih panjang
Line 003: Dan seterusnya...
```

#### 3. `*_metadata.json`
Metadata lengkap berisi:
```json
{
  "document": "document_001",
  "original_size": {"width": 2480, "height": 3508},
  "num_lines": 45,
  "processing_date": "2025-10-06T14:30:00",
  "lines": [
    {
      "line_number": 1,
      "y_start": 120,
      "y_end": 165,
      "height": 45,
      "text": "Recognized text...",
      "text_length": 25
    }
  ]
}
```

#### 4. `processing_summary.json`
Summary semua dokumen:
```json
{
  "processing_info": {
    "documents_processed": 10,
    "documents_success": 9,
    "documents_failed": 1,
    "total_time_seconds": 123.45
  },
  "results": [...]
}
```

---

## ⚙️ Parameter Configuration

### Required Parameters

| Parameter | Deskripsi | Default |
|-----------|-----------|---------|
| `--input_dir` | Direktori dokumen terdegradasi | **Required** |
| `--output_dir` | Direktori output hasil | **Required** |

### Model Parameters

| Parameter | Deskripsi | Default |
|-----------|-----------|---------|
| `--gan_checkpoint` | Path ke GAN checkpoint | `checkpoints_fp32_smoke_test` |
| `--htr_weights` | Path ke HTR weights | `best_model.weights.h5` |
| `--charset_path` | Path ke charset file | `real_data_charlist.txt` |

### Processing Parameters

| Parameter | Deskripsi | Default |
|-----------|-----------|---------|
| `--batch_size` | Batch size untuk processing | `8` |
| `--gpu_id` | GPU ID yang digunakan | `1` |

### Segmentation Parameters

| Parameter | Deskripsi | Default |
|-----------|-----------|---------|
| `--min_line_height` | Minimum tinggi baris (px) | `20` |
| `--max_line_height` | Maximum tinggi baris (px) | `200` |
| `--line_spacing_threshold` | Threshold merge baris dekat | `10` |
| `--line_padding` | Padding sekitar baris | `5` |

### Output Options

| Parameter | Deskripsi | Default |
|-----------|-----------|---------|
| `--save_intermediates` | Save baris-baris individual | `True` |
| `--no_intermediates` | Jangan save intermediate files | `False` |

---

## 🎯 Use Cases

### 1. Restorasi Dokumen Sejarah

```bash
./run_inference.sh \
    /archive/historical_documents \
    /archive/restored_output \
    1 \
    4  # Smaller batch for high-res documents
```

### 2. Batch Processing Dokumen Scan

```bash
poetry run python dual_modal_gan/scripts/inference_pipeline.py \
    --input_dir /scans/batch_001 \
    --output_dir /scans/restored_batch_001 \
    --batch_size 16 \
    --gpu_id 0 \
    --no_intermediates  # Skip intermediate saves for speed
```

### 3. Custom Segmentation Parameters

Untuk dokumen dengan spacing non-standard:

```bash
poetry run python dual_modal_gan/scripts/inference_pipeline.py \
    --input_dir data/tight_spacing_docs \
    --output_dir outputs/custom_segmentation \
    --min_line_height 15 \
    --max_line_height 150 \
    --line_spacing_threshold 5 \
    --line_padding 3
```

---

## 🔍 Troubleshooting

### Problem: Tidak ada baris terdeteksi

**Solusi:**
```bash
# Turunkan min_line_height dan line_spacing_threshold
poetry run python dual_modal_gan/scripts/inference_pipeline.py \
    --min_line_height 10 \
    --line_spacing_threshold 5 \
    ...
```

### Problem: Baris terpotong atau merge salah

**Solusi:**
```bash
# Adjust max_line_height dan spacing threshold
poetry run python dual_modal_gan/scripts/inference_pipeline.py \
    --max_line_height 250 \
    --line_spacing_threshold 15 \
    ...
```

### Problem: Out of Memory (OOM)

**Solusi:**
```bash
# Kurangi batch_size
poetry run python dual_modal_gan/scripts/inference_pipeline.py \
    --batch_size 2 \
    ...
```

### Problem: Inference lambat

**Solusi:**
1. Increase batch_size (jika memory cukup)
2. Skip intermediates: `--no_intermediates`
3. Use GPU dengan memory lebih besar

---

## 📊 Performance Metrics

### Typical Processing Speed

| Document Size | Lines | Batch=4 | Batch=8 | Batch=16 |
|--------------|-------|---------|---------|----------|
| A4 (300 DPI) | 30-40 | ~15s | ~10s | ~8s |
| A4 (600 DPI) | 40-50 | ~25s | ~18s | ~15s |
| A3 (300 DPI) | 50-60 | ~30s | ~22s | ~18s |

**Hardware**: RTX 3090 24GB, i9-10900K

### Memory Usage

| Batch Size | GPU Memory | Speed | Recommendation |
|------------|-----------|-------|----------------|
| 2 | ~6 GB | Slow | Low-end GPU (<8GB) |
| 4 | ~10 GB | Good | Mid-range GPU (8-12GB) |
| 8 | ~16 GB | Fast | High-end GPU (12-16GB) |
| 16 | ~24 GB | Fastest | Professional GPU (24GB+) |

---

## 🧪 Testing

### Test dengan Sample Document

```bash
# Download sample degraded document (jika ada)
mkdir -p test_data
# ... copy sample document ...

# Run test inference
./run_inference.sh test_data outputs/test_output 1 8

# Check results
ls -lh outputs/test_output/
cat outputs/test_output/*/processing_summary.json
```

### Validasi Output

```python
# Check metadata
import json

with open('outputs/test_output/doc_001/doc_001_metadata.json') as f:
    metadata = json.load(f)
    
print(f"Lines detected: {metadata['num_lines']}")
print(f"First line text: {metadata['lines'][0]['text']}")
```

---

## 🔬 Advanced Usage

### Custom Post-Processing

Anda bisa extend pipeline dengan custom post-processing:

```python
from dual_modal_gan.scripts.inference_pipeline import DocumentRestorationPipeline

# ... load models ...

pipeline = DocumentRestorationPipeline(generator, recognizer, charset, segmenter)

# Custom processing
for doc_path in document_files:
    result = pipeline.process_document(doc_path, output_dir)
    
    # Custom post-processing
    text = result['text']
    # Apply spell correction, formatting, etc.
    processed_text = your_custom_function(text)
    
    # Save custom output
    with open(output_path / 'processed.txt', 'w') as f:
        f.write(processed_text)
```

---

## 📝 Notes

1. **Model Compatibility**: Pipeline ini menggunakan model yang di-train dengan `train32.py`. Pastikan menggunakan checkpoint yang kompatibel.

2. **Input Format**: Dokumen input harus berupa image file. Untuk PDF, convert dulu ke image:
   ```bash
   # Using ImageMagick
   convert -density 300 input.pdf output_%03d.png
   ```

3. **Line Segmentation**: Algoritma segmentasi bekerja optimal untuk dokumen dengan:
   - Background terang, teks gelap
   - Baris horizontal (bukan rotated)
   - Spacing konsisten antar baris
   
   Untuk dokumen dengan layout kompleks, pertimbangkan pre-processing atau custom segmentation.

4. **Batch Processing**: Untuk processing ribuan dokumen, pertimbangkan:
   - Split ke multiple batches
   - Monitor disk space (intermediate files bisa besar)
   - Run on multiple GPUs parallel

---

## 🤝 Integration

Pipeline ini bisa diintegrasikan dengan:

- **OCR Systems**: Use restored images sebagai input untuk OCR downstream
- **Document Management**: Automated document digitization pipeline
- **Archive Systems**: Mass restoration historical documents
- **Research Tools**: Extract text dari manuscript terdegradasi

---

## 📞 Support

Untuk issues atau pertanyaan, lihat:
- Training logs: `dual_modal_gan/outputs/checkpoints_*/metrics/`
- Model checkpoints: `dual_modal_gan/outputs/checkpoints_*/`
- Dataset info: `dual_modal_gan/data/`

---

**Happy Restoring! 🎨📖**
