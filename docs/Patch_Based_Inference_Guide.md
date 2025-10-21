# 📚 Patch-Based Inference System for Universal GAN-HTR Processing

## 🎯 Overview

The Patch-Based Inference System enables **universal document enhancement** with the GAN-HTR model, supporting **arbitrary input dimensions** beyond the original training constraints. This system processes large documents by dividing them into smaller patches, processing each patch individually, and reconstructing the enhanced output with seamless blending.

## 🚀 Key Features

### ✅ Universal Input Support
- **Arbitrary dimensions**: Process any image size, not limited to 1024×128 pixels
- **Flexible aspect ratios**: Portrait, landscape, or square images
- **Various document types**: Historical documents, handwritten notes, printed text

### 🔧 Advanced Patch Processing
- **Multiple extraction strategies**: Sliding window, grid-based, adaptive content-aware
- **Intelligent overlap handling**: Configurable overlap ratios for seamless reconstruction
- **Batch processing optimization**: GPU-efficient processing of multiple patches

### 🎨 Seamless Reconstruction
- **Multiple blending methods**: Linear, Gaussian blending for smooth transitions
- **Edge-aware processing**: Proper handling of image boundaries
- **Quality metrics**: PSNR, SSIM, MSE, MAE calculation for reconstruction assessment

## 🏗️ System Architecture

```
UniversalGANHTRProcessor
├── PatchExtractor
│   ├── Sliding Window Strategy
│   ├── Grid Strategy
│   └── Adaptive Strategy
├── PatchProcessor
│   ├── Batch Processing
│   ├── GPU Memory Management
│   └── Error Handling
└── ImageReconstructor
    ├── Linear Blending
    ├── Gaussian Blending
    └── Quality Metrics
```

## 📋 Installation and Setup

### Prerequisites
```bash
# Ensure you have the required dependencies
poetry install
```

### File Structure
```
docRestoration/
├── scripts/
│   ├── patch_based_inference.py    # Main system implementation
│   ├── test_patch_demo.py          # Demo script (no model required)
│   └── test_patch_inference.py     # Full system test
├── dual_modal_gan/checkpoints/     # Model checkpoints
└── dibco_datasets/                 # Sample datasets
```

## 🎮 Quick Start

### 1. Demo Mode (No Model Required)
```bash
# Run demonstration with simulated enhancement
poetry run python scripts/test_patch_demo.py
```

**Output**: Enhanced documents in `demo_outputs/` directory

### 2. Full System (With Model)
```python
from scripts.patch_based_inference import UniversalGANHTRProcessor

# Initialize processor
processor = UniversalGANHTRProcessor(
    model_path="dual_modal_gan/checkpoints/your_model",
    patch_size=(512, 64),
    overlap=0.25,
    strategy='sliding',
    blend_method='linear'
)

# Process single image
result = processor.process_image(
    input_path="path/to/document.png",
    output_path="enhanced_document.png"
)

# Process entire directory
results = processor.process_directory(
    input_dir="documents/",
    output_dir="enhanced_documents/"
)
```

## ⚙️ Configuration Options

### Patch Extraction Parameters
```python
# Patch size - smaller = more detail, larger = faster processing
patch_size = (512, 64)  # Recommended for documents

# Overlap ratio - higher = better quality, slower processing
overlap = 0.25  # 25% overlap recommended

# Extraction strategy
strategy = 'sliding'  # Options: 'sliding', 'grid', 'adaptive'
```

### Strategy Comparison
| Strategy | Pros | Cons | Best For |
|----------|------|------|-----------|
| **Sliding** | Uniform coverage, good quality | More patches, slower | General documents |
| **Grid** | Faster processing | Potential gaps | Simple layouts |
| **Adaptive** | Content-focused, efficient | Complex implementation | Dense text areas |

### Blending Methods
```python
blend_method = 'linear'    # Fast, good quality
blend_method = 'gaussian'  # Smoother transitions, slower
```

## 📊 Performance Analysis

### Patch Size Impact
- **Small patches (256×32)**: High detail, more processing time
  - Patches: ~200-300 for typical documents
  - PSNR: 21+ dB
  - Processing: Slower but more precise

- **Medium patches (512×64)**: Balanced approach
  - Patches: ~40-80 for typical documents
  - PSNR: 19+ dB
  - Processing: Optimal balance

- **Large patches (1024×128)**: Fast processing
  - Patches: ~5-15 for typical documents
  - PSNR: 16+ dB
  - Processing: Fastest, good for large documents

### Memory Usage
- **GPU Memory**: ~2GB per 4 patches (FP32)
- **System Memory**: Depends on image size and patch configuration
- **Batch Processing**: Configurable batch size for memory optimization

## 🎯 Use Cases

### 1. Historical Document Enhancement
```python
# Process old manuscripts with various dimensions
processor = UniversalGANHTRProcessor(
    model_path="checkpoints/historical_model",
    patch_size=(512, 64),
    overlap=0.3  # Higher overlap for delicate documents
)

results = processor.process_directory("historical_documents/")
```

### 2. Batch Document Processing
```python
# Process office documents automatically
processor = UniversalGANHTRProcessor(
    batch_size=8,  # Larger batches for efficiency
    strategy='grid'  # Faster processing for clean documents
)

results = processor.process_directory("office_docs/")
```

### 3. Research and Evaluation
```python
# Test on benchmark datasets (H-DIBCO)
processor = UniversalGANHTRProcessor(
    patch_size=(1024, 128),
    overlap=0.25,
    blend_method='gaussian'  # Best quality for evaluation
)

results = processor.process_directory("dibco_datasets/DIPCO2016_dataset/")
```

## 📈 Quality Metrics

The system provides comprehensive quality assessment:

### Reconstruction Quality
- **PSNR (Peak Signal-to-Noise Ratio)**: Visual quality measure
- **SSIM (Structural Similarity)**: Perceptual quality
- **MSE (Mean Squared Error)**: Pixel-level accuracy
- **MAE (Mean Absolute Error)**: Average deviation

### Expected Performance
- **Excellent**: PSNR > 25 dB, SSIM > 0.9
- **Good**: PSNR > 20 dB, SSIM > 0.8
- **Acceptable**: PSNR > 15 dB, SSIM > 0.7

## 🛠️ Advanced Usage

### Custom Processing Pipeline
```python
# Extract patches manually
from scripts.patch_based_inference import PatchExtractor

extractor = PatchExtractor(
    patch_size=(512, 64),
    overlap=0.25,
    strategy='sliding'
)

patches, positions = extractor.extract_patches(image)
```

### Custom Reconstruction
```python
# Custom reconstruction with specific parameters
from scripts.patch_based_inference import ImageReconstructor

reconstructor = ImageReconstructor(
    original_size=image.shape,
    patch_size=(512, 64),
    stride=extractor.effective_stride[0],
    blend_method='gaussian'
)

enhanced_image = reconstructor.reconstruct_image(
    processed_patches, positions
)
```

## 🔧 Troubleshooting

### Common Issues

#### 1. No Patches Extracted
**Problem**: Image smaller than patch size
```python
# Solution: Use smaller patches or resize image
processor = UniversalGANHTRProcessor(
    patch_size=(256, 32),  # Smaller patches
    overlap=0.1
)
```

#### 2. Memory Issues
**Problem**: GPU memory insufficient
```python
# Solution: Reduce batch size or patch size
processor = UniversalGANHTRProcessor(
    batch_size=2,  # Smaller batches
    patch_size=(256, 32)  # Smaller patches
)
```

#### 3. Model Loading Errors
**Problem**: Incompatible model architecture
```python
# Solution: Use automatic architecture detection
processor = UniversalGANHTRProcessor(
    model_path="path/to/checkpoint",
    # System will try multiple architectures automatically
)
```

### Performance Optimization

#### For Large Documents
```python
# Optimize for speed
processor = UniversalGANHTRProcessor(
    patch_size=(1024, 128),  # Large patches
    overlap=0.1,              # Low overlap
    strategy='grid',          # Faster extraction
    batch_size=8              # Larger batches
)
```

#### For High Quality
```python
# Optimize for quality
processor = UniversalGANHTRProcessor(
    patch_size=(256, 32),    # Small patches
    overlap=0.4,              # High overlap
    strategy='sliding',       # Uniform coverage
    blend_method='gaussian'   # Smooth blending
)
```

## 📚 Integration Examples

### Command Line Interface
```bash
# Process single image
poetry run python scripts/patch_inference_cli.py \
    --input document.png \
    --output enhanced_document.png \
    --patch-size 512 64 \
    --overlap 0.25

# Process directory
poetry run python scripts/patch_inference_cli.py \
    --input-dir documents/ \
    --output-dir enhanced_documents/ \
    --strategy sliding \
    --blend-method gaussian
```

### Web API Integration
```python
from flask import Flask, request, jsonify
from scripts.patch_based_inference import UniversalGANHTRProcessor

app = Flask(__name__)
processor = UniversalGANHTRProcessor(model_path="checkpoints/model")

@app.route('/enhance', methods=['POST'])
def enhance_document():
    file = request.files['image']
    result = processor.process_image(file)
    return jsonify(result)
```

## 🎓 Research Applications

### Benchmark Testing
```python
# Test on multiple datasets
datasets = ['H-DIBCO2016', 'DIBCO2017', 'DIBCO2019']
results = {}

for dataset in datasets:
    processor = UniversalGANHTRProcessor(
        patch_size=(512, 64),
        overlap=0.25
    )

    dataset_results = processor.process_directory(f"datasets/{dataset}")
    results[dataset] = dataset_results

# Generate research report
generate_research_report(results)
```

### Ablation Studies
```python
# Test different configurations
configs = [
    {'patch_size': (256, 32), 'overlap': 0.1},
    {'patch_size': (512, 64), 'overlap': 0.25},
    {'patch_size': (1024, 128), 'overlap': 0.4}
]

for config in configs:
    processor = UniversalGANHTRProcessor(**config)
    results = processor.process_directory("test_documents/")
    analyze_performance(config, results)
```

## 📖 References and Further Reading

1. **Patch-Based Processing**: Common technique in computer vision for large image processing
2. **Image Blending**: Seam blending and multi-band blending for seamless reconstruction
3. **Document Enhancement**: GAN-based approaches for document restoration
4. **H-DIBCO Dataset**: Handwritten Document Image Binarization Competition benchmark

## 🤝 Contributing

To extend the system:

1. **Add New Extraction Strategies**: Implement in `PatchExtractor` class
2. **Custom Blending Methods**: Add to `ImageReconstructor` class
3. **Quality Metrics**: Extend `calculate_reconstruction_quality` method
4. **Model Support**: Add architectures to `_load_model` method

## 📄 License

This system is part of the GAN-HTR research project. See project license for details.

---

**🎉 Ready to enhance any document!**

The Patch-Based Inference System transforms your GAN-HTR model from a fixed-size processor into a universal document enhancement tool capable of handling any document size or format.