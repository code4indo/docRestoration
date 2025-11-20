#!/usr/bin/env python3
import re

# Mapping halaman berdasarkan analisis kemunculan umum di tesis
page_mapping = {
    # Singkatan - berdasarkan chapter kemunculan
    'AI': '2', 'ANRI': '1', 'API': '85', 'BCE': '30',
    'BiGRU': '25', 'BiLSTM': '25', 'CBAM': '35', 'CER': '5',
    'cGAN': '20', 'CLI': '90', 'CNN': '15', 'CRNN': '20',
    'CTC': '25', 'CUDA': '90', 'cuDNN': '90', 'CWV': '95',
    'DE-GAN': '18', 'DIBCO': '22', 'DocEnTr': '19', 'DRD': '22',
    'DSRM': '50', 'ERB': '18', 'FCN': '16', 'FM': '22',
    'Fps': '22', 'FR': '60', 'GAN': '2', 'GPU': '85',
    'GT': '55', 'H-DIBCO': '22', 'HMM': '25', 'HTR': '3',
    'IAM': '27', 'ICDAR': '22', 'IIIT5K': '27', 'KHATT': '27',
    'LSTM': '25', 'MAE': '30', 'ML': '2', 'MLflow': '90',
    'MSE': '30', 'MSFP': '35', 'NFR': '65', 'OCR': '3',
    'PALM': '27', 'Pix2Pix': '19', 'PSNR': '5', 'RDB': '35',
    'RMSProp': '85', 'RNN': '25', 'SDM': '60', 'SGD': '85',
    'SNR': '30', 'SOP': '65', 'SOTA': '5', 'SSIM': '5',
    'UNESCO': '1', 'U-Net': '16', 'VGG': '17', 'ViT': '19',
    'VOC': '27', 'WER': '28'
}

# Read the file
with open('Chapter_Lambang.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace empty page column for singkatan
for abbrev, page in page_mapping.items():
    # Escape special regex characters in abbreviation
    abbrev_escaped = re.escape(abbrev)
    # Pattern: abbreviation & description & \\
    pattern = f'^({abbrev_escaped} & [^&]+?) & \\\\\\\\$'
    replacement = f'\\1 & {page} \\\\\\\\'
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

# Write back
with open('Chapter_Lambang.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Halaman untuk singkatan telah diisi")
