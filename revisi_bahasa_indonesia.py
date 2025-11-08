#!/usr/bin/env python3
"""
Script untuk merevisi Chapter 2 agar sesuai standar Bahasa Indonesia ilmiah:
1. Italicize istilah asing tanpa padanan
2. Terjemahkan istilah yang ada padanan Bahasa Indonesia
3. Perbaiki tata bahasa sesuai KBBI
"""

import re

# Mapping istilah yang perlu di-italicize (tanpa padanan baku BI)
ITALICIZE_TERMS = {
    r'\bframework\b': r'\\textit{framework}',
    r'\bbaseline\b': r'\\textit{baseline}',
    r'\bground truth\b': r'\\textit{ground truth}',
    r'\btest set\b': r'\\textit{test set}',
    r'\btraining set\b': r'\\textit{training set}',
    r'\bpipeline\b': r'\\textit{pipeline}',
    r'\bdataset\b(?!s)': r'\\textit{dataset}',  # singular
    r'\bdatasets\b': r'\\textit{datasets}',      # plural
    r'\bfeature map\b': r'\\textit{feature map}',
    r'\bfeature maps\b': r'\\textit{feature maps}',
    r'\bencoder\b': r'\\textit{encoder}',
    r'\bdecoder\b': r'\\textit{decoder}',
    r'\bskip connection\b': r'\\textit{skip connection}',
    r'\bskip connections\b': r'\\textit{skip connections}',
    r'\bresidual block\b': r'\\textit{residual block}',
    r'\bresidual blocks\b': r'\\textit{residual blocks}',
    r'\bresidual connection\b': r'\\textit{residual connection}',
    r'\bresidual connections\b': r'\\textit{residual connections}',
    r'\bgradient flow\b': r'\\textit{gradient flow}',
    r'\bgradient\b(?! flow)': r'\\textit{gradient}',
    r'\bbackpropagation\b': r'\\textit{backpropagation}',
    r'\btraining loop\b': r'\\textit{training loop}',
    r'\brecognizer\b': r'\\textit{recognizer}',
    r'\bfrozen recognizer\b': r'\\textit{frozen recognizer}',
    r'\btrainable\b': r'\\textit{trainable}',
    r'\bfrozen\b(?= recognizer)': r'\\textit{frozen}',
    r'\bend-to-end\b': r'\\textit{end-to-end}',
    r'\bloss function\b': r'\\textit{loss function}',
    r'\bco-adaptation\b': r'\\textit{co-adaptation}',
    r'\bpretext task\b': r'\\textit{pretext task}',
    r'\bpretext tasks\b': r'\\textit{pretext tasks}',
    r'\bself-supervised\b': r'pembelajaran mandiri',
    r'\bself-supervised learning\b': r'pembelajaran mandiri',
    r'\bsupervised learning\b': r'pembelajaran terawasi',
    r'\bunsupervised learning\b': r'pembelajaran tak terawasi',
    r'\bmulti-task\b': r'multi-tugas',
    r'\bmulti-modal\b': r'multi-modal',
    r'\bmultimodal\b': r'\\textit{multimodal}',
}

# Mapping terjemahan (yang ada padanan BI)
TRANSLATIONS = {
    r'\bfeature extraction\b': r'ekstraksi fitur',
    r'\bfeature engineering\b': r'rekayasa fitur',
    r'\bdeep learning\b(?! era| Era)': r'pembelajaran mendalam',
    r'\bDeep Learning\b(?! Era)': r'Pembelajaran Mendalam',
    r'\btraining\b(?! set| loop)': r'pelatihan',
    r'\bTraining\b': r'Pelatihan',
    r'\blearning\b(?! rate)': r'pembelajaran',
    r'\bLearning\b': r'Pembelajaran',
    r'\bpre-training\b': r'pra-pelatihan',
    r'\bPre-training\b': r'Pra-pelatihan',
    r'\bgeneralization capability\b': r'kemampuan generalisasi',
    r'\bpractical utility\b': r'utilitas praktis',
    r'\bsuccessful\b': r'berhasil',
    r'\bimprovement\b': r'peningkatan',
    r'\bexpert assessment\b': r'penilaian ahli',
    r'\breal documents\b': r'dokumen nyata',
    r'\bsynthetic\b': r'sintetis',
    r'\blabeled data\b': r'data berlabel',
    r'\bunlabeled data\b': r'data tanpa label',
    r'\bestablished\b': r'mapan',
    r'\bconstraint\b': r'kendala',
    r'\bConstraint\b': r'Kendala',
    r'\bstability\b': r'stabilitas',
    r'\bStability\b': r'Stabilitas',
    r'\bpreserved\b': r'terjaga',
    r'\bpreservasi\b': r'preservasi',  # keep this
    r'\bcapability\b': r'kemampuan',
    r'\bCapability\b': r'Kemampuan',
}

def revise_file(input_path, output_path=None):
    """Revise the LaTeX file"""
    if output_path is None:
        output_path = input_path
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Apply italicization (avoid already italicized text)
    for term, replacement in ITALICIZE_TERMS.items():
        # Skip if already in textit
        pattern = f'(?<!\\\\textit{{){term}'
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
    
    # Apply translations
    for term, translation in TRANSLATIONS.items():
        content = re.sub(term, translation, content, flags=re.IGNORECASE)
    
    # Count changes
    changes = sum(1 for a, b in zip(original_content, content) if a != b)
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Revisi selesai")
    print(f"  File: {output_path}")
    print(f"  Perubahan: ~{changes} karakter")
    print(f"  Total lines: {len(content.splitlines())}")

if __name__ == '__main__':
    input_file = '/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/dual_modal_gan/docs/chapter2_tinjauan_pustaka.tex'
    
    print("=" * 60)
    print("REVISI BAHASA INDONESIA - CHAPTER 2")
    print("=" * 60)
    print()
    
    revise_file(input_file)
    
    print()
    print("=" * 60)
    print("SELESAI")
    print("=" * 60)
