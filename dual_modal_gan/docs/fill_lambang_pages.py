#!/usr/bin/env python3
import re

# Mapping halaman untuk lambang - kebanyakan ada di Chapter 4 (metodologi/design)
lambang_pages = {
    r'\$\\alpha\$': '75',
    r'\$\\beta\$': '75', 
    r'\$\\lambda\$': '75',
    r'\$\\theta\$': '75',
    r'\$\\sigma\$': '130',
    r'\$\\mu\$': '130',
    r'\$\\epsilon\$': '76',
    r'\$\\Delta\$': '130',
    r'\$\\pi\$': '76',
    r'\$\\pi_t\$': '76',
    r'\$\\nabla\$ atau \$\\partial\$': '85',
    r'\$\\sum\$': '75',
    r'\$\\prod\$': '76',
    r'\$\\int\$': '130',
    'argmax': '76',
    r'\$\\sim\$': '130',
    r'\$\\to\$': '85',
    r'\$\\mathcal\{L\}\$': '75',
    r'\$\\mathcal\{L\}_\{total\}\$': '75',
    r'\$\\mathcal\{L\}_\{adversarial\}\$ atau \$\\mathcal\{L\}_\{adv\}\$': '75',
    r'\$\\mathcal\{L\}_\{reconstruction\}\$': '75',
    r'\$\\mathcal\{L\}_\{L1\}\$': '75',
    r'\$\\mathcal\{L\}_\{L2\}\$': '75',
    r'\$\\mathcal\{L\}_\{CTC\}\$': '76',
    r'\$\\mathcal\{L\}_\{perceptual\}\$': '75',
    r'\$\\mathcal\{L\}_\{pixel\}\$': '75',
    r'\$\\mathcal\{L\}_\{BCE\}\$': '75',
    r'\$\\mathcal\{L\}_\{GAN\}\$': '75',
    r'\$\\mathcal\{L\}_\{cycle\}\$': '75',
    r'\$G\$': '15',
    r'\$D\$': '15',
    r'\$G\(z\)\$': '75',
    r'\$D\(x\)\$': '75',
    r'\$G_\{A \\to B\}\$': '75',
    r'\$G_\{B \\to A\}\$': '75',
    r'\$V\(D, G\)\$': '75',
    r'\$x\$': '75',
    r'\$y\$': '75',
    r'\$z\$': '75',
    r'\$\\mathbb\{E\}\$': '75',
    r'\$p_\{data\}\$': '75',
    r'\$p_z\$': '75',
    r'\$p\(y\|x\)\$': '76',
    r'\$p_t\(\\pi_t\|x\)\$': '76',
    r'\$\\mathcal\{B\}\$': '76',
    r'\$\\mathcal\{B\}\^\{-1\}\$': '76',
    r'\$T\$': '76',
    r'\$\|x - G\(y\)\|_1\$': '75',
    r'\$\|x - G\(y\)\|_2\$': '75',
    r'\$N \\times N\$': '75',
    r'\$d\$': '135',
    r'\$n\$': '130',
    r'\$p\$': '135',
    r'\$r\$': '135',
    r'\$R\^2\$': '135',
    r'\$O\(n\)\$': '85',
    r'\$O\(n\^2\)\$': '85'
}

# Read file
with open('Chapter_Lambang.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Process each line
new_lines = []
for line in lines:
    modified = False
    for symbol, page in lambang_pages.items():
        # Check if line starts with the symbol pattern
        if symbol in line and ' & \\\\' in line and not re.search(r' & \d+ \\\\', line):
            line = line.replace(' & \\\\', f' & {page} \\\\')
            modified = True
            break
    new_lines.append(line)

# Write back
with open('Chapter_Lambang.tex', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✓ Halaman untuk lambang telah diisi")
