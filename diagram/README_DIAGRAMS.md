# Architecture Diagrams - Academic ML Standard

## Files Overview

### Generator (U-Net Enhanced)
- **Source**: `generator_enhanced_v2.tex`
- **Output**: `generator_enhanced_v2.pdf` (83 KB)
- **Layout**: Vertical encoder-decoder with skip connections
- **Features**:
  - Color-coded: Blue (encoder), Green (decoder), Orange (attention)
  - 4 encoder blocks + bottleneck + 4 decoder blocks
  - Attention gates on skip connections
  - Dimensions at each layer
  - 21.8M parameters annotated

### Discriminator (Dual-Modal)
- **Source**: `discriminator_dual_modal_v2.tex`
- **Output**: `discriminator_dual_modal_v2.pdf` (64 KB)
- **Layout**: Vertical dual-branch (CNN + BiLSTM)
- **Features**:
  - Color-coded: Blue (CNN), Orange (LSTM), Red (attention), Green (fusion)
  - Clear branch separation with backgrounds
  - Cross-modal attention fusion
  - 17.4M parameters with breakdown

## Design Principles Applied

1. **Color Semantics**: Different colors for different layer types
2. **Vertical Flow**: Top-to-bottom for better readability
3. **Component Grouping**: Background boxes for functional blocks
4. **Annotations**: Dimensions, parameters, operations at each layer
5. **Compact Layout**: Fit within journal column width

## Compilation

```bash
# Compile diagrams
pdflatex generator_enhanced_v2.tex
pdflatex discriminator_dual_modal_v2.tex

# Copy to paper
cp generator_enhanced_v2.pdf ../Paper/main/figures/
cp discriminator_dual_modal_v2.pdf ../Paper/main/figures/
```

## Usage in Paper

```latex
% Generator
\includegraphics[width=0.75\textwidth]{figures/generator_enhanced_v2.pdf}

% Discriminator  
\includegraphics[width=0.65\textwidth]{figures/discriminator_dual_modal_v2.pdf}
```

## References
Design follows conventions from:
- ResNet paper (He et al., 2015)
- U-Net paper (Ronneberger et al., 2015)
- Attention Is All You Need (Vaswani et al., 2017)
