#!/usr/bin/env python3
"""
Professional English translation for GAN-HTR paper
Translates only text content while preserving LaTeX structure
"""

import re
import sys

def translate_title_metadata(text):
    """Translate title, author info, and metadata"""
    
    # Title
    text = text.replace(
        r'\title{GAN-Based Document Restoration with Frozen HTR Recognizer and Multi-Component Loss Optimization for Historical Manuscripts}',
        r'\title{GAN-Based Document Restoration with Frozen HTR Recognizer and Multi-Component Loss Optimization for Historical Manuscripts}'
    )
    
    # Author thanks - translate to English
    text = re.sub(
        r'Jatniko Nur Mutaqin adalah mahasiswa Program Studi \\textit\{Smart System\}, Sekolah Teknik Elektro dan Informatika, Institut Teknologi Bandung, Bandung 40132, Indonesia, dan bekerja di Arsip Nasional Republik Indonesia, Jakarta, Indonesia',
        r'Jatniko Nur Mutaqin is a student in the Smart System Program, School of Electrical Engineering and Informatics, Bandung Institute of Technology, Bandung 40132, Indonesia, and works at the National Archives of the Republic of Indonesia, Jakarta, Indonesia',
        text
    )
    
    text = re.sub(
        r'I Gusti Bagus Baskara Nugraha adalah dosen di Sekolah Teknik Elektro dan Informatika, Institut Teknologi Bandung, Bandung 40132, Indonesia',
        r'I Gusti Bagus Baskara Nugraha is a lecturer at the School of Electrical Engineering and Informatics, Bandung Institute of Technology, Bandung 40132, Indonesia',
        text
    )
    
    # Header
    text = text.replace(
        r'Mutaqin \MakeLowercase{\textit{et al.}}: GAN-Based Document Restoration with Frozen HTR Recognizer for Historical Manuscripts',
        r'Mutaqin \MakeLowercase{\textit{et al.}}: GAN-Based Document Restoration with Frozen HTR Recognizer for Historical Manuscripts'
    )
    
    # Manuscript dates - translate
    text = text.replace('Naskah diterima [Tanggal]; direvisi [Tanggal]', 
                       'Manuscript received [Date]; revised [Date]')
    
    return text

def translate_abstract_keywords(text):
    """Translate abstract and keywords to English"""
    
    # Abstract opening
    abstract_pattern = r'\\begin\{abstract\}(.*?)\\end\{abstract\}'
    
    english_abstract = r"""\begin{abstract}
Historical handwritten documents suffer from various degradation artifacts that affect Handwritten Text Recognition (HTR) systems. Conventional document restoration methods focus on visual quality but often fail to preserve text readability for HTR. This paper proposes an HTR-oriented document restoration framework based on Generative Adversarial Networks (GAN) with two main contributions: (1) integration of a frozen HTR recognizer providing stable text-aware gradients without joint-training instability, and (2) multi-component loss function optimization through systematic ablation studies. Unlike conventional approaches using only image-level supervision, the proposed method explicitly optimizes text readability through gradients from a frozen pre-trained recognizer. Quantitative evaluation was conducted on a realistic semi-synthetic dataset constructed from authentic 16th--18th century paleographic documents from the National Archives of the Republic of Indonesia (ANRI) with controlled degradation augmentation to enable paired ground truth evaluation. On 712 test images, the proposed framework achieved a Character Error Rate (CER) of 34.9\%, reducing CER from 83.4\% (degraded condition) with a relative reduction of 58.2\%, while maintaining high visual quality (PSNR 30.74~dB, SSIM 0.987). This result approaches the theoretical maximum performance on clean images (CER 34.1\%). Qualitative validation on 15 authentic ANRI manuscripts with historical degradation demonstrates effective generalization to damage patterns that cannot be synthetically replicated. Ablation studies reveal that the 4-component loss configuration (pixel, adversarial, perceptual, CTC) achieves optimal balance, while exploration of dual-modal discriminator shows marginal contribution ($\Delta$PSNR +0.28~dB, $p>0.05$), confirming that recognizer freezing strategy and loss optimization are more dominant than discriminator architecture complexity.
\end{abstract}"""
    
    text = re.sub(abstract_pattern, english_abstract, text, flags=re.DOTALL)
    
    # Keywords
    keywords_pattern = r'\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}'
    
    english_keywords = r"""\begin{IEEEkeywords}
Generative Adversarial Networks, Document Restoration, Handwritten Text Recognition, Loss Function Optimization, Historical Document Processing, Deep Learning.
\end{IEEEkeywords}"""
    
    text = re.sub(keywords_pattern, english_keywords, text, flags=re.DOTALL)
    
    return text


# Main execution
if __name__ == "__main__":
    input_file = "paper_english.tex"
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply translations
        content = translate_title_metadata(content)
        content = translate_abstract_keywords(content)
        
        # Translate PDF metadata
        content = content.replace(
            r'pdftitle={Restorasi Dokumen Terdegradasi Menggunakan GAN dengan Diskriminator Dual-Modal dan Optimasi Loss Function Berorientasi HTR}',
            r'pdftitle={GAN-Based Document Restoration with Frozen HTR Recognizer and Multi-Component Loss Optimization}'
        )
        
        content = content.replace('pdfauthor={[Nama Anda]}', 
                                 'pdfauthor={Jatniko Nur Mutaqin, I Gusti Bagus Baskara Nugraha}')
        
        # Write translated version
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Successfully translated title, abstract, keywords, and metadata")
        print(f"✓ File updated: {input_file}")
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)

