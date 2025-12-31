#!/usr/bin/env python3
"""
Audit Daftar Singkatan dan Lambang
==================================
Skrip ini melakukan audit terhadap daftar singkatan dan lambang dengan:
1. Mengekstrak singkatan/lambang beserta halaman yang tercantum
2. Mencari kemunculan pertama di setiap chapter
3. Membandingkan dan melaporkan inkonsistensi
"""

import re
import os
from collections import OrderedDict

# Konfigurasi
DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
LAMBANG_FILE = os.path.join(DOCS_DIR, "Chapter_Lambang_content_only.tex")

# File chapter dengan urutan penomoran halaman
# Berdasarkan struktur main_tesis_duplex.tex:
# - Halaman romawi: cover, abstrak, pengesahan, dedikasi, kata pengantar, 
#   daftar isi, daftar lampiran, daftar gambar, daftar tabel, daftar lambang
# - Halaman arabic dimulai dari: BAB I = halaman 1
CHAPTER_FILES = [
    ("chapter1_pendahuluan_content_only.tex", 1, "BAB I"),
    ("chapter2_tinjauan_pustaka_content_only.tex", None, "BAB II"),  # akan dihitung
    ("chapter3_metodologi_content_only.tex", None, "BAB III"),
    ("chapter4_analysis_design_content_only.tex", None, "BAB IV"),
    ("chapter5_hasil_content_only.tex", None, "BAB V"),
    ("chapter6_kesimpulan_content_only.tex", None, "BAB VI"),
]

def extract_singkatan_from_file():
    """Ekstrak singkatan dan halaman yang tercantum dari file daftar lambang."""
    singkatan = OrderedDict()
    lambang = OrderedDict()
    
    with open(LAMBANG_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    in_singkatan_section = False
    in_lambang_section = False
    
    for line in lines:
        original_line = line.strip()
        
        # Deteksi awal section singkatan (dari komentar)
        if 'DATA SINGKATAN' in original_line:
            in_singkatan_section = True
            in_lambang_section = False
            continue
        
        # Deteksi awal section lambang (dari komentar)  
        if 'DATA LAMBANG' in original_line or 'subsec:daftar-lambang' in original_line:
            in_singkatan_section = False
            in_lambang_section = True
            continue
        
        # Skip baris kosong dan komentar untuk parsing data
        if original_line.startswith('%') or not original_line:
            continue
        
        # Deteksi akhir tabel
        if r'\end{longtable}' in original_line:
            if in_singkatan_section:
                in_singkatan_section = False
            elif in_lambang_section:
                in_lambang_section = False
            continue
        
        if in_singkatan_section:
            # Pattern singkatan: AI & Description & 2 \\
            # Bisa juga dimulai dengan huruf kecil (cGAN, cuDNN)
            match = re.match(r'^([A-Za-z][A-Za-z0-9\-]*)\s*&\s*(.+?)\s*&\s*(\d+)\s*\\\\', original_line)
            if match:
                abbr = match.group(1).strip()
                name = match.group(2).strip()
                page = int(match.group(3).strip())
                singkatan[abbr] = {'name': name, 'claimed_page': page}
        
        if in_lambang_section:
            # Pattern lambang (mulai dengan $)
            match = re.match(r'^(\$.+?\$.*?)\s*&\s*(.+?)\s*&\s*(\d+)\s*\\\\', original_line)
            if match:
                symbol = match.group(1).strip()
                name = match.group(2).strip()
                page = int(match.group(3).strip())
                lambang[symbol] = {'name': name, 'claimed_page': page}
    
    return singkatan, lambang

def count_lines_approx(filename):
    """Hitung perkiraan jumlah halaman berdasarkan baris (asumsi ~40 baris/halaman)."""
    try:
        with open(os.path.join(DOCS_DIR, filename), 'r', encoding='utf-8') as f:
            # Hitung baris non-kosong dan non-komentar
            lines = [l for l in f.readlines() if l.strip() and not l.strip().startswith('%')]
            # Perkiraan: 40 baris per halaman (dengan ruang untuk gambar/tabel)
            return max(1, len(lines) // 40)
    except:
        return 10  # default

def find_first_occurrence(term, chapters_info):
    """
    Temukan kemunculan pertama dari term di seluruh dokumen.
    Returns: (chapter, line_number, estimated_page, context)
    """
    for filename, start_page, chapter_name in chapters_info:
        filepath = os.path.join(DOCS_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                # Skip komentar LaTeX
                if line.strip().startswith('%'):
                    continue
                
                # Untuk singkatan, cari sebagai kata utuh
                # Gunakan word boundary yang tepat untuk LaTeX
                pattern = r'(?<![A-Za-z])' + re.escape(term) + r'(?![A-Za-z])'
                if re.search(pattern, line):
                    # Estimasi halaman: start_page + (line_number / 40)
                    estimated_page = start_page + (i // 45)
                    context = line.strip()[:80]
                    return (chapter_name, i+1, estimated_page, context)
        except FileNotFoundError:
            continue
    
    return None

def read_lof_lot_for_page_mapping():
    """Baca file .lof dan .lot untuk mendapatkan mapping halaman yang tepat."""
    page_mapping = {}
    
    # Baca file toc untuk mendapatkan halaman chapter yang tepat
    toc_file = os.path.join(DOCS_DIR, "main_tesis_duplex.toc")
    try:
        with open(toc_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Pattern: \contentsline {section}{\numberline {I}BAB I PENDAHULUAN}{1}
            matches = re.findall(r'\\contentsline\s*\{section\}\{\\numberline\s*\{([IVX]+)\}.*?\}\{(\d+)\}', content)
            for roman, page in matches:
                page_mapping[roman] = int(page)
    except:
        pass
    
    return page_mapping

def main():
    print("=" * 80)
    print("AUDIT DAFTAR SINGKATAN DAN LAMBANG")
    print("=" * 80)
    print()
    
    # Ekstrak data dari file daftar lambang
    singkatan, lambang = extract_singkatan_from_file()
    
    print(f"Total singkatan ditemukan: {len(singkatan)}")
    print(f"Total lambang ditemukan: {len(lambang)}")
    print()
    
    # Baca page mapping dari TOC
    page_mapping = read_lof_lot_for_page_mapping()
    
    # Update chapter start pages berdasarkan TOC jika tersedia
    chapter_pages = {
        'I': 1,
        'II': page_mapping.get('II', 13),
        'III': page_mapping.get('III', 48),
        'IV': page_mapping.get('IV', 62),
        'V': page_mapping.get('V', 102),
        'VI': page_mapping.get('VI', 157),
    }
    
    # Update chapters_info dengan halaman yang benar
    chapters_info = [
        ("chapter1_pendahuluan_content_only.tex", chapter_pages['I'], "BAB I"),
        ("chapter2_tinjauan_pustaka_content_only.tex", chapter_pages['II'], "BAB II"),
        ("chapter3_metodologi_content_only.tex", chapter_pages['III'], "BAB III"),
        ("chapter4_analysis_design_content_only.tex", chapter_pages['IV'], "BAB IV"),
        ("chapter5_hasil_content_only.tex", chapter_pages['V'], "BAB V"),
        ("chapter6_kesimpulan_content_only.tex", chapter_pages['VI'], "BAB VI"),
    ]
    
    print(f"Halaman awal chapter (dari TOC):")
    for chap, page in chapter_pages.items():
        print(f"  BAB {chap}: halaman {page}")
    print()
    
    # Audit singkatan
    print("-" * 80)
    print("AUDIT SINGKATAN")
    print("-" * 80)
    print(f"{'Singkatan':<12} {'Tercantum':<10} {'Ditemukan':<10} {'Status':<10} {'Lokasi'}")
    print("-" * 80)
    
    issues_singkatan = []
    for abbr, info in singkatan.items():
        claimed = info['claimed_page']
        result = find_first_occurrence(abbr, chapters_info)
        
        if result:
            chapter, line, found_page, context = result
            if abs(found_page - claimed) > 3:  # Toleransi 3 halaman
                status = "❌ BEDA"
                issues_singkatan.append((abbr, claimed, found_page, chapter))
            else:
                status = "✓ OK"
            print(f"{abbr:<12} {claimed:<10} {found_page:<10} {status:<10} {chapter}")
        else:
            status = "⚠ TIDAK ADA"
            print(f"{abbr:<12} {claimed:<10} {'?':<10} {status:<10}")
    
    print()
    print("-" * 80)
    print(f"RINGKASAN SINGKATAN: {len(singkatan) - len(issues_singkatan)}/{len(singkatan)} sesuai")
    print("-" * 80)
    
    if issues_singkatan:
        print("\n⚠️  SINGKATAN YANG PERLU DIPERBAIKI:")
        print("-" * 50)
        for abbr, claimed, found, chapter in issues_singkatan:
            print(f"  {abbr}: tercantum hal.{claimed}, seharusnya ~{found} ({chapter})")
    
    # Audit lambang
    print()
    print("-" * 80)
    print("AUDIT LAMBANG (Simbolis - Perlu verifikasi manual)")
    print("-" * 80)
    print("Catatan: Lambang matematis sulit dicari secara otomatis.")
    print("Sebagian besar lambang muncul di BAB IV (Formula Matematis)")
    print("dan BAB V (Analisis Statistik).")
    print()
    
    # Ringkasan statistik lambang
    lambang_pages = [info['claimed_page'] for info in lambang.values()]
    if lambang_pages:
        print(f"Rentang halaman lambang: {min(lambang_pages)} - {max(lambang_pages)}")
        page_75_count = sum(1 for p in lambang_pages if p == 75)
        print(f"Lambang dengan halaman 75: {page_75_count}/{len(lambang_pages)}")
        print()
        print("Catatan: Jika terlalu banyak lambang dengan halaman yang sama (75),")
        print("kemungkinan ini adalah placeholder yang perlu diverifikasi ulang.")
    
    print()
    print("=" * 80)
    print("SELESAI")
    print("=" * 80)
    
    return issues_singkatan

if __name__ == "__main__":
    issues = main()
