#!/bin/bash
# Script untuk sync content dari semua chapter*.tex ke *_content_only.tex
# Usage: ./sync_all_chapters.sh [chapter_number]
#        ./sync_all_chapters.sh      # sync all chapters
#        ./sync_all_chapters.sh 5    # sync chapter 5 only

set -e

DOCS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DOCS_DIR"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          🔄 SYNC ALL CHAPTERS → content_only.tex files          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Function to sync a single chapter
sync_chapter() {
    local chapter_num=$1
    local chapter_name=$2
    local source_file="${chapter_name}.tex"
    local target_file="${chapter_name}_content_only.tex"
    
    # Check if source exists
    if [ ! -f "$source_file" ]; then
        echo "⚠️  Skipping Chapter $chapter_num: $source_file not found"
        return
    fi
    
    # Find \begin{document} line
    local begin_line=$(grep -n '\\begin{document}' "$source_file" | head -1 | cut -d: -f1)
    
    # Find \end{document} line
    local end_line=$(grep -n '\\end{document}' "$source_file" | tail -1 | cut -d: -f1)
    
    if [ -z "$begin_line" ] || [ -z "$end_line" ]; then
        echo "⚠️  Skipping Chapter $chapter_num: No \\begin{document} or \\end{document} found"
        echo "     (This might be a content-only file already)"
        return
    fi
    
    # Calculate content lines (exclude \begin{document} and \end{document})
    local start_line=$((begin_line + 1))
    local content_end_line=$((end_line - 1))
    
    # Backup if target exists
    if [ -f "$target_file" ]; then
        local backup_file="${target_file}.backup_$(date +%Y%m%d_%H%M%S)"
        cp "$target_file" "$backup_file"
        echo "  ✓ Backup: $backup_file"
    fi
    
    # Extract content
    sed -n "${start_line},${content_end_line}p" "$source_file" > "$target_file"
    
    local line_count=$(wc -l < "$target_file")
    echo "  ✓ Chapter $chapter_num synced: $line_count lines → $target_file"
}

# Determine which chapters to sync
if [ $# -eq 0 ]; then
    # Sync all chapters
    CHAPTERS=(
        "1:chapter1_pendahuluan"
        "2:chapter2_tinjauan_pustaka"
        "3:chapter3_metodologi"
        "4:chapter4_analysis_design"
        "5:chapter5_hasil"
        "6:chapter6_kesimpulan"
        "L:Chapter_Lambang"
    )
else
    # Sync specific chapter
    case $1 in
        1) CHAPTERS=("1:chapter1_pendahuluan") ;;
        2) CHAPTERS=("2:chapter2_tinjauan_pustaka") ;;
        3) CHAPTERS=("3:chapter3_metodologi") ;;
        4) CHAPTERS=("4:chapter4_analysis_design") ;;
        5) CHAPTERS=("5:chapter5_hasil") ;;
        6) CHAPTERS=("6:chapter6_kesimpulan") ;;
        L|l) CHAPTERS=("L:Chapter_Lambang") ;;
        *)
            echo "❌ ERROR: Invalid chapter number: $1"
            echo "Usage: $0 [1|2|3|4|5|6|L]"
            exit 1
            ;;
    esac
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 SYNCING CHAPTERS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for chapter in "${CHAPTERS[@]}"; do
    IFS=':' read -r num name <<< "$chapter"
    sync_chapter "$num" "$name"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ SYNC COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  1. Compile: pdflatex main_tesis.tex"
echo "  2. Biber:   biber main_tesis"
echo "  3. Compile: pdflatex main_tesis.tex (2x)"
echo "  4. View:    evince main_tesis.pdf"
echo ""
echo "💡 Tip: Sync specific file only:"
echo "   ./sync_all_chapters.sh L    # sync Chapter_Lambang only"
