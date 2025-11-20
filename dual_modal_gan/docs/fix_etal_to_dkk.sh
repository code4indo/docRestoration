#!/bin/bash
# Script untuk mengganti "et al" dengan "dkk" di file .bbl setelah biber generate
# Usage: ./fix_etal_to_dkk.sh

set -e

BBL_FILE="main_tesis.bbl"

if [ ! -f "$BBL_FILE" ]; then
    echo "Error: $BBL_FILE tidak ditemukan"
    exit 1
fi

echo "Mem-backup $BBL_FILE..."
cp "$BBL_FILE" "${BBL_FILE}.backup"

echo "Mengganti 'et al' dengan 'dkk' di $BBL_FILE..."
sed -i 's/et al\./dkk./g' "$BBL_FILE"
sed -i 's/et al\\\adddot /dkk\\adddot /g' "$BBL_FILE"

echo "✓ Selesai. File $BBL_FILE sudah dimodifikasi."
echo "Backup disimpan di ${BBL_FILE}.backup"
