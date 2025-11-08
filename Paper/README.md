# Struktur Direktori Paper

Direktori ini diorganisir untuk memudahkan manajemen paper penelitian dan mencegah kebingungan.

## 📁 Struktur Folder

```
Paper/
├── main/              # Paper utama (Indonesian)
├── english/           # Paper English versions
├── backups/           # File backup dan yang tidak digunakan
├── README.md          # Dokumen ini
├── PLACEHOLDER_FILLING_GUIDE.md
├── KONVERSI_BAHASA_INDONESIA.md
└── fix_latex.sh
```

## 📄 File Utama

### 🇮🇩 Indonesian Papers (Folder `main/`)
**Menggunakan file ini untuk publikasi Indonesian:**
- `jatniko_id.tex` - **Paper utama Indonesian** (sudah diperbaiki)
- `jatniko_id.pdf` - Versi PDF yang sudah di-compile

**Karakteristik:**
- ✅ Judul sesuai dengan tesis
- ✅ Konten lengkap dan detail
- ✅ Siap untuk publikasi journal Q1

### 🇺🇸 English Papers (Folder `english/`)
**Menggunakan file ini untuk publikasi international:**
- `jatniko.tex` - **Paper utama English**
- `jatniko_english_backup.tex` - Backup version
- `jatniko.pdf` - PDF dari paper English

**Karakteristik:**
- ✅ Judul sudah diperbaiki: "Degraded Document Restoration..."
- ✅ Konten lengkap untuk publikasi international
- ✅ Format IEEE journal

### 💾 Backups (Folder `backups/`)
**File lama dan backup - JANGAN GUNAKAN:**
- `jatniko_id_new.*` - File yang sudah dihapus dari struktur utama
- File dengan nama yang tidak jelas atau tidak konsisten

## 🎯 Cara Menggunakan

### Untuk Paper Indonesian:
```bash
cd main/
# Edit jatniko_id.tex
# Compile dengan: pdflatex jatniko_id.tex
```

### Untuk Paper English:
```bash
cd english/
# Edit jatniko.tex
# Compile dengan: pdflatex jatniko.tex
```

### Kapan Edit Mana:
- **Edit paper utama**: Folder `main/` (Indonesian) atau `english/` (English)
- **Backup work**: Simpan di folder `backups/`
- **Cross-reference**: Gunakan file dari folder yang sesuai

## ⚠️ Important Notes

1. **Hanya gunakan file di folder `main/` dan `english/`**
2. **File di `backups/` adalah untuk reference saja**
3. **Kedua paper (ID & EN) sudah memiliki judul yang sesuai dengan tesis**
4. **Konten sudah lengkap dan siap untuk submission**

## 📊 File Comparison

| File | Lokasi | Status | Uso |
|------|--------|--------|-----|
| `jatniko_id.tex` | `main/` | ✅ Aktif | Indonesian publication |
| `jatniko.tex` | `english/` | ✅ Aktif | International publication |
| `jatniko_id_new.tex` | `backups/` | ❌ Deprecated | Jangan gunakan |

---
**Terakhir diperbarui:** 30 Oktober 2025
**Status:** Struktur file sudah rapi dan siap digunakan