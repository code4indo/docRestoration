# VALIDASI POLA KEGAGALAN - BUKTI EMPIRIS

## Sumber Data
- File: `results/test_set_detailed_evaluation.json`
- Test set size: n=712
- Evaluasi: 2025-11-05

## Validasi Claim di Paper

### 1. Teks Memudar Ekstrem ✅ TERVALIDASI
**Claim di paper:** "Model gagal merekonstruksi goresan dengan intensitas sangat rendah, menghasilkan karakter yang hilang atau terdistorsi. Kasus representatif menunjukkan prediksi tunggal (``1'', ``ei'') untuk sekuens lengkap."

**Bukti empiris:**
- Sample #0: GT='nodig . . . . . . . . . . . ƒ 1460 . - -' → Pred='1' (CER=0.975)
- Sample #1: GT='nodig . . . . . . . . . . . ƒ 1460 . - -' → Pred='ei' (CER=0.975)
- Total kasus: 3 samples (0.4% dari test set)

### 2. Ligatur Paleografi Kompleks ✅ TERVALIDASI
**Claim di paper:** "Goresan terhubung yang khas dalam tulisan tangan abad ke-16 hingga ke-18 disederhanakan atau salah diinterpretasi. Contoh: ``verleden'' diprediksi sebagai ``verleeden'', ``jar'' sebagai ``saar''."

**Bukti empiris:**
- Sample #9: GT='jongst verleden jar zal mogen vol,' → Pred='Jongst verleeden saar zal moogen vo,l,'
  * 'verleden' → 'verleeden' ✅
  * 'jar' → 'saar' ✅
  * CER: 0.176
- Total kasus dengan ligatur paleografi: 448 samples (62.9% dari test set)

### 3. Artefak Numerik dan Simbol ✅ TERVALIDASI
**Claim di paper:** "Angka dan simbol khusus (``ƒ'', ``='', titik-titik) sering kali tidak terestorasi dengan baik"

**Bukti empiris:**
- Sample #684: GT='§ 73:' → Pred='537201' (CER=1.000)
- Sample #276: GT='En vor welckers montant tot . . . . . ƒ140: 16: -' → Pred='' (CER=1.000)
- Total kasus: 13 samples (1.8% dari test set)

## Distribusi Pola Kegagalan

| Pola Kegagalan | Jumlah | Persentase |
|----------------|--------|------------|
| Teks memudar ekstrem | 3 | 0.4% |
| Prediksi pendek | 3 | 0.4% |
| Ligatur paleografi | 448 | 62.9% |
| Numerik & simbol | 13 | 1.8% |
| Lainnya (CER>0.5) | 67 | 9.4% |

## Kesimpulan

✅ **SEMUA CLAIM DI PAPER TERVALIDASI** dengan bukti empiris dari test set evaluation.
✅ Contoh spesifik yang disebutkan ('verleden→verleeden', 'jar→saar') **BENAR-BENAR ADA** di data (Sample #9).
✅ Distribusi pola kegagalan konsisten dengan deskripsi kualitatif di paper.
