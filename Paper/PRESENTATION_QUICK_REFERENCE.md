# NARASI CEPAT - ARSITEKTUR RESTORASI DOKUMEN
## Versi 2 Menit (Emergency Backup)

---

Baik, saya jelaskan arsitektur sistem restorasi dokumen kami.

**[INPUT]**
Sistem menerima dokumen terdegradasi sebagai input.

**[GENERATOR - 21.8M params]**
Input diproses oleh Generator U-Net Enhanced dengan encoder-decoder architecture, menghasilkan enhanced image.

**[DISCRIMINATOR DUAL-MODAL - 17.4M params]**
Komponen kunci kami: Discriminator yang mengevaluasi DUA aspek—visual via CNN dan tekstual via BiLSTM, digabung dengan bilateral cross-modal attention. Ini memastikan hasil realistic dan text-consistent.

**[HTR RECOGNIZER - FROZEN]**
HTR frozen memberikan tiga hal: CTC gradient untuk readability, recognition features untuk consistency, dan predicted text untuk discriminator.

**[MULTI-COMPONENT LOSS]**
Lima loss components bekerja bersama: pixel, adversarial, perceptual, recognition feature, dan CTC. Yang penting, CTC gradient hanya ke generator, bukan discriminator.

**[NOVELTY]**
Tiga kontribusi utama:
1. Bilateral cross-modal attention untuk dual-modal discriminator
2. Frozen HTR sebagai multi-purpose guide
3. HTR-oriented multi-component loss dengan adaptive balancing

Hasilnya: dokumen yang terlihat bagus DAN tetap readable untuk HTR.

Terima kasih.

---

## KEY NUMBERS UNTUK DIINGAT

- **39.2M** total parameters (trainable)
- **21.8M** Generator parameters
- **17.4M** Discriminator (52% reduction)
- **33.72%** HTR baseline CER (frozen)
- **5** loss components
- **2** modalities (visual + textual)
- **3** HTR functions (CTC + RecFeat + PredText)

---

## ONE-LINER UNTUK SETIAP KOMPONEN

**Generator**: "U-Net Enhanced V2 dengan RDB dan CBAM untuk multi-scale reconstruction"

**Discriminator**: "Dual-modal dengan bilateral attention—CNN untuk visual, BiLSTM untuk text"

**HTR**: "Frozen recognizer sebagai fixed readability proxy dengan triple functions"

**Loss**: "Lima komponen balanced: visual quality + semantic + text readability"

**Novelty**: "Cross-modal fusion + frozen HTR guidance + task-oriented optimization"

---

## ELEVATOR PITCH (30 DETIK)

"Kami kembangkan arsitektur GAN dual-modal untuk restorasi dokumen kuno yang berorientasi HTR. Berbeda dari pendekatan konvensional yang fokus visual saja, kami gunakan discriminator yang mengevaluasi visual DAN textual features dengan bilateral attention, dipandu oleh frozen HTR recognizer. Hasilnya: dokumen yang tidak hanya terlihat bagus, tapi juga optimal untuk text recognition. Tiga novelty utama: cross-modal attention, frozen HTR multi-purpose, dan HTR-oriented loss balancing."

---

## CLOSING STATEMENT OPTIONS

**Opsi 1 (Formal):**
"Dengan demikian, arsitektur yang kami usulkan memberikan solusi komprehensif untuk restorasi dokumen yang mempertimbangkan downstream task HTR, achieving balance between visual quality and text readability."

**Opsi 2 (Impact-focused):**
"Sistem kami membuktikan bahwa restorasi dokumen tidak cukup hanya 'terlihat bagus'—harus juga 'terbaca dengan baik'. Arsitektur dual-modal kami mencapai keduanya."

**Opsi 3 (Technical):**
"Integrasi bilateral cross-modal attention dengan frozen HTR guidance membuka paradigma baru dalam task-oriented document restoration, di mana downstream application menjadi integral part dari optimization process."

---

## RESPONSE TEMPLATE UNTUK Q&A

**Template Umum:**
1. "Terima kasih atas pertanyaannya"
2. [Restate pertanyaan untuk konfirmasi]
3. [Jawaban singkat - core point]
4. [Elaborasi jika perlu]
5. "Apakah menjawab pertanyaan Bapak/Ibu?"

**Contoh:**
Q: "Mengapa menggunakan frozen HTR?"

A: "Terima kasih Pak. Jadi pertanyaannya kenapa HTR frozen ya. Ada tiga alasan utama: Pertama, sudah pre-trained dengan performa bagus CER 33.72%. Kedua, kita butuh fixed proxy untuk consistent readability measurement. Ketiga, menghemat computational resources dan mencegah catastrophic forgetting. Frozen HTR ini justru strength kami, karena bertindak sebagai reliable judge untuk text readability. Apakah menjawab pertanyaan Bapak?"
