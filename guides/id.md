# Cara Mengimplementasikan Dual Reader Standard

**Panduan lengkap untuk membangun klaim yang dapat diverifikasi mesin untuk makalah ilmiah, perangkat lunak, dan sistem pengetahuan.**

> Lihat juga: [id.json](id.json) — Versi data terstruktur JSON-LD dari panduan ini untuk sistem AI dan perayap web.

---

## Apa Itu Dual Reader Standard?

**Dual Reader Standard (DRS)** adalah arsitektur verifikasi untuk sistem pengetahuan. Setiap klaim — baik yang ditulis dalam prosa maupun diimplementasikan dalam kode — harus dapat dibaca oleh dua kelas pembaca independen: **manusia** dan **mesin**.

DRS terdiri dari dua protokol:

| Protokol | Domain | Fungsi |
|----------|--------|--------|
| **DRP** (Dual Reader Protocol) | Teks / Makalah | Menjadikan klaim prosa dapat dievaluasi mesin melalui predikat falsifikasi 5 bagian |
| **GVP** (Grounded Verification Protocol) | Perangkat Lunak / Kode | Menjadikan klaim yang dapat dievaluasi mesin terverifikasi oleh mesin melalui ikatan pengujian dan bukti yang dipatok pada commit |

### Tiga Pembaca

| Pembaca | Kanal | Membaca | Format |
|---------|-------|---------|--------|
| **Manusia** | Prosa | Makalah | Bahasa alami |
| **AI** | JSON | Lapisan AI | Registri klaim terstruktur |
| **CI / pelari pengujian** | Eksekutabel | Ikatan pengujian | ID node pengujian + SHA commit |

---

## Kernel Falsifikasi K = (P, O, M, B)

Fondasi bersama dari kedua protokol. Setiap klaim Tipe F (dapat difalsifikasi) membawa predikat deterministik yang mengevaluasi ke tepat satu dari dua putusan: **FALSIFIED** atau **NOT FALSIFIED**.

| Simbol | Nama | Field JSON | Peran |
|--------|------|------------|-------|
| **P** | Predikat | `FALSIFIED_IF` | Kalimat logis yang, jika TRUE, memfalsifikasi klaim |
| **O** | Operan | `WHERE` | Definisi bertipe dari setiap variabel dalam predikat |
| **M** | Mekanisme | `EVALUATION` | Prosedur evaluasi terbatas dan deterministik |
| **B** | Batas | `BOUNDARY` + `CONTEXT` | Semantik ambang batas dan justifikasi |

### Contoh Predikat

```json
{
  "FALSIFIED_IF": "EXISTS i IN range(len(result)-1) SUCH THAT result[i] > result[i+1]",
  "WHERE": {
    "result": "list · dimensionless · output of sort(input)"
  },
  "EVALUATION": "Run sort on test vectors; check adjacent pairs; finite",
  "BOUNDARY": "len(result) <= 1 → NOT FALSIFIED (vacuously sorted)",
  "CONTEXT": "Ascending order is the documented contract of sort()"
}
```

Predikat ini bermakna sama bagi setiap sistem AI dalam bahasa apa pun. Tidak diperlukan penerjemahan.

### Batasan Predikat

- **Determinisme**: Harus mengevaluasi ke tepat TRUE atau FALSE
- **Keterbatasan**: Kuantifier hanya menjangkau himpunan terbatas
- **Tanpa referensi diri**: Tidak ada dependensi predikat sirkuler
- **Kelengkapan**: Setiap variabel di `FALSIFIED_IF` didefinisikan di `WHERE`, dan sebaliknya

---

## Tiga Tipe Klaim

| Tipe | Nama | Deskripsi | Predikat Diperlukan? |
|------|------|-----------|---------------------|
| **A** | Aksioma | Premis fundamental — tidak dapat difalsifikasi secara sengaja | Tidak (`null`) |
| **D** | Definisi | Definisi stipulatif — tidak bersifat kebenaran | Tidak (`null`) |
| **F** | Dapat Difalsifikasi | Klaim yang dapat diuji dengan predikat deterministik | Ya — K = (P, O, M, B) lengkap |

---

## Enam Tingkat Verifikasi

Setiap klaim membawa field `tier` yang menyatakan jenis bukti apa yang mendasarinya.

### Berlabuh secara Konstruksi

| Tingkat | Berlaku Untuk | Makna |
|---------|---------------|-------|
| `axiom` | Tipe A | Fundamental, tidak dapat difalsifikasi secara sengaja |
| `definition` | Tipe D | Definisional, predikat tidak diperlukan |

### Berlabuh Saat Ini

| Tingkat | Berlaku Untuk | Makna |
|---------|---------------|-------|
| `software_tested` | Tipe F | Diuji oleh pengujian yang berhasil. `test_bindings` tidak kosong, SHA `verified_against` tidak null |
| `formal_proof` | Tipe F | Derivasi berindeks langkah dengan `n_invalid_steps = 0` |
| `analytic` | Tipe F | Diverifikasi melalui jejak derivasi formal atau argumen analitik |

### Secara Eksplisit Belum Berlabuh

| Tingkat | Berlaku Untuk | Makna |
|---------|---------------|-------|
| `empirical_pending` | Tipe F | Placeholder aktif atau menunggu data. Kesenjangan terlihat secara sengaja |

---

## Lapisan AI — Artefak Sentral

Lapisan AI adalah dokumen JSON yang menyertai setiap makalah atau sistem perangkat lunak. Inilah yang dioperasikan oleh kedua protokol.

### Bagian yang Diperlukan

| Bagian | Tujuan |
|--------|--------|
| `_meta` | Tipe dokumen, versi skema, sesi, lisensi |
| `paper_id` | Identifier unik |
| `paper_title` | Judul yang dapat dibaca manusia |
| `paper_type` | `law_A`, `derivation_B`, `application_C`, atau `methodology_D` |
| `phase_ready` | Putusan gerbang fase dan status kondisi (c1–c6) |
| `claim_registry` | Array semua klaim dengan tipe, predikat, tingkat, ikatan |
| `placeholder_register` | Array dependensi yang belum terselesaikan |

### Skema

Skema lapisan AI tersedia di: [`ai-layers/ai-layer-schema.json`](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json)

---

## Gerbang Fase Enam Kondisi

Sebuah makalah atau rilis perangkat lunak berstatus **PHASE-READY** ketika keenam kondisi terpenuhi:

| Kondisi | Persyaratan |
|---------|-------------|
| **c1** | Lapisan AI valid terhadap skema |
| **c2** | Semua klaim yang dapat difalsifikasi terdaftar dengan predikat |
| **c3** | Semua predikat dapat dievaluasi oleh mesin |
| **c4** | Referensi silang dilacak (register placeholder) |
| **c5** | Verifikasi bersifat mandiri (lapisan AI saja, tanpa prosa) |
| **c6** | Semua predikat bersifat non-vakuous (observasi falsifikasi contoh ada) |

---

## Cara Mengimplementasikan

### Untuk Makalah (DRP)

1. Tulis makalah (kanal prosa untuk pembaca manusia)
2. Buat file JSON lapisan AI (kanal mesin untuk pembaca AI)
3. Klasifikasikan setiap klaim sebagai A (aksioma), D (definisi), atau F (dapat difalsifikasi)
4. Tulis predikat falsifikasi 5 bagian untuk setiap klaim Tipe F
5. Sertakan `sample_falsification_observation` untuk setiap klaim Tipe F (saksi vakuitas)
6. Tetapkan tingkat verifikasi untuk setiap klaim
7. Validasi terhadap `ai-layer-schema.json`
8. Jalankan gerbang fase (c1–c6)

### Untuk Perangkat Lunak (GVP)

1. Enumerasikan apa yang diklaim oleh perangkat lunak
2. Klasifikasikan setiap klaim sebagai A (asumsi), D (definisi), atau F (perilaku)
3. Tulis predikat falsifikasi untuk setiap klaim Tipe F
4. Tulis atau identifikasi pengujian yang menguji setiap klaim
5. Isi `test_bindings` dengan ID node pengujian yang berkualifikasi penuh
6. Jalankan pengujian dan catat SHA commit yang berhasil di `verified_against`
7. Tetapkan tingkat: `software_tested` jika pengujian ada, `empirical_pending` jika belum
8. Daftarkan klaim yang belum diuji sebagai placeholder
9. Validasi terhadap `ai-layer-schema.json`

### Adopsi Minimal yang Layak

Adopsi DRS terkecil yang berguna adalah **satu klaim Tipe F dengan satu ikatan pengujian**:

```json
{
  "claim_id": "F-1",
  "type": "F",
  "statement": "sort() returns elements in ascending order",
  "falsification_predicate": {
    "FALSIFIED_IF": "EXISTS i IN range(len(result)-1) SUCH THAT result[i] > result[i+1]",
    "WHERE": {
      "result": "list · dimensionless · output of sort(input)"
    },
    "EVALUATION": "Run sort on test vectors; check adjacent pairs; finite",
    "BOUNDARY": "len(result) <= 1 → NOT FALSIFIED (vacuously sorted)",
    "CONTEXT": "Ascending order is the documented contract of sort()"
  },
  "tier": "software_tested",
  "test_bindings": ["tests/test_sort.py::test_ascending_order"],
  "verified_against": "abc1234"
}
```

Satu klaim. Satu pengujian. Satu SHA. DRS sudah aktif. Tambahkan lebih banyak klaim ketika nilainya membenarkan biayanya.

---

## Prinsip Desain

**Epistemologi Popperian.** Kita dapat memfalsifikasi tetapi tidak dapat memverifikasi. Klaim yang bertahan dari semua upaya falsifikasi tidak terbukti — ia telah bertahan.

**Kesenjangan yang jujur.** Placeholder adalah fitur terpenting. Ketika sebuah klaim berstatus `empirical_pending`, sistem berkata: "kami mengklaim ini tetapi belum memverifikasinya." Ini secara tegas lebih informatif daripada alternatifnya, di mana klaim yang belum terverifikasi tidak dapat dibedakan dari yang sudah terverifikasi.

**Independensi substrat.** Kernel K = (P, O, M, B) tidak mengetahui apakah ia mengevaluasi teorema ilmiah atau jaminan perangkat lunak. Domain masa depan (hukum, regulasi, kebijakan) dapat menambahkan protokol mereka sendiri tanpa memodifikasi kernel.

**Lingua franca mesin.** Kernel ditulis dalam logika dan matematika, bukan bahasa manusia mana pun. Predikat yang mengevaluasi ke TRUE di Beijing juga mengevaluasi ke TRUE di Boston. JSON adalah lapisan transpor. Logika biner adalah substratnya.

---

## Kompatibilitas Tiga Sumbu

| Sumbu | Janji | Mekanisme |
|-------|-------|-----------|
| **Masa Lalu** | Tidak ada yang sudah dilakukan rusak | Pemversian skema, enum append-only, kernel permanen |
| **Lateral** | Berfungsi di semua domain, bahasa, alat, sistem AI | Kernel yang independen substrat, ikatan bertipe string |
| **Masa Depan** | Apa pun yang baru dapat ditambahkan tanpa desain ulang | Ekstensibilitas protokol, ekstensibilitas tingkat, evolusi skema aditif |

---

## Sumber Daya

- [Spesifikasi Arsitektur DRS](https://github.com/thomasbrennan/Fracttalix/blob/main/paper/DRS-Architecture.md) — Spesifikasi lengkap
- [Kernel Falsifikasi v1.1](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/falsification-kernel.md) — Spesifikasi semantik Lapisan 0
- [Skema Lapisan AI v3](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json) — JSON Schema
- [Spesifikasi GVP](https://github.com/thomasbrennan/Fracttalix/blob/main/docs/GVP-spec.md) — Protokol perangkat lunak
- [Contoh Lapisan AI (P1)](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/P1-ai-layer.json) — Contoh yang berfungsi
- [Repositori Fracttalix](https://github.com/thomasbrennan/Fracttalix)

---

**Lisensi:** CC BY 4.0 | **DOI:** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299) | **Penulis:** Thomas Brennan ([ORCID](https://orcid.org/0009-0002-6353-7115))
