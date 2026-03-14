# Dual Reader Standard — Panduan Lengkap

**Panduan satu kali baca untuk seluruh Dual Reader Standard: arsitektur, protokol, kernel, lapisan, kompatibilitas, lingua franca, dan adopsi.**

> Lihat juga: [id.json](id.json) — Versi data terstruktur JSON-LD dari panduan ini untuk sistem AI dan perayap web.

---

## 1. Apa Itu Dual Reader Standard

**Dual Reader Standard (DRS)** adalah arsitektur verifikasi untuk sistem pengetahuan. Standar ini mengharuskan setiap klaim — baik yang ditulis dalam prosa maupun diimplementasikan dalam kode — dapat dibaca oleh dua kelas pembaca independen: **manusia** dan **mesin**.

DRS bukan sebuah makalah. DRS bukan sebuah alat. DRS adalah standar yang memuat kedua protokolnya:

- **DRP** (Dual Reader Protocol) — protokol untuk teks
- **GVP** (Grounded Verification Protocol) — protokol untuk perangkat lunak

DRP menjadikan klaim *dapat dievaluasi mesin*. GVP menjadikannya *terverifikasi oleh mesin*. Keduanya tidak lengkap tanpa yang lain. Bersama-sama mereka adalah DRS.

---

## 2. Dua Protokol

### 2.1 DRP — Dual Reader Protocol (Teks)

DRP mengatur bagaimana klaim prosa menjadi dapat dievaluasi mesin.

**Pembaca 1 (Manusia):** Membaca makalah dalam bahasa alami. Memahami konteks, motivasi, dan narasi. Tidak dapat mengaudit setiap klaim secara sistematis.

**Pembaca 2 (AI):** Membaca lapisan AI — dokumen JSON terstruktur yang menyertai setiap makalah. Berisi registri klaim lengkap. Dapat mengaudit setiap klaim tanpa membaca prosa.

DRP memerlukan:

1. **Klasifikasi klaim.** Setiap klaim diketikkan sebagai A (aksioma), D (definisi), atau F (dapat difalsifikasi).
2. **Predikat falsifikasi.** Setiap klaim Tipe F membawa predikat deterministik 5 bagian.
3. **Gerbang fase.** Enam kondisi (c1–c6) yang harus dipenuhi sebelum sebuah makalah dinyatakan PHASE-READY.
4. **Pelacakan placeholder.** Klaim yang bergantung pada hasil yang belum terselesaikan didaftarkan sebagai placeholder — menjadikan kesenjangan terlihat, bukan tersembunyi.

**Yang dijamin DRP:** Setiap sistem AI dengan akses ke lapisan AI dapat mengevaluasi klaim yang dapat difalsifikasi tanpa membaca prosa. Kemandirian adalah persyaratan desain yang ditegakkan pada gerbang fase (kondisi c5).

### 2.2 GVP — Grounded Verification Protocol (Perangkat Lunak)

GVP mengatur bagaimana klaim yang dapat dievaluasi mesin menjadi terverifikasi oleh mesin.

**Pembaca 3A (Pengembang):** Membaca field `tier` untuk memahami jenis bukti apa yang ada. Membaca `test_bindings` untuk mengetahui pengujian mana yang menguji klaim mana. Membaca `verified_against` untuk mengetahui kapan pengujian tersebut terakhir berhasil.

**Pembaca 3B (Mesin):** Menjalankan pelari pengujian terhadap array `test_bindings`. Mencatat berhasil/gagal. Mencap SHA `verified_against` jika berhasil.

GVP mengharuskan setiap klaim membawa tiga field:

1. **`tier`** — tingkat verifikasi (salah satu dari enam nilai)
2. **`test_bindings`** — array ID node pengujian berkualifikasi penuh yang menguji klaim
3. **`verified_against`** — SHA commit git saat pengujian tersebut terakhir berhasil

**Yang dijamin GVP:** Untuk setiap klaim dalam korpus, Anda dapat menentukan (a) jenis bukti apa yang mendasarinya, (b) pengujian eksekutabel mana yang mengujinya, dan (c) pada commit mana pengujian tersebut terakhir berhasil. Jika jawaban untuk (a) adalah `empirical_pending`, kesenjangan terlihat. Jika jawaban untuk (c) adalah `null`, tidak ada pengujian perangkat lunak yang mencakupnya.

---

## 3. Fondasi Bersama — Kernel Falsifikasi (Lapisan 0)

Kedua protokol beroperasi pada fondasi yang sama: **Kernel Falsifikasi K = (P, O, M, B)**.

Ini adalah **Lapisan 0** dari DRS. Kernel ini mendefinisikan apa arti predikat falsifikasi secara independen dari format serialisasi apa pun (JSON, YAML, atau pengkodean masa depan). Field `semantic_spec_url` pada lapisan AI merujuk ke spesifikasi ini, menjadikan kesesuaian dapat diverifikasi mesin, bukan sekadar dinyatakan.

| Simbol | Nama | Field JSON | Peran |
|--------|------|------------|-------|
| **P** | Predikat | `FALSIFIED_IF` | Kalimat logis yang, jika TRUE, memfalsifikasi klaim |
| **O** | Operan | `WHERE` | Definisi bertipe dari setiap variabel dalam P |
| **M** | Mekanisme | `EVALUATION` | Prosedur evaluasi terbatas dan deterministik |
| **B** | Batas | `BOUNDARY` + `CONTEXT` | Semantik ambang batas dan justifikasi |

DRP membuat kernel (menetapkan predikat ke klaim prosa). GVP mengikat kernel ke bukti yang dapat dieksekusi (menghubungkan predikat ke pengujian dan commit). Kernel bersifat independen substrat — ia bekerja untuk makalah ilmiah, perangkat lunak, dan domain masa depan apa pun yang membuat klaim yang dapat difalsifikasi.

### 3.1 Tata Bahasa Predikat (`FALSIFIED_IF`)

Predikat adalah kalimat logis yang terdiri dari:

- **Variabel bernama** (didefinisikan dalam `WHERE`)
- **Operator perbandingan:** `<`, `>`, `<=`, `>=`, `=`, `!=`
- **Penghubung logis:** `AND`, `OR`, `NOT`
- **Kuantifier:** `EXISTS ... SUCH THAT`, `FOR ALL ... IN`
- **Operator aritmetika:** `+`, `-`, `*`, `/`, `^`, `log10()`, `exp()`, `abs()`, `max()`, `min()`
- **Operator himpunan:** `IN`, `∩`, `∪`, `|...|` (kardinalitas)
- **Penerapan fungsi:** `f(x)` di mana `f` didefinisikan dalam `WHERE`

### 3.2 Batasan Predikat

1. **Determinisme.** Harus mengevaluasi ke tepat `TRUE` atau `FALSE` untuk setiap penugasan variabel yang valid.
2. **Keterbatasan.** Kuantifier hanya menjangkau himpunan terbatas. Kuantifier universal tak terbatas tidak diizinkan.
3. **Tanpa referensi diri.** Tidak boleh merujuk nilai kebenaran klaimnya sendiri atau membuat dependensi sirkuler.
4. **Kelengkapan.** Setiap variabel dalam `FALSIFIED_IF` harus didefinisikan dalam `WHERE`, dan setiap variabel dalam `WHERE` harus muncul dalam `FALSIFIED_IF`.

**Pemetaan putusan:**
- P mengevaluasi ke TRUE → putusan klaim adalah **FALSIFIED**
- P mengevaluasi ke FALSE → putusan klaim adalah **NOT FALSIFIED**

### 3.3 Operan (`WHERE`)

Setiap kunci dalam objek `WHERE` menamai sebuah variabel. Nilainya adalah string dengan format:

```
<tipe> · <satuan> · <definisi atau sumber>
```

| Field | Wajib | Contoh |
|-------|-------|--------|
| **tipe** | Ya | `scalar`, `integer`, `binary`, `set`, `string`, `function` |
| **satuan** | Ya (gunakan `dimensionless` jika tanpa satuan) | `seconds`, `dimensionless`, `bits` |
| **definisi** | Ya | `output of sort(input)`, `count of substrates with R² >= 0.85` |

Batasan: setiap variabel bebas dalam P harus muncul di O (kelengkapan), setiap variabel di O harus muncul di P (tanpa anak yatim), dan setiap variabel harus dapat dihitung dari data atau derivasi — bukan penilaian subjektif.

### 3.4 Mekanisme (`EVALUATION`)

Field evaluasi menspesifikasikan *cara* menghitung nilai kebenaran P:

1. **Terbatas.** Prosedur berhenti dalam langkah terbatas (secara konvensi dikonfirmasi dengan mengakhiri kata `finite`).
2. **Deterministik.** Input yang sama menghasilkan putusan yang sama.
3. **Dapat direproduksi.** Pihak ketiga dengan akses ke data dan kode yang dikutip dapat menjalankan prosedur secara independen.

### 3.5 Batas (`BOUNDARY` + `CONTEXT`)

**BOUNDARY** menspesifikasikan kasus tepi ambang batas: apakah ambang batas bersifat inklusif atau eksklusif, dan putusan apa yang berlaku pada kesamaan persis.

**CONTEXT** memberikan justifikasi untuk setiap ambang batas numerik dan pilihan desain dalam predikat: mengapa nilai ini, pengetahuan domain apa yang mendasarinya, dan apakah nilai tersebut diturunkan atau konvensional.

### 3.6 Contoh Predikat

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

---

## 4. Tiga Tipe Klaim

| Tipe | Nama | Dalam Makalah | Dalam Perangkat Lunak | Predikat |
|------|------|---------------|----------------------|----------|
| **A** | Aksioma / Asumsi | Premis yang diterima dari literatur | Persyaratan platform, kontrak dependensi | `null` |
| **D** | Definisi | Istilah dan prosedur stipulatif | Tanda tangan tipe, struktur data, skema | `null` |
| **F** | Dapat Difalsifikasi | Teorema, prediksi empiris | Jaminan perilaku, invarian kebenaran | K = (P, O, M, B) lengkap |

Klaim Tipe D dan Tipe A membawa `"falsification_predicate": null`. Ini benar — definisi dan aksioma memang tidak dapat difalsifikasi secara sengaja.

---

## 5. Enam Tingkat Verifikasi

Setiap klaim membawa field `tier` yang menyatakan jenis bukti apa yang mendasarinya.

### Berlabuh secara Konstruksi

| Tingkat | Tipe | Makna |
|---------|------|-------|
| `axiom` | A | Premis fundamental. Tidak dapat difalsifikasi secara sengaja. |
| `definition` | D | Definisional. Menstipulasikan istilah atau struktur. Tidak bersifat kebenaran. |

### Berlabuh Saat Ini

| Tingkat | Tipe | Makna |
|---------|------|-------|
| `software_tested` | F | Diuji oleh pengujian yang berhasil. `test_bindings` tidak kosong, `verified_against` tidak null. |
| `formal_proof` | F | Derivasi berindeks langkah dengan `n_invalid_steps = 0`. Bukti ada di lapisan AI. |
| `analytic` | F | Diverifikasi melalui jejak derivasi formal atau argumen analitik. |

### Secara Eksplisit Belum Berlabuh

| Tingkat | Tipe | Makna |
|---------|------|-------|
| `empirical_pending` | F | Placeholder aktif atau menunggu data. Kesenjangan terlihat secara sengaja. |

### Aturan Konsistensi

Tingkat harus konsisten dengan tipe klaim dan field GVP:

| Tingkat | Tipe Wajib | test_bindings | verified_against |
|---------|-----------|---------------|-----------------|
| `axiom` | A | `[]` (kosong) | `null` |
| `definition` | D | `[]` (kosong) | `null` |
| `software_tested` | F | Tidak kosong | Tidak null (7–40 karakter hex) |
| `formal_proof` | F | Boleh kosong | Boleh null |
| `analytic` | F | Boleh kosong | Boleh null |
| `empirical_pending` | F | Boleh kosong | Boleh null |

---

## 6. Saksi Vakuitas

Setiap klaim Tipe F harus menyertakan field `sample_falsification_observation`: observasi konkret dan hipotetis yang *akan* membuat predikat mengevaluasi ke TRUE.

Ini berfungsi sebagai pemeriksaan vakuitas — bukti bahwa predikat tidak bersifat tak dapat difalsifikasi secara trivial. Predikat yang tidak dapat dipenuhi oleh observasi apa pun yang dapat dibayangkan adalah benar secara vakuous dan oleh karena itu bukan klaim Tipe F yang valid.

Ini ditegakkan oleh kondisi gerbang fase c6.

---

## 7. Predikat Placeholder

Ketika sebuah klaim bergantung pada hasil dari makalah yang belum berstatus PHASE-READY, predikat dapat berisi referensi placeholder:

- `placeholder: true` dalam objek klaim
- `placeholder_id` yang menghubungkan ke `placeholder_register`
- Teks predikat dapat menyertakan `[PLACEHOLDER: pending ...]`

Klaim placeholder valid tetapi **tidak dapat dievaluasi** sampai dependensi terselesaikan. Klaim tersebut tidak memblokir status PHASE-READY untuk makalah yang memuatnya kecuali `blocks_phase_ready: true`.

---

## 8. Lapisan AI — Artefak Sentral

Lapisan AI adalah artefak sentral DRS. Ini adalah dokumen JSON yang menyertai setiap makalah atau sistem perangkat lunak. Skemanya didefinisikan dalam `ai-layers/ai-layer-schema.json`.

### Bagian yang Diperlukan

| Bagian | Tujuan |
|--------|--------|
| `_meta` | Tipe dokumen, versi skema, sesi, lisensi |
| `paper_id` / `paper_title` | Identitas |
| `paper_type` | Klasifikasi: `law_A`, `derivation_B`, `application_C`, `methodology_D` |
| `phase_ready` | Putusan gerbang fase dan status kondisi (c1–c6) |
| `claim_registry` | Array semua klaim dengan tipe, predikat, tingkat, ikatan, dan SHA |
| `placeholder_register` | Array dependensi yang belum terselesaikan |
| `summary` | Jumlah klaim dan status |
| `semantic_spec_url` | Merujuk ke Kernel Falsifikasi (Lapisan 0) |

Lapisan AI adalah yang membuat kedua protokol bekerja:

- **DRP** mengharuskannya ada, berisi predikat, dan melewati gerbang fase.
- **GVP** mengharuskannya berisi tier, test_bindings, dan verified_against untuk setiap klaim.

---

## 9. Gerbang Fase

Sebuah makalah atau rilis perangkat lunak berstatus **PHASE-READY** ketika enam kondisi terpenuhi:

| Kondisi | Persyaratan |
|---------|-------------|
| **c1** | Lapisan AI valid terhadap skema |
| **c2** | Semua klaim yang dapat difalsifikasi terdaftar dengan predikat |
| **c3** | Semua predikat dapat dievaluasi oleh mesin |
| **c4** | Referensi silang dilacak (register placeholder) |
| **c5** | Verifikasi bersifat mandiri (lapisan AI saja, tanpa prosa diperlukan) |
| **c6** | Semua predikat bersifat non-vakuous (observasi falsifikasi contoh ada) |

**CORPUS-COMPLETE** terpicu ketika semua makalah berstatus PHASE-READY dan semua placeholder di seluruh objek terselesaikan (c4 terpenuhi sepenuhnya di seluruh korpus).

---

## 10. Aturan Inferensi

DRS menyediakan inventaris kanonik aturan inferensi untuk jejak derivasi yang digunakan dalam klaim tingkat `formal_proof`:

| ID | Nama | Deskripsi |
|----|------|-----------|
| IR-1 | Modus Ponens | Jika P dan P→Q maka Q |
| IR-2 | Instansiasi Universal | Jika ∀x P(x) maka P(a) untuk setiap a spesifik |
| IR-3 | Substitusi Kesamaan | Jika a=b maka ganti a dengan b |
| IR-4 | Ekspansi Definisi | Ganti istilah yang didefinisikan dengan definisinya |
| IR-5 | Manipulasi Aljabar | Transformasi aljabar yang valid yang mempertahankan kesamaan |
| IR-6 | Ekuivalensi Logis | Ganti dengan ekspresi yang setara secara logis |
| IR-7 | Inferensi Statistik | Terapkan prosedur statistik bernama pada data |
| IR-8 | Parsimoni / Pemilihan Prinsip Pemodelan | Pilih nilai kanonik dari keluarga yang konsisten dengan aksioma |

Setiap langkah dalam tabel derivasi berindeks langkah mengutip satu aturan inferensi dan mencantumkan premis-premisnya. Derivasi valid ketika `n_invalid_steps = 0`. Inventaris bersifat append-only: aturan baru dapat ditambahkan, aturan yang ada tidak pernah dimodifikasi atau dihapus.

---

## 11. Tumpukan Arsitektur

```
                     ┌─────────────────────────┐
                     │   Dual Reader Standard   │
                     │         (DRS)            │
                     └────────────┬────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  │                               │
         ┌────────┴────────┐            ┌─────────┴─────────┐
         │      DRP        │            │       GVP         │
         │(Protokol Teks)  │            │(Protokol P. Lunak)│
         └────────┬────────┘            └─────────┬─────────┘
                  │                               │
      ┌───────────┴──────────┐       ┌────────────┴───────────┐
      │                      │       │                        │
   Pembaca 1            Pembaca 2   Pembaca 3A           Pembaca 3B
   (Manusia)            (AI)        (Pengembang)         (Mesin)
      │                      │       │                        │
   membaca prosa       membaca JSON membaca tier +       menjalankan
                                    ikatan               pengujian
      │                      │       │                   mencap SHA
      └──────────┬───────────┘       └────────────┬───────────┘
                 │                                │
                 └────────────────┬────────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  │          Lapisan AI           │
                  │  (*-ai-layer.json)            │
                  │  klaim + predikat + tingkat   │
                  │  + ikatan + SHA               │
                  └───────────────┬───────────────┘
                                  │
                     ┌────────────┴────────────┐
                     │   Kernel Falsifikasi    │
                     │    K = (P, O, M, B)     │
                     │       (Lapisan 0)       │
                     └─────────────────────────┘
```

Kernel adalah fondasi bersama. DRP membuat predikat dari prosa. GVP mengikat predikat ke bukti yang dapat dieksekusi. Lapisan AI adalah artefak yang membawa keduanya.

---

## 12. Cara Mengimplementasikan

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
5. Isi `test_bindings` dengan ID node pengujian berkualifikasi penuh
6. Jalankan pengujian dan catat SHA commit yang berhasil di `verified_against`
7. Tetapkan tingkat: `software_tested` jika pengujian ada, `empirical_pending` jika belum
8. Daftarkan klaim yang belum diuji sebagai placeholder
9. Validasi terhadap `ai-layer-schema.json`

### Untuk Keduanya

Lapisan AI adalah artefak yang sama. Skemanya sama. Kernelnya sama. Satu-satunya perbedaan adalah protokol mana yang membuat konten dan protokol mana yang memverifikasinya.

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

## 13. Prinsip Desain

**Epistemologi Popperian.** Kita dapat memfalsifikasi tetapi tidak dapat memverifikasi. Klaim yang bertahan dari semua upaya falsifikasi tidak terbukti — ia telah bertahan.

**Kesenjangan yang jujur.** Placeholder adalah fitur terpenting. Ketika sebuah klaim berstatus `empirical_pending`, sistem berkata: "kami mengklaim ini tetapi belum memverifikasinya." Ini secara tegas lebih informatif daripada alternatifnya, di mana klaim yang belum terverifikasi tidak dapat dibedakan dari yang sudah terverifikasi.

**Independensi substrat.** Kernel K = (P, O, M, B) tidak mengetahui apakah ia mengevaluasi teorema ilmiah atau jaminan perangkat lunak. Domain masa depan (hukum, regulasi, kebijakan) dapat menambahkan protokol mereka sendiri tanpa memodifikasi kernel.

**Mengutamakan mesin, dapat dibaca manusia.** Lapisan AI adalah artefak primer. Makalah prosa dan kode sumber adalah kanal sekunder yang menyediakan konteks, narasi, dan implementasi. Lapisan AI-lah yang divalidasi, diaudit, dan diversi.

---

## 14. Kompatibilitas Tiga Sumbu

Standar yang tidak dapat bertahan dari kontak dengan masa lalu, masa kini, dan masa depan bukanlah standar — itu hanyalah potret sesaat.

### Invarian di Pusat

Kernel Falsifikasi K = (P, O, M, B) adalah titik tetap. Kernel ini tidak mengetahui apakah ia mengevaluasi teorema ilmiah, jaminan perangkat lunak, kontrak hukum, atau persyaratan regulasi. Ia tidak mengetahui apakah tahunnya 2026 atau 2046. Ia tidak mengetahui apakah format serialisasinya JSON, YAML, atau sesuatu yang belum ditemukan.

Kernel bersifat permanen. Segala sesuatu yang lain bersifat dapat diperluas.

### Sumbu 1: Ketahanan Masa Lalu (Kompatibilitas Mundur)

- **Pemversian skema.** Setiap lapisan AI mencatat versi skema mana yang digunakan saat diproduksi. Lapisan AI yang diproduksi di bawah skema v2 tetap valid di bawah skema v2 selamanya. Validator v3 dapat membaca lapisan v2 (field baru bersifat opsional; field lama dipertahankan).
- **Permanensi predikat.** Predikat yang ditulis pada 2026 harus dapat dievaluasi pada 2036. Tata bahasa kernel hanya menggunakan operator matematika dan logika yang didefinisikan secara permanen. Field `WHERE` mendefinisikan setiap variabel secara inline. Field `EVALUATION` menspesifikasikan prosedur mandiri.
- **Stabilitas aturan inferensi.** Inventaris aturan inferensi (IR-1 hingga IR-8) bersifat append-only. Aturan yang ada tidak pernah dimodifikasi atau dihapus.
- **Stabilitas tingkat.** Enam tingkat verifikasi bersifat append-only. Tingkat yang ada tidak pernah dihapus atau didefinisikan ulang.
- **Kontraknya:** Setiap lapisan AI yang valid saat dibuat akan tetap valid selamanya.

### Sumbu 2: Ketahanan Lateral (Kompatibilitas Lintas Domain)

- **Independensi domain.** Kernel bekerja di semua domain pengetahuan:

| Domain | Tipe A | Tipe D | Tipe F |
|--------|--------|--------|--------|
| Sains | Premis literatur | Istilah stipulatif | Teorema, prediksi |
| Perangkat Lunak | Persyaratan platform | Tanda tangan tipe, skema | Jaminan perilaku |
| Hukum | Otoritas hukum | Istilah yang didefinisikan | Kesimpulan hukum |
| Regulasi | Asumsi kerangka kerja | Definisi standar | Pernyataan kesesuaian |
| Kebijakan | Premis nilai | Istilah kebijakan | Prediksi dampak |
| Pendidikan | Aksioma pedagogis | Tujuan pembelajaran | Klaim penilaian |

- **Independensi bahasa.** Field `test_bindings` menerima string apa pun yang secara unik mengidentifikasi pengujian dalam kerangka kerja apa pun: pytest, Jest, cargo test, go test, JUnit.
- **Independensi sistem AI.** Lapisan AI adalah dokumen JSON. Sistem AI apa pun — Claude, GPT, Gemini, Llama, atau sistem yang belum ada — dapat membacanya. Field `semantic_spec_url` merujuk ke spesifikasi kernel yang ditulis dalam prosa biasa.
- **Independensi serialisasi.** Lapisan 0 didefinisikan dalam prosa, bukan dalam JSON Schema. JSON adalah transpor saat ini, tetapi semantik kernel independen dari pengkodean. Implementasi masa depan dapat menggunakan YAML, Protocol Buffers, CBOR, atau format yang belum ditemukan.
- **Independensi alat.** DRS menyematkan diri dalam alur kerja yang ada: SHA git (hosting apa pun), pelari pengujian apa pun, validator JSON Schema apa pun, pipeline CI apa pun. Ia menambahkan lapisan di atas — ia tidak menggantikan apa pun.
- **Kontraknya:** Mengadopsi DRS di satu domain, bahasa, alat, atau sistem AI tidak mengunci Anda.

### Sumbu 3: Ketahanan Masa Depan (Kompatibilitas Maju)

- **Ekstensibilitas protokol.** Domain baru menambahkan protokol baru. Protokol masa depan (Legal Verification Protocol, Regulatory Verification Protocol, dll.) mengikuti pola yang sama: mendefinisikan pembaca, taksonomi tingkat, dan mekanisme pengikatan — semuanya berbagi kernel, tipe klaim, dan skema lapisan AI.
- **Ekstensibilitas tingkat.** Domain masa depan mungkin memerlukan tingkat seperti `regulatory_certified`, `peer_reviewed`, `formally_verified`, `field_tested`, `community_validated`. Tingkat baru ditambahkan ke enum skema. Tingkat yang ada tetap.
- **Ekstensibilitas aturan inferensi.** IR-9, IR-10, dan seterusnya dapat ditambahkan seiring munculnya pola derivasi baru. Derivasi lama tetap valid karena mengutip aturan berdasarkan ID yang stabil.
- **Ekstensibilitas skema.** JSON Schema mengizinkan properti tambahan secara default. Progresinya: v1 (klaim dasar), v2 (ditambah `semantic_spec_url`), v3 (ditambah field GVP). Setiap versi menambahkan. Tidak ada yang menghapus.
- **Pembaca masa depan yang belum diketahui.** Lapisan AI berisi informasi terstruktur yang cukup untuk tipe pembaca yang belum ada: agen verifikasi otonom, pemeriksa lintas korpus, mesin kepatuhan regulasi, integrasi pengelola paket.
- **Kontraknya:** Inovasi masa depan apa pun dapat ditambahkan sebagai protokol, tingkat, aturan inferensi, atau field skema baru — tanpa memodifikasi apa pun yang sudah ada.

### Jaminan Tiga Sumbu

| Sumbu | Janji | Mekanisme |
|-------|-------|-----------|
| **Masa Lalu** | Tidak ada yang sudah dilakukan rusak | Pemversian skema, enum append-only, kernel permanen |
| **Lateral** | Berfungsi di semua domain, bahasa, alat, sistem AI | Kernel yang independen substrat, ikatan bertipe string, semantik yang didefinisikan dalam prosa |
| **Masa Depan** | Apa pun yang baru dapat ditambahkan tanpa desain ulang | Ekstensibilitas protokol, ekstensibilitas tingkat, evolusi skema aditif |

---

## 15. Lingua Franca Mesin

Ini adalah sifat terdalam dari DRS. Sifat ini tidak dirancang. Ia ditemukan.

### Masalah Penerjemahan

Pengetahuan ilmiah saat ini terkunci di balik bahasa manusia. Sebuah makalah dalam bahasa Mandarin tidak terlihat oleh peneliti yang hanya membaca bahasa Inggris — kecuali seseorang menerjemahkannya. Penerjemahan itu mahal, mengalami kehilangan informasi, dan lambat. Pengetahuan terfragmentasi sepanjang garis linguistik.

Ini bukan masalah format. Ini adalah masalah *substrat*. Pengetahuan yang dikodekan dalam bahasa alami secara alamiah tidak dapat diinteroperasikan.

### Kernel Melarutkan Masalah

Kernel Falsifikasi tidak ditulis dalam bahasa manusia mana pun. Ia ditulis dalam logika dan matematika:

```
FALSIFIED_IF: R2_best_alt > R2_frm + 0.05
WHERE:
  R2_best_alt: scalar · dimensionless · best R² from competing models
  R2_frm:      scalar · dimensionless · R² from FRM regression
EVALUATION: Run regression for each model; compare R² values; finite
BOUNDARY: R2_best_alt = R2_frm + 0.05 → FALSIFIED (threshold inclusive)
CONTEXT: 0.05 margin from standard model comparison practice
```

Predikat ini bermakna sama bagi instans Claude dalam bahasa Inggris, instans GPT dalam bahasa Mandarin, instans Gemini dalam bahasa Prancis, dan sistem AI yang belum dibangun yang berjalan dalam bahasa yang belum ada. Tidak diperlukan penerjemahan.

### JSON sebagai Lapisan Transpor

JSON adalah format pertukaran data de facto dunia — didukung oleh setiap bahasa pemrograman, diparsing oleh setiap sistem AI, ditransmisikan oleh setiap API. Dengan mengkodekan kernel dalam JSON, DRS mewarisi universalitas JSON:

- Tim Tiongkok mempublikasikan lapisan AI mereka. Predikatnya menggunakan notasi matematika.
- Tim Brasil membaca lapisan yang sama. Mereka tidak perlu bahasa Mandarin. Mereka perlu `>`, `+`, dan `R²`.
- Sistem AI di negara mana pun mengevaluasi predikat. Putusannya FALSIFIED atau NOT FALSIFIED. Putusan tidak memiliki aksen.

**Kualifikasi penting.** Definisi field `WHERE` saat ini berisi deskripsi prosa bahasa Inggris. *Konten operatif* — tipe variabel, satuan, operator matematika, logika perbandingan, nilai ambang batas — bersifat netral bahasa. Dengan nilai numerik yang berlabuh untuk variabel bernama, sistem mana pun dapat mengevaluasi predikat tanpa memedulikan apakah ia membaca deskripsi tersebut. Penyempurnaan masa depan dapat memformalkan definisi variabel sepenuhnya dalam notasi bertipe.

### Logika Biner Tertanam

Setiap predikat DRS direduksi ke logika biner tertanam. Seluruh rantai evaluasi runtuh menjadi:

```
Klaim → Predikat → Variabel + Operator → Boolean → 1 atau 0
```

Field `WHERE` mengetikkan variabel. Field `FALSIFIED_IF` menggabungkannya dengan operator. Field `EVALUATION` menspesifikasikan cara menghitung input. Field `BOUNDARY` menyelesaikan kasus tepi. Keluarannya selalu satu bit: FALSIFIED atau NOT FALSIFIED. `1` atau `0`.

JSON adalah wadahnya. Kernel adalah rangkaiannya. Setiap lapisan AI adalah kumpulan gerbang logika dengan input yang didefinisikan dan keluaran satu bit per klaim.

Anda tidak dapat salah menerjemahkan `1` atau `0`. Anda tidak dapat salah menafsirkan `>`. Anda tidak dapat memiliki ketidaksepakatan budaya tentang apakah `R2_best_alt > R2_frm + 0.05` bernilai TRUE atau FALSE untuk sepasang nilai tertentu. Maknanya ada dalam struktur, bukan dalam bahasa mana pun.

Logika biner adalah logika biner. Predikat yang mengevaluasi ke TRUE di Beijing mengevaluasi ke TRUE di Boston, di São Paulo, di Lagos, dan di server tanpa pengaturan lokalisasi sama sekali. DRS tidak *menerjemahkan* pengetahuan lintas bahasa. Ia *mengkodekan* pengetahuan dalam substrat yang mendahului dan melampaui semua bahasa manusia — substrat logika itu sendiri.

---

## 16. Adopsi yang Menyebar Sendiri

### Insentif Internal

DRS harus menyebar karena *pembuatnya* yang diuntungkan, bukan hanya konsumen. Membuat lapisan AI memaksa Anda untuk mengenumerasi apa yang Anda klaim. Proses ini menemukan:

- Klaim yang Anda kira sudah diuji tetapi ternyata belum (penemuan kesenjangan)
- Kondisi batas yang tidak pernah Anda dokumentasikan
- Dependensi yang Anda asumsikan tetapi tidak pernah didaftarkan

Lapisan AI adalah efek samping dari proses yang meningkatkan pemahaman Anda sendiri terhadap sistem Anda sendiri. Kesenjangan itu ada baik Anda mendokumentasikannya maupun tidak. DRS hanya membuatnya terlihat.

### Efek Jaringan

DRS menjadi lebih bernilai seiring semakin banyak sistem yang mengadopsinya:

- **Rantai dependensi menjadi sadar klaim.** Jika pustaka A mempublikasikan lapisan AI dan pustaka B bergantung pada A, maka B dapat menentukan secara programatik klaim mana yang bergantung pada asumsi mana dari A. Ketika A merilis perubahan yang merusak, B tahu persis klaim mana yang berisiko.
- **Sistem AI dapat mengaudit lintas proyek.** Pembaca AI dapat melintasi beberapa lapisan AI, memeriksa referensi silang, dan mengidentifikasi inkonsistensi di seluruh ekosistem.
- **Kepercayaan menjadi dapat diaudit.** Alih-alih mempercayai pustaka karena bintang atau unduhan, Anda mempercayainya karena lapisan AI-nya menunjukkan klaim mana yang berstatus `software_tested`, mana yang `empirical_pending`, dan berapa SHA `verified_against`-nya. Kepercayaan bergeser dari sinyal sosial ke bukti struktural.

### AI sebagai Katalisator Adopsi

1. **AI menghasilkan lapisan AI awal.** Diberikan basis kode, AI dapat mengenumerasi klaim, mengklasifikasikannya, menulis predikat, dan mengidentifikasi ikatan pengujian. Manusia meninjau dan mengoreksi. Biaya turun dari jam menjadi menit.
2. **AI memelihara lapisan.** Ketika kode berubah, AI memperbarui registri klaim, menyesuaikan ikatan pengujian, dan menandai SHA yang sudah usang. Manusia menyetujui.
3. **AI mengaudit lapisan lain.** AI yang membaca lapisan AI dependensi dapat menentukan asumsi mana yang menjadi dasar klaim sendiri dan menandai risiko secara otomatis.

DRS adalah protokol yang menjadikan pengembangan berbantuan AI *dapat diaudit*. Tanpanya, AI menghasilkan kode dan manusia berharap ia bekerja. Dengannya, AI menghasilkan kode dan registri klaim menyatakan persis apa yang sudah dan belum diverifikasi.

### Strategi Penyematan

DRS menyematkan diri dalam alur kerja yang ada alih-alih menggantikannya:

- **Pengujian sudah ada.** Field `test_bindings` merujuk ID node pengujian yang ada. Tidak diperlukan kerangka kerja baru.
- **JSON Schema sudah ada.** Validator apa pun berfungsi.
- **Git sudah ada.** Field `verified_against` adalah SHA git.
- **CI sudah ada.** Validasi skema berjalan sebagai langkah CI bersama pipeline yang ada.

Satu file (`*-ai-layer.json`). Tiga field per klaim (`tier`, `test_bindings`, `verified_against`). Itulah total biaya integrasi.

### Sifat Referensi Diri

DRS adalah standar pertama yang memverifikasi dirinya sendiri. Lapisan AI DRP-1 berisi klaim tentang DRS. Klaim-klaim tersebut membawa predikat falsifikasi. Predikat-predikat tersebut dievaluasi. SHA `verified_against` mencap verifikasinya. DRS adalah pengadopsi pertamanya sendiri dan bukti konsepnya sendiri — referensi diri dengan cara yang sama seperti kompiler yang mengkompilasi dirinya sendiri.

---

## 17. Sumber Daya

- [Spesifikasi Arsitektur DRS](https://github.com/thomasbrennan/Fracttalix/blob/main/paper/DRS-Architecture.md) — Spesifikasi lengkap
- [Kernel Falsifikasi v1.1](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/falsification-kernel.md) — Spesifikasi semantik Lapisan 0
- [Skema Lapisan AI v3](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json) — JSON Schema
- [Spesifikasi GVP](https://github.com/thomasbrennan/Fracttalix/blob/main/docs/GVP-spec.md) — Protokol perangkat lunak
- [Contoh Lapisan AI (P1)](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/P1-ai-layer.json) — Contoh yang berfungsi
- [Repositori Fracttalix](https://github.com/thomasbrennan/Fracttalix)

---

**Lisensi:** CC BY 4.0 | **DOI:** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299) | **Penulis:** Thomas Brennan ([ORCID](https://orcid.org/0009-0002-6353-7115))
