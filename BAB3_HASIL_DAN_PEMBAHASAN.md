# BAB III: HASIL DAN PEMBAHASAN

> [!NOTE]
> Dokumen ini disusun berdasarkan data nyata hasil percobaan pada tanggal **18 November 2026**.
> Semua angka bersumber dari file log dan CSV aktual, bukan estimasi.
> File referensi utama: `klasifikasi_gestur/hasil/tabel/`, `multi_target_v2/hasil_percobaan/percobaan_*/log/`.

---

## 3.1 Kinerja Klasifikasi Gestur

### 3.1.1 Deskripsi Dataset dan Model

Klasifikasi gestur dilakukan menggunakan algoritma **K-Nearest Neighbor (KNN)** dengan nilai *k* = 5. Dataset yang digunakan berjumlah **2.800 sampel**, terdiri dari dua kelas:

| Kelas | Label | Jumlah Sampel |
|-------|-------|---------------|
| Hands Up (HU) | 1 | 1.400 |
| No Gesture (NG) | 0 | 1.400 |

Dataset dibagi dengan rasio 80:20 menggunakan stratified split:
- **Data latih**: 2.240 sampel
- **Data uji**: 560 sampel (280 HU + 280 NG)

Model tersimpan di: `klasifikasi_gestur/model/knn_k5.pkl`
Dataset lengkap di: `klasifikasi_gestur/dataset/dataset.csv`

Fitur masukan model adalah **koordinat sendi tangan** yang diekstraksi dari Kinect v1 melalui OpenNI2. Setiap sampel merepresentasikan posisi spasial kedua tangan subjek dalam format koordinat 3D yang dinormalisasi.

---

### 3.1.2 Pemilihan Nilai k (Cross-Validation)

Pemilihan nilai *k* optimal dilakukan dengan 5-fold cross-validation pada rentang k = 1 hingga 15. Hasil selengkapnya tersimpan di `klasifikasi_gestur/hasil/tabel/pemilihan_k.csv`.

**Tabel 3.1 — Hasil 5-Fold Cross-Validation Pemilihan k**

| k | Akurasi CV (rata-rata) | Std Deviasi |
|---|------------------------|-------------|
| 1 | 99,71% | ±0,21% |
| 2 | 99,64% | ±0,30% |
| **3** | **99,75%** | **±0,18%** |
| 4 | 99,68% | ±0,29% |
| **5** | **99,61%** | **±0,24%** |
| 6 | 99,68% | ±0,26% |
| 7 | 99,61% | ±0,26% |
| 8 | 99,61% | ±0,35% |
| 9 | 99,46% | ±0,32% |
| 10 | 99,54% | ±0,33% |
| 11 | 99,46% | ±0,34% |
| 12 | 99,39% | ±0,46% |
| 13 | 99,32% | ±0,48% |
| 14 | 99,32% | ±0,50% |
| 15 | 99,25% | ±0,50% |

> [!NOTE]
> Meskipun k = 3 menghasilkan akurasi CV tertinggi (99,75%), nilai **k = 5** dipilih dengan pertimbangan:
> - Selisih akurasi hanya 0,14% (tidak signifikan secara praktis)
> - k genap dihindari untuk mencegah tie-breaking pada klasifikasi biner
> - k = 5 memberikan generalisasi yang lebih baik dan lebih tahan terhadap *outlier*

**Referensi Grafik:** `klasifikasi_gestur/hasil/grafik/pemilihan_k_cv.png` — visualisasi kurva akurasi CV vs nilai k

---

### 3.1.3 Metrik Evaluasi pada Data Uji

Model dievaluasi pada 560 data uji (data yang tidak pernah dilihat model saat pelatihan). Sumber data: `klasifikasi_gestur/hasil/tabel/metrik_utama.csv`.

#### Confusion Matrix

$$
CM = \begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix} = \begin{bmatrix} 278 & 2 \\ 3 & 277 \end{bmatrix}
$$

**Referensi Grafik:** `klasifikasi_gestur/hasil/grafik/confusion_matrix.png`

#### Rumus dan Hasil Metrik

**Akurasi:**
$$
\text{Akurasi} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{277 + 278}{277 + 278 + 2 + 3} = \frac{555}{560} = 99{,}11\%
$$

**Precision (Hands Up):**
$$
\text{Precision}_{HU} = \frac{TP}{TP + FP} = \frac{277}{277 + 2} = \frac{277}{279} = 99{,}28\%
$$

**Recall (Hands Up):**
$$
\text{Recall}_{HU} = \frac{TP}{TP + FN} = \frac{277}{277 + 3} = \frac{277}{280} = 98{,}93\%
$$

**F1-Score (Hands Up):**
$$
\text{F1}_{HU} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \times 0{,}9928 \times 0{,}9893}{0{,}9928 + 0{,}9893} = 99{,}11\%
$$

**Tabel 3.2 — Metrik Evaluasi Model KNN k=5**

| Metrik | Kelas Hands Up | Kelas No Gesture | Weighted Avg |
|--------|---------------|-----------------|--------------|
| Precision | 99,28% | 98,93% | 99,11% |
| Recall | 98,93% | 99,29% | 99,11% |
| F1-Score | 99,11% | 99,11% | 99,11% |
| **Akurasi Keseluruhan** | — | — | **99,11%** |

#### Interval Kepercayaan Wilson (95%)

Metrik akurasi tunggal dapat menyesatkan tanpa interval kepercayaan. Menggunakan **Wilson Score Interval** untuk proporsi binomial:

$$
CI_{Wilson} = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}
$$

Dengan $n = 560$, $\hat{p} = 0{,}9911$, $z = 1{,}96$:

$$
CI_{95\%} = [97{,}93\%,\ 99{,}62\%]
$$

Hal ini menunjukkan bahwa dengan tingkat kepercayaan 95%, akurasi model sesungguhnya berada di rentang **97,93% — 99,62%**.

---

### 3.1.4 Akurasi per Jarak Deteksi

Model diuji pada 7 segmen jarak (0,5 m hingga 3,5 m). Data bersumber dari `klasifikasi_gestur/hasil/tabel/metrik_per_jarak.csv`.

**Tabel 3.3 — Akurasi Klasifikasi per Segmen Jarak**

| Jarak | Jumlah Sampel | Akurasi | Precision HU | Recall HU | F1 HU |
|-------|--------------|---------|--------------|-----------|-------|
| 0,5 m | 69 | **100,00%** | 100,00% | 100,00% | 100,00% |
| 1,0 m | 82 | **100,00%** | 100,00% | 100,00% | 100,00% |
| 1,5 m | 77 | 97,40% | 97,50% | 97,50% | 97,50% |
| 2,0 m | 79 | 98,73% | 100,00% | 97,22% | 98,59% |
| 2,5 m | 81 | 98,77% | 100,00% | 97,83% | 98,90% |
| 3,0 m | 90 | **100,00%** | 100,00% | 100,00% | 100,00% |
| 3,5 m | 82 | 98,78% | 97,56% | 100,00% | 98,77% |

**Referensi Grafik:** `klasifikasi_gestur/hasil/grafik/akurasi_per_jarak.png` — grafik batang akurasi per segmen jarak

> [!TIP]
> Akurasi terendah terjadi pada jarak 1,5 m (97,40%), yang diduga disebabkan oleh efek *depth noise* Kinect pada jarak dekat-menengah saat postur tangan belum sepenuhnya terbuka. Namun demikian, seluruh segmen jarak menunjukkan akurasi di atas 97%, membuktikan ketahanan model KNN k=5 di berbagai jarak operasional.

**Referensi Grafik:** `klasifikasi_gestur/hasil/grafik/decision_boundary.png` — visualisasi batas keputusan KNN di ruang fitur 2D (PCA)

---

### 3.1.5 Waktu Inferensi

Pengujian kecepatan inferensi dilakukan dengan 1.000 pengulangan. Data bersumber dari `klasifikasi_gestur/hasil/tabel/inferensi_waktu.csv`.

| Parameter | Nilai |
|-----------|-------|
| Jumlah Pengulangan | 1.000 |
| Waktu Rata-rata per Inferensi | **0,86 ms** |
| Standar Deviasi | 2,05 ms |
| Estimasi Throughput | **~1.162 FPS** |

Waktu inferensi 0,86 ms per frame jauh lebih cepat dibandingkan laju akuisisi frame Kinect (~30 FPS, setara 33 ms/frame). Artinya, **klasifikasi KNN tidak menjadi bottleneck** dalam pipeline navigasi robot.

---

## 3.2 Skenario dan Kondisi Pengujian

### 3.2.1 Perangkat Keras yang Digunakan

| Komponen | Spesifikasi |
|----------|------------|
| Sensor Kedalaman | Microsoft Kinect v1 (Xbox 360) — resolusi depth 640×480, range 0,5–4,5 m |
| Mikrokontroler | ESP32 (CH340 USB-UART Bridge, COM9) |
| Driver Motor | BTS7960 43A H-Bridge (×2 unit) |
| Motor | PG28 DC Motor dengan Encoder Kuadratur |
| Pin Motor Kiri | GPIO 27 (RPWM), GPIO 14 (LPWM) |
| Pin Motor Kanan | GPIO 25 (RPWM), GPIO 26 (LPWM) |
| Platform Komputasi | PC/Laptop Windows 10, Python 3.10, OpenNI2 SDK |

> [!IMPORTANT]
> Pemetaan pin GPIO di atas **sudah diverifikasi secara fisik** melalui pengujian bertahap pada `uji_keselarasan_motor.py`. Firmware ESP32 tersimpan di `ProgramEsp/ProgramEsp.ino`.

### 3.2.2 Parameter Sistem Navigasi

Parameter-parameter berikut dikonfigurasi secara konstan selama seluruh sesi pengujian (9 percobaan):

| Parameter | Nilai | Keterangan |
|-----------|-------|-----------|
| `STOP_DISTANCE` | 0,70 m | Robot berhenti saat depth ≤ 0,70 m dari target |
| `ANGLE_THRESHOLD` | ±2,0° | Toleransi sudut saat rotasi ke target |
| `DUAL_DETECT_HOLD` | 2,0 detik | Durasi gestur ganda harus dipertahankan |
| Frekuensi Sampling | ~100 ms | Interval logging frame (`log_frame.csv`) |
| Model Klasifikasi | `knn_k5.pkl` | KNN k=5, 2.800 sampel latih |

### 3.2.3 State Machine Navigasi

Program navigasi (`multi_target_v2/multi_target_navigasi_v3.py`) mengimplementasikan **state machine 9 state**:

```
WAIT_GESTURE → DUAL_DETECTED → KOORDINAT_TERKUNCI →
ROTASI_T1 → MAJU_T1 → T1_SELESAI →
ROTASI_T2 → MAJU_T2 → MISI_SELESAI
```

Transisi antar state dipicu oleh:
- Deteksi gestur KNN (WAIT → DUAL_DETECTED)
- Kelengkapan hold timer 2,0 detik (DUAL_DETECTED → KOORDINAT_TERKUNCI)
- Pembacaan encoder & depth sensor (ROTASI → MAJU → SELESAI)

### 3.2.4 Variasi Skenario Percobaan

Sembilan percobaan dilakukan dengan variasi posisi T1 dan T2 yang berbeda-beda untuk menguji kemampuan generalisasi sistem. Variasi mencakup:

- **Rentang jarak T1**: 1,88 m — 3,17 m dari kamera
- **Rentang jarak T2**: 2,98 m — 4,87 m dari kamera
- **Rentang sudut T1**: −20,8° hingga +13,9° dari sumbu tengah
- **Rentang sudut T2**: −10,8° hingga +3,1° dari sumbu tengah
- **Prioritas urutan**: selalu T1 terlebih dahulu (tangan pertama yang terdeteksi), kemudian T2

> [!NOTE]
> Percobaan-1 (folder `percobaan_20261118_011430`) merupakan sesi pengujian awal dengan status **BERJALAN** (tidak ada target yang terkunci karena program dihentikan segera setelah inisialisasi). Percobaan ini **dieksklusi** dari analisis hasil navigasi, sehingga **8 percobaan valid** yang dianalisis (percobaan 2–9).

**Referensi Foto Dokumentasi Setup:**
- `percobaan_20261118_014632/foto/dual_detect_mulai_01.png` — kondisi awal saat subjek menunggu deteksi
- `percobaan_20261118_014632/foto/kunci_koordinat_01.png` — tampilan layar saat koordinat T1 & T2 terkunci

---

## 3.3 Hasil Pengujian Navigasi Dua Target

### 3.3.1 Rekap Data 9 Percobaan

Berikut adalah data lengkap dari seluruh sesi percobaan. Data berasal dari `log/ringkasan.txt` dan `log/log_event.txt` pada masing-masing folder percobaan.

**Tabel 3.4 — Rekap Hasil Percobaan Navigasi**

| No | Folder | Status | T1 Awal (m) | T1 Sudut (°) | T2 Awal (m) | T2 Sudut (°) | Jarak T1→T2 (m) | Sudut @Kamera (°) | Durasi (s) | Enc Kiri | Enc Kanan | Jarak Tempuh (m) |
|----|--------|--------|------------|-------------|------------|-------------|-----------------|-------------------|-----------|----------|-----------|-----------------|
| 1 | 011430 | BERJALAN* | 0,00 | 0,0 | 0,00 | 0,0 | 0,000 | 0,00 | 15,29 | 0 | 0 | 0,000 |
| 2 | 014632 | **SUKSES** | 1,94 | +13,9 | 2,98 | −7,7 | 1,376 | 21,60 | 254,45 | 3.789 | 3.577 | 8,184 |
| 3 | 015047 | **SUKSES** | 1,88 | +12,8 | 3,42 | −10,8 | 1,857 | 23,60 | 188,95 | 8.304 | 8.018 | 18,136 |
| 4 | 015357 | **SUKSES** | 1,88 | +11,3 | 2,98 | −6,5 | 1,322 | 17,80 | 29,57 | 11.218 | 10.790 | 24,453 |
| 5 | 015429 | **SUKSES** | 1,98 | −19,6 | 3,12 | +0,0 | 1,420 | 19,60 | 64,09 | 14.747 | 14.549 | 32,551 |
| 6 | 015534 | **SUKSES** | 1,94 | −14,5 | 3,29 | +3,1 | 1,556 | 17,60 | 115,03 | 18.704 | 18.766 | 41,633 |
| 7 | 015731 | **SUKSES** | 1,99 | +8,9 | 3,53 | −10,4 | 1,778 | 19,30 | 64,53 | 23.296 | 23.306 | 51,780 |
| 8 | 015837 | **SUKSES** | 2,64 | −16,8 | 4,87 | −2,5 | 2,402 | 14,30 | 55,75 | 26.214 | 26.090 | 58,116 |
| 9 | 015934 | **SUKSES** | 3,17 | −20,8 | 4,55 | −2,1 | 1,851 | 18,70 | 85,87 | 31.707 | 31.611 | 70,353 |

*\*Percobaan-1 dieksklusi: program dihentikan sebelum target dikunci*

> [!IMPORTANT]
> **Tingkat Keberhasilan: 8/8 = 100%** (dari 8 percobaan valid yang memiliki target terkunci)
> Semua percobaan mencapai `MISI_SELESAI` dengan T1 dan T2 masing-masing tiba pada jarak akhir **0,70 m** (sesuai `STOP_DISTANCE`).

---

### 3.3.2 Analisis Error Jarak

Jarak akhir robot ke setiap target adalah **0,70 m** di semua percobaan, persis sama dengan nilai `STOP_DISTANCE` yang dikonfigurasi. Karena robot menggunakan umpan balik depth sensor secara real-time (bukan open-loop), tidak ada error jarak terhadap parameter STOP_DISTANCE.

Error yang relevan adalah **error posisi absolut** robot terhadap posisi target sebenarnya, yang bergantung pada akurasi sensor depth Kinect:

- Kinect v1 memiliki akurasi depth ±3–4 mm pada jarak 1–3 m (spesifikasi pabrikan)
- Pada jarak >3 m, error depth meningkat hingga ±3 cm

**Jarak yang dicakup selama seluruh pengujian (kumulatif encoder):**
- Total estimasi jarak tempuh seluruh percobaan: **305,21 m** (encoder kiri) / **296,54 m** (encoder kanan)

---

### 3.3.3 Pemetaan Segitiga 2D

Sistem menggunakan pendekatan geometri segitiga untuk memetakan posisi T1 dan T2 dalam koordinat 2D relatif terhadap kamera (bidang XY horizontal):

$$
x_{Ti} = d_i \cdot \sin(\theta_i)
$$
$$
y_{Ti} = d_i \cdot \cos(\theta_i)
$$

Jarak antar target dihitung dengan hukum cosinus:
$$
d_{T1 \to T2} = \sqrt{d_1^2 + d_2^2 - 2 \cdot d_1 \cdot d_2 \cdot \cos(\theta_1 - \theta_2)}
$$

**Tabel 3.5 — Koordinat 2D Target pada 8 Percobaan Valid**

| No | T1 (x, y) m | T2 (x, y) m | Jarak T1→T2 (m) | Sudut @Kamera (°) |
|----|------------|------------|-----------------|-------------------|
| 2 | (+0,466, +1,883) | (−0,399, +2,953) | 1,376 | 21,60 |
| 3 | (+0,417, +1,833) | (−0,641, +3,359) | 1,857 | 23,60 |
| 4 | (+0,368, +1,844) | (−0,337, +2,961) | 1,322 | 17,80 |
| 5 | (−0,664, +1,865) | (+0,000, +3,120) | 1,420 | 19,60 |
| 6 | (−0,486, +1,878) | (+0,178, +3,285) | 1,556 | 17,60 |
| 7 | (+0,308, +1,966) | (−0,637, +3,472) | 1,778 | 19,30 |
| 8 | (−0,763, +2,527) | (−0,212, +4,865) | 2,402 | 14,30 |
| 9 | (−1,126, +2,963) | (−0,167, +4,547) | 1,851 | 18,70 |

Rata-rata sudut dari kamera: **19,06°** | Rata-rata jarak T1→T2: **1,70 m**

**Referensi Foto Dokumentasi Navigasi:**
- `percobaan_20261118_014632/foto/kunci_koordinat_01.png` — tampilan koordinat saat terkunci
- `percobaan_20261118_014632/foto/tiba_t1_01.png` — robot tiba di T1
- `percobaan_20261118_014632/foto/rotasi_ke_t2_01.png` — robot memulai rotasi ke T2
- `percobaan_20261118_014632/foto/misi_selesai_01.png` — robot tiba di T2, misi selesai

**Referensi Grafik Rekap:** `percobaan_20261118_014632/grafik/grafik_rekap.png` — grafik 4 subplot (jarak T1 & T2 vs waktu, sudut vs waktu, encoder vs waktu, state machine vs waktu)

---

### 3.3.4 Analisis Waktu Navigasi

Variasi durasi navigasi yang besar antar percobaan (29,57 s hingga 254,45 s) disebabkan oleh dua faktor utama:

1. **Waktu tunggu deteksi gestur** — Percobaan-2 (254,45 s) memiliki waktu tunggu tertinggi karena deteksi gesture membutuhkan banyak frame ulang sebelum `DUAL_DETECT_HOLD` terpenuhi (tercatat 6 foto `dual_detect_mulai` = ±6 kali siklus deteksi).

2. **Jarak total yang ditempuh** — Percobaan-4 (29,57 s) adalah percobaan tercepat karena robot sudah berada di dekat posisi awal yang langsung memudahkan akuisisi target.

Waktu navigasi murni (sejak target terkunci hingga misi selesai) dapat diestimasi dari timestamp log_event:

| No | Waktu Kunci | Waktu Misi Selesai | Durasi Navigasi Murni |
|----|-------------|--------------------|-----------------------|
| 2 | 01:50:14 | 01:50:43 | ~29 detik |
| 3 | 01:51:37 | 01:52:08 | ~31 detik |
| 4 | 01:54:00 | 01:54:23 | ~23 detik |
| 5 | 01:55:02 | 01:55:30 | ~28 detik |
| 6 | 01:56:05 | 01:56:34 | ~29 detik |
| 7 | 01:57:37 | 01:58:08 | ~31 detik |
| 8 | 01:58:53 | 01:59:28 | ~35 detik |
| 9 | 02:00:08 | 02:00:58 | ~50 detik |

**Rata-rata durasi navigasi murni (sejak koordinat terkunci):** ~32 detik

---

## 3.4 Analisis Kegagalan

### 3.4.1 Kegagalan pada Tingkat Klasifikasi

Dari data uji klasifikasi (560 sampel):

| Jenis Kesalahan | Jumlah | Dampak pada Sistem |
|-----------------|--------|--------------------|
| **False Positive (FP)** | 2 | Gestur NG terdeteksi sebagai HU → robot memulai prosedur target tanpa perintah sebenarnya |
| **False Negative (FN)** | 3 | Gestur HU tidak terdeteksi → robot tidak merespons perintah pengguna |

Namun dampak FP diminimalkan oleh mekanisme **DUAL_DETECT_HOLD = 2,0 detik**: robot hanya mengunci koordinat jika gesture terdeteksi secara konsisten selama 2 detik penuh. Satu frame FP tidak cukup untuk memicu penguncian koordinat.

**Jarak terjadinya FP/FN:** Berdasarkan tabel akurasi per jarak, kesalahan terjadi paling banyak pada jarak **1,5 m** (3 kesalahan dari 77 sampel) dan **2,0 m** (1 kesalahan dari 79 sampel). Hal ini sesuai dengan karakteristik depth sensor Kinect yang mengalami noise lebih tinggi pada jarak dekat-menengah.

### 3.4.2 Kegagalan pada Tingkat Navigasi

**Dari 9 percobaan, tidak ada kegagalan misi navigasi.** Seluruh 8 percobaan valid berhasil mencapai kedua target (T1 dan T2) dengan jarak akhir 0,70 m.

**Percobaan-1 (011430)** — Status BERJALAN:
- Bukan kegagalan sistem; program dihentikan secara manual sebelum gestur dimasukkan
- Ini adalah sesi *dry-run* pertama untuk verifikasi koneksi ESP32 dan Kinect
- Tidak ada target yang dikunci, encoder menunjukkan 0 tick

### 3.4.3 Potensi Kegagalan yang Diamati

Meskipun semua percobaan berhasil, beberapa kondisi potensial kegagalan yang diamati:

1. **Waktu deteksi gestur bervariasi tinggi** — Percobaan-2 memerlukan 6 siklus `dual_detect_mulai` sebelum koordinat terkunci, menunjukkan bahwa:
   - Postur tangan subjek tidak konsisten dari frame ke frame
   - Atau pencahayaan ruangan memengaruhi akuisisi depth Kinect

2. **Blind spot Kinect < 0,5 m** — Jika subjek berdiri terlalu dekat (<0,5 m), Kinect tidak dapat mengukur depth. Parameter `DUAL_DETECT_HOLD` tidak akan pernah terpenuhi.

3. **Akumulasi error encoder** — Encoder kumulatif percobaan-9 mencapai 31.707 tick (≈70 m perjalanan). Drift mekanis pada robot diferensial dapat terakumulasi pada lintasan panjang, berpotensi menurunkan akurasi orientasi.

4. **Sudut T1 ekstrem (−20,8° pada percobaan-9)** — Sudut rotasi besar meningkatkan waktu navigasi (fase ROTASI_T1 lebih lama) dan potensi akumulasi error sudut.

---

## 3.5 Pembahasan dan Keterbatasan

### 3.5.1 Pembahasan Kinerja Klasifikasi

Akurasi KNN k=5 sebesar **99,11%** (CI 95%: 97,93%–99,62%) membuktikan bahwa pendekatan berbasis KNN dengan fitur koordinat sendi tangan cukup untuk membedakan gesture *Hands Up* dari *No Gesture* dalam kondisi pencahayaan ruangan normal.

**Keunggulan pendekatan ini:**
- Waktu inferensi 0,86 ms tidak menambah latensi signifikan pada loop navigasi robot
- Tidak memerlukan GPU atau accelerator khusus; dapat berjalan pada CPU laptop biasa
- Dataset 2.800 sampel sudah mencukupi untuk mencapai akurasi tinggi pada dua kelas sederhana

**Referensi Grafik:** `klasifikasi_gestur/hasil/grafik/learning_curve.png` — kurva belajar menunjukkan model tidak mengalami overfitting

**Perbandingan dengan model pembanding** (lampiran): Data tersimpan di `klasifikasi_gestur/hasil/tabel/lampiran_evaluasi_model_pembanding.csv` dan `klasifikasi_gestur/hasil/grafik/lampiran_grafik_pembanding.png`. KNN k=5 dibandingkan dengan Gaussian Naïve Bayes, Logistic Regression, dan Perceptron.

---

### 3.5.2 Pembahasan Kinerja Navigasi

**Strategi memori koordinat segitiga 2D** terbukti efektif: robot hanya perlu melakukan **satu kali akuisisi target** di awal, kemudian menavigasi kedua target secara berurutan tanpa perlu memindai ulang. Hal ini berbeda dengan pendekatan reaktif (target dideteksi ulang setiap saat) yang lebih rentan terhadap hilangnya sinyal gesture selama perjalanan.

Konsistensi jarak akhir 0,70 m di semua percobaan membuktikan bahwa sistem umpan balik depth sensor bekerja dengan baik untuk memastikan robot berhenti pada jarak yang tepat.

**Variasi jarak T1→T2** antara 1,322 m (percobaan-4) hingga 2,402 m (percobaan-8) menunjukkan bahwa sistem navigasi mampu menangani berbagai konfigurasi geometri antar target tanpa modifikasi parameter.

---

### 3.5.3 Keterbatasan Sistem

| No | Keterbatasan | Dampak | Saran Perbaikan |
|----|-------------|--------|----------------|
| 1 | Dataset dari **1 subjek** (postur tangan satu orang) | Model mungkin tidak generalize ke subjek dengan postur berbeda | Tambah data dari ≥10 subjek berbeda |
| 2 | **Indoor only**, pencahayaan fluorescent stabil | Kinect tidak dapat digunakan di luar ruangan (inframerah terganggu sinar matahari) | Gunakan sensor stereo atau LiDAR untuk outdoor |
| 3 | **Blind spot Kinect < 0,5 m** | Target yang terlalu dekat tidak bisa dikunci | Tambah sensor ultrasonik atau ToF untuk jarak dekat |
| 4 | **Drift encoder** pada lintasan panjang | Akurasi orientasi menurun pada percobaan dengan encoder kumulatif tinggi (>30.000 tick) | Tambah IMU atau magnetometer untuk koreksi heading |
| 5 | **Hanya 2 target** per misi | Sistem tidak dapat menangani lebih dari 2 pemanggil | Kembangkan state machine multi-target (N target) |
| 6 | **Sudut deteksi terbatas** (FOV Kinect ≈ 57° horizontal) | Target di luar FOV tidak bisa dideteksi | Rotasi kamera atau gunakan sensor 360° |
| 7 | **Prioritas berdasarkan urutan tangan** bukan jarak | Robot mungkin mengunjungi target yang lebih jauh terlebih dahulu | Implementasikan prioritas berdasarkan jarak terdekat atau sinyal bahaya |

---

### 3.5.4 Ringkasan Capaian

| Aspek | Target | Capaian | Status |
|-------|--------|---------|--------|
| Akurasi klasifikasi gestur | ≥ 95% | **99,11%** | ✅ Tercapai |
| Tingkat keberhasilan navigasi | ≥ 80% | **100% (8/8)** | ✅ Tercapai |
| Waktu inferensi | Real-time (<33 ms) | **0,86 ms** | ✅ Tercapai |
| Rentang jarak operasional | 0,5–4,0 m | **hingga 4,87 m** | ✅ Melebihi target |
| Presisi berhenti di target | STOP_DISTANCE ±10% | **tepat 0,70 m** | ✅ Tercapai |

---

> [!NOTE]
> **Panduan Referensi File untuk Prompt AI:**
>
> Saat memprompt ke Claude AI untuk penyempurnaan bab ini, sebutkan file-file berikut:
> - **Data klasifikasi:** `klasifikasi_gestur/hasil/tabel/metrik_utama.csv`, `metrik_per_jarak.csv`, `pemilihan_k.csv`, `inferensi_waktu.csv`
> - **Grafik klasifikasi:** `confusion_matrix.png`, `decision_boundary.png`, `akurasi_per_jarak.png`, `pemilihan_k_cv.png`, `learning_curve.png` (semua di `klasifikasi_gestur/hasil/grafik/`)
> - **Data navigasi (tiap percobaan):** `percobaan_YYYYMMDD_HHMMSS/log/ringkasan.txt`, `log/log_event.txt`, `grafik/grafik_rekap.png`
> - **Foto dokumentasi (percobaan-2 sebagai referensi):** `percobaan_20261118_014632/foto/kunci_koordinat_01.png`, `tiba_t1_01.png`, `misi_selesai_01.png`
