# DOKUMEN BEDAH TEKNIS & FORMULASI MATEMATIS KLASIFIKASI GESTUR HANDS UP (KNN)

Dokumen ini disusun khusus untuk menjawab revisi dosen/penguji skripsi: **membedah tuntas aspek klasifikasi gestur dengan fokus mendalam pada K-Nearest Neighbors (KNN)**, membuktikan dasar pemilihan parameter, menjabarkan alur matematis langkah demi langkah, serta menyajikan data pembuktian empiris secara komprehensif.

---

## 1. ALUR KERJA LENGKAP SISTEM KLASIFIKASI (PIPELINE)

Alur kerja pengenalan gestur dari sensor hingga menghasilkan keputusan terdiri dari 4 tahapan berurutan:

```
[Citra Kamera Kinect v1] (RGB 640x480)
           │
           ▼
[Tahap 1: Deteksi Manusia (YOLOv8n)]
  - Model: yolov8n.pt (Confidence >= 0.5)
  - Filter Dimensi: Lebar >= 35 px, Tinggi >= 70 px
  - Luaran: Bounding Box Pemanggil (x1, y1, x2, y2)
           │
           ▼ (Crop Bounding Box)
[Tahap 2: Ekstraksi Pose Kerangka (MediaPipe Pose)]
  - Model Complexity: 1
  - Deteksi Keypoints:
      * Sendi Bahu Kiri:        LS = (x_11, y_11)
      * Sendi Bahu Kanan:       RS = (x_12, y_12)
      * Pergelangan Tangan Kiri:  LW = (x_15, y_15)
      * Pergelangan Tangan Kanan: RW = (x_16, y_16)
           │
           ▼
[Tahap 3: Rekayasa Fitur Geometris 2D (Feature Engineering)]
  - Formulasi Selisih Ketinggian Vertikal Ternormalisasi:
      f_L = y_LW - y_LS  = y_15 - y_11
      f_R = y_RW - y_RS  = y_16 - y_12
  - Vektor Fitur: X = [f_L, f_R] in R^2
           │
           ▼
[Tahap 4: Klasifikasi Ruang Spasial K-Nearest Neighbors (KNN)]
  - Parameter: k = 5, Metrik: Euclidean Distance
  - Voting Mayoritas:
      y = argmax_{c in {0, 1}} sum_{i=1}^k I(y_i == c)
           │
           ▼
[Keputusan Akhir: 1 (Hands Up - Panggilan Aktif) / 0 (No Gesture - Abaikan)]
```

---

## 2. FORMULASI MATEMATIS LENGKAP

### A. Fitur Geometris Ternormalisasi ($f_L, f_R$)
MediaPipe menghasilkan koordinat ternormalisasi terhadap dimensi *bounding box* orang:
$$y \in [0.0, 1.0]$$
dengan titik koordinat $(0, 0)$ berada di pojok kiri atas citra dan $(1, 1)$ di pojok kanan bawah.

1. **Fitur Lengan Kiri ($f_L$):**
   $$f_L = y_{LW} - y_{LS}$$
2. **Fitur Lengan Kanan ($f_R$):**
   $$f_R = y_{RW} - y_{RS}$$

**Interpretasi Nilai Fisis:**
* **$f < 0$ (Bernilai Negatif):** Koordinat pergelangan tangan lebih kecil dari bahu $\rightarrow$ posisi tangan secara fisis berada **LEBIH TINGGI daripada bahu (Hands Up)**.
* **$f > 0$ (Bernilai Positif):** Posisi tangan berada **LEBIH RENDAH daripada bahu (Rileks / No Gesture)**.
* **Invariansi Jarak:** Karena $y$ merupakan rasio proporsional terhadap tinggi tubuh orang, rasio ini **tidak berubah** meskipun pemanggil berdiri dekat ($0,5$ m) maupun jauh ($3,5$ m).

---

### B. Perhitungan Jarak Euclidean pada KNN
Diberikan titik sampel baru hasil ekstraksi fitur $X_{uji} = (f_{L, uji}, f_{R, uji})$ dan sekumpulan titik data latih $X_i = (f_{L, i}, f_{R, i})$ untuk $i = 1, 2, \dots, N$:

Jarak spasial Euclidean $d(X_{uji}, X_i)$ dihitung dengan:
$$d(X_{uji}, X_i) = \sqrt{(f_{L, uji} - f_{L, i})^2 + (f_{R, uji} - f_{R, i})^2}$$

### C. Pemilihan Tetangga Terdekat & Voting Mayoritas
1. Urutkan seluruh jarak dari yang terkecil:
   $$d_{(1)} \le d_{(2)} \le \dots \le d_{(k)} \le \dots \le d_{(N)}$$
2. Ambil himpunan indeks $k$ tetangga terdekat:
   $$\mathcal{N}_k(X_{uji}) = \{i_1, i_2, \dots, i_k\}$$
3. Hitung probabilitas posterior kelas $c \in \{0, 1\}$:
   $$P(Y = c \mid X_{uji}) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(X_{uji})} \mathbb{I}(y_i = c)$$
   di mana $\mathbb{I}(\cdot)$ adalah fungsi indikator ($1$ jika kondisi terpenuhi, $0$ jika tidak).
4. Keputusan kelas akhir dengan voting mayoritas:
   $$\hat{y} = \begin{cases} 1 \text{ (Hands Up)}, & \text{jika } P(Y=1 \mid X_{uji}) \ge 0.5 \\ 0 \text{ (No Gesture)}, & \text{jika } P(Y=1 \mid X_{uji}) < 0.5 \end{cases}$$

---

## 3. CONTOH NUMERIK PERHITUNGAN LANGKAH DEMI LANGKAH (STEP-BY-STEP)

Misalkan robot mendeteksi pemanggil baru pada jarak $2.0$ meter dengan hasil ekstraksi koordinat MediaPipe:
* Bahu Kiri ($y_{LS}$): $0,385$
* Pergelangan Tangan Kiri ($y_{LW}$): $0,145$ (tangan kiri diangkat)
* Bahu Kanan ($y_{RS}$): $0,390$
* Pergelangan Tangan Kanan ($y_{RW}$): $0,620$ (tangan kanan di bawah)

### Langkah 1: Hitung Fitur
$$f_L = 0,145 - 0,385 = -0,240$$
$$f_R = 0,620 - 0,390 = +0,230$$
Vektor fitur uji: $X_{uji} = (-0,240, +0,230)$

### Langkah 2: Hitung Jarak ke Titik Latih Terdekat
Misalkan 5 sampel latih terdekat di database adalah:
* Titik 1: $(-0,250, +0,220)$ dengan kelas $y_1 = 1$
  $$d_1 = \sqrt{(-0,240 - (-0,250))^2 + (0,230 - 0,220)^2} = \sqrt{(0,010)^2 + (0,010)^2} = \sqrt{0,0002} \approx 0,0141$$
* Titik 2: $(-0,235, +0,240)$ dengan kelas $y_2 = 1$
  $$d_2 = \sqrt{(-0,240 - (-0,235))^2 + (0,230 - 0,240)^2} = \sqrt{(-0,005)^2 + (-0,010)^2} = \sqrt{0,000125} \approx 0,0112$$
* Titik 3: $(-0,220, +0,210)$ dengan kelas $y_3 = 1$
  $$d_3 = \sqrt{(-0,240 - (-0,220))^2 + (0,230 - 0,210)^2} = \sqrt{(-0,020)^2 + (0,020)^2} = \sqrt{0,0008} \approx 0,0283$$
* Titik 4: $(-0,260, +0,250)$ dengan kelas $y_4 = 1$
  $$d_4 = \sqrt{(-0,240 - (-0,260))^2 + (0,230 - 0,250)^2} = \sqrt{(0,020)^2 + (-0,020)^2} = \sqrt{0,0008} \approx 0,0283$$
* Titik 5: $(-0,010, +0,280)$ dengan kelas $y_5 = 0$ (titik batas)
  $$d_5 = \sqrt{(-0,240 - (-0,010))^2 + (0,230 - 0,280)^2} = \sqrt{(-0,230)^2 + (-0,050)^2} = \sqrt{0,0554} \approx 0,2354$$

### Langkah 3: Voting Mayoritas ($k = 5$)
* Tetangga Kelas 1 (Hands Up): Titik 1, 2, 3, 4 (Total = 4 suara)
* Tetangga Kelas 0 (No Gesture): Titik 5 (Total = 1 suara)
* Probabilitas: $P(Y=1) = \frac{4}{5} = 80\%$

**Kesimpulan:** $\hat{y} = 1$ (**Hands Up Terkonfirmasi**). Robot memproses posisi pemanggil sebagai target pelayanan.

---

## 4. PEMBUKTIAN ILMIAH: MENGAPA $k = 5$ ADALAH PILIHAN PALING TEPAT?

Berdasarkan pengujian *Stratified 5-Fold Cross Validation* pada dataset 2.800 sampel:

| Nilai $k$ | Akurasi Rata-rata CV | Simpangan Baku ($\pm \sigma$) | Karakteristik / Analisis Fisis |
| :---: | :---: | :---: | :--- |
| **$k = 1$** | $99,71\%$ | $\pm 0,14\%$ | **Rentan Overfitting:** Sangat sensitif terhadap *outlier* (misal sentakan gerakan tangan sesaat). |
| **$k = 3$** | $99,75\%$ | $\pm 0,11\%$ | Akurasi numerik tertinggi, namun batas wilayah keputusan terlalu rapat. |
| **$k = 5$ (Dipilih)** | **$99,61\%$** | **$\pm 0,16\%$** | **Optimal & Stabil:** Selisih akurasi hanya $0,14\%$ dari $k=3$, namun memiliki wilayah keputusan (*decision boundary*) yang jauh lebih halus (*smooth*) dan tahan terhadap derau koordinat (*noise*) pergerakan tangan saat berjalan. |
| **$k = 7$** | $99,57\%$ | $\pm 0,18\%$ | Mulai terjadi pergeseran batas kelas pada wilayah transisi. |
| **$k = 9$** | $99,54\%$ | $\pm 0,18\%$ | Akurasi mulai menurun secara bertahap. |
| **$k = 15$** | $99,39\%$ | $\pm 0,22\%$ | **Underfitting:** Wilayah minoritas mulai tereduksi oleh kerapatan mayoritas. |

---

## 5. ALASAN MEMILIH KNN DIBANDINGKAN MODEL LAIN (JUSTIFIKASI DOKUMENTASI)

1. **Bentuk Batas Keputusan Non-Linear Sederhana:**
   Batas pemisah antara tangan di atas bahu dan tangan di bawah bahu membentuk bidang non-linear kuadran pada ruang $\mathbb{R}^2$. Model linear seperti Logistic Regression ($86,43\%$) dan Perceptron ($82,86\%$) gagal menangkap fleksibilitas gestur satu tangan vs dua tangan, sementara KNN ($99,11\%$) mampu membentuk batas pemisah melengkung secara alamiah (lihat file [`hasil/grafik/decision_boundary.png`](file:///D:/Project/Skripsi/Mobile-Robot-Differential-Navigation-with-Dual-Caller-Hands-Up-Priority-2D-Position-Estimation/klasifikasi_gestur/hasil/grafik/decision_boundary.png)).
2. **Kecepatan Inferensi Ultra Cepat ($0,86$ ms):**
   Dengan dimensi fitur hanya 2 ($f_L, f_R$) dan $2.800$ titik latih, pencarian 5 tetangga terdekat hanya membutuhkan waktu **$0,8607$ milidetik** per frame ($\approx 1.162$ FPS). Ini tidak membebani prosesor saat robot berjalan.
3. **Bebas Komputasi Pelatihan Ulang (Instance-Based):**
   Jika di masa depan ada subjek baru dengan rentang gerak khusus, data cukup ditambahkan ke database tanpa perlu melatih ulang arsitektur jaringan (*non-parametric*).

---

## 6. METRIK EVALUASI KUANTITATIF (LENGKAP UNTUK BAB 4)

Pengujian pada data uji independen ($N = 560$ sampel):

* **Akurasi:** $99,11\%$ ($555 / 560$ benar)
* **Wilson Score Interval 95%:** $[97,93\%, 99,62\%]$
* **Confusion Matrix:**
  * $\text{True Positives (TP)} = 277$
  * $\text{True Negatives (TN)} = 278$
  * $\text{False Positives (FP)} = 2$
  * $\text{False Negatives (FN)} = 3$
* **Presisi Kelas Hands Up:** $99,28\%$
* **Recall Kelas Hands Up:** $98,93\%$
* **F1-Score Kelas Hands Up:** $99,11\%$
* **Konsistensi Jarak:** Akurasi pada $0,5$m ($100\%$), $1,0$m ($100\%$), $1,5$m ($97,40\%$), $2,0$m ($98,73\%$), $2,5$m ($98,77\%$), $3,0$m ($100\%$), $3,5$m ($98,78\%$).

