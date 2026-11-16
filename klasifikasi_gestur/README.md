# Sistem Evaluasi Klasifikasi Gestur Hands Up (MediaPipe Pose + KNN)

Repositori modul ini merupakan sistem evaluasi mandiri yang dirancang khusus untuk menguji, memvalidasi, dan mendokumentasikan performa klasifikasi gestur **Hands Up** berbasis ekstraksi pose MediaPipe dan algoritma *K-Nearest Neighbors* (KNN).

Modul ini **terpisah total** dari skrip navigasi robot, menghasilkan seluruh tabel dan grafik resolusi tinggi (300 DPI) yang siap pakai untuk artikel jurnal atau naskah skripsi.

---

## 📁 Struktur Direktori

```text
klasifikasi_gestur/
├── config.py                 # Parameter terpusat (SEED=42, k, path, threshold)
├── 01_kumpul_data.py         # Perekaman dataset cerdas berbasis gestur trigger (Kinect v1 via OpenNI2)
├── 02_latih_evaluasi.py      # Pelatihan, tuning k (CV), learning curve, metrik & analisis per jarak
├── 03_visualisasi.py         # Visualisasi foto hasil klasifikasi (TP/TN/FP/FN) & kolase figur jurnal
├── 04_pembanding.py          # Lampiran: Pembanding model (GNB, LogReg, Perceptron) + Uji McNemar
├── dataset/
│   ├── dataset.csv           # Master dataset 2.800 sampel lengkap (jarak 0,5m - 3,5m)
│   └── images/raw/           # Foto frame asli kamera Kinect ({id}_{kelas}_{jarak}m.jpg)
├── model/
│   └── knn_k5.pkl            # Model KNN terlatih (k=5) hasil evaluasi 2.800 sampel
├── hasil/
│   ├── log/                  # Log eksekusi ber-timestamp & dump environment (pip freeze)
│   ├── grafik/               # Seluruh grafik visualisasi (300 DPI)
│   ├── tabel/                # Seluruh tabel metrik (format CSV)
│   └── foto_klasifikasi/     # Foto teranotasi hasil uji per kategori (TP/, TN/, FP/, FN/)
└── README.md                 # Panduan penggunaan dan laporan hasil pengujian lengkap
```

---

## 📊 Hasil Evaluasi Aktual Model (Dataset 2.800 Sampel Baru)

> **Format Angka:** Memakai desimal standar Indonesia (koma `,`).

### 1. Metrik Evaluasi Utama (Held-out Test Set 20%, $N = 560$ sampel)

| Parameter / Metrik | Nilai Hasil Pengujian | Keterangan |
| :--- | :---: | :--- |
| **Total Sampel Dataset** | **2.800 sampel** | 1.400 Hands Up (Kelas 1), 1.400 No Gesture (Kelas 0) |
| **Rentang Jarak Pengujian** | **0,5 m – 3,5 m** | 7 variasi jarak (masing-masing 400 sampel) |
| **Nilai $k$ Optimal (5-Fold CV)** | $k = 3$ ($99,75\%$) & $k = 5$ ($99,61\%$) | Nilai $k=5$ terbukti sangat stabil dan optimal |
| **Akurasi Data Uji (Held-out 20%)** | **$99,11\%$** | **555 dari 560 sampel uji terklasifikasi benar** |
| **Wilson Score Interval 95%** | **[$97,93\%$, $99,62\%$]** | Selang kepercayaan statistik akurasi |
| **True Positives (TP)** | 277 sampel | Gestur Hands Up benar dikenali |
| **True Negatives (TN)** | 278 sampel | Kondisi No Gesture benar dikenali |
| **False Positives (FP)** | 2 sampel | Salah deteksi (hanya 2 dari 560) |
| **False Negatives (FN)** | 3 sampel | Gagal deteksi (hanya 3 dari 560) |
| **Presisi (Hands Up)** | **$99,28\%$** ($0,9928$) | Kerapatan ketepatan panggilan positif |
| **Recall (Hands Up)** | **$98,93\%$** ($0,9893$) | Sensitivitas penangkapan panggilan gestur |
| **F1-Score (Hands Up)** | **$99,11\%$** ($0,9911$) | Harmonisasi sempurna presisi dan recall |
| **Waktu Inferensi Rata-rata** | **$0,8607$ ms** per frame | Throughput estimasi: $\approx 1.162$ FPS |

---

### 2. Analisis Performa Berdasarkan Jarak Sensor (0,5 m – 3,5 m)

Pengujian terhadap 560 sampel uji pada tiap variasi jarak menunjukkan ketahanan luar biasa dari sensor Kinect v1 dan normalisasi MediaPipe:

| Jarak Pengujian | Jumlah Sampel Uji | Akurasi | Presisi Hands Up | Recall Hands Up | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0,5 m** | 69 | **$100,00\%$** | $100,00\%$ | $100,00\%$ | $1,0000$ |
| **1,0 m** | 82 | **$100,00\%$** | $100,00\%$ | $100,00\%$ | $1,0000$ |
| **1,5 m** | 77 | **$97,40\%$** | $97,50\%$ | $97,50\%$ | $0,9750$ |
| **2,0 m** | 79 | **$98,73\%$** | $100,00\%$ | $97,22\%$ | $0,9859$ |
| **2,5 m** | 81 | **$98,77\%$** | $100,00\%$ | $97,83\%$ | $0,9890$ |
| **3,0 m** | 90 | **$100,00\%$** | $100,00\%$ | $100,00\%$ | $1,0000$ |
| **3,5 m** | 82 | **$98,78\%$** | $97,56\%$ | $100,00\%$ | $0,9877$ |

---

### 3. Hasil Perbandingan Model & Uji McNemar (Lampiran Naskah)

Berdasarkan pengujian pada data uji yang sama ($N=560$):
* **KNN ($k=5$):** Akurasi **$99,11\%$** | F1-Score **$99,11\%$**
* **Gaussian Naive Bayes:** Akurasi $94,64\%$ | F1-Score $94,63\%$
* **Logistic Regression:** Akurasi $86,43\%$ | F1-Score $86,43\%$
* **Perceptron:** Akurasi $82,86\%$ | F1-Score $82,55\%$

Uji statistik **McNemar** membuktikan bahwa keunggulan model KNN atas Gaussian Naive Bayes ($p = 1,62 \times 10^{-5}$), Logistic Regression ($p = 3,44 \times 10^{-15}$), dan Perceptron ($p < 10^{-16}$) adalah **signifikan secara statistik ($p < 0,05$)**.

---

## 📝 Saran Kalimat untuk Naskah Jurnal / Skripsi

### Bab Metode Penelitian:
> *"Sistem klasifikasi gestur panggilan Hands Up menggunakan arsitektur hibrida YOLOv8n dan MediaPipe Pose. Kamera RGB sensor Kinect v1 ($640 \times 480$ piksel) mendeteksi keberadaan tubuh manusia menggunakan YOLOv8n dengan batas ambang keyakinan $0,5$. Area tubuh terpangkas kemudian diproses oleh MediaPipe Pose untuk mengekstraksi koordinat vertikal sendi bahu kiri ($y_{LS}$), bahu kanan ($y_{RS}$), pergelangan tangan kiri ($y_{LW}$), dan pergelangan tangan kanan ($y_{RW}$). Dua fitur geometris ternormalisasi dirumuskan melalui $f_L = y_{LW} - y_{LS}$ dan $f_R = y_{RW} - y_{RS}$, di mana nilai negatif merepresentasikan pergelangan tangan berada lebih tinggi daripada sendi bahu. Model K-Nearest Neighbors (KNN) dengan parameter $k=5$ dan jarak Euclidean digunakan untuk mengklasifikasikan status panggilan secara instan."*

### Bab Hasil dan Pembahasan:
> *"Evaluasi empiris model KNN dilakukan menggunakan total 2.800 sampel data yang direkam secara seimbang antara gestur Hands Up dan No Gesture pada 7 variasi jarak pengujian mulai dari $0,5$ meter hingga $3,5$ meter (interval $0,5$ m). Melalui validasi silang bertingkat (stratified 5-fold cross-validation), konfigurasi hiperparameter $k=5$ membuktikan kestabilan optimal dengan akurasi validasi mencapai $99,61\%$. Pada evaluasi data uji independen ($20\%$ held-out test set, $N = 560$), model KNN berhasil meraih akurasi sebesar $99,11\%$ dengan selang kepercayaan Wilson $95\%$ berada pada rentang $[97,93\%, 99,62\%]$. Rincian matriks konfusi mencatatkan 277 True Positives, 278 True Negatives, 2 False Positives, dan 3 False Negatives, yang menghasilkan presisi $99,28\%$ serta recall $98,93\%$ untuk kelas panggilan Hands Up. Analisis performa berdasarkan variasi jarak membuktikan tingkat akurasi konsisten di atas $97\%$ pada seluruh titik pengujian, bahkan mencapai $100\%$ pada jarak $0,5$ m, $1,0$ m, dan $3,0$ m. Tolok ukur latensi komputasi mencatatkan rata-rata waktu inferensi sebesar $0,8607$ milidetik per frame (throughput setara $\approx 1.162$ FPS), membuktikan efisiensi komputasi yang sangat tinggi untuk navigasi otonom waktu-nyata."*

---

## 📌 Lokasi Berkas Keluaran

* **Model Terlatih Baru:** [`model/knn_k5.pkl`](file:///D:/Project/Skripsi/Mobile-Robot-Differential-Navigation-with-Dual-Caller-Hands-Up-Priority-2D-Position-Estimation/klasifikasi_gestur/model/knn_k5.pkl)
* **Dataset Master:** [`dataset/dataset.csv`](file:///D:/Project/Skripsi/Mobile-Robot-Differential-Navigation-with-Dual-Caller-Hands-Up-Priority-2D-Position-Estimation/klasifikasi_gestur/dataset/dataset.csv)
* **Grafik Publikasi (300 DPI):**
  * [`hasil/grafik/akurasi_per_jarak.png`](file:///D:/Project/Skripsi/Mobile-Robot-Differential-Navigation-with-Dual-Caller-Hands-Up-Priority-2D-Position-Estimation/klasifikasi_gestur/hasil/grafik/akurasi_per_jarak.png) (Grafik Akurasi Jarak 0,5m - 3,5m)
  * [`hasil/grafik/confusion_matrix.png`](file:///D:/Project/Skripsi/Mobile-Robot-Differential-Navigation-with-Dual-Caller-Hands-Up-Priority-2D-Position-Estimation/klasifikasi_gestur/hasil/grafik/confusion_matrix.png) (Matriks Konfusi)
  * [`hasil/grafik/decision_boundary.png`](file:///D:/Project/Skripsi/Mobile-Robot-Differential-Navigation-with-Dual-Caller-Hands-Up-Priority-2D-Position-Estimation/klasifikasi_gestur/hasil/grafik/decision_boundary.png) (Batas Keputusan KNN)
  * [`hasil/grafik/kolase_klasifikasi_jurnal.png`](file:///D:/Project/Skripsi/Mobile-Robot-Differential-Navigation-with-Dual-Caller-Hands-Up-Priority-2D-Position-Estimation/klasifikasi_gestur/hasil/grafik/kolase_klasifikasi_jurnal.png) (Kolase Foto TP, TN, FP, FN)
  * [`hasil/grafik/lampiran_grafik_pembanding.png`](file:///D:/Project/Skripsi/Mobile-Robot-Differential-Navigation-with-Dual-Caller-Hands-Up-Priority-2D-Position-Estimation/klasifikasi_gestur/hasil/grafik/lampiran_grafik_pembanding.png) (Grafik Pembanding Model)
* **Tabel Metrik CSV:** [`hasil/tabel/`](file:///D:/Project/Skripsi/Mobile-Robot-Differential-Navigation-with-Dual-Caller-Hands-Up-Priority-2D-Position-Estimation/klasifikasi_gestur/hasil/tabel)
