# Ringkasan Analisis Percobaan 1–9 dan Peran KNN dalam Sistem Navigasi

> **Pernyataan Integritas Data Akademik**:
> Laporan ini disusun secara objektif dan ketat tanpa memanipulasi, mengarang, atau memperhalus data. Seluruh angka, waktu, dan label berasal dari perhitungan nyata atas berkas log (`log_event.txt`, `log_frame*.csv`, `ringkasan.txt`) dan inferensi ulang model `knn_k5.pkl` pada foto dokumentasi percobaan.

---

## A. Inventaris Berkas per Percobaan

Pemeriksaan kelengkapan berkas pada direktori `multi_target_v2/hasil_percobaan/`:

| No. Percobaan | Folder Sesi | Foto (`foto/`) | Log Frame (`log/`) | Event Log (`log/`) | Ringkasan (`log/`) | Grafik (`grafik/`) | Status Kelengkapan |
|---|---|---|---|---|---|---|---|
| **1** | `percobaan_20261118_011430` | 0 foto | `log_frame1.csv` (71 baris) | `log_event.txt` (Ada) | `ringkasan.txt` (Ada) | `grafik_rekap.png` (Ada) | **Foto tidak ada** (Sesi dry-run, program dihentikan sebelum deteksi selesai) |
| **2** | `percobaan_20261118_014632` | 15 foto | `log_frame2.csv` (1.558 baris) | `log_event.txt` (Ada) | `ringkasan.txt` (Ada) | `grafik_rekap.png` (Ada) | Lengkap |
| **3** | `percobaan_20261118_015047` | 14 foto* | `log_frame3.csv` (1.413 baris) | `log_event.txt` (Ada) | `ringkasan.txt` (Ada) | `grafik_rekap.png` (Ada) | Lengkap (*memiliki `foto_original_backup/` 14 foto) |
| **4** | `percobaan_20261118_015357` | 9 foto | `log_frame4.csv` (201 baris) | `log_event.txt` (Ada) | `ringkasan.txt` (Ada) | `grafik_rekap.png` (Ada) | Lengkap |
| **5** | `percobaan_20261118_015429` | 19 foto | `log_frame5.csv` (433 baris) | `log_event.txt` (Ada) | `ringkasan.txt` (Ada) | `grafik_rekap.png` (Ada) | Lengkap |
| **6** | `percobaan_20261118_015534` | 10 foto | `log_frame6.csv` (782 baris) | `log_event.txt` (Ada) | `ringkasan.txt` (Ada) | `grafik_rekap.png` (Ada) | Lengkap |
| **7** | `percobaan_20261118_015731` | 9 foto | `log_frame7.csv` (453 baris) | `log_event.txt` (Ada) | `ringkasan.txt` (Ada) | `grafik_rekap.png` (Ada) | Lengkap |
| **8** | `percobaan_20261118_015837` | 11 foto | `log_frame8.csv` (371 baris) | `log_event.txt` (Ada) | `ringkasan.txt` (Ada) | `grafik_rekap.png` (Ada) | Lengkap |
| **9** | `percobaan_20261118_015934` | 14 foto | `log_frame9.csv` (582 baris) | `log_event.txt` (Ada) | `ringkasan.txt` (Ada) | `grafik_rekap.png` (Ada) | Lengkap |

**Total Foto Dokumentasi Tersimpan**: 101 foto PNG (Percobaan 2–9).

---

## B. Analisis Asal Usul Label dan Anotasi pada Foto

Berdasarkan penelusuran kode sumber pada `multi_target_navigasi_v3.py` dan skrip `anotasi_foto.py`:

1. **Foto pada Fase `dual_detect_mulai`**:
   - Berasal dari luaran visual sistem langsung (*real-time live stream*). Pada baris 837–839 `multi_target_navigasi_v3.py`, fungsi `draw_hud(...)` dipanggil **sebelum** `logger.simpan_foto(...)`.
   - Bounding box, teks `Kandidat #X: X.XXm`, titik centroid (cyan), dan tulisan gestur `HANDS_UP` pada foto-foto `dual_detect_mulai_*.png` digambar secara otomatis oleh sistem saat pengujian berlangsung.

2. **Foto Setelah Penguncian Koordinat (`kunci_koordinat` hingga `misi_selesai`)**:
   - **Asli dari Sistem**: Pada kode `multi_target_navigasi_v3.py` asli (baris 871, 964, 1008, 1044), fungsi `logger.simpan_foto(...)` dipanggil menggunakan `frame.copy()` **mentah** tanpa HUD dan tanpa bounding box (karena `draw_hud` baru dipanggil di akhir perulangan utama pada baris 1120 sebelum `cv2.imshow`). Akibatnya, seluruh foto asli setelah penguncian berstatus **frame mentah (raw RGB frame)**.
   - **Anotasi Tambahan pada Percobaan 3 (`percobaan_20261118_015047`)**:
     Pada sesi modifikasi sebelumnya, file `anotasi_foto.py` dibuat untuk menimpa foto-foto `kunci_koordinat_01.png` s.d. `misi_selesai_01.png` dengan bounding box merah dan teks HUD simulasi.
   - **Kejujuran Teknis**: Label dan bounding box pada file `kunci_koordinat` hingga `tiba_t2` di folder `percobaan_20261118_015047/foto/` **bukan keluaran HUD asli saat robot berjalan**, melainkan hasil injeksi skrip overlay `anotasi_foto.py` berdasarkan konfigurasi posisi manual (`box_pct`) dan metadata log ringkasan. Foto mentah aslinya tersimpan utuh di `foto_original_backup/`.

3. **Log CSV Tidak Mencatat Label Gestur per Bingkai**:
   - Berkas `log_frame*.csv` hanya mencatat kolom: `Timestamp`, `Waktu_Detik`, `Fase_State`, `Target_Aktif`, `Jarak_Sensor_m`, `Sudut_Deg`, `Cmd_Jarak`, `Cmd_Sudut`, `Encoder_L`, `Encoder_R`, `Jarak_Tempuh_m`.
   - Nilai fitur ($f_L, f_R$) dan kelas prediksi KNN tidak dicatat per baris ke CSV log, sehingga pembuktian peran KNN harus dilakukan melalui re-inferensi pada foto dokumentasi.

---

## C & D. Hasil Inferensi Ulang KNN dan Penilaian Visual

Inferensi ulang dijalankan pada seluruh 101 foto percobaan 2–9 menggunakan:
- **Detektor Orang**: YOLOv8n (`conf >= 0.50`, lebar $\ge 35$ px, tinggi $\ge 70$ px).
- **Ekstraksi Landmark**: MediaPipe Pose (`model_complexity=1`, `min_detection_confidence=0.5`).
- **Fitur Masukan**: $f_L = y_{wrist\_L} - y_{shoulder\_L}$ dan $f_R = y_{wrist\_R} - y_{shoulder\_R}$ (koordinat ternormalisasi crop).
- **Model Pengklasifikasi**: `klasifikasi_gestur/model/knn_k5.pkl` ($k=5$).
- **Luaran Tersimpan**: `analisis_knn_sistem/inferensi_foto.csv` (156 baris instansi orang terdeteksi).

### Ringkasan Status Penilaian Visual:
- **Total Instansi Orang Terdeteksi YOLO**: 156 instansi
- **Landmark Pose Berhasil Diekstrak**: 128 instansi
- **Landmark Pose Gagal Diekstrak / Non-Person**: 28 instansi
- **Kategori Penilaian Visual Manusia**:
  - `ANGKAT` (Tangan terangkat di atas bahu): 47 instansi
  - `TURUN` (Tangan berada di bawah bahu / samping tubuh): 36 instansi
  - `TIDAK DAPAT DINILAI` (Buram, terhalang meja/kursi, di luar FOV vertikal): 67 instansi
  - `PERLU VERIFIKASI MANUSIA` (Posisi tangan tepat di garis bahu / ambigu): 6 instansi

---

## E. Tabel Perbandingan per Status Mesin

Berikut adalah perbandingan antara kondisi gestur visual nyata (`penilaian_visual`) dengan prediksi model `knn_k5.pkl` (`label_knn`):

| Status Mesin | Label Gestur yang Diharapkan | Total Deteksi Orang | Dapat Dinilai Visual | Cocok (Visual vs KNN) | Tidak Cocok | Tidak Dapat Dinilai | Catatan Visual Lapangan |
|---|---|---|---|---|---|---|---|
| **dual_detect_mulai** | HANDS_UP | 66 | 43 | 43 | 0 | 23 | 2 pemanggil mengangkat tangan; kandidat background tereliminasi |
| **kunci_koordinat** | HANDS_UP | 17 | 14 | 14 | 0 | 3 | T1 dan T2 terlihat jelas mengangkat tangan |
| **statelock_t1** | HANDS_UP | 17 | 10 | 10 | 0 | 7 | Robot berorientasi ke T1; T1 tangan terangkat |
| **tiba_t1** | HANDS_UP / TURUN | 8 | 4 | 4 | 0 | 4 | Jarak 0,7 m; tubuh atas pemanggil terpotong batas atas frame |
| **t1_selesai** | NO_GESTURE | 9 | 0 | 0 | 0 | 9 | Jarak 0,7 m; hanya tubuh bawah yang masuk kamera (`vis_min < 0.05`) |
| **rotasi_ke_t2** | HANDS_UP | 10 | 0 | 0 | 0 | 10 | Kamera bergerak rotasi cepat (motion blur) / target di luar FOV |
| **statelock_t2** | HANDS_UP | 13 | 7 | 7 | 0 | 6 | Robot berorientasi ke T2; T2 tangan terangkat |
| **tiba_t2** | HANDS_UP / TURUN | 8 | 4 | 4 | 0 | 4 | Jarak 0,7 m; pemanggil mulai menurunkan tangan |
| **misi_selesai** | NO_GESTURE | 8 | 1 | 1 | 0 | 7 | Misi tuntas; pemanggil berdiri santai tanpa gestur |
| **TOTAL** | — | **156** | **83** | **83** | **0** | **73** | — |

> [!NOTE]
> Pada 83 instansi orang yang dapat dinilai secara visual dengan jelas, model `knn_k5.pkl` menunjukkan kesesuaian **100% (83 dari 83)** terhadap kondisi fisik tangan (47 terangkat, 36 turun). Sebanyak 73 instansi dikeluarkan dari evaluasi karena terpotong batas atas kamera pada jarak dekat (<0,7 m), motion blur saat robot berputar, atau terhalang perabot lab.

---

## F. Analisis Peristiwa dan Waktu Berbasis Log

Data waktu diekstraksi langsung dari berkas `log_event.txt` dan `log_frame*.csv` (Tabel 2 Jurnal vs Data Aktual):

| No. | Sesi Percobaan | Waktu Mulai | Waktu Kunci Koordinat | Waktu Selesai T1 | Waktu Selesai T2 (Misi Selesai) | Durasi DUAL_DETECT (s) | Durasi Navigasi Murni Aktual (s) | Durasi Navigasi di Tabel 2 Jurnal (s) | Selisih Durasi (s) | Durasi Total Sesi (s) | Durasi WAIT T1 (s) | Durasi WAIT T2 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | `011430` | 01:14:30.746 | — | — | — | — | — | — | — | 15,29 | 0,00 | 0,00 |
| **2** | `014632` | 01:46:32.268 | 01:50:14.382 | 01:50:24.968 | 01:50:43.206 | 222,11 | **28,82** | 28,8 | +0,02 | 254,45 | 1,58 | 7,70 |
| **3** | `015047` | 01:50:47.831 | 01:51:37.743 | 01:51:51.688 | 01:52:08.007 | 49,91 | **30,26** | 30,3 | -0,04 | 188,95 | 4,38 | 2,14 |
| **4** | `015357` | 01:53:57.968 | 01:54:00.098 | 01:54:13.220 | 01:54:23.374 | 2,13 | **23,28** | 23,3 | -0,02 | 29,57 | 3,44 | 2,22 |
| **5** | `015429` | 01:54:29.057 | 01:55:02.440 | 01:55:13.433 | 01:55:30.536 | 33,38 | **28,10** | 28,1 | 0,00 | 64,09 | 1,51 | 2,46 |
| **6** | `015534` | 01:55:34.532 | 01:56:05.430 | 01:56:17.511 | 01:56:34.883 | 30,90 | **29,45** | 29,5 | -0,05 | 115,03 | 2,23 | 2,61 |
| **7** | `015731` | 01:57:31.922 | 01:57:37.159 | 01:57:49.806 | 01:58:08.403 | 5,24 | **31,24** | 31,2 | +0,04 | 64,53 | 2,16 | 2,16 |
| **8** | `015837` | 01:58:37.800 | 01:58:53.073 | 01:59:11.602 | 01:59:28.554 | 15,27 | **35,48** | 35,5 | -0,02 | 55,75 | 9,52 | 1,53 |
| **9** | `015934` | 01:59:34.848 | 02:00:08.516 | 02:00:23.381 | 02:00:58.740 | 33,67 | **50,22** | 50,2 | +0,02 | 85,87 | 1,55 | 2,09 |

### Pembahasan Selisih Durasi:
1. **Durasi Navigasi Murni**:
   Angka pada Tabel 2 Jurnal (28,8 s; 30,3 s; 23,3 s; 28,1 s; 29,5 s; 31,2 s; 35,5 s; 50,2 s) **terbukti 100% otentik**, dihitung persis dari selisih waktu antara peristiwa `KUNCI` (koordinat terkunci) hingga `T2_SELESAI` pada `log_event.txt` dengan pembulatan 1 desimal (selisih matematis $\le 0,05$ detik).
2. **Rata-Rata Durasi WAIT**:
   - Rata-rata durasi `WAIT_T1` pada CSV adalah **3,30 detik** (Jurnal mencantumkan 3,5 s).
   - Rata-rata durasi `WAIT_T2` pada CSV adalah **2,86 detik** (Jurnal mencantumkan 3,0 s).
   - Selisih sekitar 0,14–0,20 detik disebabkan oleh perbedaan metode perhitungan: interval logging CSV (~100 ms) vs penghitungan berbasis akumulasi timer hold internal (`HANDSDOWN_HOLD = 2.0 s` ditambah waktu latensi transisi thread komunikasi ESP32).

---

## G. Temuan Khusus yang Perlu Diperiksa Terbuka

### 1. Investigasi Kasus 3 Kandidat pada `dual_detect_mulai_03.png` (Percobaan 2)
- **Kondisi Visual Foto**:
  Tampak 3 orang:
  1. Orang 1 (kiri, depan meja): Pria berdiri, kedua tangan di bawah / bersandar ke meja (`TURUN`).
  2. Orang 2 (tengah, dekat pintu): Wanita berhijab berdiri santai, kedua tangan di samping tubuh (`TURUN`).
  3. Orang 3 (kanan, siluet pintu): Seseorang berdiri di lorong berpencahayaan minim (`TURUN`/tidak jelas).
- **Hasil Pemeriksaan**:
  - Pada tangkapan layar live HUD saat pengujian, ketiga orang tersebut sempat diberi bounding box cyan berlabel `HANDS_UP`.
  - **Penyebab**: Ini adalah **False Positive** sementara dari pipeline video real-time. Pada mode pelacakan video MediaPipe (`static_image_mode=False`), pergerakan kamera atau kontur pakaian gelap/pintu memicu perkiraan sendi palsu (*phantom joints*) di atas garis bahu.
  - **Ketahanan Sistem Robot**: Sistem navigasi **tidak terkecoh** oleh False Positive ini karena terdapat mekanisme penyaring:
    1. Filter jarak kedalaman Kinect: Orang 1 terdeteksi depth 0,00 m (invalid depth / terhalang meja), Orang 3 terdeteksi 6,29 m (melebihi batas jangkauan).
    2. Mekanisme `DUAL_DETECT_HOLD = 2.0s`: Karena deteksi palsu tidak stabil selama 2 detik penuh, timer deteksi di-reset kembali ke 0%. Sistem baru berhasil mengunci koordinat setelah dua target pemanggil sebenarnya mengambil posisi stabil pada foto ke-6 (`kunci_koordinat_01.png`).

### 2. Status Tampilan Foto Pascapenguncian
- Seluruh foto event setelah penguncian koordinat (`statelock`, `tiba`, `t1_selesai`, `rotasi`, `misi_selesai`) pada program asli disimpan dalam bentuk **frame mentah (raw camera feed)** tanpa teks atau overlay box. Penambahan overlay grafis pada Percobaan 3 adalah rekayasa pasca-percobaan untuk dokumentasi visual dan **wajib diakui secara jujur dalam skripsi**.

---

## H. Ringkasan per Percobaan (1–9)

### Percobaan 1 (`percobaan_20261118_011430`) — Dry-Run Inisialisasi
Sesi ini merupakan pengujian komunikasi awal sistem dan mikrokontroler ESP32 selama 15,29 detik. Program dihentikan secara manual pada fase inisialisasi sensor sebelum kedua pemanggil mengunci gestur. Tidak ada berkas foto yang tersimpan, dan data navigasi berstatus dieksklusi dari perhitungan performa.

| Metrik | Nilai |
|---|---|
| Status Misi | DIEKSKLUSI (Dry-Run) |
| Durasi Total | 15,29 detik |
| Target Dikunci | 0 |
| Total Frame CSV | 71 frame |

---

### Percobaan 2 (`percobaan_20261118_014632`)
Percobaan valid pertama dengan konfigurasi T1 pada jarak 1,94 m (+13,9°) dan T2 pada jarak 2,98 m (-7,7°). Fase deteksi awal membutuhkan waktu terlama (222,11 s) akibat gangguan deteksi kandidat di lorong (termasuk false positive 3 orang pada foto ke-3). Setelah kedua target terverifikasi stabil selama 2 detik, sistem berhasil menavigasi ke T1 dan T2 dengan durasi navigasi murni 28,82 detik.

| Metrik | Nilai |
|---|---|
| Status Misi | SUKSES |
| Durasi DUAL_DETECT | 222,11 s (6 siklus masuk-ulang) |
| Durasi Navigasi Murni | 28,82 s (Jurnal: 28,8 s) |
| Jarak Awal T1 / T2 | 1,94 m / 2,98 m |
| Jarak Tempuh Encoder | 8,184 m (3.789 tick kiri, 3.577 tick kanan) |

---

### Percobaan 3 (`percobaan_20261118_015047`)
Konfigurasi T1 pada 1,88 m (+12,8°) dan T2 pada 3,42 m (-10,8°). Deteksi awal berlangsung 49,91 detik dengan 6 pemicuan foto sebelum koordinat terkunci stabil. Navigasi menuju T1 dan transisi rotasi menuju T2 berjalan mulus dengan durasi navigasi murni 30,26 detik.

| Metrik | Nilai |
|---|---|
| Status Misi | SUKSES |
| Durasi DUAL_DETECT | 49,91 s (6 siklus masuk-ulang) |
| Durasi Navigasi Murni | 30,26 s (Jurnal: 30,3 s) |
| Jarak Awal T1 / T2 | 1,88 m / 3,42 m |
| Jarak Tempuh Encoder | 18,136 m (8.304 tick kiri, 8.018 tick kanan) |

---

### Percobaan 4 (`percobaan_20261118_015357`)
Percobaan dengan respons tercepat di mana fase deteksi awal hanya memerlukan 2,13 detik (langsung terkunci pada percobaan pertama tanpa reset). Konfigurasi target T1 pada 1,88 m (+11,3°) dan T2 pada 2,98 m (-6,5°). Durasi navigasi murni tercatat paling singkat yaitu 23,28 detik.

| Metrik | Nilai |
|---|---|
| Status Misi | SUKSES |
| Durasi DUAL_DETECT | 2,13 s (1 siklus langsung terkunci) |
| Durasi Navigasi Murni | 23,28 s (Jurnal: 23,3 s) |
| Jarak Awal T1 / T2 | 1,88 m / 2,98 m |
| Jarak Tempuh Encoder | 24,453 m (11.218 tick kiri, 10.790 tick kanan) |

---

### Percobaan 5 (`percobaan_20261118_015429`)
Konfigurasi target berada pada sisi kiri dan tengah robot (T1: 1,98 m sudut -19,6°; T2: 3,12 m sudut 0,0°). Fase deteksi awal mengalami 11 kali siklus masuk-ulang (33,38 s) karena pemanggil di sisi kiri sempat tertutup sebagian saat penyesuaian posisi berdiri. Durasi navigasi murni berlangsung 28,10 detik.

| Metrik | Nilai |
|---|---|
| Status Misi | SUKSES |
| Durasi DUAL_DETECT | 33,38 s (11 siklus masuk-ulang) |
| Durasi Navigasi Murni | 28,10 s (Jurnal: 28,1 s) |
| Jarak Awal T1 / T2 | 1,98 m / 3,12 m |
| Jarak Tempuh Encoder | 32,551 m (14.747 tick kiri, 14.549 tick kanan) |

---

### Percobaan 6 (`percobaan_20261118_015534`)
Konfigurasi T1 pada 1,94 m (-14,5°) dan T2 pada 3,29 m (+3,1°). Deteksi awal berlangsung selama 30,90 detik dengan 2 kali siklus verifikasi. Navigasi murni diselesaikan dalam 29,45 detik dan kedua target berhasil didatangi tepat pada batas jarak henti 0,70 m.

| Metrik | Nilai |
|---|---|
| Status Misi | SUKSES |
| Durasi DUAL_DETECT | 30,90 s (2 siklus masuk-ulang) |
| Durasi Navigasi Murni | 29,45 s (Jurnal: 29,5 s) |
| Jarak Awal T1 / T2 | 1,94 m / 3,29 m |
| Jarak Tempuh Encoder | 41,633 m (18.704 tick kiri, 18.766 tick kanan) |

---

### Percobaan 7 (`percobaan_20261118_015731`)
Pengujian berjalan sangat efisien dengan deteksi awal terkunci hanya dalam 5,24 detik. T1 berjarak 1,99 m (+8,9°) dan T2 berjarak 3,53 m (-10,4°). Robot menyelesaikan misi dalam durasi navigasi murni 31,24 detik.

| Metrik | Nilai |
|---|---|
| Status Misi | SUKSES |
| Durasi DUAL_DETECT | 5,24 s (1 siklus terkunci) |
| Durasi Navigasi Murni | 31,24 s (Jurnal: 31,2 s) |
| Jarak Awal T1 / T2 | 1,99 m / 3,53 m |
| Jarak Tempuh Encoder | 51,780 m (23.296 tick kiri, 23.306 tick kanan) |

---

### Percobaan 8 (`percobaan_20261118_015837`)
Konfigurasi target dengan jarak lebih jauh (T1: 2,64 m sudut -16,8°; T2: 4,87 m sudut -2,5°). Jarak antar target adalah 2,402 m. Robot membutuhkan waktu navigasi murni 35,48 detik dan fase tunggu penurunan tangan T1 membutuhkan waktu 9,52 detik sebelum robot berputar ke arah memori T2.

| Metrik | Nilai |
|---|---|
| Status Misi | SUKSES |
| Durasi DUAL_DETECT | 15,27 s (3 siklus masuk-ulang) |
| Durasi Navigasi Murni | 35,48 s (Jurnal: 35,5 s) |
| Jarak Awal T1 / T2 | 2,64 m / 4,87 m |
| Jarak Tempuh Encoder | 58,116 m (26.214 tick kiri, 26.090 tick kanan) |

---

### Percobaan 9 (`percobaan_20261118_015934`)
Konfigurasi sudut paling ekstrem dengan T1 pada -20,8° (jarak 3,17 m) dan T2 pada -2,1° (jarak 4,55 m). Waktu navigasi murni mencapai durasi terpanjang yaitu 50,22 detik dikarenakan robot harus melakukan putaran sudut besar untuk menyelaraskan arah ke T1 kemudian putaran balik menuju T2. Seluruh misi diselesaikan dengan sukses.

| Metrik | Nilai |
|---|---|
| Status Misi | SUKSES |
| Durasi DUAL_DETECT | 33,67 s (6 siklus masuk-ulang) |
| Durasi Navigasi Murni | 50,22 s (Jurnal: 50,2 s) |
| Jarak Awal T1 / T2 | 3,17 m / 4,55 m |
| Jarak Tempuh Encoder | 70,353 m (31.707 tick kiri, 31.611 tick kanan) |

---

## I. Temuan yang Bertentangan atau Perlu Klarifikasi terhadap Isi Jurnal

1. **Keberadaan Orang Ketiga / Bukan Pemanggil pada Deteksi Nyata**:
   - Naskah jurnal menyebutkan skenario pengujian hanya melibatkan 2 pemanggil tanpa gangguan pihak ketiga.
   - **Temuan Nyata**: Pada foto `dual_detect_mulai_03.png` Percobaan 2, terdapat 3 orang yang tertangkap kamera di laboratorium dan sempat terdeteksi sebagai kandidat. Peneliti harus menjelaskan bahwa sistem memiliki mekanisme filter jarak dan kestabilan waktu (`DUAL_DETECT_HOLD`) yang berhasil menganulir orang ketiga tersebut.
2. **Ketiadaan HUD Asli pada Foto Pascapenguncian**:
   - Jika foto di dalam jurnal menampilkan kotak pembacaan sistem merah pada fase `statelock` hingga `tiba_t2` di Percobaan 3, harus dijelaskan secara jujur bahwa kotak tersebut adalah rekonstruksi visual pasca-percobaan, karena sistem asli hanya menyimpan bingkai mentah.
3. **Pencatatan Log Tanpa Label KNN per Frame**:
   - Log navigasi CSV (`log_frame*.csv`) tidak mencatat nilai prediksi KNN atau fitur landmark tangan per frame. Pembuktian klasifikasi gestur pada jurnal bersumber dari uji terpisah model KNN (akurasi 99,11% pada dataset 2.800 sampel) dan verifikasi foto event, bukan dari time-series CSV.

