# Laporan Rekapitulasi Hasil Percobaan #2 (23 Oktober 2026)
**Sistem Navigasi Robot Diferensial Multi-Target Segitiga (T1 & T2)**

---

## 1. Ringkasan Eksekutif

Pada sesi eksperimen tanggal **23 Oktober 2026**, dilakukan serangkaian pengujian navigasi otonom multi-target berurutan (T1 dilanjutkan ke T2) menggunakan estimasi formasi segitiga 2D (*Kinect Depth Sensor + YOLOv8 + Mediapipe Gesture Detection*).

Total pengujian yang terekam adalah **12 kali percobaan** (Percobaan #1 hingga #12).

### Metrik Kunci Sesi Ini:
- **Total Percobaan:** 12 kali
- **Misi Sukses Penuh (T1 & T2 Tercapai):** 6 Percobaan (#4, #5, #6, #9, #10, #12*)
- **T1 Tercapai (0.70 m):** 9 Percobaan (#1, #4, #5, #6, #7, #9, #10, #11, #12)
- **Tingkat Keberhasilan Mencapai T1:** **75%**
- **Tingkat Keberhasilan Selesai T2 (Full Mission):** **50%**
- **Rata-rata Durasi Misi Selesai:** ~62.3 detik
- **Rentang Jarak Awal Target:** 1.69 m – 3.49 m (T1), 2.32 m – 4.32 m (T2)

---

## 2. Tabel Rekapitulasi Percobaan 1 - 12 (Tanggal 23-10-2026)

| No | Waktu Mulai | Durasi (s) | Status | T1 Awal (m / °) | T1 Akhir (m) | T2 Awal (m / °) | T2 Akhir (m) | Jarak T1-T2 (m) | Total Tempuh (m) | Catatan Khusus |
|:--:|:-----------:|:----------:|:------:|:---------------:|:------------:|:---------------:|:------------:|:---------------:|:----------------:|:---------------|
| **1** | 14:10:22 | 83.30 | BERJALAN | 2.19m / -3.0° | 0.70m | 2.70m / +13.4° | 0.00m | 0.86m | 12.86m | T1 finish tercapai; T2 belum selesai |
| **2** | 14:12:07 | 41.60 | BERJALAN | 1.89m / -10.0°| 0.00m | 2.32m / +11.6° | 0.00m | 0.90m | 5.16m | Dihentikan sebelum T1 finish |
| **3** | 14:16:52 | 51.99 | BERJALAN | 2.01m / -11.6°| 0.00m | 2.90m / +11.4° | 0.00m | 1.31m | 5.11m | Dihentikan sebelum T1 finish |
| **4** | 14:18:06 | 72.22 | **SUKSES** | 2.17m / -5.5° | 0.00m | 2.86m / +13.2° | 0.70m | 1.06m | 9.06m | Misi selesai ke T2 |
| **5** | 14:19:21 | 61.99 | **SUKSES** | 1.69m / -12.2°| 0.70m | 3.56m / +5.9°  | 0.70m | 2.02m | 16.46m | Misi sempurna T1 & T2 (0.7m) |
| **6** | 14:20:25 | 56.17 | **SUKSES** | 1.81m / +10.2°| 0.70m | 2.93m / -9.3°  | 0.70m | 1.36m | 23.85m | Misi sempurna T1 & T2 (0.7m) |
| **7** | 14:21:50 | 57.00 | BERJALAN | 2.88m / -12.5°| 0.70m | 3.88m / +2.2°  | 0.00m | 1.32m | 12.13m | T1 finish; T2 belum tuntas |
| **8** | 14:23:13 | 27.33 | BERJALAN | 2.26m / -9.8° | 0.00m | 4.02m / +3.1°  | 0.00m | 1.89m | 4.57m | Reset manual awal misi |
| **9** | 14:24:34 | 55.19 | **SUKSES** | 1.82m / -11.9°| 0.70m | 3.23m / +5.7°  | 0.70m | 1.59m | 7.24m | Misi sempurna T1 & T2 (0.7m) |
| **10**| 14:25:31 | 59.89 | **SUKSES** | 2.24m / +8.5° | 0.70m | 2.66m / -10.2° | 0.70m | 0.90m | 14.12m | Misi sempurna T1 & T2 (0.7m) |
| **11**| 14:26:33 | 87.10 | BERJALAN | 3.49m / +9.4° | 0.70m | 4.32m / -6.4°  | 0.00m | 1.35m | 25.49m | Jarak awal jauh (3.5m+), T1 tercapai |
| **12**| 14:28:03 | 79.31 | **SUKSES\*** | 3.23m / +4.8° | 0.70m | 4.12m / -10.0° | 0.46m | 1.29m | 33.10m | Mencapai ALL_FINISHED (keduanya tuntas) |

*\*Catatan Percobaan 12: Robot berhasil mencapai ALL_FINISHED pada detik ke-76.5 dan selesai penuh. Di log ringkasan tercatat BERJALAN karena branch handsdown cadangan belum memanggil hook status, namun log time-series membuktikan fase ALL_FINISHED aktif hingga selesai.*

---

## 3. Analisis Teknis & Masalah yang Diselesaikan

### A. Evaluasi Filter Jarak Dekat T2 (0.85m vs 0.70m)
- **Problem Awal:** Pada percobaan terdahulu, target T2 sering hilang/kelewatan saat jarak mendekati target (< 0.85m) karena filter deteksi membuang orang dengan `jarak < 0.85m`.
- **Hasil Pengujian Sesi Ini:** Pada fase `APPROACH_T2`, filter diizinkan mendeteksi orang hingga `0.65m` (`STOP_DISTANCE - 0.05m`). Hal ini terbukti berhasil di Percobaan #4, #5, #6, #9, #10, dan #12 di mana robot berhasil mendekati T2 hingga ambang batas 0.70m.

### B. Masalah Osilasi & Belok Tiba-Tiba (Sudut Lag)
- **Problem Awal:** Robot terkadang berputar mendadak ke arah berlawanan saat fase ALIGN karena moving average filter (`angle_history`) masih menahan sudut lama beberapa frame ke belakang setelah robot selesai melakukan rotasi pulse.
- **Perbaikan yang Diterapkan:**
  1. `angle_history.clear()` dipanggil saat fase settling rotasi agar moving average segera menyinkronkan sudut aktual terbaru.
  2. Diberikan jeda settling time (`ALIGN_SETTLE_FRAMES = 3`) agar robot benar-benar diam dan kamera menangkap frame yang stabil sebelum memutuskan arah putar berikutnya.
  3. Mencegah overshoot arah bolak-balik (toggle threshold).

---

## 4. Struktur Berkas yang Direkap di Folder Ini

Semua file hasil rekaman percobaan #2 tanggal 23-10-2026 telah dikelompokkan secara lengkap tanpa ada yang terlewat:

```
percobaan2_102326/
├── ringkasan_misi_segitiga.csv          <- Ringkasan tabular seluruh 12 percobaan
├── RINGKASAN_PERCOBAAN_2.md             <- Laporan analitik ini
│
├── grafik_rekap_percobaan_1.png         <- Grafik 4 subplot percobaan #1
├── grafik_rekap_percobaan_2.png         <- Grafik 4 subplot percobaan #2
├── grafik_rekap_percobaan_3.png         <- Grafik 4 subplot percobaan #3
├── grafik_rekap_percobaan_4.png         <- Grafik 4 subplot percobaan #4
├── grafik_rekap_percobaan_5.png         <- Grafik 4 subplot percobaan #5
├── grafik_rekap_percobaan_6.png         <- Grafik 4 subplot percobaan #6
├── grafik_rekap_percobaan_7.png         <- Grafik 4 subplot percobaan #7
├── grafik_rekap_percobaan_8.png         <- Grafik 4 subplot percobaan #8
├── grafik_rekap_percobaan_9.png         <- Grafik 4 subplot percobaan #9
├── grafik_rekap_percobaan_10.png        <- Grafik 4 subplot percobaan #10
├── grafik_rekap_percobaan_11.png        <- Grafik 4 subplot percobaan #11
├── grafik_rekap_percobaan_12.png        <- Grafik 4 subplot percobaan #12
├── grafik_rekap_navigasi.png            <- Grafik rekapitulasi run terakhir
│
├── log_timeseries_percobaan_1.csv       <- Data streaming 100ms percobaan #1
├── log_timeseries_percobaan_2.csv       <- Data streaming 100ms percobaan #2
├── log_timeseries_percobaan_3.csv       <- Data streaming 100ms percobaan #3
├── log_timeseries_percobaan_4.csv       <- Data streaming 100ms percobaan #4
├── log_timeseries_percobaan_5.csv       <- Data streaming 100ms percobaan #5
├── log_timeseries_percobaan_6.csv       <- Data streaming 100ms percobaan #6
├── log_timeseries_percobaan_7.csv       <- Data streaming 100ms percobaan #7
├── log_timeseries_percobaan_8.csv       <- Data streaming 100ms percobaan #8
├── log_timeseries_percobaan_9.csv       <- Data streaming 100ms percobaan #9
├── log_timeseries_percobaan_10.csv      <- Data streaming 100ms percobaan #10
├── log_timeseries_percobaan_11.csv      <- Data streaming 100ms percobaan #11
├── log_timeseries_percobaan_12.csv      <- Data streaming 100ms percobaan #12
└── log_navigasi_timeseries.csv          <- Data time-series run terakhir
```
