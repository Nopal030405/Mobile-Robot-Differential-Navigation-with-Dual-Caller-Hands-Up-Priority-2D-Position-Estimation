# Laporan Rekapitulasi Hasil Percobaan #1 (17 Oktober 2026)
**Sistem Navigasi Robot Diferensial Multi-Target Segitiga (T1 & T2)**

---

## 1. Ringkasan Eksekutif

Pada sesi eksperimen tanggal **17 Oktober 2026** (sebelumnya tersimpan di folder `rekapan_percobaan_1_sampai_9`), dilakukan pengujian awal navigasi otonom multi-target berurutan.

Total percobaan yang tercatat adalah **9 kali percobaan** (Percobaan #1 hingga #9).

### Metrik Kunci Sesi Ini:
- **Total Percobaan:** 9 kali
- **T1 Tercapai (0.70 m):** 5 Percobaan (#1, #5, #6, #8, #9)
- **T2 Tercapai (0.70 m):** 0 Percobaan (0% - T2 belum pernah finish pada 0.7m akibat batasan filter jarak < 0.85m pada algoritma awal)
- **Catatan Masalah Utama:** 
  1. Target T2 hilang/kelewatan ketika mendekati ambang batas finish karena filter `jarak > 0.85m` menyaring T2 sebelum mencapai `STOP_DISTANCE` (0.70m).
  2. Masalah osilasi rotasi saat penyelarasan arah (alignment lag).

---

## 2. Tabel Rekapitulasi Percobaan 1 - 9 (Tanggal 17-10-2026)

| No | Waktu Mulai | Durasi (s) | Status | T1 Awal (m / °) | T1 Akhir (m) | T2 Awal (m / °) | T2 Akhir (m) | Jarak T1-T2 (m) | Total Tempuh (m) | Keterangan |
|:--:|:-----------:|:----------:|:------:|:---------------:|:------------:|:---------------:|:------------:|:---------------:|:----------------:|:-----------|
| **1** | 16:09:42 | 64.24 | BERJALAN | 1.32m / -11.6°| 0.70m | 2.60m / +10.4°| 0.00m | 1.46m | 14.37m | T1 finish; T2 belum selesai |
| **2** | 13:48:30 | 111.04| BERJALAN | 1.71m / +11.0°| 0.00m | 2.17m / -8.9° | 0.00m | 0.81m | 3.94m  | Uji coba penyesuaian |
| **3** | 13:51:38 | 49.75 | BERJALAN | 1.65m / +8.4° | 0.00m | 2.07m / -9.3° | 0.00m | 0.71m | 4.30m  | Uji coba penyesuaian |
| **4** | 13:58:38 | 43.97 | BERJALAN | 1.17m / +12.5°| 0.00m | 1.35m / -9.4° | 0.00m | 0.51m | 9.27m  | Uji coba penyesuaian |
| **5** | 14:00:45 | 59.40 | BERJALAN | 1.31m / +11.7°| 0.70m | 1.80m / -9.3° | 0.00m | 0.74m | 6.78m  | T1 finish; T2 belum tercapai |
| **6** | 14:02:33 | 59.31 | BERJALAN | 1.93m / +9.9° | 0.70m | 2.81m / -13.7°| 0.00m | 1.30m | 4.25m  | T1 finish; T2 belum tercapai |
| **7** | 14:04:07 | 71.92 | BERJALAN | 1.38m / +10.2°| 0.00m | 1.86m / -11.3°| 0.00m | 0.77m | 0.19m  | Uji coba penyesuaian |
| **8** | 14:06:01 | 77.72 | BERJALAN | 1.15m / +9.6° | 0.70m | 1.84m / -13.3°| 0.00m | 0.90m | 7.06m  | T1 finish; T2 belum tercapai |
| **9** | 14:09:34 | 89.73 | BERJALAN | 2.38m / +11.7°| 0.70m | 2.79m / -7.4° | 0.00m | 0.95m | 8.82m  | T1 finish; T2 belum tercapai |

---

## 3. Struktur Berkas yang Tersimpan di Folder Ini

```
percobaan1_101726/
├── ringkasan_misi_segitiga.csv          <- Ringkasan tabular percobaan 1 s/d 9
├── RINGKASAN_PERCOBAAN_1.md             <- Laporan analitik sesi 1
│
├── grafik_rekap_percobaan_1.png s/d grafik_rekap_percobaan_9.png (9 file)
└── log_timeseries_percobaan_1.csv s/d log_timeseries_percobaan_9.csv (9 file)
```
