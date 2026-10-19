import os
import csv
import time
import math
from datetime import datetime
import matplotlib.pyplot as plt

class NavigationDataLogger:
    def __init__(self, output_dir=None):
        if output_dir is None:
            self.output_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        
        self.summary_csv = os.path.join(self.output_dir, "ringkasan_misi_segitiga.csv")
        
        # Tentukan nomor percobaan berikutnya berdasarkan isi file summary jika ada
        self.trial_number = self._detect_next_trial_number()
        
        # Inisialisasi sesi percobaan
        self._init_session()

    def _detect_next_trial_number(self):
        """Menghitung nomor percobaan berikutnya dari file CSV ringkasan."""
        if not os.path.exists(self.summary_csv):
            return 1
        try:
            count = 0
            with open(self.summary_csv, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if row and len(row) > 0 and row[0].strip():
                        count += 1
            return max(1, count + 1)
        except Exception:
            return 1

    def _init_session(self):
        self.start_wall_time = time.time()
        self.start_datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_log_time = 0.0
        self.log_interval = 0.1  # Rekam data setiap 100ms
        self.is_finalized = False

        # Nama file khusus per percobaan
        self.trial_timeseries_csv = os.path.join(
            self.output_dir, f"log_timeseries_percobaan_{self.trial_number}.csv"
        )
        self.latest_timeseries_csv = os.path.join(
            self.output_dir, "log_navigasi_timeseries.csv"
        )
        self.trial_graph_png = os.path.join(
            self.output_dir, f"grafik_rekap_percobaan_{self.trial_number}.png"
        )
        self.latest_graph_png = os.path.join(
            self.output_dir, "grafik_rekap_navigasi.png"
        )

        # Buffer data time-series
        self.records = []
        
        # Informasi Target & Segitiga
        self.locked_t1 = None
        self.locked_t2 = None
        self.triangle_info = None
        
        self.t1_finish_dist = 0.0
        self.t2_finish_dist = 0.0
        self.mission_status = "BERJALAN"

        # Buat header untuk file time-series sesi ini
        header = [
            "Percobaan_Ke", "Timestamp", "Waktu_Detik", "Fase_State", "Target_Aktif",
            "Jarak_Sensor_m", "Sudut_Sensor_deg",
            "Cmd_Jarak", "Cmd_Sudut",
            "Encoder_Left", "Encoder_Right", "Jarak_Tempuh_m"
        ]
        with open(self.trial_timeseries_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

        with open(self.latest_timeseries_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

        print(f"\n>> [LOGGER] Memulai Sesi Percobaan #{self.trial_number}")
        print(f"   Log Time-Series: log_timeseries_percobaan_{self.trial_number}.csv")

    def reset(self):
        """Dipanggil saat pengguna menekan 'r' untuk reset."""
        # Jika ada data signifikan di sesi sebelumnya, finalize terlebih dahulu
        if len(self.records) >= 5 and not self.is_finalized:
            self.finalize()
        
        # Mulai sesi percobaan berikutnya
        self.trial_number += 1
        self._init_session()

    def log_frame(self, state_str, target_lbl, jarak, sudut, cmd_j, cmd_s, enc_l, enc_r):
        now = time.time()
        if now - self.last_log_time < self.log_interval:
            return
        self.last_log_time = now

        elapsed = round(now - self.start_wall_time, 2)
        ts_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        avg_tick = (abs(enc_l) + abs(enc_r)) / 2.0
        dist_travelled = round(avg_tick / 450.0, 3)

        row = [
            self.trial_number, ts_str, elapsed, state_str, target_lbl,
            round(jarak, 3), round(sudut, 2),
            round(cmd_j, 3), round(cmd_s, 2),
            enc_l, enc_r, dist_travelled
        ]
        self.records.append(row)

        try:
            with open(self.trial_timeseries_csv, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            with open(self.latest_timeseries_csv, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception:
            pass

    def record_locked_data(self, t1_dict, t2_dict, tri_dict):
        self.locked_t1 = dict(t1_dict) if t1_dict else None
        self.locked_t2 = dict(t2_dict) if t2_dict else None
        self.triangle_info = dict(tri_dict) if tri_dict else None

    def record_t1_finish(self, dist):
        self.t1_finish_dist = dist

    def record_t2_finish(self, dist):
        self.t2_finish_dist = dist
        self.mission_status = "SUKSES"

    def finalize(self):
        if self.is_finalized:
            return
        self.is_finalized = True

        end_datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_duration = round(time.time() - self.start_wall_time, 2)

        # 1. Simpan Ringkasan Misi ke CSV
        file_exists = os.path.exists(self.summary_csv)
        header = [
            "Percobaan_Ke", "Waktu_Mulai", "Waktu_Selesai", "Durasi_Total_dtk", "Status_Misi",
            "T1_Jarak_Awal_m", "T1_Sudut_Awal_deg", "T1_Jarak_Akhir_m",
            "T2_Jarak_Awal_m", "T2_Sudut_Awal_deg", "T2_Jarak_Akhir_m",
            "Segitiga_Cam_T1_m", "Segitiga_Cam_T2_m", "Segitiga_T1_T2_m", "Sudut_Kamera_deg",
            "Total_Tick_Enc_L", "Total_Tick_Enc_R", "Total_Jarak_Tempuh_m"
        ]

        t1_j_awal = self.locked_t1['jarak'] if self.locked_t1 else 0.0
        t1_s_awal = self.locked_t1['sudut'] if self.locked_t1 else 0.0
        t2_j_awal = self.locked_t2['jarak'] if self.locked_t2 else 0.0
        t2_s_awal = self.locked_t2['sudut'] if self.locked_t2 else 0.0

        c_t1 = self.triangle_info.get('cam_to_t1', 0.0) if self.triangle_info else 0.0
        c_t2 = self.triangle_info.get('cam_to_t2', 0.0) if self.triangle_info else 0.0
        t1_t2 = self.triangle_info.get('t1_to_t2', 0.0) if self.triangle_info else 0.0
        ang_cam = self.triangle_info.get('angle_at_cam_deg', 0.0) if self.triangle_info else 0.0

        last_enc_l = self.records[-1][9] if self.records else 0
        last_enc_r = self.records[-1][10] if self.records else 0
        last_dist = self.records[-1][11] if self.records else 0.0

        summary_row = [
            self.trial_number, self.start_datetime_str, end_datetime_str, total_duration, self.mission_status,
            t1_j_awal, t1_s_awal, round(self.t1_finish_dist, 2),
            t2_j_awal, t2_s_awal, round(self.t2_finish_dist, 2),
            c_t1, c_t2, t1_t2, ang_cam,
            last_enc_l, last_enc_r, last_dist
        ]

        with open(self.summary_csv, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(summary_row)

        print(f"\n>> [DATA LOGGER - PERCOBAAN #{self.trial_number}] Selesai.")
        print(f">> Timeseries : {os.path.basename(self.trial_timeseries_csv)}")
        print(f">> Ringkasan   : {os.path.basename(self.summary_csv)}")

        # 2. Gambar Grafik Rekapitulasi (4 Subplot) khusus nomor percobaan ini
        self.generate_plots()

    def generate_plots(self):
        if len(self.records) < 3:
            print(f">> [DATA LOGGER] Data percobaan #{self.trial_number} terlalu sedikit untuk grafik.")
            return

        times = [r[2] for r in self.records]
        states = [r[3] for r in self.records]
        dists = [r[5] for r in self.records]
        angles = [r[6] for r in self.records]
        enc_ls = [r[9] for r in self.records]
        enc_rs = [r[10] for r in self.records]
        travels = [r[11] for r in self.records]

        # Setup 4-Subplot Layout
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f"Rekapitulasi Navigasi Multi-Target & Odometri Robot - Percobaan #{self.trial_number}\n({self.start_datetime_str})",
                     fontsize=15, fontweight='bold')

        # --- Subplot 1: Jarak Target terhadap Waktu ---
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(times, dists, label="Jarak Sensor (m)", color='#007acc', linewidth=2.0)
        ax1.axhline(0.7, color='red', linestyle='--', linewidth=1.5, label='Batas Aman Finish (0.7m)')
        ax1.set_title(f"1. Profil Perubahan Jarak Target (Percobaan #{self.trial_number})", fontweight='bold')
        ax1.set_xlabel("Waktu (detik)")
        ax1.set_ylabel("Jarak (meter)")
        ax1.set_ylim(bottom=0.0)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='upper right')

        # --- Subplot 2: Sudut Target terhadap Waktu (State-Lock) ---
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(times, angles, label="Sudut Target (°)", color='#e67e22', linewidth=1.8)
        ax2.axhline(0.0, color='gray', linestyle='-', linewidth=1.0)
        ax2.axhspan(-2.0, 2.0, color='green', alpha=0.15, label='Zona Presisi Statelock (±2°)')
        ax2.set_title("2. Respon Penyelarasan Sudut Robot (Alignment)", fontweight='bold')
        ax2.set_xlabel("Waktu (detik)")
        ax2.set_ylabel("Sudut (derajat)")
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(loc='upper right')

        # --- Subplot 3: Grafik Encoder Motor Kiri & Kanan (Awal - Akhir) ---
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(times, enc_ls, label="Encoder Kiri (Tick)", color='#2980b9', linewidth=1.8)
        ax3.plot(times, enc_rs, label="Encoder Kanan (Tick)", color='#c0392b', linewidth=1.8)
        ax3.set_title("3. Respons Encoder Motor PG28 (Awal - Akhir)", fontweight='bold')
        ax3.set_xlabel("Waktu (detik)")
        ax3.set_ylabel("Akumulasi Ticks")
        ax3.grid(True, linestyle=':', alpha=0.6)

        # Tambahkan sumbu kedua untuk jarak tempuh robot
        ax3_twin = ax3.twinx()
        ax3_twin.plot(times, travels, label="Estimasi Tempuh (m)", color='#27ae60', linestyle=':', linewidth=2.0)
        ax3_twin.set_ylabel("Jarak Tempuh (meter)", color='#27ae60')
        ax3_twin.tick_params(axis='y', labelcolor='#27ae60')

        # Gabungkan legend
        lines_3, labels_3 = ax3.get_legend_handles_labels()
        lines_3b, labels_3b = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines_3 + lines_3b, labels_3 + labels_3b, loc='upper left')

        # --- Subplot 4: Peta 2D Geometri Segitiga Kamera-T1-T2 ---
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(0, 0, marker='^', markersize=12, color='black', label='Kamera Robot (0,0)')

        if self.triangle_info and self.locked_t1 and self.locked_t2:
            t1x = self.triangle_info.get('t1x', 0.0)
            t1y = self.triangle_info.get('t1y', 0.0)
            t2x = self.triangle_info.get('t2x', 0.0)
            t2y = self.triangle_info.get('t2y', 0.0)

            # Gambar titik target
            ax4.plot(t1x, t1y, marker='o', markersize=10, color='#27ae60', label=f"T1 ({self.locked_t1['jarak']:.2f}m)")
            ax4.plot(t2x, t2y, marker='s', markersize=10, color='#e74c3c', label=f"T2 ({self.locked_t2['jarak']:.2f}m)")

            # Gambar garis segitiga
            ax4.plot([0, t1x], [0, t1y], 'g--', alpha=0.7, label=f"Cam->T1: {self.triangle_info['cam_to_t1']:.2f}m")
            ax4.plot([0, t2x], [0, t2y], 'r--', alpha=0.7, label=f"Cam->T2: {self.triangle_info['cam_to_t2']:.2f}m")
            ax4.plot([t1x, t2x], [t1y, t2y], 'b-', linewidth=2.0, label=f"T1->T2: {self.triangle_info['t1_to_t2']:.2f}m")

            # Label teks pada titik
            ax4.text(t1x + 0.05, t1y + 0.05, f"T1\n({t1x:.2f}, {t1y:.2f})", fontsize=9, color='darkgreen', fontweight='bold')
            ax4.text(t2x + 0.05, t2y + 0.05, f"T2\n({t2x:.2f}, {t2y:.2f})", fontsize=9, color='darkred', fontweight='bold')
            ax4.text(0.05, 0.05, f"Robot (Kamera)\nSudut: {self.triangle_info['angle_at_cam_deg']:.1f}°", fontsize=9, fontweight='bold')

            max_range = max(1.5, max(abs(t1x), abs(t2x), t1y, t2y) + 0.5)
            ax4.set_xlim(-max_range, max_range)
            ax4.set_ylim(-0.3, max_range)
        else:
            ax4.text(0, 0.5, "Data Penguncian Segitiga\nBelum Lengkap", ha='center', fontsize=11, color='gray')
            ax4.set_xlim(-2, 2)
            ax4.set_ylim(-0.5, 3)

        ax4.set_title("4. Pemetaan 2D Segitiga Kamera - T1 - T2", fontweight='bold')
        ax4.set_xlabel("X - Lateral (meter)")
        ax4.set_ylabel("Y - Maju / Depth (meter)")
        ax4.grid(True, linestyle=':', alpha=0.6)
        ax4.legend(loc='lower right', fontsize=8)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        try:
            # Simpan file spesifik untuk percobaan ini
            plt.savefig(self.trial_graph_png, dpi=300)
            # Simpan juga sebagai grafik terbaru
            plt.savefig(self.latest_graph_png, dpi=300)
            print(f">> Grafik Percobaan #{self.trial_number} tersimpan: {os.path.basename(self.trial_graph_png)}")
        except Exception as e:
            print(f">> [ERROR] Gagal menyimpan grafik: {e}")
        finally:
            plt.close(fig)
