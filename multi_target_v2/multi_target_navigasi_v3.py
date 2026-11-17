# =============================================================
# MULTI-TARGET NAVIGASI v3 - SEGITIGA KAMERA-T1-T2
# =============================================================
# CHANGELOG dari v2:
#  [v3.1] Model KNN diperbarui ke knn_k5.pkl (2800 sampel, semua variasi gesture)
#         Model lama (320 sampel) HANYA mendeteksi tangan kiri saja.
#         Model baru deteksi: 1 tangan kiri, 1 tangan kanan, ATAU keduanya terangkat.
#  [v3.2] Folder hasil percobaan TERPISAH per sesi:
#         hasil_percobaan/percobaan_YYYYMMDD_HHMMSS/
#           ├── log/        log_frame.csv, log_event.txt, ringkasan.txt
#           ├── foto/       PNG otomatis per event (dual_detect, kunci, align, finish)
#           └── grafik/     4 grafik PNG 300 DPI (jarak, sudut, encoder, segitiga 2D)
#  [v3.3] Auto-capture PNG Kinect frame pada setiap event kunci:
#         DUAL_DETECT_LOCKED, ALIGN_T1_DONE, FINISH_T1, ROTATE_START,
#         ALIGN_T2_DONE, FINISH_T2, MISI_SELESAI
#  [v3.4] Log event timestamped di .txt (bukan hanya CSV timeseries)
#  [v3.5] Ringkasan Misi TXT otomatis setelah misi selesai atau 'q'
# =============================================================

import warnings, os, sys
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2, time, math, csv, threading
from datetime import datetime
import mediapipe as mp
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')   # Render tanpa GUI agar tidak konflik dengan cv2
import matplotlib.pyplot as plt
from openni import openni2
from ultralytics import YOLO
from Esp32.send_to_esp32 import ESP32Controller

# =============================================================
# KONSTANTA SISTEM (sama dengan v2)
# =============================================================
ANGLE_THRESHOLD     = 2.0
COARSE_PULSE_DUR    = 0.12
COARSE_PAUSE_DUR    = 0.40
FINE_PULSE_DURATION = 0.08
FINE_PAUSE_DURATION = 0.35
LOCK_CONFIRM_FRAMES = 2
STOP_DISTANCE       = 0.7
SMOOTH_WINDOW       = 5
STREAM_INTERVAL     = 0.2
DUAL_DETECT_HOLD    = 2.0
DUAL_DETECT_GRACE   = 0.5
HANDSDOWN_HOLD      = 1.5
HANDSDOWN_GRACE     = 0.5
SCAN_ROTATION_ANGLE = -10.0
SCAN_STEP_PULSE     = 0.12
SCAN_STEP_PAUSE     = 0.28

# =============================================================
# STATE MACHINE
# =============================================================
STATE_DUAL_DETECT        = 0
STATE_ALIGN_T1           = 1
STATE_APPROACH_T1        = 2
STATE_WAIT_HANDSDOWN_T1  = 3
STATE_ROTATE_TO_T2       = 4
STATE_ALIGN_T2           = 5
STATE_APPROACH_T2        = 6
STATE_WAIT_HANDSDOWN_T2  = 7
STATE_ALL_FINISHED       = 8

STATE_NAMES = {
    STATE_DUAL_DETECT: "DUAL_DETECT",
    STATE_ALIGN_T1: "ALIGN_T1",
    STATE_APPROACH_T1: "APPROACH_T1",
    STATE_WAIT_HANDSDOWN_T1: "WAIT_T1",
    STATE_ROTATE_TO_T2: "ROTATE_TO_T2",
    STATE_ALIGN_T2: "ALIGN_T2",
    STATE_APPROACH_T2: "APPROACH_T2",
    STATE_WAIT_HANDSDOWN_T2: "WAIT_T2",
    STATE_ALL_FINISHED: "ALL_FINISHED"
}

# =============================================================
# [v3.2] KELAS LOGGER PERCOBAAN (pengganti multi_target_logger.py)
# Menyimpan SEMUA data ke sub-folder terpisah per sesi.
# =============================================================
class PercobaaanLogger:
    """
    Logger komprehensif per sesi percobaan.
    Output:
      hasil_percobaan/percobaan_YYYYMMDD_HHMMSS/
        log/log_frame.csv        — time-series per frame
        log/log_event.txt        — event log dengan timestamp
        log/ringkasan.txt        — ringkasan akhir percobaan
        foto/                    — PNG kamera per event kunci
        grafik/grafik_rekap.png  — 4 subplot rekap percobaan
    """

    BASE_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "hasil_percobaan")

    def __init__(self):
        # Buat folder sesi baru berbasis timestamp
        sess_name = "percobaan_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sess_dir  = os.path.join(self.BASE_RESULTS_DIR, sess_name)
        self.log_dir   = os.path.join(self.sess_dir, "log")
        self.foto_dir  = os.path.join(self.sess_dir, "foto")
        self.grafik_dir = os.path.join(self.sess_dir, "grafik")

        for d in [self.log_dir, self.foto_dir, self.grafik_dir]:
            os.makedirs(d, exist_ok=True)

        # Path file output
        self.frame_csv  = os.path.join(self.log_dir, "log_frame.csv")
        self.event_txt  = os.path.join(self.log_dir, "log_event.txt")
        self.ringkasan_txt = os.path.join(self.log_dir, "ringkasan.txt")

        # Buffer timeseries
        self.records = []
        self.frame_header = [
            "Timestamp", "Waktu_Detik", "Fase_State", "Target_Aktif",
            "Jarak_Sensor_m", "Sudut_Deg",
            "Cmd_Jarak", "Cmd_Sudut",
            "Encoder_L", "Encoder_R", "Jarak_Tempuh_m"
        ]
        with open(self.frame_csv, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(self.frame_header)

        # State data
        self.start_time = time.time()
        self.start_dt   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.locked_t1  = None
        self.locked_t2  = None
        self.triangle_info = None
        self.t1_finish_dist = 0.0
        self.t2_finish_dist = 0.0
        self.mission_status = "BERJALAN"
        self.last_log_t = 0.0
        self.is_finalized = False
        self._foto_counters = {}

        self._log_event("SISTEM", f"Sesi percobaan dimulai. Folder: {self.sess_dir}")
        self._log_event("MODEL",  "KNN v3: knn_k5.pkl (2800 sampel — semua variasi gesture)")
        print(f"\n>> [LOGGER v3] Sesi percobaan: {sess_name}")
        print(f"   Output folder: {self.sess_dir}")

    # --- EVENT LOG ---
    def _log_event(self, tag, msg):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{ts}] [{tag}] {msg}"
        print(f">> {line}")
        with open(self.event_txt, 'a', encoding='utf-8') as f:
            f.write(line + "\n")

    # --- FRAME LOG ---
    def log_frame(self, state_str, target_lbl, jarak, sudut, cmd_j, cmd_s, enc_l, enc_r):
        now = time.time()
        if now - self.last_log_t < 0.10:   # Throttle 100 ms
            return
        self.last_log_t = now

        elapsed = round(now - self.start_time, 2)
        ts_str  = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        avg_tick = (abs(enc_l) + abs(enc_r)) / 2.0
        dist_m   = round(avg_tick / 450.0, 3)

        row = [ts_str, elapsed, state_str, target_lbl,
               round(jarak, 3), round(sudut, 2),
               round(cmd_j, 3), round(cmd_s, 2),
               enc_l, enc_r, dist_m]
        self.records.append(row)

        try:
            with open(self.frame_csv, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(row)
        except Exception:
            pass

    # --- FOTO EVENT ---
    def simpan_foto(self, event_name: str, frame: np.ndarray):
        """
        Simpan frame kamera pada momen event kunci.
        Nama file: {event_name}_{counter:02d}.png
        """
        cnt = self._foto_counters.get(event_name, 0) + 1
        self._foto_counters[event_name] = cnt
        fname = f"{event_name}_{cnt:02d}.png"
        path = os.path.join(self.foto_dir, fname)
        try:
            cv2.imwrite(path, frame)
            self._log_event("FOTO", f"Tersimpan: foto/{fname}")
        except Exception as e:
            self._log_event("FOTO_ERROR", str(e))

    # --- RECORD EVENTS ---
    def record_locked_data(self, t1, t2, tri):
        self.locked_t1 = dict(t1) if t1 else None
        self.locked_t2 = dict(t2) if t2 else None
        self.triangle_info = dict(tri) if tri else None
        if self.locked_t1 and self.locked_t2:
            self._log_event("KUNCI",
                f"T1={self.locked_t1['jarak']:.2f}m sudut={self.locked_t1['sudut']:+.1f}° | "
                f"T2={self.locked_t2['jarak']:.2f}m sudut={self.locked_t2['sudut']:+.1f}°")
            if self.triangle_info:
                self._log_event("SEGITIGA",
                    f"Cam→T1={self.triangle_info['cam_to_t1']:.2f}m | "
                    f"Cam→T2={self.triangle_info['cam_to_t2']:.2f}m | "
                    f"T1→T2={self.triangle_info['t1_to_t2']:.2f}m | "
                    f"Sudut@Cam={self.triangle_info['angle_at_cam_deg']:.1f}°")

    def record_t1_finish(self, dist):
        self.t1_finish_dist = dist
        self._log_event("T1_SELESAI", f"Robot tiba di T1. Jarak akhir: {dist:.2f} m")

    def record_t2_finish(self, dist):
        self.t2_finish_dist = dist
        self.mission_status = "SUKSES"
        self._log_event("T2_SELESAI", f"Robot tiba di T2. Jarak akhir: {dist:.2f} m")
        self._log_event("MISI", "SEMUA TARGET BERHASIL DIKUNJUNGI!")

    # --- FINALIZE ---
    def finalize(self):
        if self.is_finalized:
            return
        self.is_finalized = True

        end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_dur = round(time.time() - self.start_time, 2)

        self._log_event("AKHIR", f"Sesi selesai. Durasi: {total_dur}s | Status: {self.mission_status}")

        # Tulis ringkasan TXT
        t1 = self.locked_t1 or {}
        t2 = self.locked_t2 or {}
        tri = self.triangle_info or {}
        last_enc_l = self.records[-1][8] if self.records else 0
        last_enc_r = self.records[-1][9] if self.records else 0
        last_dist  = self.records[-1][10] if self.records else 0.0

        with open(self.ringkasan_txt, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(" RINGKASAN PERCOBAAN NAVIGASI MULTI-TARGET v3\n")
            f.write("=" * 60 + "\n")
            f.write(f" Status Misi      : {self.mission_status}\n")
            f.write(f" Waktu Mulai      : {self.start_dt}\n")
            f.write(f" Waktu Selesai    : {end_dt}\n")
            f.write(f" Durasi Total     : {total_dur} detik\n")
            f.write(f" Total Frame Log  : {len(self.records)} sampel\n")
            f.write("-" * 60 + "\n")
            f.write(" DATA TARGET\n")
            f.write(f" T1 Jarak Awal    : {t1.get('jarak',0):.2f} m\n")
            f.write(f" T1 Sudut Awal    : {t1.get('sudut',0):+.1f}°\n")
            f.write(f" T1 Jarak Akhir   : {self.t1_finish_dist:.2f} m\n")
            f.write(f" T2 Jarak Awal    : {t2.get('jarak',0):.2f} m\n")
            f.write(f" T2 Sudut Awal    : {t2.get('sudut',0):+.1f}°\n")
            f.write(f" T2 Jarak Akhir   : {self.t2_finish_dist:.2f} m\n")
            f.write("-" * 60 + "\n")
            f.write(" PEMETAAN SEGITIGA 2D\n")
            f.write(f" Cam → T1         : {tri.get('cam_to_t1',0):.3f} m\n")
            f.write(f" Cam → T2         : {tri.get('cam_to_t2',0):.3f} m\n")
            f.write(f" T1  → T2         : {tri.get('t1_to_t2',0):.3f} m\n")
            f.write(f" Sudut @Kamera    : {tri.get('angle_at_cam_deg',0):.2f}°\n")
            f.write(f" T1 (x,y)         : ({tri.get('t1x',0):.3f}, {tri.get('t1y',0):.3f}) m\n")
            f.write(f" T2 (x,y)         : ({tri.get('t2x',0):.3f}, {tri.get('t2y',0):.3f}) m\n")
            f.write("-" * 60 + "\n")
            f.write(" ODOMETRI ENCODER\n")
            f.write(f" Encoder Kiri     : {last_enc_l} tick\n")
            f.write(f" Encoder Kanan    : {last_enc_r} tick\n")
            f.write(f" Estimasi Tempuh  : {last_dist:.3f} m\n")
            f.write("=" * 60 + "\n")
            f.write(" File Output:\n")
            f.write(f"  log/log_frame.csv\n")
            f.write(f"  log/log_event.txt\n")
            f.write(f"  foto/*.png\n")
            f.write(f"  grafik/grafik_rekap.png\n")
            f.write("=" * 60 + "\n")

        print(f"\n>> [LOGGER v3] Ringkasan tersimpan: {self.ringkasan_txt}")

        # Buat grafik
        self._generate_plots()

    def _generate_plots(self):
        if len(self.records) < 3:
            return

        times   = [r[1]  for r in self.records]
        states  = [r[2]  for r in self.records]
        dists   = [r[4]  for r in self.records]
        angles  = [r[5]  for r in self.records]
        enc_ls  = [r[8]  for r in self.records]
        enc_rs  = [r[9]  for r in self.records]
        travels = [r[10] for r in self.records]

        tri = self.triangle_info or {}
        t1  = self.locked_t1 or {}
        t2  = self.locked_t2 or {}

        # Warna per state untuk background shading
        STATE_COLORS = {
            "DUAL_DETECT": "#d0eeff",
            "ALIGN_T1":    "#ffe8cc",
            "APPROACH_T1": "#ccffcc",
            "WAIT_T1":     "#e8e8ff",
            "ROTATE_TO_T2":"#fff4cc",
            "ALIGN_T2":    "#ffd5b8",
            "APPROACH_T2": "#b8ffd5",
            "WAIT_T2":     "#d5b8ff",
            "ALL_FINISHED":"#c8ffc8",
        }

        fig = plt.figure(figsize=(16, 11))
        fig.suptitle(
            f"Rekapitulasi Navigasi Multi-Target v3 — {self.start_dt}\n"
            f"Model: knn_k5.pkl (2800 sampel) | Status: {self.mission_status}",
            fontsize=13, fontweight='bold'
        )

        # --- Subplot 1: Jarak vs Waktu ---
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(times, dists, color='#1a6fbd', lw=2.0, label="Jarak Sensor (m)")
        ax1.axhline(STOP_DISTANCE, color='red', ls='--', lw=1.5,
                    label=f"Batas Stop ({STOP_DISTANCE} m)")
        # Shading state
        prev_s, prev_t = states[0], times[0]
        for i in range(1, len(states)):
            if states[i] != prev_s or i == len(states)-1:
                ax1.axvspan(prev_t, times[i],
                             alpha=0.15,
                             color=STATE_COLORS.get(prev_s, "#eeeeee"))
                prev_s, prev_t = states[i], times[i]
        ax1.set_title("1. Profil Jarak Target vs Waktu", fontweight='bold')
        ax1.set_xlabel("Waktu (detik)"); ax1.set_ylabel("Jarak (m)")
        ax1.set_ylim(bottom=0)
        ax1.grid(True, ls=':', alpha=0.5); ax1.legend(fontsize=8)

        # --- Subplot 2: Sudut vs Waktu ---
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(times, angles, color='#e67e22', lw=1.8, label="Sudut Target (°)")
        ax2.axhline(0, color='gray', lw=1.0)
        ax2.axhspan(-ANGLE_THRESHOLD, ANGLE_THRESHOLD, color='green', alpha=0.15,
                    label=f"Zona Presisi STATELOCK (±{ANGLE_THRESHOLD}°)")
        ax2.set_title("2. Respon Penyelarasan Sudut (STATELOCK)", fontweight='bold')
        ax2.set_xlabel("Waktu (detik)"); ax2.set_ylabel("Sudut (°)")
        ax2.grid(True, ls=':', alpha=0.5); ax2.legend(fontsize=8)

        # --- Subplot 3: Encoder & Jarak Tempuh ---
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(times, enc_ls, color='#2980b9', lw=1.8, label="Encoder Kiri (tick)")
        ax3.plot(times, enc_rs, color='#c0392b', lw=1.8, label="Encoder Kanan (tick)")
        ax3.set_title("3. Encoder Motor & Jarak Tempuh Robot", fontweight='bold')
        ax3.set_xlabel("Waktu (detik)"); ax3.set_ylabel("Tick Encoder")
        ax3.grid(True, ls=':', alpha=0.5)
        ax3b = ax3.twinx()
        ax3b.plot(times, travels, color='#27ae60', ls=':', lw=2.2, label="Jarak Tempuh (m)")
        ax3b.set_ylabel("Jarak Tempuh (m)", color='#27ae60')
        ax3b.tick_params(axis='y', labelcolor='#27ae60')
        lines3, labs3 = ax3.get_legend_handles_labels()
        lines3b, labs3b = ax3b.get_legend_handles_labels()
        ax3.legend(lines3 + lines3b, labs3 + labs3b, fontsize=8, loc='upper left')

        # --- Subplot 4: Peta 2D Segitiga ---
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(0, 0, marker='^', ms=12, color='black', label='Robot/Kamera (0,0)', zorder=5)

        if tri:
            t1x = tri.get('t1x', 0); t1y = tri.get('t1y', 0)
            t2x = tri.get('t2x', 0); t2y = tri.get('t2y', 0)
            ax4.plot(t1x, t1y, 'go', ms=10, label=f"T1 ({t1.get('jarak',0):.2f}m)", zorder=5)
            ax4.plot(t2x, t2y, 'rs', ms=10, label=f"T2 ({t2.get('jarak',0):.2f}m)", zorder=5)
            ax4.plot([0, t1x], [0, t1y], 'g--', alpha=0.7,
                     label=f"Cam→T1: {tri.get('cam_to_t1',0):.2f}m")
            ax4.plot([0, t2x], [0, t2y], 'r--', alpha=0.7,
                     label=f"Cam→T2: {tri.get('cam_to_t2',0):.2f}m")
            ax4.plot([t1x, t2x], [t1y, t2y], 'b-', lw=2.0,
                     label=f"T1→T2: {tri.get('t1_to_t2',0):.2f}m")
            ax4.text(t1x+0.05, t1y+0.05,
                     f"T1\n({t1x:.2f},{t1y:.2f})", fontsize=9, color='darkgreen', fontweight='bold')
            ax4.text(t2x+0.05, t2y+0.05,
                     f"T2\n({t2x:.2f},{t2y:.2f})", fontsize=9, color='darkred', fontweight='bold')
            ax4.text(0.05, -0.15,
                     f"Sudut Cam: {tri.get('angle_at_cam_deg',0):.1f}°", fontsize=9, fontweight='bold')
            mr = max(1.5, max(abs(t1x), abs(t2x), t1y, t2y) + 0.5)
            ax4.set_xlim(-mr, mr); ax4.set_ylim(-0.3, mr)
        else:
            ax4.text(0, 0.5, "Data segitiga belum tersedia", ha='center', color='gray', fontsize=10)
            ax4.set_xlim(-2, 2); ax4.set_ylim(-0.5, 3)

        ax4.set_title("4. Pemetaan 2D Segitiga Kamera-T1-T2", fontweight='bold')
        ax4.set_xlabel("X Lateral (m)"); ax4.set_ylabel("Y Depth / Maju (m)")
        ax4.grid(True, ls=':', alpha=0.5); ax4.legend(fontsize=8, loc='lower right')

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out = os.path.join(self.grafik_dir, "grafik_rekap.png")
        try:
            plt.savefig(out, dpi=300)
            print(f">> [LOGGER v3] Grafik tersimpan: grafik/grafik_rekap.png")
        except Exception as e:
            print(f">> [ERROR] Gagal simpan grafik: {e}")
        finally:
            plt.close(fig)

    def reset(self):
        """Dipanggil saat pengguna menekan 'r' — finalize sesi ini, mulai sesi baru."""
        if len(self.records) >= 5 and not self.is_finalized:
            self.finalize()
        self.__init__()   # Reset penuh ke sesi baru

    @property
    def trial_number(self):
        """Kompatibilitas dengan draw_hud yang menggunakan logger.trial_number."""
        # Hitung dari nama folder sesi
        try:
            sessions = [d for d in os.listdir(self.BASE_RESULTS_DIR)
                        if d.startswith("percobaan_")]
            return len(sessions)
        except Exception:
            return 1


# =============================================================
# INISIALISASI SISTEM
# =============================================================
print("=" * 65)
print("   SISTEM NAVIGASI MULTI-TARGET v3 (Segitiga Kamera-T1-T2)")
print("   Model KNN: knn_k5.pkl — 2800 sampel (semua variasi gesture)")
print("   Output   : hasil_percobaan/percobaan_YYYYMMDD_HHMMSS/")
print("=" * 65)

esp32 = ESP32Controller(port='COM5', baudrate=115200)

print("Memuat YOLOv8n dan Model KNN v3 (knn_k5.pkl)...")
yolo_model = YOLO('yolov8n.pt')

# [v3.1] Path model KNN diperbarui ke knn_k5.pkl (2800 sampel)
# Model baru mendeteksi: tangan kiri, tangan kanan, ATAU keduanya terangkat.
_base = os.path.dirname(os.path.abspath(__file__))
_model_path_new = os.path.join(_base, '..', 'klasifikasi_gestur', 'model', 'knn_k5.pkl')
_model_path_new = os.path.normpath(_model_path_new)

if os.path.exists(_model_path_new):
    knn_model = joblib.load(_model_path_new)
    print(f">> [KNN v3] Model berhasil dimuat dari: {_model_path_new}")
    print(f"   Jumlah sampel latih: {knn_model._fit_X.shape[0]}")
else:
    # Fallback ke model lama dengan peringatan
    print(f">> [PERINGATAN] Model baru tidak ditemukan di: {_model_path_new}")
    print("   Fallback ke model lama (model_knn_gesture.pkl) — 320 sampel.")
    try:
        knn_model = joblib.load('model_knn_gesture.pkl')
    except FileNotFoundError:
        knn_model = joblib.load('../Skenario Multi Target/model_knn_gesture.pkl')

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

print("Menghubungkan ke sensor Kinect v1...")
openni2.initialize()
dev = openni2.Device.open_any()
color_stream = dev.create_color_stream(); color_stream.start()
depth_stream = dev.create_depth_stream(); depth_stream.start()

logger = PercobaaanLogger()

print("\n>> Kamera & Depth Sensor aktif!")
print(">> Tekan 'q' di jendela video untuk Keluar.")
print(">> Setelah misi selesai, ketik 'r' + Enter di terminal untuk Reset.\n")

current_cmd_j = 0.0
current_cmd_s = 0.0

# =============================================================
# FUNGSI BANTU (tidak berubah dari v2)
# =============================================================
def get_depth_in_meters(depth_frame, px, py, frame_w=640, frame_h=480):
    px = max(0, min(int(px), frame_w - 1))
    py = max(0, min(int(py), frame_h - 1))
    depth_array = np.frombuffer(
        depth_frame.get_buffer_as_uint16(), dtype=np.uint16
    ).reshape(480, 640)
    y1, y2 = max(0, py - 3), min(frame_h, py + 4)
    x1, x2 = max(0, px - 3), min(frame_w, px + 4)
    patch = depth_array[y1:y2, x1:x2]
    valid = patch[patch > 0]
    if len(valid) > 0:
        return round(float(np.median(valid)) / 1000.0, 2)
    return 0.0


def send_async(j1, s1):
    global current_cmd_j, current_cmd_s
    current_cmd_j = j1; current_cmd_s = s1
    if esp32:
        esp32.send_data(j1, s1, 0.0, 0.0)


def classify_pose(pose_landmarks, knn_model):
    """
    Klasifikasi gestur menggunakan model KNN v3 (2800 sampel).
    Mendeteksi: tangan kiri, tangan kanan, atau KEDUA tangan terangkat.
    Fitur: f_L = y_LW - y_LS, f_R = y_RW - y_RS (negatif = tangan di atas bahu)
    """
    if pose_landmarks is None:
        return 'neutral'
    lm = pose_landmarks.landmark
    ls, rs = lm[11], lm[12]   # Bahu
    lw, rw = lm[15], lm[16]   # Pergelangan tangan
    lh, rh = lm[23], lm[24]   # Pinggul (untuk hands_down)

    l_diff = lw.y - ls.y      # f_L
    r_diff = rw.y - rs.y      # f_R

    pred = knn_model.predict(np.array([[l_diff, r_diff]]))[0]
    if pred == 1:
        return 'hands_up'

    hip_avg_y = (lh.y + rh.y) / 2
    if lw.y > hip_avg_y and rw.y > hip_avg_y:
        return 'hands_down'
    return 'neutral'


def compute_triangle(a1_deg, d1, a2_deg, d2):
    a1 = math.radians(a1_deg); a2 = math.radians(a2_deg)
    t1x = d1 * math.sin(a1);   t1y = d1 * math.cos(a1)
    t2x = d2 * math.sin(a2);   t2y = d2 * math.cos(a2)
    d_t1t2 = math.hypot(t2x - t1x, t2y - t1y)
    dot = t1x * t2x + t1y * t2y
    cos_a = max(-1.0, min(1.0, dot / (d1 * d2 + 1e-9)))
    return {
        'cam_to_t1': d1, 'cam_to_t2': d2,
        't1_to_t2': round(d_t1t2, 3),
        'angle_at_cam_deg': round(math.degrees(math.acos(cos_a)), 2),
        't1x': round(t1x, 3), 't1y': round(t1y, 3),
        't2x': round(t2x, 3), 't2y': round(t2y, 3),
    }


def check_keyboard_reset():
    import msvcrt
    if msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in ('r', 'R'):
            return True
    return False


# =============================================================
# HUD (tidak berubah dari v2, kecuali penambahan label model)
# =============================================================
def draw_hud(frame, state, curr_sudut, curr_jarak, start_dist,
             detected_persons, locked_t1, locked_t2,
             triangle_info, dual_prog, handsdown_prog, enc_l=0, enc_r=0, trial_num=1):
    fh, fw = frame.shape[:2]
    cx = fw // 2
    cv2.line(frame, (cx, 34), (cx, fh - 48), (70, 70, 85), 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 6), (fw - 10, 36), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (10, 6), (fw - 10, 36), (50, 50, 65), 1)

    cv2.circle(frame, (22, 21), 5, (0, 0, 255), -1)
    cv2.putText(frame, "REC", (32, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Label model versi — penanda visual v3
    cv2.putText(frame, "v3 KNN-2800", (fw - 100, fh - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 255), 1)

    if state == STATE_DUAL_DETECT:
        pct = int(dual_prog * 100)
        status_txt = (f"[FASE 1] DETEKSI AWAL: Tunggu 2 target Hands-Up ({pct}%)"
                      if dual_prog < 1.0 else "[FASE 1] MENGUNCI KOORDINAT...")
        scolor = (0, 215, 255)
    elif state in (STATE_ALIGN_T1, STATE_ALIGN_T2):
        lbl = "T1" if state == STATE_ALIGN_T1 else "T2"
        side = "KIRI" if curr_sudut > 0 else "KANAN"
        if abs(curr_sudut) <= ANGLE_THRESHOLD:
            status_txt = f"[{lbl}] STATELOCK - TERPUSAT ({curr_sudut:+.1f}) -> KUNCI..."; scolor = (50, 230, 100)
        elif abs(curr_sudut) <= 4.5:
            status_txt = f"[{lbl}] ALIGN HALUS (Nudge {side}: {abs(curr_sudut):.1f})"; scolor = (255, 200, 30)
        else:
            status_txt = f"[{lbl}] ALIGN KASAR ({side}: {abs(curr_sudut):.1f})"; scolor = (255, 175, 20)
    elif state in (STATE_APPROACH_T1, STATE_APPROACH_T2):
        lbl = "T1" if state == STATE_APPROACH_T1 else "T2"
        status_txt = f"[{lbl}] MAJU -> Jarak Aman {STOP_DISTANCE:.1f} m"; scolor = (50, 230, 100)
    elif state in (STATE_WAIT_HANDSDOWN_T1, STATE_WAIT_HANDSDOWN_T2):
        lbl = "T1" if state == STATE_WAIT_HANDSDOWN_T1 else "T2"
        pct = int(handsdown_prog * 100)
        status_txt = f"[{lbl} TIBA] Turunkan Tangan {lbl} untuk Lanjut... ({pct}%)"; scolor = (0, 255, 220)
    elif state == STATE_ROTATE_TO_T2:
        status_txt = "[TRANSISI] Robot berputar ke koordinat memori T2..."; scolor = (255, 140, 0)
    else:
        status_txt = "[MISI SELESAI] Semua target berhasil! Ketik 'r'+Enter untuk Reset."; scolor = (50, 255, 150)

    cv2.putText(frame, status_txt, (65, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, scolor, 1)
    cv2.putText(frame, "Q: Keluar | r: Reset", (fw - 180, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 170, 170), 1)
    info_txt = f"Percobaan #{trial_num} | ENC L:{enc_l} | R:{enc_r}"
    cv2.putText(frame, info_txt, (20, fh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1)

    if state == STATE_DUAL_DETECT and dual_prog > 0:
        bw = int((fw - 20) * min(1.0, dual_prog))
        cv2.rectangle(frame, (10, 34), (10 + bw, 36), (0, 215, 255), -1)
    if state in (STATE_WAIT_HANDSDOWN_T1, STATE_WAIT_HANDSDOWN_T2) and handsdown_prog > 0:
        bw = int((fw - 20) * min(1.0, handsdown_prog))
        cv2.rectangle(frame, (10, 34), (10 + bw, 36), (0, 255, 180), -1)

    for p in detected_persons:
        x1, y1, x2, y2 = p['box']; mx, my = p['mid']
        j = p['jarak']; pid = p['id']
        gesture = p.get('gesture', 'neutral')
        is_t1 = locked_t1 and p['id'] == locked_t1.get('id')
        is_t2 = locked_t2 and p['id'] == locked_t2.get('id')

        if is_t1:   color = (0, 230, 255);  label = f"T1 [AKTIF]: {j:.2f}m";     thick = 2
        elif is_t2: color = (255, 100, 50); label = f"T2 [MEMORI]: {j:.2f}m";    thick = 2
        elif gesture == 'hands_up': color = (255, 180, 50); label = f"Kandidat #{pid}: {j:.2f}m"; thick = 1
        else:       color = (100, 100, 100); label = f"P#{pid}: {j:.2f}m";        thick = 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        cv2.circle(frame, (mx, my), 6, color, -1 if thick == 1 else 2)
        cv2.putText(frame, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, thick)
        # Tambahan v3: tampilkan f_L, f_R pada setiap kandidat
        cv2.putText(frame, gesture.upper(), (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    if triangle_info:
        tri_lines = [
            "Segitiga Kamera-T1-T2:",
            f"  Cam->T1 : {triangle_info['cam_to_t1']:.2f} m",
            f"  Cam->T2 : {triangle_info['cam_to_t2']:.2f} m",
            f"  T1->T2  : {triangle_info['t1_to_t2']:.2f} m",
            f"  Sudut Cam: {triangle_info['angle_at_cam_deg']:.1f}°",
        ]
        ty = fh - 10 - len(tri_lines) * 16
        for i, tl in enumerate(tri_lines):
            cv2.putText(frame, tl, (fw - 250, ty + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 200, 255), 1)

    if state in (STATE_ALIGN_T1, STATE_APPROACH_T1,
                 STATE_ALIGN_T2, STATE_APPROACH_T2) and curr_jarak > 0:
        by1, by2 = fh - 44, fh - 12
        bw_total = fw - 40; bx1 = 20
        ov2 = frame.copy()
        cv2.rectangle(ov2, (bx1, by1), (bx1 + bw_total, by2), (15, 15, 20), -1)
        cv2.addWeighted(ov2, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx1 + bw_total, by2), (50, 50, 65), 1)
        cv2.putText(frame, f"Jarak: {curr_jarak:.2f} m", (bx1 + 12, by1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 240, 90), 2)
        bar_x1 = bx1 + 130; bar_x2 = bx1 + bw_total - 120
        bar_w = max(10, bar_x2 - bar_x1)
        bar_y1, bar_y2 = by1 + 9, by2 - 9
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (30, 30, 40), -1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (65, 65, 80), 1)
        base_start = max(start_dist if start_dist > STOP_DISTANCE else 3.0, 1.2)
        progress = max(0.0, min(1.0, (base_start - curr_jarak) /
                                max(0.01, base_start - STOP_DISTANCE)))
        fill_w = int(progress * bar_w)
        fill_color = (50, 230, 100) if curr_jarak <= STOP_DISTANCE else (30, 190, 240)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), fill_color, -1)
        cv2.putText(frame, f"{int(progress * 100)}%",
                    (bar_x1 + bar_w // 2 - 12, by1 + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        f_color = (50, 230, 100) if curr_jarak <= STOP_DISTANCE else (120, 140, 255)
        cv2.putText(frame, f"Finish: {STOP_DISTANCE:.1f}m", (bar_x2 + 10, by1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, f_color, 1)


# =============================================================
# RESET
# =============================================================
_finished_banner_shown = False

def reset_all():
    global state, locked_t1, locked_t2, triangle_info
    global dual_detect_start, dual_detect_active, dual_last_seen
    global handsdown_start, handsdown_active, handsdown_last_seen
    global angle_history, smooth_sudut, last_valid_jarak, start_distance
    global align_pulse_active, align_pulse_start, align_settle_count
    global scan_pulse_active, scan_start_time
    global last_stream_time, last_terminal_log
    global rotate_target_sudut, _finished_banner_shown

    state = STATE_DUAL_DETECT
    locked_t1 = None; locked_t2 = None; triangle_info = None
    dual_detect_start = 0.0; dual_detect_active = False; dual_last_seen = 0.0
    handsdown_start = 0.0; handsdown_active = False; handsdown_last_seen = 0.0
    angle_history = []; smooth_sudut = 0.0
    last_valid_jarak = 0.0; start_distance = 0.0
    align_pulse_active = False; align_pulse_start = time.time()
    align_settle_count = 0; scan_pulse_active = False
    scan_start_time = 0.0; last_stream_time = 0.0
    last_terminal_log = 0.0; rotate_target_sudut = 0.0
    _finished_banner_shown = False

    send_async(0.0, 0.0)
    logger.reset()
    print("\n" + "=" * 65)
    print(">> [RESET TOTAL] Sistem dikembalikan ke awal (Fase 1 Deteksi).")
    print(">> [LOGGER v3]   Sesi percobaan baru dimulai.")
    print("=" * 65 + "\n")


# =============================================================
# VARIABEL RUNTIME
# =============================================================
state = STATE_DUAL_DETECT
locked_t1 = None; locked_t2 = None; triangle_info = None
dual_detect_start = 0.0; dual_detect_active = False; dual_last_seen = 0.0
handsdown_start = 0.0; handsdown_active = False; handsdown_last_seen = 0.0
angle_history = []; smooth_sudut = 0.0
last_valid_jarak = 0.0; start_distance = 0.0
align_pulse_active = False; align_pulse_start = time.time()
align_settle_count = 0; scan_pulse_active = False
scan_start_time = 0.0; last_stream_time = 0.0
last_terminal_log = 0.0; rotate_target_sudut = 0.0

# =============================================================
# MAIN LOOP
# =============================================================
while True:
    c_frame = color_stream.read_frame()
    d_frame = depth_stream.read_frame()

    img   = np.frombuffer(c_frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    fh, fw = frame.shape[:2]
    cx_frame  = fw // 2
    current_t = time.time()

    # YOLO detection
    yolo_res = yolo_model(frame, verbose=False, classes=[0], conf=0.5)
    detected_persons = []
    person_idx = 1

    for r in yolo_res:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, min(x1, fw-1)); y1 = max(0, min(y1, fh-1))
            x2 = max(0, min(x2, fw));   y2 = max(0, min(y2, fh))
            bw = x2 - x1; bh = y2 - y1
            if bw < 35 or bh < 70:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pose_res = pose_detector.process(crop_rgb)

            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            gesture = 'neutral'

            if pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks.landmark
                ls, rs = lm[11], lm[12]
                gesture = classify_pose(pose_res.pose_landmarks, knn_model)
                mid_x = int(x1 + ((ls.x + rs.x) / 2) * bw)
                mid_y = int(y1 + ((ls.y + rs.y) / 2) * bh)
                mid_x = max(x1, min(mid_x, x2))
                mid_y = max(y1, min(mid_y, y2))

            jarak = get_depth_in_meters(d_frame, mid_x, mid_y, fw, fh)
            if jarak <= 0:
                jarak = get_depth_in_meters(d_frame, (x1+x2)//2, (y1+y2)//2, fw, fh)

            sudut = round((cx_frame - mid_x) * (57.5 / fw), 1)
            detected_persons.append({
                'id': person_idx, 'box': (x1, y1, x2, y2),
                'mid': (mid_x, mid_y), 'jarak': jarak,
                'sudut': sudut, 'gesture': gesture,
            })
            person_idx += 1

    min_dist_filter = (STOP_DISTANCE + 0.15) if state in (
        STATE_ALIGN_T2, STATE_ROTATE_TO_T2) else 0.0
    valid_persons  = [p for p in detected_persons if p['jarak'] > min_dist_filter]
    hands_up_cands = [p for p in valid_persons if p['gesture'] == 'hands_up']

    selected_target = None
    if len(hands_up_cands) > 0:
        hands_up_cands.sort(key=lambda p: p['jarak'])
        selected_target = hands_up_cands[0]
    elif state in (STATE_ALIGN_T1, STATE_APPROACH_T1,
                   STATE_ALIGN_T2, STATE_APPROACH_T2) and len(valid_persons) > 0:
        valid_persons.sort(key=lambda p: p['jarak'])
        selected_target = valid_persons[0]

    curr_sudut = 0.0; curr_jarak = 0.0
    if selected_target:
        angle_history.append(selected_target['sudut'])
        if len(angle_history) > SMOOTH_WINDOW:
            angle_history.pop(0)
        smooth_sudut = round(sum(angle_history) / len(angle_history), 1)
        curr_sudut = smooth_sudut
        if selected_target['jarak'] > 0:
            last_valid_jarak = selected_target['jarak']
        curr_jarak = last_valid_jarak

    if not selected_target and state in (STATE_DUAL_DETECT, STATE_ROTATE_TO_T2):
        angle_history.clear(); smooth_sudut = 0.0

    dual_prog = 0.0; handsdown_prog = 0.0

    # ==========================================================
    # STATE MACHINE (sama dengan v2, ditambah simpan_foto di event kunci)
    # ==========================================================

    # [FASE 1] DUAL_DETECT
    if state == STATE_DUAL_DETECT:
        hu_list_valid = [p for p in detected_persons
                         if p['gesture'] == 'hands_up' and p['jarak'] > 0]
        hu_list_any   = [p for p in detected_persons
                         if p['gesture'] == 'hands_up']
        hu_list = hu_list_valid if len(hu_list_valid) >= 2 else hu_list_any
        got_two = len(hu_list) >= 2

        if got_two:
            dual_last_seen = current_t
            if not dual_detect_active:
                dual_detect_start = current_t
                dual_detect_active = True
                print(f"\n>> [DETEKSI AWAL] Dua target hands-up terdeteksi. "
                      f"Tahan {DUAL_DETECT_HOLD:.1f}s untuk mengunci koordinat...")
                # [v3.3] Foto saat pertama kali deteksi 2 target
                draw_hud(frame, state, 0, 0, 0, detected_persons, None, None,
                         None, 0.0, 0.0, 0, 0, logger.trial_number)
                logger.simpan_foto("dual_detect_mulai", frame.copy())

            elapsed = current_t - dual_detect_start
            dual_prog = min(1.0, elapsed / DUAL_DETECT_HOLD)

            if elapsed >= DUAL_DETECT_HOLD:
                lock_list = hu_list_valid if len(hu_list_valid) >= 2 else hu_list
                lock_list.sort(key=lambda p: p['jarak'] if p['jarak'] > 0 else 999)
                locked_t1 = dict(lock_list[0])
                locked_t2 = dict(lock_list[1])
                rotate_target_sudut = locked_t2['sudut']

                triangle_info = compute_triangle(
                    locked_t1['sudut'], locked_t1['jarak'],
                    locked_t2['sudut'], locked_t2['jarak']
                )
                start_distance = locked_t1['jarak']

                print("\n" + "=" * 65)
                print(">> [KOORDINAT DIKUNCI!]")
                print(f"   T1 (Terdekat)  : Jarak={locked_t1['jarak']:.2f}m | "
                      f"Sudut={locked_t1['sudut']:+.1f} | ID=#{locked_t1['id']}")
                print(f"   T2 (Lebih Jauh): Jarak={locked_t2['jarak']:.2f}m | "
                      f"Sudut={locked_t2['sudut']:+.1f} | ID=#{locked_t2['id']}")
                print(f"   Pemetaan Segitiga: Cam->T1={triangle_info['cam_to_t1']:.2f}m | "
                      f"Cam->T2={triangle_info['cam_to_t2']:.2f}m | "
                      f"T1->T2={triangle_info['t1_to_t2']:.2f}m | "
                      f"Sudut@Cam={triangle_info['angle_at_cam_deg']:.1f}°")
                print("=" * 65 + "\n")

                logger.record_locked_data(locked_t1, locked_t2, triangle_info)
                # [v3.3] Foto saat koordinat dikunci
                logger.simpan_foto("kunci_koordinat", frame.copy())

                dual_detect_active = False
                align_pulse_start = current_t
                align_pulse_active = False
                align_settle_count = 0
                angle_history.clear(); smooth_sudut = 0.0
                last_stream_time = current_t
                state = STATE_ALIGN_T1
        else:
            if dual_detect_active:
                gap = current_t - dual_last_seen
                if gap > DUAL_DETECT_GRACE:
                    dual_detect_active = False
                    print(f">> [DETEKSI AWAL] Target berkurang > "
                          f"{DUAL_DETECT_GRACE:.1f}s, reset timer...")
                else:
                    elapsed = current_t - dual_detect_start
                    dual_prog = min(1.0, elapsed / DUAL_DETECT_HOLD)

    elif state in (STATE_ALIGN_T1, STATE_ALIGN_T2):
        next_approach = STATE_APPROACH_T1 if state == STATE_ALIGN_T1 else STATE_APPROACH_T2
        next_wait     = STATE_WAIT_HANDSDOWN_T1 if state == STATE_ALIGN_T1 else STATE_WAIT_HANDSDOWN_T2
        lbl = "T1" if state == STATE_ALIGN_T1 else "T2"

        if selected_target:
            abs_s = abs(curr_sudut)
            in_coarse_zone = abs_s > 4.5
            in_fine_zone   = ANGLE_THRESHOLD < abs_s <= 4.5

            if in_coarse_zone or in_fine_zone:
                if align_settle_count > 0:
                    angle_history.clear()
                    if selected_target:
                        angle_history.append(selected_target['sudut'])
                    smooth_sudut = selected_target['sudut'] if selected_target else 0.0
                    curr_sudut = smooth_sudut
                    abs_s = abs(curr_sudut)
                    in_coarse_zone = abs_s > 4.5
                    in_fine_zone   = ANGLE_THRESHOLD < abs_s <= 4.5
                align_settle_count = 0

                pulse_dur = COARSE_PULSE_DUR if in_coarse_zone else FINE_PULSE_DURATION
                pause_dur = COARSE_PAUSE_DUR if in_coarse_zone else FINE_PAUSE_DURATION
                cycle_dur = pulse_dur + pause_dur

                if not align_pulse_active:
                    t_in_cycle = (current_t - align_pulse_start) % cycle_dur
                else:
                    t_in_cycle = (current_t - align_pulse_start) % cycle_dur

                if t_in_cycle < pulse_dur:
                    if not align_pulse_active:
                        align_pulse_start = current_t - t_in_cycle
                        side_txt = "KIRI" if curr_sudut > 0 else "KANAN"
                        cmd_sudut = curr_sudut if in_coarse_zone else (
                            5.0 if curr_sudut > 0 else -5.0)
                        lbl_mode = "COARSE" if in_coarse_zone else "FINE"
                        print(f">> [ALIGN {lbl_mode} {lbl}] Sudut: {curr_sudut:+.1f} "
                              f"({side_txt}) -> Denyut {int(pulse_dur*1000)}ms")
                        t = threading.Thread(target=send_async, args=(1.0, cmd_sudut))
                        t.daemon = True; t.start()
                        align_pulse_active = True; last_stream_time = current_t
                else:
                    if align_pulse_active:
                        t = threading.Thread(target=send_async, args=(0.0, 0.0))
                        t.daemon = True; t.start()
                        align_pulse_active = False; last_stream_time = current_t
                        print(f">> [ALIGN BERHENTI {lbl}] Sudut: {curr_sudut:+.1f} | "
                              f"Jeda {int(pause_dur*1000)}ms...")
            else:
                if align_pulse_active:
                    t = threading.Thread(target=send_async, args=(0.0, 0.0))
                    t.daemon = True; t.start()
                    align_pulse_active = False; last_stream_time = current_t
                    align_settle_count = 0

                if current_t - last_stream_time >= STREAM_INTERVAL:
                    t = threading.Thread(target=send_async, args=(0.0, 0.0))
                    t.daemon = True; t.start(); last_stream_time = current_t

                align_settle_count += 1
                print(f">> [ALIGN CENTER {lbl}] Sudut: {curr_sudut:+.1f} | "
                      f"Stabil: {align_settle_count}/{LOCK_CONFIRM_FRAMES}")

                if align_settle_count >= LOCK_CONFIRM_FRAMES:
                    start_distance = curr_jarak if curr_jarak > 0 else start_distance
                    print("\n" + "=" * 65)
                    print(f">> [STATELOCK {lbl} TERKUNCI!]")
                    print(f"   Sudut: {curr_sudut:+.1f} | Jarak: {start_distance:.2f} m")
                    print(f"   Robot MULAI MAJU menuju {lbl} (Finish {STOP_DISTANCE:.1f}m)...")
                    print("=" * 65 + "\n")
                    # [v3.3] Foto saat STATELOCK terkunci
                    logger.simpan_foto(f"statelock_{lbl.lower()}", frame.copy())
                    align_settle_count = 0; align_pulse_active = False
                    if start_distance <= STOP_DISTANCE:
                        t = threading.Thread(target=send_async, args=(0.0, 0.0))
                        t.daemon = True; t.start()
                        handsdown_active = False; handsdown_start = 0.0
                        state = next_wait
                    else:
                        state = next_approach
                        t = threading.Thread(target=send_async, args=(start_distance, 0.0))
                        t.daemon = True; t.start()
        else:
            align_settle_count = 0
            if align_pulse_active or (current_t - last_stream_time >= STREAM_INTERVAL):
                t = threading.Thread(target=send_async, args=(0.0, 0.0))
                t.daemon = True; t.start()
                align_pulse_active = False; last_stream_time = current_t

    elif state in (STATE_APPROACH_T1, STATE_APPROACH_T2):
        next_wait = STATE_WAIT_HANDSDOWN_T1 if state == STATE_APPROACH_T1 else STATE_WAIT_HANDSDOWN_T2
        lbl = "T1" if state == STATE_APPROACH_T1 else "T2"

        if current_t - last_stream_time >= STREAM_INTERVAL:
            if state == STATE_APPROACH_T2:
                t2_raw = [p for p in detected_persons if p['jarak'] > 0]
                if t2_raw:
                    t2_raw.sort(key=lambda p: p['jarak'])
                    t2_best = min(t2_raw, key=lambda p: abs(p['jarak'] - last_valid_jarak))
                    c_dist = t2_best['jarak']
                    last_valid_jarak = c_dist
                else:
                    c_dist = last_valid_jarak
            else:
                c_dist = curr_jarak if curr_jarak > 0 else last_valid_jarak

            if c_dist > 0 and c_dist <= STOP_DISTANCE:
                print("\n" + "=" * 65)
                print(f">> [FINISH {lbl} TERCAPAI!] Jarak: {c_dist:.2f} m")
                print(f"   Robot BERHENTI. Tunggu gesture HANDS DOWN dari {lbl}...")
                print("=" * 65 + "\n")
                t = threading.Thread(target=send_async, args=(0.0, 0.0))
                t.daemon = True; t.start()
                handsdown_active = False; handsdown_start = 0.0
                # [v3.3] Foto saat robot tiba di target
                logger.simpan_foto(f"tiba_{lbl.lower()}", frame.copy())
                state = next_wait
            else:
                t = threading.Thread(target=send_async, args=(c_dist, 0.0))
                t.daemon = True; t.start()
                if current_t - last_terminal_log >= 1.0:
                    sisa = max(0.0, c_dist - STOP_DISTANCE)
                    print(f">> [{lbl} MAJU] Jarak: {c_dist:.2f}m | Sisa: {sisa:.2f}m")
                    last_terminal_log = current_t
            last_stream_time = current_t

    elif state == STATE_WAIT_HANDSDOWN_T1:
        if current_t - last_stream_time >= STREAM_INTERVAL:
            send_async(0.0, 0.0); last_stream_time = current_t

        near = [p for p in detected_persons
                if p['jarak'] <= (STOP_DISTANCE + 0.5) or p['jarak'] == 0]
        is_still_hands_up = any(p['gesture'] == 'hands_up' for p in near)
        target_released = (len(near) > 0 and not is_still_hands_up) or \
                          (len(detected_persons) > 0 and not is_still_hands_up)

        if target_released:
            handsdown_last_seen = current_t
            if not handsdown_active:
                handsdown_start = current_t; handsdown_active = True
                print(f">> [T1] Target sudah menurunkan tangan! Tahan {HANDSDOWN_HOLD:.1f}s...")
            elapsed = current_t - handsdown_start
            handsdown_prog = min(1.0, elapsed / HANDSDOWN_HOLD)
            if elapsed >= HANDSDOWN_HOLD:
                print("\n" + "=" * 65)
                print(">> [T1 SELESAI!] Tangan diturunkan / interaksi selesai.")
                print(f"   Robot berputar menuju koordinat memori T2 "
                      f"(sudut: {rotate_target_sudut:+.1f})")
                print("=" * 65 + "\n")
                logger.record_t1_finish(STOP_DISTANCE)
                # [v3.3] Foto saat T1 dikonfirmasi selesai
                logger.simpan_foto("t1_selesai", frame.copy())
                handsdown_active = False
                scan_start_time = current_t; scan_pulse_active = False
                angle_history.clear(); smooth_sudut = 0.0
                state = STATE_ROTATE_TO_T2
        else:
            if handsdown_active:
                gap = current_t - handsdown_last_seen
                if gap > HANDSDOWN_GRACE:
                    handsdown_active = False
                    print(f">> [T1] Target masih Hands-Up, menunggu tangan diturunkan...")
                else:
                    elapsed = current_t - handsdown_start
                    handsdown_prog = min(1.0, elapsed / HANDSDOWN_HOLD)
                    if elapsed >= HANDSDOWN_HOLD:
                        logger.record_t1_finish(STOP_DISTANCE)
                        logger.simpan_foto("t1_selesai", frame.copy())
                        handsdown_active = False
                        scan_start_time = current_t; scan_pulse_active = False
                        angle_history.clear(); smooth_sudut = 0.0
                        state = STATE_ROTATE_TO_T2

    elif state == STATE_ROTATE_TO_T2:
        # [v3.3] Foto satu kali saat rotasi dimulai
        if not scan_pulse_active and (current_t - scan_start_time < 0.3):
            logger.simpan_foto("rotasi_ke_t2", frame.copy())

        t2_visible = [p for p in valid_persons if p['jarak'] > (STOP_DISTANCE + 0.15)]
        t2_candidate = None
        if t2_visible:
            t2_visible.sort(key=lambda p: p['jarak'])
            t2_candidate = t2_visible[0]

        if t2_candidate:
            print("\n" + "=" * 65)
            print(f">> [T2 TERDETEKSI di FOV!] ID=#{t2_candidate['id']}")
            print(f"   Jarak: {t2_candidate['jarak']:.2f}m | "
                  f"Sudut: {t2_candidate['sudut']:+.1f}")
            print("   Robot BERHENTI BERPUTAR -> Masuk STATELOCK T2...")
            print("=" * 65 + "\n")
            t = threading.Thread(target=send_async, args=(0.0, 0.0))
            t.daemon = True; t.start()
            locked_t2 = dict(t2_candidate)
            start_distance = t2_candidate['jarak']
            angle_history.clear(); smooth_sudut = 0.0
            align_pulse_start = current_t; align_pulse_active = False
            align_settle_count = 0; last_stream_time = current_t
            state = STATE_ALIGN_T2
        else:
            scan_dir = -abs(SCAN_ROTATION_ANGLE) if rotate_target_sudut < 0 \
                       else abs(SCAN_ROTATION_ANGLE)
            cycle_time = SCAN_STEP_PULSE + SCAN_STEP_PAUSE
            t_in_cycle = (current_t - scan_start_time) % cycle_time

            if t_in_cycle < SCAN_STEP_PULSE:
                if not scan_pulse_active:
                    t = threading.Thread(target=send_async, args=(1.0, scan_dir))
                    t.daemon = True; t.start()
                    scan_pulse_active = True; last_stream_time = current_t
            else:
                if scan_pulse_active:
                    t = threading.Thread(target=send_async, args=(0.0, 0.0))
                    t.daemon = True; t.start()
                    scan_pulse_active = False; last_stream_time = current_t

            if current_t - last_terminal_log >= 1.5:
                dir_txt = "CW/kanan" if scan_dir < 0 else "CCW/kiri"
                print(f">> [ROTASI KE T2] Berputar {dir_txt}, "
                      f"sudut memori {rotate_target_sudut:+.1f}...")
                last_terminal_log = current_t

    elif state == STATE_WAIT_HANDSDOWN_T2:
        if current_t - last_stream_time >= STREAM_INTERVAL:
            send_async(0.0, 0.0); last_stream_time = current_t

        near = [p for p in detected_persons
                if p['jarak'] <= (STOP_DISTANCE + 0.5) or p['jarak'] == 0]
        is_still_hands_up = any(p['gesture'] == 'hands_up' for p in near)
        target_released = (len(near) > 0 and not is_still_hands_up) or \
                          (len(detected_persons) > 0 and not is_still_hands_up)

        if target_released:
            handsdown_last_seen = current_t
            if not handsdown_active:
                handsdown_start = current_t; handsdown_active = True
                print(f">> [T2] Target sudah menurunkan tangan! Tahan {HANDSDOWN_HOLD:.1f}s...")
            elapsed = current_t - handsdown_start
            handsdown_prog = min(1.0, elapsed / HANDSDOWN_HOLD)
            if elapsed >= HANDSDOWN_HOLD:
                logger.record_t2_finish(STOP_DISTANCE)
                # [v3.3] Foto saat misi selesai
                logger.simpan_foto("misi_selesai", frame.copy())
                state = STATE_ALL_FINISHED
        else:
            if handsdown_active:
                gap = current_t - handsdown_last_seen
                if gap > HANDSDOWN_GRACE:
                    handsdown_active = False
                    print(f">> [T2] Target masih Hands-Up, menunggu tangan diturunkan...")
                else:
                    elapsed = current_t - handsdown_start
                    handsdown_prog = min(1.0, elapsed / HANDSDOWN_HOLD)
                    if elapsed >= HANDSDOWN_HOLD:
                        logger.record_t2_finish(STOP_DISTANCE)
                        logger.simpan_foto("misi_selesai", frame.copy())
                        state = STATE_ALL_FINISHED

    elif state == STATE_ALL_FINISHED:
        send_async(0.0, 0.0)
        if not _finished_banner_shown:
            print("\n" + "X" * 65)
            print("X                                                               X")
            print("X   [OK] MISI NAVIGASI MULTI-TARGET v3 SELESAI DENGAN SUKSES!  X")
            print("X   Semua target (T1 & T2) dikunjungi & dikonfirmasi.          X")
            print("X                                                               X")
            print("X   Ketik  r + Enter  di terminal ini untuk Reset ke awal.     X")
            print("X   Tekan  q  di jendela video untuk Keluar program.            X")
            print("X" * 65 + "\n")
            _finished_banner_shown = True
        if check_keyboard_reset():
            reset_all()

    # --- LOG FRAME ---
    disp_j = curr_jarak if curr_jarak > 0 else last_valid_jarak
    enc_l, enc_r = esp32.get_encoder_ticks() if esp32 else (0, 0)
    st_name = STATE_NAMES.get(state, "UNKNOWN")
    tgt_lbl = "T1" if state in (STATE_ALIGN_T1, STATE_APPROACH_T1, STATE_WAIT_HANDSDOWN_T1) else (
              "T2" if state in (STATE_ALIGN_T2, STATE_APPROACH_T2, STATE_WAIT_HANDSDOWN_T2) else "-")
    logger.log_frame(st_name, tgt_lbl, disp_j, curr_sudut, current_cmd_j, current_cmd_s, enc_l, enc_r)

    # --- RENDER ---
    draw_hud(frame, state, curr_sudut, disp_j, start_distance,
             detected_persons, locked_t1, locked_t2,
             triangle_info, dual_prog, handsdown_prog, enc_l, enc_r, logger.trial_number)

    cv2.imshow("Multi-Target Navigasi v3 (KNN-2800 | Semua Variasi Gesture)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        reset_all()

# --- CLEANUP ---
color_stream.stop()
depth_stream.stop()
if esp32:
    esp32.close()
openni2.unload()
cv2.destroyAllWindows()

print("\n" + "=" * 65)
print(">> MENYIMPAN HASIL REKAMAN & MEMBUAT GRAFIK PNG...")
logger.finalize()
print("=" * 65)
print("\n>> Program ditutup dan koneksi hardware dilepas dengan aman.")
print(f">> Hasil tersimpan di: {logger.sess_dir}")

