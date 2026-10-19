# =============================================================
# TEST STATE-LOCK & NAVIGASI MAJU (SINGLE TARGET - FINISH 0.7M)
# Fokus: State Lock Titik Tengah -> Terminal Koordinat -> Maju 0.7m
# =============================================================
import warnings, os
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2, time, mediapipe as mp, numpy as np, joblib, threading
from openni import openni2
from Esp32.send_to_esp32 import ESP32Controller

# ─── KONSTANTA SISTEM ─────────────────────────────────────────
ANGLE_THRESHOLD     = 1.0   # Toleransi sudut titik tengah sangat ketat (±1.0 deg)
COARSE_PULSE_DUR    = 0.12  # Durasi denyut putar kasar (sudut besar, 120ms)
COARSE_PAUSE_DUR    = 0.50  # Jeda setelah putar kasar agar kamera stabil (500ms)
FINE_PULSE_DURATION = 0.08  # Durasi denyut mikro putar halus (sudut kecil, 80ms)
FINE_PAUSE_DURATION = 0.40  # Jeda setelah denyut mikro agar kamera stabil (400ms)
LOCK_CONFIRM_FRAMES = 5     # Jumlah frame berturut-turut stabil di tengah sebelum kunci
STOP_DISTANCE       = 0.7   # Jarak finish aman (meter)
SMOOTH_WINDOW       = 7     # Jumlah frame moving average sudut
REQUIRED_HOLD       = 5.0   # Waktu tahan gestur Hands Up (detik)
STREAM_INTERVAL     = 0.2   # Interval pengiriman data ke ESP32 (detik)

STATE_IDLE        = 0     # Menunggu gestur Hands Up 5 detik
STATE_ALIGNING    = 1     # Menyelaraskan sudut kamera ke titik tengah target
STATE_APPROACHING = 2     # Target terkunci: Robot maju lurus ke finish point 0.7m
STATE_FINISHED    = 3     # Finish point tercapai (<= 0.7m) -> Stop aman

# ─── INISIALISASI HARDWARE & MODEL ────────────────────────────
print("="*55)
print("   SISTEM NAVIGASI: TITIK TENGAH -> LOCK -> MAJU 0.7M")
print("="*55)

esp32 = ESP32Controller(port='COM5', baudrate=115200)

try:
    knn = joblib.load('model_knn_gesture.pkl')
except FileNotFoundError:
    knn = joblib.load('Skenario Multi Target/model_knn_gesture.pkl')

mp_pose = mp.solutions.pose
pose_det = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw  = mp.solutions.drawing_utils

# Inisialisasi Sensor Kinect (Color + Depth)
print("Menginisialisasi sensor Kinect v1...")
openni2.initialize()
dev = openni2.Device.open_any()

color_stream = dev.create_color_stream()
color_stream.start()

depth_stream = dev.create_depth_stream()
depth_stream.start()

print(">> Kamera & Depth Sensor aktif.")
print(">> Lakukan gestur Hands Up dan tahan 5 detik.")
print(">> Shortcut: Tekan 'r' untuk Reset, 'q' untuk Keluar.\n")

# ─── FUNGSI BANTU ─────────────────────────────────────────────
def get_depth_in_meters(depth_frame, px, py, frame_w=640, frame_h=480):
    """Membaca jarak kedalaman (meter) dengan filter patch 7x7 agar stabil"""
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
    """Mengirim data secara non-blocking ke ESP32"""
    if esp32:
        esp32.send_data(j1, s1, 0.0, 0.0)

# ─── HUD MINIMALIS & BERSIH (TIDAK MEMENUHI KAMERA) ───────────
def draw_clean_hud(frame, sudut, jarak, start_dist, has_target, target_pt, state, hold_prog):
    fh, fw = frame.shape[:2]
    cx = fw // 2

    # 1. Garis tengah frame halus (referensi vertikal tipis)
    cv2.line(frame, (cx, 32), (cx, fh - 48), (80, 80, 100), 1)

    # 2. Header Status Minimalis (Top Bar Ramping)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 6), (fw - 10, 36), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (10, 6), (fw - 10, 36), (50, 50, 65), 1)

    if state == STATE_IDLE:
        status_txt = "IDLE - Tahan gestur Hands Up 5 detik"
        scolor = (0, 215, 255)
    elif state == STATE_ALIGNING:
        target_side = "KIRI" if sudut > 0 else "KANAN"
        status_txt = f"ALIGNING - Menyelaraskan Sudut ({target_side}: {abs(sudut):.1f} deg)"
        scolor = (255, 175, 20)
    elif state == STATE_APPROACHING:
        status_txt = f"MAJU KE TARGET - Menuju Jarak Aman {STOP_DISTANCE:.1f}m"
        scolor = (50, 230, 100)
    else:  # STATE_FINISHED
        status_txt = f"FINISH POINT TERCAPAI - Robot Stop Aman ({STOP_DISTANCE:.1f}m)"
        scolor = (0, 255, 180)

    cv2.putText(frame, status_txt, (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, scolor, 1)
    cv2.putText(frame, "R: Reset | Q: Keluar", (fw - 175, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 170), 1)

    # Garis tipis progress tahan gestur di bawah top bar saat IDLE
    if state == STATE_IDLE and hold_prog > 0:
        bar_w = int((fw - 20) * min(1.0, hold_prog))
        cv2.rectangle(frame, (10, 34), (10 + bar_w, 36), (0, 215, 255), -1)

    # 3. Crosshair Target Minimalis
    if has_target and target_pt:
        tx, ty = target_pt
        t_color = (50, 230, 100) if state in (STATE_APPROACHING, STATE_FINISHED) else (0, 220, 255)
        # Lingkaran kecil & crosshair halus
        cv2.circle(frame, (tx, ty), 10, t_color, 2)
        cv2.line(frame, (tx - 15, ty), (tx + 15, ty), t_color, 1)
        cv2.line(frame, (tx, ty - 15), (tx, ty + 15), t_color, 1)

        # Label koordinat & jarak ringkas
        j_text = f"{jarak:.2f}m" if jarak > 0 else "--"
        info_target = f"({tx},{ty}) | {j_text}"
        cv2.putText(frame, info_target, (tx + 14, ty - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, t_color, 1)

    # 4. Progress Bar Jarak ke Finish (Bottom Bar Tipis & Ramping)
    # Muncul saat ALIGNING, APPROACHING, atau FINISHED
    if state in (STATE_ALIGNING, STATE_APPROACHING, STATE_FINISHED) and jarak > 0:
        by1, by2 = fh - 44, fh - 12
        bw_total = fw - 40
        bx1 = 20

        # Background transparan
        overlay_b = frame.copy()
        cv2.rectangle(overlay_b, (bx1, by1), (bx1 + bw_total, by2), (15, 15, 20), -1)
        cv2.addWeighted(overlay_b, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx1 + bw_total, by2), (50, 50, 65), 1)

        # Label Jarak Terbaca
        cv2.putText(frame, f"Jarak: {jarak:.2f} m", (bx1 + 12, by1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 240, 90), 2)

        # Kotak Progress Bar
        bar_x1 = bx1 + 130
        bar_x2 = bx1 + bw_total - 120
        bar_w  = max(10, bar_x2 - bar_x1)
        bar_y1 = by1 + 9
        bar_y2 = by2 - 9

        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (30, 30, 40), -1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (65, 65, 80), 1)

        # Hitung progres jarak terbaca menuju 0.7m
        base_start = max(start_dist if start_dist > STOP_DISTANCE else 3.0, 1.2)
        progress = max(0.0, min(1.0, (base_start - jarak) / max(0.01, (base_start - STOP_DISTANCE))))
        fill_w = int(progress * bar_w)

        fill_color = (50, 230, 100) if jarak <= STOP_DISTANCE else (30, 190, 240)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), fill_color, -1)

        # Persentase di dalam bar
        pct_txt = f"{int(progress * 100)}%"
        cv2.putText(frame, pct_txt, (bar_x1 + (bar_w // 2) - 12, by1 + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        # Label Finish Target di kanan bar
        finish_txt = f"Finish: {STOP_DISTANCE:.1f}m"
        f_color = (50, 230, 100) if jarak <= STOP_DISTANCE else (120, 140, 255)
        cv2.putText(frame, finish_txt, (bar_x2 + 10, by1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, f_color, 1)

# ─── VARIABEL STATE ───────────────────────────────────────────
state            = STATE_IDLE
gesture_start    = None
is_holding       = False
last_stream      = 0
last_terminal_log= 0
start_distance   = 0.0
last_valid_jarak   = 0.0
angle_history      = []
smooth_sudut       = 0.0
align_pulse_active = False
align_pulse_start  = 0.0
align_settle_count = 0

# ─── MAIN LOOP ────────────────────────────────────────────────
while True:
    # 1. Baca Frame Kamera & Depth
    c_frame = color_stream.read_frame()
    d_frame = depth_stream.read_frame()

    img   = np.frombuffer(c_frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    fh, fw = frame.shape[:2]
    cx_frame   = fw // 2
    current_t  = time.time()

    # 2. Deteksi Gestur & Titik Tengah (MediaPipe Pose)
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_res = pose_det.process(rgb)

    sudut        = 0.0
    has_target   = False
    target_point = None
    jarak_now    = 0.0

    if pose_res.pose_landmarks:
        # Gambar skeleton tipis
        mp_draw.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm  = pose_res.pose_landmarks.landmark
        ls, rs = lm[11], lm[12]
        lw, rw = lm[15], lm[16]

        pred = knn.predict(np.array([[lw.y - ls.y, rw.y - rs.y]]))[0]

        if pred == 1:
            # Koordinat titik tengah torso (antara bahu kiri dan bahu kanan)
            mid_x = int((ls.x + rs.x) / 2 * fw)
            mid_y = int((ls.y + rs.y) / 2 * fh)
            target_point = (mid_x, mid_y)
            has_target = True

            # Hitung sudut: (cx_frame - mid_x)
            # Target di KIRI  (mid_x < cx_frame) -> Sudut POSITIF (+) -> ESP32 Berputar KIRI (CCW)
            # Target di KANAN (mid_x > cx_frame) -> Sudut NEGATIF (-) -> ESP32 Berputar KANAN (CW)
            sudut_raw = (cx_frame - mid_x) * (57.5 / fw)
            angle_history.append(sudut_raw)
            if len(angle_history) > SMOOTH_WINDOW:
                angle_history.pop(0)
            smooth_sudut = round(sum(angle_history) / len(angle_history), 1)
            sudut = smooth_sudut

            # Baca jarak dari depth sensor
            d_val = get_depth_in_meters(d_frame, mid_x, mid_y, fw, fh)
            if d_val > 0:
                last_valid_jarak = d_val
            jarak_now = last_valid_jarak

    # Reset angle history jika target tidak tampak saat IDLE
    if not has_target and state == STATE_IDLE:
        angle_history.clear()
        smooth_sudut = 0.0

    hold_progress = 0.0

    # ── 3. STATE MACHINE NAVIGASI ──────────────────────────────

    # [STATE 0: IDLE] Menunggu Gestur Hands Up 5 Detik
    if state == STATE_IDLE:
        if has_target:
            if not is_holding:
                gesture_start = current_t
                is_holding    = True
            elapsed = current_t - gesture_start
            hold_progress = min(1.0, elapsed / REQUIRED_HOLD)

            if elapsed >= REQUIRED_HOLD:
                start_distance = jarak_now if jarak_now > 0 else 2.5
                print("\n" + "="*55)
                print(f">> [GESTUR TERKONFIRMASI 5 DETIK]")
                print(f"   Target di Sudut Awal : {sudut:+.1f} deg | Jarak Awal: {start_distance:.2f} m")
                print(f"   Robot Masuk Mode ALIGNING (Menyelaraskan Sudut)...")
                print("="*55)

                is_holding         = False
                last_stream        = current_t
                align_pulse_active = False
                align_pulse_start  = current_t
                align_settle_count = 0
                state              = STATE_ALIGNING
        else:
            if is_holding:
                is_holding = False

    # [STATE 1: ALIGNING] Menyelaraskan Sudut ke Titik Tengah (Pulse-Pause)
    elif state == STATE_ALIGNING:
        if has_target:
            abs_s = abs(sudut)
            in_coarse_zone = abs_s > 4.5
            in_fine_zone   = ANGLE_THRESHOLD < abs_s <= 4.5

            if in_coarse_zone or in_fine_zone:
                # Target belum di tengah -> reset counter kestabilan
                align_settle_count = 0
                pulse_dur = COARSE_PULSE_DUR if in_coarse_zone else FINE_PULSE_DURATION
                pause_dur = COARSE_PAUSE_DUR if in_coarse_zone else FINE_PAUSE_DURATION
                cycle_dur = pulse_dur + pause_dur
                t_in_cycle = (current_t - align_pulse_start) % cycle_dur

                if t_in_cycle < pulse_dur:
                    # ── FASE DENYUT: Motor berputar sesaat ──
                    if not align_pulse_active:
                        side_txt = "KIRI" if sudut > 0 else "KANAN"
                        putar_ke = "KIRI" if sudut > 0 else "KANAN"
                        if in_coarse_zone:
                            cmd_sudut = sudut
                            lbl = "COARSE"
                        else:
                            # Halus: amplifikasi ke ±5° agar menembus deadband ESP32 (±3°)
                            cmd_sudut = 5.0 if sudut > 0 else -5.0
                            lbl = "FINE"
                        print(f">> [ALIGN {lbl}] Sudut: {sudut:+.1f} deg ({side_txt}) -> Denyut {putar_ke} ({int(pulse_dur*1000)}ms)")
                        t = threading.Thread(target=send_async, args=(1.0, cmd_sudut))
                        t.daemon = True; t.start()
                        align_pulse_active = True
                        last_stream = current_t
                else:
                    # ── FASE JEDA: Motor diam, kamera baca frame stabil ──
                    if align_pulse_active:
                        t = threading.Thread(target=send_async, args=(0.0, 0.0))
                        t.daemon = True; t.start()
                        align_pulse_active = False
                        last_stream = current_t
                        print(f">> [ALIGN BERHENTI] Sudut setelah denyut: {sudut:+.1f} deg | Jeda kamera {int(pause_dur*1000)}ms...")
            else:
                # ── TARGET DI TITIK TENGAH (abs_s <= 1.0°) -> STABILISASI & KONFIRMASI KUNCI ──
                if align_pulse_active:
                    t = threading.Thread(target=send_async, args=(0.0, 0.0))
                    t.daemon = True; t.start()
                    align_pulse_active = False
                    last_stream = current_t
                    align_settle_count = 0

                # Robot diam, jaga agar ESP32 tetap stop
                if current_t - last_stream >= STREAM_INTERVAL:
                    t = threading.Thread(target=send_async, args=(0.0, 0.0))
                    t.daemon = True; t.start()
                    last_stream = current_t

                align_settle_count += 1
                print(f">> [ALIGN CENTER] Sudut: {sudut:+.1f} deg | Stabil: {align_settle_count}/{LOCK_CONFIRM_FRAMES} frame")

                if align_settle_count >= LOCK_CONFIRM_FRAMES:
                    # ── STATE LOCK TERCAPAI ──
                    start_distance = jarak_now if jarak_now > 0 else start_distance
                    print("\n" + "="*58)
                    print(">> [TARGET TERKUNCI PRESISI DI TITIK TENGAH!]")
                    print(f"   Koordinat Titik Tengah Target : (X: {target_point[0]}, Y: {target_point[1]})")
                    print(f"   Sudut Terpusat                : {sudut:+.1f} deg (toleransi +/-{ANGLE_THRESHOLD} deg, {LOCK_CONFIRM_FRAMES} frame stabil)")
                    print(f"   Jarak Terbaca Awal            : {start_distance:.2f} m")
                    print(f"   Batas Finish Aman Robot       : {STOP_DISTANCE:.1f} m")
                    print(f"   Status: Robot MULAI BERGERAK MAJU LURUS!")
                    print("="*58 + "\n")

                    align_settle_count = 0
                    align_pulse_active = False
                    if start_distance <= STOP_DISTANCE:
                        t = threading.Thread(target=send_async, args=(0.0, 0.0))
                        t.daemon = True; t.start()
                        state = STATE_FINISHED
                    else:
                        state = STATE_APPROACHING
                        t = threading.Thread(target=send_async, args=(start_distance, 0.0))
                        t.daemon = True; t.start()
        else:
            # Target hilang saat aligning
            print(">> [HILANG] Target tidak terdeteksi saat ALIGNING -> STOP -> kembali IDLE")
            t = threading.Thread(target=send_async, args=(0.0, 0.0))
            t.daemon = True; t.start()
            align_pulse_active = False
            align_settle_count = 0
            state = STATE_IDLE

    # [STATE 2: APPROACHING] Maju Lurus Menuju Finish Point 0.7m
    elif state == STATE_APPROACHING:
        current_j = jarak_now if jarak_now > 0 else last_valid_jarak

        if current_t - last_stream >= STREAM_INTERVAL:
            # Evaluasi jarak aman finish
            if current_j > 0 and current_j <= STOP_DISTANCE:
                # ── FINISH POINT TERCAPAI: STOP AMAN ──
                print("\n" + "="*58)
                print(f">> [FINISH POINT TERCAPAI!]")
                print(f"   Jarak Akhir Terbaca : {current_j:.2f} m (<= {STOP_DISTANCE:.1f} m)")
                print(f"   Perintah ke ESP32   : STOP TOTAL (0.0, 0.0)")
                print(f"   Status             : MISI SELESAI & ROBOT BERHENTI AMAN.")
                print(f"   -> Tekan 'r' di window untuk Reset / Uji Ulang")
                print(f"   -> Tekan 'q' untuk Keluar")
                print("="*58 + "\n")

                t = threading.Thread(target=send_async, args=(0.0, 0.0))
                t.daemon = True; t.start()
                state = STATE_FINISHED

            else:
                # Masih berjalan maju menuju target
                t = threading.Thread(target=send_async, args=(current_j, 0.0))
                t.daemon = True; t.start()

                # Cetak progres di terminal secara berkala setiap 1 detik
                if current_t - last_terminal_log >= 1.0:
                    sisa = max(0.0, current_j - STOP_DISTANCE)
                    print(f">> [MAJU LURUS] Jarak: {current_j:.2f} m  |  Sisa ke Finish: {sisa:.2f} m")
                    last_terminal_log = current_t

            last_stream = current_t

    # [STATE 3: FINISHED] Misi Selesai (Robot Stop Total)
    elif state == STATE_FINISHED:
        # Menjaga ESP32 tetap diam
        pass

    # 4. Gambar Tampilan HUD Bersih & Ramping
    disp_jarak = jarak_now if jarak_now > 0 else last_valid_jarak
    draw_clean_hud(frame, sudut, disp_jarak, start_distance, has_target, target_point, state, hold_progress)

    cv2.imshow("Navigasi Titik Tengah - Finish 0.7m", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Shortcut Reset untuk pengujian ulang tanpa restart script
        state              = STATE_IDLE
        is_holding         = False
        align_pulse_active = False
        align_settle_count = 0
        angle_history.clear()
        smooth_sudut       = 0.0
        start_distance     = 0.0
        send_async(0.0, 0.0)
        print("\n>> [RESET] Sistem dikembalikan ke STATE_IDLE. Lakukan gestur untuk uji ulang.\n")

# ─── CLEANUP ──────────────────────────────────────────────────
color_stream.stop()
depth_stream.stop()
if esp32:
    esp32.close()
openni2.unload()
cv2.destroyAllWindows()
