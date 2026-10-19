# =============================================================
# TEST MULTI-TARGET NAVIGASI OTOMATIS (SKENARIO LENGKAP)
# Alur:
# 1. Seleksi Target Terdekat (Nearest Target First) via YOLO + MP Pose
# 2. State-Lock Titik Tengah Target 1 -> Maju Lurus -> Stop di 0.7m
# 3. Jeda 3 detik (Target 1 menurunkan tangan)
# 4. Robot berputar searah jarum jam (CW) memindai ruangan mencari Target 2
# 5. Lock Target 2 -> Maju Lurus -> Stop di 0.7m -> Misi Selesai Total
# =============================================================
import warnings, os
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2, time, csv, mediapipe as mp, numpy as np, joblib, threading
from openni import openni2
from ultralytics import YOLO
from Esp32.send_to_esp32 import ESP32Controller

# ─── KONSTANTA SISTEM ─────────────────────────────────────────
ANGLE_THRESHOLD     = 2.0   # Toleransi sudut titik tengah presisi & stabil (±2.0 deg)
COARSE_PULSE_DUR    = 0.12  # Durasi denyut putar kasar (sudut besar, 120ms)
COARSE_PAUSE_DUR    = 0.40  # Jeda setelah putar kasar agar kamera stabil (400ms)
FINE_PULSE_DURATION = 0.08  # Durasi denyut mikro putar halus (sudut kecil, 80ms)
FINE_PAUSE_DURATION = 0.35  # Jeda setelah denyut mikro agar kamera stabil (350ms)
LOCK_CONFIRM_FRAMES = 2     # Konfirmasi stabil 2 frame berturut-turut sebelum mengunci & maju
STOP_DISTANCE       = 0.7   # Jarak finish aman (meter)
SMOOTH_WINDOW       = 5     # Moving average responsif & bersih
REQUIRED_HOLD       = 4.0   # Waktu tahan gestur untuk konfirmasi (detik)
STREAM_INTERVAL     = 0.2   # Interval komunikasi ESP32 (detik)
WAIT_AFTER_FINISH   = 3.0   # Jeda setelah finish target 1 (detik)
SCAN_ROTATION_ANGLE = -10.0 # Sudut perintah memutar robot CW (searah jarum jam / KANAN)
SCAN_STEP_PULSE     = 0.12  # Durasi motor berputar pendek per langkah (detik)
SCAN_STEP_PAUSE     = 0.28  # Durasi jeda diam per langkah agar kamera stabil & jernih (detik)

# Definisi State Machine
STATE_IDLE_SELECT   = 0     # Memindai semua orang & memilih target terdekat (Tahan 4s)
STATE_ALIGNING      = 1     # Menyelaraskan sudut kamera ke titik tengah target aktif
STATE_APPROACHING   = 2     # Maju lurus menuju target aktif hingga 0.7m
STATE_WAIT_NEXT     = 3     # Stop di target aktif, jeda 3s agar target turunkan tangan
STATE_SCANNING_NEXT = 4     # Berputar searah jarum jam mencari target berikutnya
STATE_ALL_FINISHED  = 5     # Seluruh target selesai (Misi Selesai Total)

# ─── INISIALISASI MODEL & HARDWARE ────────────────────────────
print("="*65)
print("   SISTEM NAVIGASI MULTI-TARGET (YOLOv8 + MP POSE + ESP32)")
print("   Alur: Seleksi Terdekat -> Target 1 -> Putar CW -> Target 2")
print("="*65)

esp32 = ESP32Controller(port='COM5', baudrate=115200)

print("Memuat model YOLOv8n dan Model KNN Gesture...")
yolo_model = YOLO('yolov8n.pt')

try:
    knn_model = joblib.load('model_knn_gesture.pkl')
except FileNotFoundError:
    knn_model = joblib.load('Skenario Multi Target/model_knn_gesture.pkl')

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Inisialisasi Sensor Kinect (Color + Depth)
print("Menghubungkan ke sensor Kinect v1...")
openni2.initialize()
dev = openni2.Device.open_any()

color_stream = dev.create_color_stream()
color_stream.start()

depth_stream = dev.create_depth_stream()
depth_stream.start()

print(">> Kamera & Depth Sensor aktif!")
print(">> Shortcut: Tekan 'r' untuk Reset Misi, 'q' untuk Keluar.\n")

# ─── FUNGSI BANTU ─────────────────────────────────────────────
def get_depth_in_meters(depth_frame, px, py, frame_w=640, frame_h=480):
    """Membaca jarak depth dengan median patch 7x7 agar stabil"""
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
    """Kirim data serial ke ESP32 secara non-blocking"""
    if esp32:
        esp32.send_data(j1, s1, 0.0, 0.0)

# ─── HUD MINIMALIS & BERSIH (LEGA) ────────────────────────────
def draw_hud(frame, state, active_target_num, current_sudut, current_jarak, start_dist, detected_persons, selected_target, hold_prog):
    fh, fw = frame.shape[:2]
    cx = fw // 2

    # 1. Garis tengah frame vertikal tipis
    cv2.line(frame, (cx, 34), (cx, fh - 48), (70, 70, 85), 1)

    # 2. Top Banner Status Bar (Tipis 30px)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 6), (fw - 10, 36), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (10, 6), (fw - 10, 36), (50, 50, 65), 1)

    if state == STATE_IDLE_SELECT:
        status_txt = f"[FASE 1] SELEKSI TARGET AWAL: Tahan gestur Hands Up ({int(hold_prog*REQUIRED_HOLD)}s/{int(REQUIRED_HOLD)}s)"
        scolor = (0, 215, 255)
    elif state == STATE_ALIGNING:
        target_side = "KIRI" if current_sudut > 0 else "KANAN"
        if abs(current_sudut) <= ANGLE_THRESHOLD:
            status_txt = f"[TARGET {active_target_num}] TERPUSAT ({current_sudut:+.1f} deg) -> MENGUNCI ARAH LURUS..."
            scolor = (50, 230, 100)
        elif abs(current_sudut) <= 4.5:
            status_txt = f"[TARGET {active_target_num}] PENYELARASAN HALUS (Nudge {target_side}: {abs(current_sudut):.1f} deg)"
            scolor = (255, 200, 30)
        else:
            status_txt = f"[TARGET {active_target_num}] PENYELARASAN KASAR ({target_side}: {abs(current_sudut):.1f} deg)"
            scolor = (255, 175, 20)
    elif state == STATE_APPROACHING:
        status_txt = f"[TARGET {active_target_num}] MAJU KE TARGET (Menuju Jarak Aman {STOP_DISTANCE:.1f}m)"
        scolor = (50, 230, 100)
    elif state == STATE_WAIT_NEXT:
        status_txt = f"[TARGET 1 SELESAI] Target 1 turunkan tangan... Bersiap memindai Target 2"
        scolor = (0, 255, 220)
    elif state == STATE_SCANNING_NEXT:
        status_txt = f"[SCANNING] Robot berputar searah jarum jam (CW) mencari Target 2..."
        scolor = (255, 140, 0)
    else: # STATE_ALL_FINISHED
        status_txt = f"[MISI SELESAI TOTAL] Seluruh target berhasil diselesaikan! [STOP]"
        scolor = (50, 255, 150)

    cv2.putText(frame, status_txt, (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, scolor, 1)
    cv2.putText(frame, "R: Reset | Q: Keluar", (fw - 165, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 170), 1)

    # Progress bar tahan gestur saat IDLE
    if state == STATE_IDLE_SELECT and hold_prog > 0:
        bar_w = int((fw - 20) * min(1.0, hold_prog))
        cv2.rectangle(frame, (10, 34), (10 + bar_w, 36), (0, 215, 255), -1)

    # 3. Gambar Bounding Box Orang
    for p in detected_persons:
        x1, y1, x2, y2 = p['box']
        mx, my = p['mid']
        j = p['jarak']
        s = p['sudut']
        pid = p['id']

        if selected_target and p['id'] == selected_target['id']:
            # TARGET AKTIF TERPILIH (Emas / Hijau Terang Tebal)
            b_color = (0, 230, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), b_color, 2)
            cv2.circle(frame, (mx, my), 10, b_color, 2)
            cv2.line(frame, (mx - 15, my), (mx + 15, my), b_color, 2)
            cv2.line(frame, (mx, my - 15), (mx, my + 15), b_color, 2)

            t_tag = f"TARGET {active_target_num} [AKTIF]: {j:.2f}m | {s:+.1f}d"
            cv2.putText(frame, t_tag, (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, b_color, 2)

        elif p['is_gesture']:
            # Target lain yang bergestur (Biru / Cyan)
            b_color = (255, 180, 50)
            cv2.rectangle(frame, (x1, y1), (x2, y2), b_color, 1)
            cv2.circle(frame, (mx, my), 6, b_color, -1)
            cv2.putText(frame, f"Kandidat: {j:.2f}m", (x1, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, b_color, 1)

        else:
            # Person biasa (Abu-abu tipis)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
            cv2.putText(frame, f"P#{pid}: {j:.2f}m", (x1, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    # 4. Bottom Distance Progress Bar (Hanya saat mendekati target)
    if state in (STATE_ALIGNING, STATE_APPROACHING) and current_jarak > 0:
        by1, by2 = fh - 44, fh - 12
        bw_total = fw - 40
        bx1 = 20

        overlay_b = frame.copy()
        cv2.rectangle(overlay_b, (bx1, by1), (bx1 + bw_total, by2), (15, 15, 20), -1)
        cv2.addWeighted(overlay_b, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx1 + bw_total, by2), (50, 50, 65), 1)

        cv2.putText(frame, f"Jarak: {current_jarak:.2f} m", (bx1 + 12, by1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 240, 90), 2)

        bar_x1 = bx1 + 130
        bar_x2 = bx1 + bw_total - 120
        bar_w  = max(10, bar_x2 - bar_x1)
        bar_y1 = by1 + 9
        bar_y2 = by2 - 9

        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (30, 30, 40), -1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (65, 65, 80), 1)

        base_start = max(start_dist if start_dist > STOP_DISTANCE else 3.0, 1.2)
        progress = max(0.0, min(1.0, (base_start - current_jarak) / max(0.01, (base_start - STOP_DISTANCE))))
        fill_w = int(progress * bar_w)

        fill_color = (50, 230, 100) if current_jarak <= STOP_DISTANCE else (30, 190, 240)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), fill_color, -1)

        pct_txt = f"{int(progress * 100)}%"
        cv2.putText(frame, pct_txt, (bar_x1 + (bar_w // 2) - 12, by1 + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        finish_txt = f"Finish: {STOP_DISTANCE:.1f}m"
        f_color = (50, 230, 100) if current_jarak <= STOP_DISTANCE else (120, 140, 255)
        cv2.putText(frame, finish_txt, (bar_x2 + 10, by1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, f_color, 1)

# ─── VARIABEL RUNTIME STATE MACHINE ───────────────────────────
state              = STATE_IDLE_SELECT
active_target_num  = 1
gesture_hold_start = None
is_holding_gesture = False
last_stream_time   = 0
last_terminal_log  = 0
wait_state_start   = 0
start_distance     = 0.0
last_valid_jarak   = 0.0
angle_history      = []
smooth_sudut       = 0.0
scan_start_time    = 0
scan_pulse_active  = False
align_pulse_active   = False
align_pulse_start    = 0
align_settle_count   = 0
target2_detect_count = 0

# ─── MAIN LOOP ────────────────────────────────────────────────
while True:
    c_frame = color_stream.read_frame()
    d_frame = depth_stream.read_frame()

    img   = np.frombuffer(c_frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    fh, fw = frame.shape[:2]
    cx_frame   = fw // 2
    current_t  = time.time()

    # 1. YOLOv8 Deteksi Person
    yolo_res = yolo_model(frame, verbose=False, classes=[0], conf=0.5)

    detected_persons = []
    person_idx = 1

    for r in yolo_res:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, min(x1, fw - 1))
            y1 = max(0, min(y1, fh - 1))
            x2 = max(0, min(x2, fw))
            y2 = max(0, min(y2, fh))

            box_w = x2 - x1
            box_h = y2 - y1

            if box_w < 35 or box_h < 70:
                continue

            # 2. Crop area person untuk dianalisis MediaPipe Pose
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pose_res = pose_detector.process(crop_rgb)

            is_hands_up = False
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            if pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks.landmark
                ls = lm[11]
                rs = lm[12]
                lw = lm[15]
                rw = lm[16]

                l_diff = lw.y - ls.y
                r_diff = rw.y - rs.y

                pred = knn_model.predict(np.array([[l_diff, r_diff]]))[0]
                if pred == 1:
                    is_hands_up = True

                mid_x = int(x1 + ((ls.x + rs.x) / 2) * box_w)
                mid_y = int(y1 + ((ls.y + rs.y) / 2) * box_h)
                mid_x = max(x1, min(mid_x, x2))
                mid_y = max(y1, min(mid_y, y2))

            # 3. Jarak & Sudut
            jarak = get_depth_in_meters(d_frame, mid_x, mid_y, fw, fh)
            if jarak <= 0:
                jarak = get_depth_in_meters(d_frame, (x1 + x2) // 2, (y1 + y2) // 2, fw, fh)

            # Hubungan sudut kamera vs kemudi ESP32:
            # mid_x < cx_frame (Target di KIRI)  -> Sudut POSITIF (+) -> ESP32 Putar KIRI (CCW)
            # mid_x > cx_frame (Target di KANAN) -> Sudut NEGATIF (-) -> ESP32 Putar KANAN (CW)
            sudut = round((cx_frame - mid_x) * (57.5 / fw), 1)

            detected_persons.append({
                'id': person_idx,
                'box': (x1, y1, x2, y2),
                'mid': (mid_x, mid_y),
                'jarak': jarak,
                'sudut': sudut,
                'is_gesture': is_hands_up
            })
            person_idx += 1

    # ── 4. LOGIKA SELEKSI TARGET AKTIF ────────────────────────────
    # Cari kandidat target yang mengangkat tangan
    # Jika sedang memproses Target 2, selalu abaikan objek dekat (<= 0.85m) karena itu adalah Target 1
    min_dist_filter = 0.85 if active_target_num == 2 else 0.0
    valid_persons = [p for p in detected_persons if p['jarak'] > min_dist_filter]
    gesture_candidates = [p for p in valid_persons if p['is_gesture']]

    selected_target = None
    if len(gesture_candidates) > 0:
        # Prioritas: Person dengan gestur Hands Up terdekat
        gesture_candidates.sort(key=lambda p: p['jarak'])
        selected_target = gesture_candidates[0]
    elif state in (STATE_ALIGNING, STATE_APPROACHING) and len(valid_persons) > 0:
        # Saat fase ALIGNING atau APPROACHING: jika tangan target turun sejenak,
        # tetap pertahankan target terdekat agar state lock & maju tidak terputus!
        valid_persons.sort(key=lambda p: p['jarak'])
        selected_target = valid_persons[0]

    # Ambil sudut dan jarak target aktif saat ini
    curr_sudut = 0.0
    curr_jarak = 0.0

    if selected_target:
        # Smoothing sudut target terpilih
        angle_history.append(selected_target['sudut'])
        if len(angle_history) > SMOOTH_WINDOW:
            angle_history.pop(0)
        smooth_sudut = round(sum(angle_history) / len(angle_history), 1)
        curr_sudut   = smooth_sudut

        if selected_target['jarak'] > 0:
            last_valid_jarak = selected_target['jarak']
        curr_jarak = last_valid_jarak

    # Reset riwayat sudut jika target tidak terdeteksi saat IDLE atau SCANNING
    if not selected_target and state in (STATE_IDLE_SELECT, STATE_SCANNING_NEXT):
        angle_history.clear()
        smooth_sudut = 0.0

    hold_prog = 0.0

    # ── 5. STATE MACHINE MULTI-TARGET ─────────────────────────────

    # [FASE 1: IDLE_SELECT] Menunggu gestur & memilih target terdekat (Tahan 4 detik)
    if state == STATE_IDLE_SELECT:
        if selected_target:
            if not is_holding_gesture:
                gesture_hold_start = current_t
                is_holding_gesture = True
            elapsed = current_t - gesture_hold_start
            hold_prog = min(1.0, elapsed / REQUIRED_HOLD)

            if elapsed >= REQUIRED_HOLD:
                start_distance = curr_jarak if curr_jarak > 0 else 2.5
                print("\n" + "="*60)
                print(f">> [TARGET 1 TERKONFIRMASI!]")
                print(f"   Target Terpilih      : Person #{selected_target['id']} (Terdekat)")
                print(f"   Jarak Awal           : {start_distance:.2f} m")
                print(f"   Sudut Awal           : {curr_sudut:+.1f} deg")
                print(f"   Robot Mulai Penyelarasan Sudut (State-Lock)...")
                print("="*60 + "\n")

                is_holding_gesture = False
                last_stream_time   = current_t
                align_pulse_start  = current_t
                align_pulse_active = False
                align_settle_count = 0
                state              = STATE_ALIGNING
        else:
            if is_holding_gesture:
                is_holding_gesture = False

    # [FASE 2: ALIGNING] Penyelarasan Presisi: Pulse-Pause Anti-Overshoot
    # Seluruh gerakan (kasar & halus) memakai siklus: denyut pendek -> berhenti -> baca ulang
    # Mencegah robot overshooting dan memastikan kamera membaca posisi setelah motor diam
    elif state == STATE_ALIGNING:
        if selected_target:
            abs_s = abs(curr_sudut)

            # Semua putaran, baik kasar maupun halus, hanya boleh dilakukan saat bukan
            # sedang dalam jeda diam (align_pulse_active == False & waktu jeda belum habis).
            # Fase JEDA: motor diam, kamera baca sudut segar.
            in_coarse_zone = abs_s > 4.5
            in_fine_zone   = ANGLE_THRESHOLD < abs_s <= 4.5
            in_center_zone = abs_s <= ANGLE_THRESHOLD

            if in_coarse_zone or in_fine_zone:
                # Target belum di tengah -> reset penghitung konfirmasi
                align_settle_count = 0

                pulse_dur = COARSE_PULSE_DUR if in_coarse_zone else FINE_PULSE_DURATION
                pause_dur = COARSE_PAUSE_DUR if in_coarse_zone else FINE_PAUSE_DURATION
                cycle_dur = pulse_dur + pause_dur
                t_in_cycle = (current_t - align_pulse_start) % cycle_dur

                if t_in_cycle < pulse_dur:
                    # ── FASE DENYUT: Motor berputar sesaat ──
                    if not align_pulse_active:
                        side_txt = "KIRI" if curr_sudut > 0 else "KANAN"
                        putar_ke = "KIRI" if curr_sudut > 0 else "KANAN"
                        if in_coarse_zone:
                            # Kasar: kirim sudut aktual langsung ke ESP32
                            cmd_sudut = curr_sudut
                            lbl = "COARSE"
                        else:
                            # Halus: amplifikasi ke ±5° agar menembus deadband ESP32 (±3°)
                            cmd_sudut = 5.0 if curr_sudut > 0 else -5.0
                            lbl = "FINE"
                        print(f">> [ALIGN {lbl} T{active_target_num}] Sudut: {curr_sudut:+.1f} deg ({side_txt}) -> Denyut {putar_ke} ({int(pulse_dur*1000)}ms)")
                        t = threading.Thread(target=send_async, args=(1.0, cmd_sudut))
                        t.daemon = True; t.start()
                        align_pulse_active = True
                        last_stream_time = current_t
                else:
                    # ── FASE JEDA: Motor diam, tunggu kamera stabil ──
                    if align_pulse_active:
                        t = threading.Thread(target=send_async, args=(0.0, 0.0))
                        t.daemon = True; t.start()
                        align_pulse_active = False
                        last_stream_time = current_t
                        remaining = pause_dur - (t_in_cycle - pulse_dur)
                        print(f">> [ALIGN BERHENTI T{active_target_num}] Sudut setelah denyut: {curr_sudut:+.1f} deg | Jeda kamera {int(pause_dur*1000)}ms...")

            else:
                # ── TARGET DI TITIK TENGAH (abs_s <= 1.0°) -> STABILISASI & KONFIRMASI KUNCI ──
                if align_pulse_active:
                    # Pastikan motor diam sebelum mulai menghitung frame stabil
                    t = threading.Thread(target=send_async, args=(0.0, 0.0))
                    t.daemon = True; t.start()
                    align_pulse_active = False
                    last_stream_time = current_t
                    align_settle_count = 0  # Reset: mulai hitung dari posisi diam

                # Robot diam, terus kirim STOP untuk memastikan tidak bergerak
                if current_t - last_stream_time >= STREAM_INTERVAL:
                    t = threading.Thread(target=send_async, args=(0.0, 0.0))
                    t.daemon = True; t.start()
                    last_stream_time = current_t

                align_settle_count += 1
                print(f">> [ALIGN CENTER T{active_target_num}] Sudut: {curr_sudut:+.1f} deg | Stabil: {align_settle_count}/{LOCK_CONFIRM_FRAMES} frame")

                if align_settle_count >= LOCK_CONFIRM_FRAMES:
                    # ── STATE LOCK TERCAPAI: Terkonfirmasi stabil selama N frame ──
                    start_distance = curr_jarak if curr_jarak > 0 else start_distance
                    print("\n" + "="*60)
                    print(f">> [STATE LOCK: TARGET {active_target_num} TERKUNCI PRESISI DI TITIK TENGAH!]")
                    print(f"   Koordinat Titik Tengah Target : (X: {selected_target['mid'][0]}, Y: {selected_target['mid'][1]})")
                    print(f"   Sudut Terpusat                : {curr_sudut:+.1f} deg (toleransi +/-{ANGLE_THRESHOLD} deg, {LOCK_CONFIRM_FRAMES} frame stabil)")
                    print(f"   Jarak Terbaca Awal            : {start_distance:.2f} m")
                    print(f"   Batas Finish Aman Robot       : {STOP_DISTANCE:.1f} m")
                    print(f"   Status: Robot MULAI BERGERAK MAJU LURUS menuju Finish Point ({STOP_DISTANCE:.1f}m)...")
                    print("="*60 + "\n")

                    align_settle_count = 0
                    align_pulse_active = False
                    if start_distance <= STOP_DISTANCE:
                        t = threading.Thread(target=send_async, args=(0.0, 0.0))
                        t.daemon = True; t.start()
                        state = STATE_WAIT_NEXT if active_target_num == 1 else STATE_ALL_FINISHED
                        wait_state_start = current_t
                    else:
                        state = STATE_APPROACHING
                        t = threading.Thread(target=send_async, args=(start_distance, 0.0))
                        t.daemon = True; t.start()
        else:
            # Target hilang saat aligning -> Stop & reset counter
            align_settle_count = 0
            if align_pulse_active or (current_t - last_stream_time >= STREAM_INTERVAL):
                t = threading.Thread(target=send_async, args=(0.0, 0.0))
                t.daemon = True; t.start()
                align_pulse_active = False
                last_stream_time = current_t

    # [FASE 3: APPROACHING] Maju Lurus Menuju Target Aktif Hingga 0.7m
    elif state == STATE_APPROACHING:
        if current_t - last_stream_time >= STREAM_INTERVAL:
            c_dist = curr_jarak if curr_jarak > 0 else last_valid_jarak

            if c_dist > 0 and c_dist <= STOP_DISTANCE:
                # ── FINISH POINT TARGET AKTIF TERCAPAI ──
                print("\n" + "="*60)
                print(f">> [FINISH TARGET {active_target_num} TERCAPAI!]")
                print(f"   Jarak Akhir : {c_dist:.2f} m (<= {STOP_DISTANCE:.1f} m)")
                print(f"   Robot BERHENTI TOTAL di depan Target {active_target_num}.")
                print("="*60 + "\n")

                t = threading.Thread(target=send_async, args=(0.0, 0.0))
                t.daemon = True; t.start()

                if active_target_num == 1:
                    print(">> Menunggu jeda 3 detik: Target 1 mohon turunkan tangan...")
                    state = STATE_WAIT_NEXT
                    wait_state_start = current_t
                else:
                    print(">> SELURUH TARGET SELESAI! Misi Navigasi Multi-Target Berhasil.")
                    state = STATE_ALL_FINISHED

            else:
                # Terus maju lurus
                t = threading.Thread(target=send_async, args=(c_dist, 0.0))
                t.daemon = True; t.start()

                if current_t - last_terminal_log >= 1.0:
                    sisa = max(0.0, c_dist - STOP_DISTANCE)
                    print(f">> [TARGET {active_target_num} MAJU] Jarak: {c_dist:.2f}m | Sisa: {sisa:.2f}m")
                    last_terminal_log = current_t

            last_stream_time = current_t

    # [FASE 4: WAIT_NEXT] Jeda 3 Detik setelah Target 1 Selesai
    elif state == STATE_WAIT_NEXT:
        # Robot tetap diam
        send_async(0.0, 0.0)

        if current_t - wait_state_start >= WAIT_AFTER_FINISH:
            # Selesai jeda: Mulai scanning Target 2
            active_target_num = 2
            angle_history.clear()
            smooth_sudut = 0.0
            print("\n" + "="*60)
            print(">> [MEMULAI SCANNING TARGET 2]")
            print("   Robot mulai berputar perlahan searah jarum jam (Step-Scan CW)...")
            print("="*60 + "\n")
            scan_start_time   = current_t
            scan_pulse_active = False
            state = STATE_SCANNING_NEXT
            last_stream_time  = current_t

    # [FASE 5: SCANNING_NEXT] Berputar Perlahan Searah Jarum Jam Mencari Target 2
    elif state == STATE_SCANNING_NEXT:
        if selected_target:
            # ── TARGET 2 BERHASIL DITEMUKAN! ──
            print("\n" + "="*60)
            print(f">> [TARGET 2 DITEMUKAN!] ID #{selected_target['id']}")
            print(f"   Jarak: {selected_target['jarak']:.2f}m | Sudut: {selected_target['sudut']:+.1f} deg")
            print("   Robot BERHENTI BERPUTAR -> Masuk Penyelarasan Sudut Target 2...")
            print("="*60 + "\n")

            # Hentikan putaran segera
            t = threading.Thread(target=send_async, args=(0.0, 0.0))
            t.daemon = True; t.start()

            angle_history.clear()
            smooth_sudut       = 0.0
            start_distance     = selected_target['jarak']
            scan_pulse_active  = False
            align_pulse_start  = current_t
            align_pulse_active = False
            align_settle_count = 0
            state              = STATE_ALIGNING
            last_stream_time   = current_t

        else:
            # Belum menemukan Target 2: Lakukan rotasi pelan bertahap (Step-Scan)
            # Siklus: Pulsa putar pendek (120ms) -> Jeda diam (280ms) agar kamera stabil & tidak blur
            cycle_time = SCAN_STEP_PULSE + SCAN_STEP_PAUSE
            t_in_cycle = (current_t - scan_start_time) % cycle_time

            if t_in_cycle < SCAN_STEP_PULSE:
                # Fase Pulsa: Putar sedikit
                if not scan_pulse_active:
                    t = threading.Thread(target=send_async, args=(1.0, SCAN_ROTATION_ANGLE))
                    t.daemon = True; t.start()
                    scan_pulse_active = True
                    last_stream_time = current_t
            else:
                # Fase Jeda: Diam sejenak agar kamera mengambil frame jernih
                if scan_pulse_active:
                    t = threading.Thread(target=send_async, args=(0.0, 0.0))
                    t.daemon = True; t.start()
                    scan_pulse_active = False
                    last_stream_time = current_t

    # [FASE 6: ALL_FINISHED] Seluruh Misi Selesai Total
    elif state == STATE_ALL_FINISHED:
        send_async(0.0, 0.0)

    # 6. Gambar Tampilan HUD
    disp_j = curr_jarak if curr_jarak > 0 else last_valid_jarak
    draw_hud(frame, state, active_target_num, curr_sudut, disp_j, start_distance,
             detected_persons, selected_target, hold_prog)

    cv2.imshow("Multi-Target Navigasi Otomatis (Finish 0.7m)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset seluruh misi ke awal
        state              = STATE_IDLE_SELECT
        active_target_num  = 1
        is_holding_gesture = False
        angle_history.clear()
        smooth_sudut       = 0.0
        start_distance     = 0.0
        scan_pulse_active  = False
        align_pulse_active = False
        align_settle_count = 0
        send_async(0.0, 0.0)
        print("\n>> [RESET TOTAL] Seluruh misi dikembalikan ke awal (Fase 1 Seleksi Target).\n")

# ─── CLEANUP ──────────────────────────────────────────────────
color_stream.stop()
depth_stream.stop()
if esp32:
    esp32.close()
openni2.unload()
cv2.destroyAllWindows()
print("\n>> Program ditutup dan koneksi hardware dilepas dengan aman.")
