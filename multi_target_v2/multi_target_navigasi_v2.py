# =============================================================
# MULTI-TARGET NAVIGASI v2 - SEGITIGA KAMERA-T1-T2
# =============================================================
# Alur Baru:
#  1. [DETEKSI AWAL]     Kamera menangkap maks. 2 target dalam FOV
#                        (keduanya harus melakukan gesture Hands Up)
#  2. [KUNCI KOORDINAT]  Jarak & sudut T1+T2 dikunci di memori;
#                        pemetaan segitiga Kamera-T1-T2 dihitung
#  3. [PRIORITAS TARGET] T1 = target terdekat; T2 = target lebih jauh
#  4. [EKSEKUSI T1]
#        a. STATELOCK pada T1 (penyelarasan sudut presisi)
#        b. Robot maju hingga 0.7 m dari T1
#        c. T1 lakukan gesture Hands DOWN -> status T1 "selesai"
#  5. [TRANSISI ke T2]   Robot berputar ke arah sudut T2 yang sudah
#                        dikunci di memori
#  6. [EKSEKUSI T2]
#        a. Setelah T2 masuk FOV: STATELOCK pada T2
#        b. Robot maju hingga 0.7 m dari T2
#        c. T2 lakukan gesture Hands DOWN -> misi selesai
#  7. [RESET]            Terminal menampilkan indikator selesai dan
#                        menunggu keyboard 'r' + Enter untuk ulang
# =============================================================

import warnings, os
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2, time, math, mediapipe as mp, numpy as np, joblib, threading
from openni import openni2
from ultralytics import YOLO
from Esp32.send_to_esp32 import ESP32Controller
from multi_target_logger import NavigationDataLogger

# --- KONSTANTA SISTEM ---
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
DUAL_DETECT_GRACE   = 0.5          # toleransi flicker: boleh hilang maks 0.5 detik
HANDSDOWN_HOLD      = 1.5
HANDSDOWN_GRACE     = 0.5          # toleransi flicker gesture hands-down
SCAN_ROTATION_ANGLE = -10.0
SCAN_STEP_PULSE     = 0.12
SCAN_STEP_PAUSE     = 0.28

# --- STATE MACHINE ---
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

# Inisialisasi Data Logger
logger = NavigationDataLogger()
current_cmd_j = 0.0
current_cmd_s = 0.0

# --- INISIALISASI ---
print("=" * 65)
print("   SISTEM NAVIGASI MULTI-TARGET v2 (Segitiga Kamera-T1-T2)")
print("   Alur: Deteksi Awal -> Kunci -> T1 -> Rotasi -> T2 -> Selesai")
print("=" * 65)

esp32 = ESP32Controller(port='COM5', baudrate=115200)

print("Memuat model YOLOv8n dan Model KNN Gesture...")
yolo_model = YOLO('yolov8n.pt')

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

color_stream = dev.create_color_stream()
color_stream.start()
depth_stream = dev.create_depth_stream()
depth_stream.start()

print(">> Kamera & Depth Sensor aktif!")
print(">> Tekan 'q' di jendela video untuk Keluar.")
print(">> Setelah misi selesai, ketik 'r' + Enter di terminal untuk Reset.\n")


# --- FUNGSI BANTU ---
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
    current_cmd_j = j1
    current_cmd_s = s1
    if esp32:
        esp32.send_data(j1, s1, 0.0, 0.0)


def classify_pose(pose_landmarks, knn_model):
    if pose_landmarks is None:
        return 'neutral'
    lm = pose_landmarks.landmark
    ls, rs = lm[11], lm[12]
    lw, rw = lm[15], lm[16]
    lh, rh = lm[23], lm[24]

    l_diff = lw.y - ls.y
    r_diff = rw.y - rs.y
    pred = knn_model.predict(np.array([[l_diff, r_diff]]))[0]
    if pred == 1:
        return 'hands_up'

    hip_avg_y = (lh.y + rh.y) / 2
    if lw.y > hip_avg_y and rw.y > hip_avg_y:
        return 'hands_down'
    return 'neutral'


def compute_triangle(a1_deg, d1, a2_deg, d2):
    a1 = math.radians(a1_deg)
    a2 = math.radians(a2_deg)
    t1x = d1 * math.sin(a1); t1y = d1 * math.cos(a1)
    t2x = d2 * math.sin(a2); t2y = d2 * math.cos(a2)
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


# --- HUD ---
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

    # Indikator REC di pojok kiri atas
    cv2.circle(frame, (22, 21), 5, (0, 0, 255), -1)
    cv2.putText(frame, "REC", (32, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    if state == STATE_DUAL_DETECT:
        pct = int(dual_prog * 100)
        status_txt = (f"[FASE 1] DETEKSI AWAL: Tunggu 2 target Hands-Up ({pct}%)"
                      if dual_prog < 1.0 else "[FASE 1] MENGUNCI KOORDINAT...")
        scolor = (0, 215, 255)
    elif state in (STATE_ALIGN_T1, STATE_ALIGN_T2):
        lbl = "T1" if state == STATE_ALIGN_T1 else "T2"
        side = "KIRI" if curr_sudut > 0 else "KANAN"
        if abs(curr_sudut) <= ANGLE_THRESHOLD:
            status_txt = f"[{lbl}] STATELOCK - TERPUSAT ({curr_sudut:+.1f}) -> KUNCI..."
            scolor = (50, 230, 100)
        elif abs(curr_sudut) <= 4.5:
            status_txt = f"[{lbl}] ALIGN HALUS (Nudge {side}: {abs(curr_sudut):.1f})"
            scolor = (255, 200, 30)
        else:
            status_txt = f"[{lbl}] ALIGN KASAR ({side}: {abs(curr_sudut):.1f})"
            scolor = (255, 175, 20)
    elif state in (STATE_APPROACH_T1, STATE_APPROACH_T2):
        lbl = "T1" if state == STATE_APPROACH_T1 else "T2"
        status_txt = f"[{lbl}] MAJU -> Jarak Aman {STOP_DISTANCE:.1f} m"
        scolor = (50, 230, 100)
    elif state in (STATE_WAIT_HANDSDOWN_T1, STATE_WAIT_HANDSDOWN_T2):
        lbl = "T1" if state == STATE_WAIT_HANDSDOWN_T1 else "T2"
        pct = int(handsdown_prog * 100)
        status_txt = f"[{lbl} TIBA] Turunkan Tangan {lbl} untuk Lanjut... ({pct}%)"
        scolor = (0, 255, 220)
    elif state == STATE_ROTATE_TO_T2:
        status_txt = "[TRANSISI] Robot berputar ke koordinat memori T2..."
        scolor = (255, 140, 0)
    else:
        status_txt = "[MISI SELESAI] Semua target berhasil! Ketik 'r'+Enter untuk Reset."
        scolor = (50, 255, 150)

    cv2.putText(frame, status_txt, (65, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, scolor, 1)
    cv2.putText(frame, "Q: Keluar | r: Reset", (fw - 180, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 170, 170), 1)

    # Info Percobaan & Encoder di bagian bawah kiri
    info_txt = f"Percobaan #{trial_num} | ENC L:{enc_l} | R:{enc_r}"
    cv2.putText(frame, info_txt, (20, fh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1)

    if state == STATE_DUAL_DETECT and dual_prog > 0:
        bw = int((fw - 20) * min(1.0, dual_prog))
        cv2.rectangle(frame, (10, 34), (10 + bw, 36), (0, 215, 255), -1)

    if state in (STATE_WAIT_HANDSDOWN_T1, STATE_WAIT_HANDSDOWN_T2) and handsdown_prog > 0:
        bw = int((fw - 20) * min(1.0, handsdown_prog))
        cv2.rectangle(frame, (10, 34), (10 + bw, 36), (0, 255, 180), -1)

    for p in detected_persons:
        x1, y1, x2, y2 = p['box']
        mx, my = p['mid']
        j = p['jarak']
        pid = p['id']
        gesture = p.get('gesture', 'neutral')
        is_t1 = locked_t1 and p['id'] == locked_t1.get('id')
        is_t2 = locked_t2 and p['id'] == locked_t2.get('id')

        if is_t1:
            color = (0, 230, 255); label = f"T1 [AKTIF]: {j:.2f}m"; thick = 2
        elif is_t2:
            color = (255, 100, 50); label = f"T2 [MEMORI]: {j:.2f}m"; thick = 2
        elif gesture == 'hands_up':
            color = (255, 180, 50); label = f"Kandidat #{pid}: {j:.2f}m"; thick = 1
        else:
            color = (100, 100, 100); label = f"P#{pid}: {j:.2f}m"; thick = 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        cv2.circle(frame, (mx, my), 6, color, -1 if thick == 1 else 2)
        cv2.putText(frame, label, (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, thick)

    if triangle_info:
        tri_lines = [
            "Segitiga Kamera-T1-T2:",
            f"  Cam->T1 : {triangle_info['cam_to_t1']:.2f} m",
            f"  Cam->T2 : {triangle_info['cam_to_t2']:.2f} m",
            f"  T1->T2  : {triangle_info['t1_to_t2']:.2f} m",
            f"  Sudut Cam: {triangle_info['angle_at_cam_deg']:.1f}",
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


# --- RESET ---
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
    print(">> [LOGGER] Sesi perekaman data di-reset.")
    print("=" * 65 + "\n")


# --- VARIABEL RUNTIME ---
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


# ============================================================
# MAIN LOOP
# ============================================================
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
            x1 = max(0, min(x1, fw - 1)); y1 = max(0, min(y1, fh - 1))
            x2 = max(0, min(x2, fw));     y2 = max(0, min(y2, fh))
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

    # Filter jarak minimum HANYA pada fase ALIGN_T2 dan ROTATE_TO_T2
    # agar T1 yang masih berdiri dekat tidak ikut dipilih sebagai T2.
    # Saat APPROACH_T2, filter DIMATIKAN supaya T2 tidak tersingkir saat jarak < 0.85m.
    min_dist_filter = (STOP_DISTANCE + 0.15) if state in (
        STATE_ALIGN_T2, STATE_ROTATE_TO_T2) else 0.0
    valid_persons = [p for p in detected_persons if p['jarak'] > min_dist_filter]
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
    # STATE MACHINE
    # ==========================================================

    # [FASE 1] DUAL_DETECT
    if state == STATE_DUAL_DETECT:
        # Ambil semua kandidat hands_up (jarak > 0 ATAU jarak == 0 tapi gesture benar)
        hu_list_valid = [p for p in detected_persons
                         if p['gesture'] == 'hands_up' and p['jarak'] > 0]
        hu_list_any   = [p for p in detected_persons
                         if p['gesture'] == 'hands_up']

        # Utamakan yang punya jarak valid; fallback ke semua jika >=2 terdeteksi
        # (depth sensor kadang memberi 0 sesaat)
        hu_list = hu_list_valid if len(hu_list_valid) >= 2 else hu_list_any
        got_two = len(hu_list) >= 2

        if got_two:
            dual_last_seen = current_t          # catat kapan terakhir lihat 2 target
            if not dual_detect_active:
                dual_detect_start = current_t
                dual_detect_active = True
                print(f"\n>> [DETEKSI AWAL] Dua target hands-up terdeteksi. "
                      f"Tahan {DUAL_DETECT_HOLD:.1f}s untuk mengunci koordinat...")

            elapsed = current_t - dual_detect_start
            dual_prog = min(1.0, elapsed / DUAL_DETECT_HOLD)

            if elapsed >= DUAL_DETECT_HOLD:
                # Pastikan pakai list yang ada jarak valid untuk penguncian
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
                print(f"   Pemetaan Segitiga Kamera-T1-T2:")
                print(f"     Cam->T1 : {triangle_info['cam_to_t1']:.2f} m")
                print(f"     Cam->T2 : {triangle_info['cam_to_t2']:.2f} m")
                print(f"     T1->T2  : {triangle_info['t1_to_t2']:.2f} m")
                print(f"     Sudut@Kamera: {triangle_info['angle_at_cam_deg']:.1f} deg")
                print(f"   Robot Mulai STATELOCK -> Penyelarasan ke T1...")
                print("=" * 65 + "\n")

                # Catat data kunci ke logger
                logger.record_locked_data(locked_t1, locked_t2, triangle_info)

                dual_detect_active = False
                align_pulse_start = current_t
                align_pulse_active = False
                align_settle_count = 0
                angle_history.clear(); smooth_sudut = 0.0
                last_stream_time = current_t
                state = STATE_ALIGN_T1
        else:
            # Grace period: hanya reset jika sudah > DUAL_DETECT_GRACE detik
            # tanpa melihat 2 target (menghindari flicker 1-2 frame)
            if dual_detect_active:
                gap = current_t - dual_last_seen
                if gap > DUAL_DETECT_GRACE:
                    dual_detect_active = False
                    print(">> [DETEKSI AWAL] Target berkurang > "
                          f"{DUAL_DETECT_GRACE:.1f}s, reset timer...")
                else:
                    # Masih dalam toleransi flicker, pertahankan timer
                    elapsed = current_t - dual_detect_start
                    dual_prog = min(1.0, elapsed / DUAL_DETECT_HOLD)

    # --- helper align (dipakai untuk T1 dan T2 agar DRY) ---
    elif state in (STATE_ALIGN_T1, STATE_ALIGN_T2):
        next_approach = STATE_APPROACH_T1 if state == STATE_ALIGN_T1 else STATE_APPROACH_T2
        next_wait     = STATE_WAIT_HANDSDOWN_T1 if state == STATE_ALIGN_T1 else STATE_WAIT_HANDSDOWN_T2
        lbl = "T1" if state == STATE_ALIGN_T1 else "T2"

        if selected_target:
            abs_s = abs(curr_sudut)
            in_coarse_zone = abs_s > 4.5
            in_fine_zone   = ANGLE_THRESHOLD < abs_s <= 4.5

            if in_coarse_zone or in_fine_zone:
                # Jika settle count direset (terjadi overshoot), flush angle_history
                # agar nilai sudut tidak "lag" akibat moving average menyimpan nilai lama
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

                # Reset align_pulse_start setiap kali mulai pulse aktif baru
                # agar timing siklus tidak desync saat berpindah zona
                if not align_pulse_active:
                    t_in_cycle = (current_t - align_pulse_start) % cycle_dur
                else:
                    t_in_cycle = (current_t - align_pulse_start) % cycle_dur

                if t_in_cycle < pulse_dur:
                    if not align_pulse_active:
                        # Reset timer siklus saat mulai pulse baru
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

    # [APPROACH T1 & T2]
    elif state in (STATE_APPROACH_T1, STATE_APPROACH_T2):
        next_wait = STATE_WAIT_HANDSDOWN_T1 if state == STATE_APPROACH_T1 else STATE_WAIT_HANDSDOWN_T2
        lbl = "T1" if state == STATE_APPROACH_T1 else "T2"

        if current_t - last_stream_time >= STREAM_INTERVAL:
            # Untuk T2: Saat APPROACH tidak ada filter minimum jarak, sehingga
            # depth reading mendekati 0.7m tetap valid. Gunakan selected_target
            # langsung jika ada, jika tidak fallback ke last_valid_jarak.
            if state == STATE_APPROACH_T2:
                # Ambil raw depth dari semua detected_persons (tanpa filter 0.85m)
                t2_raw = [p for p in detected_persons if p['jarak'] > 0]
                if t2_raw:
                    t2_raw.sort(key=lambda p: p['jarak'])
                    # Pilih yang paling dekat dengan jarak terakhir dikunci
                    t2_best = min(t2_raw, key=lambda p: abs(p['jarak'] - last_valid_jarak))
                    c_dist = t2_best['jarak']
                    # Update last_valid_jarak agar tidak stuck di nilai lama
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
                state = next_wait
            else:
                t = threading.Thread(target=send_async, args=(c_dist, 0.0))
                t.daemon = True; t.start()
                if current_t - last_terminal_log >= 1.0:
                    sisa = max(0.0, c_dist - STOP_DISTANCE)
                    print(f">> [{lbl} MAJU] Jarak: {c_dist:.2f}m | Sisa: {sisa:.2f}m")
                    last_terminal_log = current_t
            last_stream_time = current_t

    # [WAIT HANDSDOWN T1]
    elif state == STATE_WAIT_HANDSDOWN_T1:
        # Throttle ESP32 send — tidak perlu spam setiap frame
        if current_t - last_stream_time >= STREAM_INTERVAL:
            send_async(0.0, 0.0)
            last_stream_time = current_t

        # Target dekat: jarak <= 1.2m ATAU jarak == 0 (karena sangat dekat kamera)
        near = [p for p in detected_persons
                if p['jarak'] <= (STOP_DISTANCE + 0.5) or p['jarak'] == 0]
        
        # Konfirmasi selesai: target sudah TIDAK hands-up (bisa hands_down, netral, santai)
        # Jika tidak ada yang hands_up di area dekat, anggap target sudah menurunkan tangan
        is_still_hands_up = any(p['gesture'] == 'hands_up' for p in near)
        target_released = (len(near) > 0 and not is_still_hands_up) or (len(detected_persons) > 0 and not is_still_hands_up)

        if target_released:
            handsdown_last_seen = current_t
            if not handsdown_active:
                handsdown_start = current_t; handsdown_active = True
                print(f">> [T1] Target sudah menurunkan tangan (tidak Hands-Up)! Tahan {HANDSDOWN_HOLD:.1f}s...")
            elapsed = current_t - handsdown_start
            handsdown_prog = min(1.0, elapsed / HANDSDOWN_HOLD)
            if elapsed >= HANDSDOWN_HOLD:
                print("\n" + "=" * 65)
                print(">> [T1 SELESAI!] Tangan diturunkan / interaksi selesai.")
                print(f"   Robot berputar menuju koordinat memori T2 "
                      f"(sudut: {rotate_target_sudut:+.1f})")
                print("=" * 65 + "\n")
                logger.record_t1_finish(STOP_DISTANCE)
                handsdown_active = False
                scan_start_time = current_t; scan_pulse_active = False
                angle_history.clear(); smooth_sudut = 0.0
                state = STATE_ROTATE_TO_T2
        else:
            # Jika target masih mengangkat tangan (hands_up), reset timer
            if handsdown_active:
                gap = current_t - handsdown_last_seen
                if gap > HANDSDOWN_GRACE:
                    handsdown_active = False
                    print(f">> [T1] Target masih Hands-Up, menunggu tangan diturunkan...")
                else:
                    elapsed = current_t - handsdown_start
                    handsdown_prog = min(1.0, elapsed / HANDSDOWN_HOLD)
                    if elapsed >= HANDSDOWN_HOLD:
                        print("\n" + "=" * 65)
                        print(">> [T1 SELESAI!] Interaksi selesai.")
                        print(f"   Robot berputar menuju koordinat memori T2 "
                              f"(sudut: {rotate_target_sudut:+.1f})")
                        print("=" * 65 + "\n")
                        handsdown_active = False
                        scan_start_time = current_t; scan_pulse_active = False
                        angle_history.clear(); smooth_sudut = 0.0
                        state = STATE_ROTATE_TO_T2

    # [ROTATE TO T2]
    elif state == STATE_ROTATE_TO_T2:
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

    # [WAIT HANDSDOWN T2]
    elif state == STATE_WAIT_HANDSDOWN_T2:
        # Throttle ESP32 send
        if current_t - last_stream_time >= STREAM_INTERVAL:
            send_async(0.0, 0.0)
            last_stream_time = current_t

        near = [p for p in detected_persons
                if p['jarak'] <= (STOP_DISTANCE + 0.5) or p['jarak'] == 0]
        
        # Konfirmasi selesai: target sudah TIDAK hands-up
        is_still_hands_up = any(p['gesture'] == 'hands_up' for p in near)
        target_released = (len(near) > 0 and not is_still_hands_up) or (len(detected_persons) > 0 and not is_still_hands_up)

        if target_released:
            handsdown_last_seen = current_t
            if not handsdown_active:
                handsdown_start = current_t; handsdown_active = True
                print(f">> [T2] Target sudah menurunkan tangan (tidak Hands-Up)! Tahan {HANDSDOWN_HOLD:.1f}s...")
            elapsed = current_t - handsdown_start
            handsdown_prog = min(1.0, elapsed / HANDSDOWN_HOLD)
            if elapsed >= HANDSDOWN_HOLD:
                logger.record_t2_finish(STOP_DISTANCE)
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
                        state = STATE_ALL_FINISHED

    # [ALL FINISHED]
    elif state == STATE_ALL_FINISHED:
        send_async(0.0, 0.0)
        if not _finished_banner_shown:
            print("\n" + "X" * 65)
            print("X                                                               X")
            print("X   [OK] MISI NAVIGASI MULTI-TARGET SELESAI DENGAN SUKSES!     X")
            print("X   Semua target (T1 & T2) dikunjungi & dikonfirmasi.          X")
            print("X                                                               X")
            print("X   Ketik  r + Enter  di terminal ini untuk Reset ke awal.     X")
            print("X   Tekan  q  di jendela video untuk Keluar program.            X")
            print("X" * 65 + "\n")
            _finished_banner_shown = True
        if check_keyboard_reset():
            reset_all()

    # --- LOG REALTIME KE CSV ---
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

    cv2.imshow("Multi-Target Navigasi v2 (Segitiga Kamera-T1-T2)", frame)
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

# --- GENERATE REKAPITULASI CSV & GRAFIK PNG ---
print("\n" + "=" * 65)
print(">> MENYIMPAN HASIL REKAMAN & MEMBUAT GRAFIK PNG...")
logger.finalize()
print("=" * 65)
print("\n>> Program ditutup dan koneksi hardware dilepas dengan aman.")
