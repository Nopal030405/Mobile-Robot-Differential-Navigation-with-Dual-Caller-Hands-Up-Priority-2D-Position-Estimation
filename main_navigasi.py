# --- PEMBLOKIR WARNING (HARUS DI BARIS PALING ATAS) ---
import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="google.protobuf.symbol_database")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- IMPORT LIBRARY UTAMA ---
import cv2
import time
import csv
import mediapipe as mp
import numpy as np
import joblib
import threading
from openni import openni2
from ultralytics import YOLO

# Impor pengendali ESP32 dari folder Esp32
from Esp32.send_to_esp32 import ESP32Controller

print("="*60)
print("     SISTEM UTAMA NAVIGASI & MEKANIK (FINAL - COM5)")
print("="*60)

# --- 1. INISIALISASI KONEKSI KE ESP32 (PORT COM5) ---
esp32 = ESP32Controller(port='COM5', baudrate=115200)

# --- 2. MUAT MODEL YOLOv8 & KNN POSE ---
print("Memuat model YOLOv8n dan Model KNN Pose...")
yolo_model = YOLO('yolov8n.pt')

try:
    knn_model = joblib.load('model_knn_gesture.pkl')
except FileNotFoundError:
    try:
        knn_model = joblib.load('Skenario Multi Target/model_knn_gesture.pkl')
    except FileNotFoundError:
        print(">> [ERROR] File 'model_knn_gesture.pkl' tidak ditemukan!")
        exit()

# --- 3. INISIALISASI MEDIAPIPE POSE ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# --- 4. INISIALISASI SENSOR KINECT V1 ---
print("Menghubungkan ke sensor Kinect v1...")
openni2.initialize()
kinect_device = openni2.Device.open_any()

color_stream = kinect_device.create_color_stream()
color_stream.start()
depth_stream = kinect_device.create_depth_stream()
depth_stream.start()

print(">> Sistem Aktif! Tahan gestur Hands Up 5 detik. Tekan 'q' untuk keluar.")

# ══════════════════════════════════════════════════════════════════
# KONSTANTA SISTEM
# ══════════════════════════════════════════════════════════════════
STATE_IDLE     = 0
STATE_TRACKING = 1

STOP_DISTANCE    = 0.7
MAX_DISTANCE     = 4.5
ANGLE_THRESHOLD  = 3.0
STREAM_INTERVAL  = 0.2
TRACKING_GRACE   = 2.0
REQUIRED_HOLD_TIME = 5.0

# ══════════════════════════════════════════════════════════════════
# VARIABEL STATE
# ══════════════════════════════════════════════════════════════════
state              = STATE_IDLE
gesture_start_time = None
is_holding         = False
cooldown_timer     = 0
last_stream_time   = 0
tracking_lost_time = None
csv_filename       = "rekap_data_eksperimen_fleksibel.csv"


# ══════════════════════════════════════════════════════════════════
# FUNGSI BANTU
# ══════════════════════════════════════════════════════════════════
def get_depth_in_meters(depth_frame, px, py, frame_w, frame_h):
    px = max(0, min(int(px), frame_w - 1))
    py = max(0, min(int(py), frame_h - 1))
    depth_array = np.frombuffer(
        depth_frame.get_buffer_as_uint16(), dtype=np.uint16
    ).reshape(480, 640)
    depth_mm = depth_array[py, px]
    if depth_mm <= 0:
        return 1.5
    return round(depth_mm / 1000.0, 2)


def send_data_async(j1, s1, j2, s2):
    if esp32:
        esp32.send_data(j1, s1, j2, s2)


def draw_hud(frame, jarak, sudut):
    fh, fw = frame.shape[:2]

    # Panel fase
    if abs(sudut) > ANGLE_THRESHOLD:
        phase_text  = "  FASE A : MENYELARASKAN SUDUT  "
        phase_color = (20, 140, 255)
    elif jarak <= STOP_DISTANCE and jarak > 0:
        phase_text  = "  BERHENTI - FINISH POINT  "
        phase_color = (30, 30, 220)
    else:
        phase_text  = "  FASE B : MAJU KE TARGET  "
        phase_color = (30, 200, 40)

    (tw, th), _ = cv2.getTextSize(phase_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    px0, py0 = 8, 106
    cv2.rectangle(frame, (px0, py0), (px0 + tw + 6, py0 + th + 10), (15, 15, 25), -1)
    cv2.rectangle(frame, (px0, py0), (px0 + tw + 6, py0 + th + 10), phase_color, 1)
    cv2.putText(frame, phase_text, (px0 + 3, py0 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, phase_color, 2)

    arrow = ">" if sudut > 0 else ("<" if sudut < 0 else "^")
    info_txt = f"Sudut: {sudut:+.1f} deg {arrow}   |   Jarak: {jarak:.2f} m"
    cv2.putText(frame, info_txt, (px0 + 3, py0 + th + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 60), 1)

    # Bar horizontal speedometer digital
    N_SEGS      = 28
    MARGIN      = 10
    PANEL_H     = 98
    panel_y     = fh - PANEL_H - 6
    bar_total_w = fw - MARGIN * 2
    seg_w       = bar_total_w // N_SEGS
    bar_y       = panel_y + 32
    bar_h       = 44

    overlay = frame.copy()
    cv2.rectangle(overlay, (MARGIN - 4, panel_y - 4),
                  (fw - MARGIN + 4, fh - 4), (12, 12, 22), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (MARGIN - 4, panel_y - 4),
                  (fw - MARGIN + 4, fh - 4), (60, 60, 80), 1)

    cv2.putText(frame, "JARAK SISA", (MARGIN, panel_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 160, 180), 1)

    jarak_str = f"{jarak:.2f} m"
    (jw, jh), _ = cv2.getTextSize(jarak_str, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
    cv2.putText(frame, jarak_str, (fw // 2 - jw // 2, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 230, 50), 2)

    cv2.putText(frame, f"STOP {STOP_DISTANCE}m",
                (fw - MARGIN - 95, panel_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (60, 80, 255), 1)

    ratio  = max(0.0, min(1.0, (jarak - STOP_DISTANCE) / (MAX_DISTANCE - STOP_DISTANCE)))
    filled = int(round(ratio * N_SEGS))

    for i in range(N_SEGS):
        sx  = MARGIN + i * seg_w + 1
        pct = i / N_SEGS
        if i < filled:
            if pct < 0.25:
                color = (30,  50, 235)
            elif pct < 0.55:
                color = (20, 195, 215)
            else:
                color = (25, 215,  40)
            cv2.rectangle(frame,
                          (sx, bar_y + 4), (sx + seg_w - 2, bar_y + bar_h - 4),
                          color, -1)
            lighter = tuple(min(255, c + 55) for c in color)
            cv2.rectangle(frame,
                          (sx, bar_y + 4), (sx + seg_w - 2, bar_y + 9),
                          lighter, -1)
        else:
            cv2.rectangle(frame,
                          (sx, bar_y + 4), (sx + seg_w - 2, bar_y + bar_h - 4),
                          (40, 40, 52), 1)

    cv2.line(frame, (MARGIN, bar_y + 2), (MARGIN, bar_y + bar_h - 2), (80, 100, 255), 2)

    cv2.putText(frame, f"STOP {STOP_DISTANCE}m",
                (MARGIN + 2, bar_y + bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, (60, 80, 255), 1)
    cv2.putText(frame, f"{MAX_DISTANCE:.1f}m",
                (MARGIN + bar_total_w - 34, bar_y + bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, (25, 200, 40), 1)

    pct_val = int(ratio * 100)
    cv2.putText(frame, f"{pct_val}%",
                (fw - MARGIN - 38, bar_y + bar_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)


# ══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════
while True:
    frame_data  = color_stream.read_frame()
    frame_array = frame_data.get_buffer_as_uint8()
    img         = np.frombuffer(frame_array, dtype=np.uint8).reshape(480, 640, 3)
    frame       = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    depth_frame_data = depth_stream.read_frame()
    fh, fw, _        = frame.shape
    current_time     = time.time()

    # A. Deteksi YOLO
    yolo_results = yolo_model(frame, verbose=False, classes=[0], conf=0.5)

    # B. Deteksi Pose MediaPipe
    rgb_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)

    person_boxes = []
    for r in yolo_results:
        for box in r.boxes:
            if len(person_boxes) < 4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    valid_gesture_targets = []
    if pose_results.pose_landmarks:
        mp_draw.draw_landmarks(frame, pose_results.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS)
        landmarks = pose_results.pose_landmarks.landmark

        left_shoulder  = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist     = landmarks[15]
        right_wrist    = landmarks[16]

        l_diff   = left_wrist.y  - left_shoulder.y
        r_diff   = right_wrist.y - right_shoulder.y
        features = np.array([[l_diff, r_diff]])
        prediction = knn_model.predict(features)[0]

        if prediction == 1:
            mid_x = int((left_shoulder.x + right_shoulder.x) / 2 * fw)
            mid_y = int((left_shoulder.y + right_shoulder.y) / 2 * fh)
            valid_gesture_targets.append((mid_x, mid_y))

    total_person  = len(person_boxes)
    total_gesture = len(valid_gesture_targets)

    # C. Hitung jarak & sudut REAL-TIME
    j1, s1 = 0.0, 0.0
    j2, s2 = 0.0, 0.0

    if len(valid_gesture_targets) >= 1:
        tx1, ty1 = valid_gesture_targets[0]
        s1 = round((tx1 - (fw / 2)) * (57.5 / fw), 1)
        j1 = get_depth_in_meters(depth_frame_data, tx1, ty1, fw, fh)
        cv2.circle(frame, (tx1, ty1), 10, (0, 255, 255), 2)
        cv2.line(frame, (tx1 - 14, ty1), (tx1 + 14, ty1), (0, 255, 255), 1)
        cv2.line(frame, (tx1, ty1 - 14), (tx1, ty1 + 14), (0, 255, 255), 1)
        cv2.putText(frame, f"T1 {j1:.2f}m",
                    (tx1 + 14, ty1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    if len(valid_gesture_targets) >= 2:
        tx2, ty2 = valid_gesture_targets[1]
        s2 = round((tx2 - (fw / 2)) * (57.5 / fw), 1)
        j2 = get_depth_in_meters(depth_frame_data, tx2, ty2, fw, fh)
        cv2.circle(frame, (tx2, ty2), 10, (255, 120, 0), 2)
        cv2.putText(frame, f"T2 {j2:.2f}m",
                    (tx2 + 14, ty2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)

    # ══════════════════════════════════════════════════════════════
    # STATE MACHINE
    # ══════════════════════════════════════════════════════════════
    can_trigger = (current_time - cooldown_timer > 3.0)

    if state == STATE_IDLE:
        if (total_person > 0) and (1 <= total_gesture <= 2) and can_trigger:
            if not is_holding:
                gesture_start_time = current_time
                is_holding = True

            elapsed_time = current_time - gesture_start_time

            prog     = min(1.0, elapsed_time / REQUIRED_HOLD_TIME)
            bar_w_px = int((fw - 40) * prog)
            cv2.rectangle(frame, (20, 92), (fw - 20, 108), (40, 40, 40), -1)
            cv2.rectangle(frame, (20, 92), (20 + bar_w_px, 108), (0, 165, 255), -1)
            cv2.putText(frame,
                        f"MENAHAN GESTUR: {elapsed_time:.1f}s / {REQUIRED_HOLD_TIME:.0f}s",
                        (24, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

            if elapsed_time >= REQUIRED_HOLD_TIME:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n>> [SUKSES 5 DETIK TEREKAM]")
                print(f"   Total Person (YOLO) : {total_person}")
                print(f"   Target Gestur Aktif : {total_gesture}")
                print(f"   Target 1            -> Jarak: {j1}m | Sudut: {s1} deg")
                print(f"   Target 2            -> Jarak: {j2}m | Sudut: {s2} deg")
                print(f">> [STREAMING] Mulai streaming real-time ke ESP32...")

                file_exists = os.path.isfile(csv_filename)
                with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['Timestamp', 'Total_Person', 'Total_Gesture',
                                         'Jarak_1', 'Sudut_1', 'Jarak_2', 'Sudut_2'])
                    writer.writerow([timestamp, total_person, total_gesture,
                                     j1, s1, j2, s2])

                t = threading.Thread(target=send_data_async, args=(j1, s1, j2, s2))
                t.daemon = True
                t.start()

                is_holding         = False
                cooldown_timer     = 0
                last_stream_time   = current_time
                tracking_lost_time = None
                state              = STATE_TRACKING

        else:
            if is_holding:
                is_holding = False
            cv2.putText(frame, "Status: Menunggu Gestur Hands Up...",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 220), 2)

    elif state == STATE_TRACKING:
        # Cek target masih terdeteksi
        if total_gesture > 0:
            tracking_lost_time = None
        else:
            if tracking_lost_time is None:
                tracking_lost_time = current_time
            elif current_time - tracking_lost_time > TRACKING_GRACE:
                print("\n>> [TRACKING] Target hilang > 2s. Kembali ke IDLE.")
                t = threading.Thread(target=send_data_async, args=(0.0, 0.0, 0.0, 0.0))
                t.daemon = True
                t.start()
                state          = STATE_IDLE
                cooldown_timer = current_time
                is_holding     = False

        # Cek finish point
        if j1 > 0 and j1 <= STOP_DISTANCE:
            print(f"\n>> [TRACKING] Finish point tercapai! Jarak: {j1}m. Berhenti.")
            t = threading.Thread(target=send_data_async, args=(0.0, 0.0, 0.0, 0.0))
            t.daemon = True
            t.start()
            state          = STATE_IDLE
            cooldown_timer = current_time
            is_holding     = False

        # Stream real-time setiap STREAM_INTERVAL detik
        elif current_time - last_stream_time >= STREAM_INTERVAL:
            t = threading.Thread(target=send_data_async, args=(j1, s1, j2, s2))
            t.daemon = True
            t.start()
            last_stream_time = current_time

        # HUD
        draw_hud(frame, j1, s1)

    # Header info
    label  = "[ TRACKING AKTIF ]" if state == STATE_TRACKING else "[ IDLE ]"
    lcolor = (0, 200, 220)        if state == STATE_TRACKING else (0, 200, 0)
    cv2.putText(frame, "Main Navigasi Sistem",
                (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
    cv2.putText(frame, f"Person: {total_person} | Gestur: {total_gesture}  {label}",
                (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, lcolor, 1)

    cv2.imshow("Main Navigasi Sistem", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# CLEANUP
color_stream.stop()
depth_stream.stop()
if esp32:
    esp32.close()
openni2.unload()
cv2.destroyAllWindows()
