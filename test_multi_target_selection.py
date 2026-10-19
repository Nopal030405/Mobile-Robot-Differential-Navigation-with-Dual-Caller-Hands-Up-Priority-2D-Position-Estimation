# =============================================================
# TEST MULTI-TARGET SELECTION (YOLOv8 + MediaPipe Pose + Depth)
# File Uji Coba Terpisah: Deteksi Multi-Person -> Cek Gestur
# -> Seleksi Target Terdekat -> Rekam Data CSV & Screenshot
# =============================================================
import warnings, os
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2, time, csv, mediapipe as mp, numpy as np, joblib
from openni import openni2
from ultralytics import YOLO

print("="*65)
print("   UJI HIBRIDA: YOLOv8 (Multi-Person) + MP Pose (Gestur)")
print("   Tujuan: Seleksi Target Terdekat & Perekaman Data")
print("="*65)

# ─── 1. MUAT MODEL YOLOv8 & KNN POSE ──────────────────────────
print("Memuat model YOLOv8n dan Model KNN Gesture...")
yolo_model = YOLO('yolov8n.pt')

try:
    knn_model = joblib.load('model_knn_gesture.pkl')
    print(">> Model KNN pose berhasil dimuat dari 'model_knn_gesture.pkl'")
except FileNotFoundError:
    knn_model = joblib.load('Skenario Multi Target/model_knn_gesture.pkl')
    print(">> Model KNN pose berhasil dimuat dari folder 'Skenario Multi Target/'")

# ─── 2. INISIALISASI MEDIAPIPE POSE ───────────────────────────
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ─── 3. INISIALISASI SENSOR KINECT V1 (COLOR + DEPTH) ─────────
print("Menghubungkan ke sensor Kinect v1...")
openni2.initialize()
kinect_dev = openni2.Device.open_any()

color_stream = kinect_dev.create_color_stream()
color_stream.start()

depth_stream = kinect_dev.create_depth_stream()
depth_stream.start()

print(">> Kamera & Depth Sensor aktif!")
print(">> Petunjuk Keyboard:")
print("   • Tekan 's' : Simpan Snapshot & Catat ke CSV")
print("   • Tekan 'q' : Keluar dari program\n")

# ─── 4. INISIALISASI FILE CSV REKAPITULASI ─────────────────────
csv_filename = "rekap_multi_target_selection.csv"
file_exists  = os.path.isfile(csv_filename)

with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            'Timestamp', 'Total_Person', 'Total_Gesture_Aktif',
            'ID_Target_Terpilih', 'Jarak_Target_Terpilih_m', 'Sudut_Target_Terpilih_deg',
            'Keterangan_Semua_Person'
        ])

folder_screenshot = "screenshot_multi_target"
os.makedirs(folder_screenshot, exist_ok=True)

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

# ─── VARIABEL RUNTIME ─────────────────────────────────────────
last_save_time   = 0
save_notif_timer = 0
notif_text       = ""
sample_counter   = 0

# ─── MAIN LOOP ────────────────────────────────────────────────
while True:
    c_frame = color_stream.read_frame()
    d_frame = depth_stream.read_frame()

    img   = np.frombuffer(c_frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    fh, fw = frame.shape[:2]
    cx_frame   = fw // 2
    current_t  = time.time()

    # 1. YOLOv8 mendeteksi semua person
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

            # Filter deteksi terlalu kecil
            if box_w < 35 or box_h < 70:
                continue

            # 2. Crop area person untuk dianalisis oleh MediaPipe Pose
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
                ls = lm[11] # Bahu kiri
                rs = lm[12] # Bahu kanan
                lw = lm[15] # Pergelangan tangan kiri
                rw = lm[16] # Pergelangan tangan kanan

                # Ekstraksi fitur KNN (relatif terhadap bahu di dalam crop)
                l_diff = lw.y - ls.y
                r_diff = rw.y - rs.y

                pred = knn_model.predict(np.array([[l_diff, r_diff]]))[0]
                if pred == 1:
                    is_hands_up = True

                # Titik tengah torso dalam koordinat frame global
                mid_x = int(x1 + ((ls.x + rs.x) / 2) * box_w)
                mid_y = int(y1 + ((ls.y + rs.y) / 2) * box_h)
                mid_x = max(x1, min(mid_x, x2))
                mid_y = max(y1, min(mid_y, y2))

            # 3. Baca Jarak & Sudut Person
            jarak = get_depth_in_meters(d_frame, mid_x, mid_y, fw, fh)
            # Jika depth di titik tengah 0, coba titik tengah bounding box
            if jarak <= 0:
                jarak = get_depth_in_meters(d_frame, (x1 + x2) // 2, (y1 + y2) // 2, fw, fh)

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

    # ── 4. LOGIKA SELEKSI TARGET TERDEKAT ─────────────────────────
    # Filter hanya person yang melakukan gestur Hands Up dan jaraknya terbaca valid (> 0)
    gesture_candidates = [p for p in detected_persons if p['is_gesture'] and p['jarak'] > 0]

    selected_target = None
    if len(gesture_candidates) > 0:
        # Urutkan berdasarkan jarak terkecil (Nearest Target First)
        gesture_candidates.sort(key=lambda p: p['jarak'])
        selected_target = gesture_candidates[0] # Target terdekat terpilih!

    # ── 5. VISUALISASI FRAME (BOUNDING BOX & INFORMASI) ───────────
    # Garis tengah frame referensi
    cv2.line(frame, (cx_frame, 35), (cx_frame, fh - 10), (80, 80, 80), 1)

    for p in detected_persons:
        x1, y1, x2, y2 = p['box']
        mx, my = p['mid']
        j = p['jarak']
        s = p['sudut']
        pid = p['id']

        if selected_target and p['id'] == selected_target['id']:
            # TARGET TERPILIH (Paling Dekat): Warna Emas / Hijau Terang + Tebal
            b_color = (0, 230, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), b_color, 3)

            # Crosshair target
            cv2.circle(frame, (mx, my), 12, b_color, 2)
            cv2.line(frame, (mx - 18, my), (mx + 18, my), b_color, 2)
            cv2.line(frame, (mx, my - 18), (mx, my + 18), b_color, 2)

            label_tag = f">> [TARGET TERPILIH: #{pid}] <<"
            cv2.rectangle(frame, (x1, max(0, y1 - 42)), (x1 + 250, max(24, y1)), (15, 20, 25), -1)
            cv2.putText(frame, label_tag, (x1 + 4, max(16, y1 - 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, b_color, 2)
            cv2.putText(frame, f"JARAK: {j:.2f}m | SUDUT: {s:+.1f}d [HANDS UP]",
                        (x1 + 4, max(34, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 150), 1)

        elif p['is_gesture']:
            # TARGET CADANGAN (Gestur aktif tapi jarak lebih jauh): Warna Biru / Cyan
            b_color = (255, 180, 50)
            cv2.rectangle(frame, (x1, y1), (x2, y2), b_color, 2)
            cv2.circle(frame, (mx, my), 8, b_color, -1)

            cv2.rectangle(frame, (x1, max(0, y1 - 24)), (x1 + 220, max(12, y1)), (20, 20, 20), -1)
            cv2.putText(frame, f"Target #{pid} (Lebih Jauh): {j:.2f}m | {s:+.1f}d",
                        (x1 + 4, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, b_color, 1)

        else:
            # BUKAN TARGET (Tidak ada gestur Hands Up): Warna Abu-abu / Putih Tipis
            b_color = (130, 130, 130)
            cv2.rectangle(frame, (x1, y1), (x2, y2), b_color, 1)
            cv2.putText(frame, f"Person #{pid}: {j:.2f}m (Biasa)",
                        (x1 + 4, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

    # ── 6. HEADER STATUS BAR (TOP PANEL) ──────────────────────────
    overlay_top = frame.copy()
    cv2.rectangle(overlay_top, (10, 6), (fw - 10, 36), (15, 15, 20), -1)
    cv2.addWeighted(overlay_top, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (10, 6), (fw - 10, 36), (50, 50, 65), 1)

    total_p = len(detected_persons)
    total_g = len(gesture_candidates)

    if selected_target:
        head_txt = f"TOTAL: {total_p} Person | GESTUR: {total_g} | TERPILIH: Person #{selected_target['id']} ({selected_target['jarak']:.2f}m)"
        h_color  = (0, 230, 255)
    else:
        head_txt = f"TOTAL: {total_p} Person | GESTUR: {total_g} | Menunggu gestur Hands Up..."
        h_color  = (180, 180, 180)

    cv2.putText(frame, head_txt, (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, h_color, 1)
    cv2.putText(frame, "S: Simpan Data | Q: Keluar", (fw - 215, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

    # ── 7. NOTIFIKASI SIMPAN KE CSV ───────────────────────────────
    if current_t - save_notif_timer < 2.0 and notif_text:
        cv2.rectangle(frame, (10, fh - 38), (fw - 10, fh - 8), (20, 120, 30), -1)
        cv2.putText(frame, notif_text, (20, fh - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    cv2.imshow("Test Multi-Target Selection (YOLOv8 + MP Pose)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s'):
        # ── REKAM DATA KE CSV & SIMPAN SCREENSHOT ──
        sample_counter += 1
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        sel_id    = selected_target['id']    if selected_target else "-"
        sel_dist  = selected_target['jarak'] if selected_target else "-"
        sel_angle = selected_target['sudut'] if selected_target else "-"

        # Rangkuman semua person
        detail_list = []
        for p in detected_persons:
            g_str = "HANDS_UP" if p['is_gesture'] else "BIASA"
            detail_list.append(f"P#{p['id']}[{p['jarak']}m,{p['sudut']}d,{g_str}]")
        keterangan_str = "; ".join(detail_list) if detail_list else "Tidak ada person"

        # Tulis baris CSV
        with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, total_p, total_g,
                sel_id, sel_dist, sel_angle, keterangan_str
            ])

        # Simpan gambar screenshot
        img_name = f"{folder_screenshot}/sample_{sample_counter:03d}_p{total_p}_g{total_g}.png"
        cv2.imwrite(img_name, frame)

        notif_text = f">> [DATA TERSIMPAN #{sample_counter}] CSV & Screenshot: {img_name}"
        save_notif_timer = current_t

        print("\n" + "="*60)
        print(f">> [DATA TERSIMPAN #{sample_counter}] Timestamp: {timestamp}")
        print(f"   Total Person YOLO    : {total_p}")
        print(f"   Total Gestur Aktif   : {total_g}")
        if selected_target:
            print(f"   ★ TARGET TERPILIH   : Person #{sel_id}")
            print(f"     • Jarak Terdekat   : {sel_dist} m")
            print(f"     • Sudut Horisontal : {sel_angle} deg")
        else:
            print("   Tidak ada target gestur Hands Up.")
        print(f"   Detail Semua Person  : {keterangan_str}")
        print(f"   Screenshot           : {img_name}")
        print("="*60 + "\n")

# ─── CLEANUP ──────────────────────────────────────────────────
color_stream.stop()
depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()
print("\n>> Pengujian selesai. Program ditutup dengan aman.")
