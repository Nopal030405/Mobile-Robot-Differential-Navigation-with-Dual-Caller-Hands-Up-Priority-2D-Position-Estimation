import cv2
import time
import csv
import os
import joblib
import mediapipe as mp
import numpy as np
from openni import openni2
from ultralytics import YOLO

print("="*60)
print("     EKSPERIMEN HIBRIDA: YOLOv8 (Max 4 Person) + MP Pose (Max 2 Gestures)")
print("="*60)

# --- 1. MUAT MODEL YOLOv8 & MODEL KNN POSE ---
print("Memuat model YOLOv8n dan Model KNN Pose...")
yolo_model = YOLO('yolov8n.pt') 

model_filename = 'model_knn_gesture.pkl'
try:
    knn_model = joblib.load(model_filename)
    print(f">> Model KNN pose '{model_filename}' berhasil dimuat!")
except FileNotFoundError:
    print(f">> [ERROR] File model '{model_filename}' tidak ditemukan! Jalankan train_eval_2.py dulu.")
    exit()

# --- 2. INISIALISASI MEDIAPIPE POSE ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# --- 3. INISIALISASI CSV REKAPITULASI ---
csv_filename = "rekap_data_eksperimen_fleksibel.csv"
file_exists = os.path.isfile(csv_filename)

with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            'Timestamp', 'Ulangan_Ke', 'Total_Person_YOLO', 'Total_Gesture_Aktif',
            'Jarak_Target_1', 'Sudut_Target_1', 
            'Jarak_Target_2', 'Sudut_Target_2'
        ])

print(f">> Data rekap dicatat terpusat ke '{csv_filename}'.")

# --- 4. INISIALISASI KINECT V1 (WARNA + DEPTH) ---
print("Menghubungkan ke sensor Kinect v1...")
openni2.initialize()     
kinect_device = openni2.Device.open_any()

color_stream = kinect_device.create_color_stream()
color_stream.start()
depth_stream = kinect_device.create_depth_stream()
depth_stream.start()

print(">> Berhasil terhubung! Sistem Hibrida Aktif. Tekan 'q' untuk keluar.")

iteration_count = 0
cooldown_timer = 0  
MAX_ULANGAN = 20

def get_depth_in_meters(depth_frame, px, py, frame_w, frame_h):
    px = max(0, min(int(px), frame_w - 1))
    py = max(0, min(int(py), frame_h - 1))
    depth_array = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
    depth_mm = depth_array[py, px]
    if depth_mm <= 0:
        return 1.5 # Estimasi cadangan jika depth kosong
    return round(depth_mm / 1000.0, 2)

while iteration_count < MAX_ULANGAN:
    frame_data = color_stream.read_frame()
    frame_array = frame_data.get_buffer_as_uint8()
    img = np.frombuffer(frame_array, dtype=np.uint8).reshape(480, 640, 3)
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    depth_frame_data = depth_stream.read_frame()
    h, w, _ = frame.shape
    current_time = time.time()
    
    # A. YOLOv8 Mendeteksi Person (Maksimal 4 orang)
    yolo_results = yolo_model(frame, verbose=False, classes=[0], conf=0.5)
    
    # B. MediaPipe Pose Mendeteksi Pose Tubuh
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    
    # Tampilan Informasi di Layar
    cv2.putText(frame, f"Hibrida YOLO (Max 4) + MP Pose (Max 2)", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"Terkumpul: {iteration_count} / {MAX_ULANGAN}", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # 1. Kumpulkan Bounding Box Person dari YOLO (Batasi maks 4)
    person_boxes = []
    for r in yolo_results:
        for box in r.boxes:
            if len(person_boxes) < 4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 2. Analisis Pose untuk setiap orang yang terdeteksi
    valid_gesture_targets = [] # Menyimpan koordinat target yang terbukti melakukan Hands Up

    if pose_results.pose_landmarks:
        mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = pose_results.pose_landmarks.landmark
        
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        
        # Ekstraksi fitur untuk model KNN Pose
        l_diff = left_wrist.y - left_shoulder.y
        r_diff = right_wrist.y - right_shoulder.y
        features = np.array([[l_diff, r_diff]])
        
        prediction = knn_model.predict(features)[0]
        
        if prediction == 1: # Jika terdeteksi melakukan Hands Up
            # Ambil titik tengah pinggang/badan sebagai representasi posisi target
            mid_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
            mid_y = int((left_shoulder.y + right_shoulder.y) / 2 * h)
            valid_gesture_targets.append((mid_x, mid_y))

    total_person = len(person_boxes)
    total_gesture = len(valid_gesture_targets)

    # REKAM OTOMATIS: Jika ada orang terdeteksi oleh YOLO DAN minimal ada 1 hingga 2 gestur aktif terdeteksi
    if (total_person > 0) and (1 <= total_gesture <= 2) and (current_time - cooldown_timer > 2.0):
        iteration_count += 1
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        j1, s1 = "-", "-"
        j2, s2 = "-", "-"
        
        # Target 1 yang bergestur
        if len(valid_gesture_targets) >= 1:
            tx1, ty1 = valid_gesture_targets[0]
            s1 = round((tx1 - (w / 2)) * (57.5 / w), 1)
            j1 = get_depth_in_meters(depth_frame_data, tx1, ty1, w, h)
            
        # Target 2 yang bergestur (jika ada 2)
        if len(valid_gesture_targets) >= 2:
            tx2, ty2 = valid_gesture_targets[1]
            s2 = round((tx2 - (w / 2)) * (57.5 / w), 1)
            j2 = get_depth_in_meters(depth_frame_data, tx2, ty2, w, h)

        # Simpan ke file CSV rekapitulasi
        with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, iteration_count, total_person, total_gesture, j1, s1, j2, s2])
        
        # Simpan Screenshot Dokumentasi
        folder_img = "screenshot_dokumentasi"
        os.makedirs(folder_img, exist_ok=True)
        img_path = f"{folder_img}/ulangan_{iteration_count}_gestur_{total_gesture}.png"
        cv2.imwrite(img_path, frame)
        
        print(f">> [TEREKAM HIBRIDA #{iteration_count}] Total Person YOLO: {total_person} | Gestur Aktif: {total_gesture}")
        cooldown_timer = time.time()

    cv2.imshow("Eksperimen Hibrida YOLOv8 + MP Pose", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

color_stream.stop()
depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()

print(f"\n>> [SELESAI] Data hibrida tersimpan rapi di '{csv_filename}' dan folder 'screenshot_dokumentasi/'.")