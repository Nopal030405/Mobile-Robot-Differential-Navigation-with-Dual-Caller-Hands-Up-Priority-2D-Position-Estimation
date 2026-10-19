import cv2
import csv
import os
import time
import mediapipe as mp
import numpy as np
from openni import openni2

# Inisialisasi OpenNI & Kinect v1
openni2.initialize()
kinect_device = openni2.Device.open_any()
color_stream = kinect_device.create_color_stream()
color_stream.start()

# Menggunakan MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

CSV_DATASET = "dataset_gestur_pose.csv"

# Buat file CSV jika belum ada beserta headernya (Fitur: Posisi relatif pergelangan tangan terhadap bahu)
if not os.path.exists(CSV_DATASET):
    with open(CSV_DATASET, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["left_wrist_shoulder_y", "right_wrist_shoulder_y", "label"])

print("==================================================")
print("     PROGRAM PEREKAMAN DATASET POSE (HANDS UP)")
print("==================================================")
print("Pilih mode rekaman di terminal:")
print(" 1 -> Rekam Kelas GESTURE (Tangan Diangkat di Atas Bahu)")
print(" 0 -> Rekam Kelas NO GESTURE (Tangan Rileks di Samping)")
pilihan = input("Masukkan pilihan Anda (1 / 0): ").strip()

if pilihan not in ['0', '1']:
    print("Pilihan tidak valid! Jalankan ulang program.")
    color_stream.stop()
    openni2.unload()
    exit()

target_label = int(pilihan)
nama_label = "HANDS UP (1)" if target_label == 1 else "NO GESTURE (0)"
print(f"\nMode terpilih: Merekam {nama_label} sebanyak 200 sampel.")
print("Bersiaplah... Perekaman akan dimulai otomatis dalam 3 detik!")
time.sleep(3)

count = 0
MAX_SAMPLES = 200

while count < MAX_SAMPLES:
    frame_data = color_stream.read_frame()
    frame_array = frame_data.get_buffer_as_uint8()
    img = np.frombuffer(frame_array, dtype=np.uint8).reshape(480, 640, 3)
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)
    
    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        landmarks = result.pose_landmarks.landmark
        
        # Ambil koordinat bahu dan pergelangan tangan
        # 11: Left Shoulder, 12: Right Shoulder
        # 15: Left Wrist, 16: Right Wrist
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        
        # Fitur: Selisih koordinat Y (Wrist Y - Shoulder Y)
        # Nilai negatif berarti posisi pergelangan tangan berada di atas bahu
        l_diff = left_wrist.y - left_shoulder.y
        r_diff = right_wrist.y - right_shoulder.y
        
        current_features = [l_diff, r_diff]
        
        # Simpan otomatis ke CSV
        with open(CSV_DATASET, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(current_features + [target_label])
        
        count += 1
        print(f"[{count}/{MAX_SAMPLES}] Merekam {nama_label}...")
        time.sleep(0.05)

    cv2.putText(frame, f"REKAM POSE: {nama_label} ({count}/{MAX_SAMPLES})", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Auto-Collector Pose Hands Up", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\nBerhasil! 200 sampel untuk {nama_label} telah tersimpan di '{CSV_DATASET}'.")
color_stream.stop()
openni2.unload()
cv2.destroyAllWindows()