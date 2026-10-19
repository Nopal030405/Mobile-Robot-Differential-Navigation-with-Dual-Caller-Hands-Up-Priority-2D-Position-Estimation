import cv2
import time
import csv
import os
import mediapipe as mp
import numpy as np
from openni import openni2
import matplotlib.pyplot as plt

# --- 1. INISIALISASI PENCATATAN CSV (LOG) ---
CSV_FILENAME = "log_percobaan_kinect.csv"

# Membuat file CSV dan menulis header jika belum ada
if not os.path.exists(CSV_FILENAME):
    with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "No_Data", "Jarak_Meter", "Sudut_Derajat"])

print(f"File log otomatis disiapkan: {CSV_FILENAME}")

# --- 2. INISIALISASI OPENNI2 & KINECT V1 (WARNA + DEPTH) ---
print("Menginisialisasi sensor Kinect v1...")
openni2.initialize()
kinect_device = openni2.Device.open_any()

color_stream = kinect_device.create_color_stream()
color_stream.start()
depth_stream = kinect_device.create_depth_stream()
depth_stream.start()

# --- 3. INISIALISASI MEDIAPIPE HANDS ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

lock_start_time = 0
LOCK_DURATION = 3.0     
targets_queue = []      
data_counter = 0        # Untuk menghitung jumlah sampel data terekam

def hitung_jarak_dan_sudut_mentah(landmark_wrist, frame_width, frame_height, depth_frame_data):
    px = int(landmark_wrist.x * frame_width)
    py = int(landmark_wrist.y * frame_height)
    
    px = max(0, min(px, frame_width - 1))
    py = max(0, min(py, frame_height - 1))
    
    depth_array = np.frombuffer(depth_frame_data.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
    depth_mm = depth_array[py, px]
    
    if depth_mm <= 0:
        jarak_meter = 0.0  
    else:
        jarak_meter = depth_mm / 1000.0
        
    sudut_derajat = (px - (frame_width / 2)) * (57.5 / frame_width)
    
    return round(jarak_meter, 3), round(sudut_derajat, 1)

print("Pencatatan otomatis aktif. Tahan Open Palm selama 3 detik untuk merekam data jarak & sudut. Tekan 'q' untuk keluar dan generate grafik.")

while True:
    color_frame_data = color_stream.read_frame()
    color_array = color_frame_data.get_buffer_as_uint8()
    img = np.frombuffer(color_array, dtype=np.uint8).reshape(480, 640, 3)
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    depth_frame_data = depth_stream.read_frame()
    
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    detected_open_palm = False
    current_d, current_theta = 0.0, 0.0
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            fingers_extended = (
                hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and
                hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y and
                hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y and
                hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
            )
            
            if fingers_extended:
                detected_open_palm = True
                wrist = hand_landmarks.landmark[0]
                
                current_d, current_theta = hitung_jarak_dan_sudut_mentah(wrist, w, h, depth_frame_data)
                
                cv2.putText(frame, f"JARAK: {current_d}m | SUDUT: {current_theta} deg", (30, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                if lock_start_time == 0:
                    lock_start_time = time.time()
                
                elapsed_time = time.time() - lock_start_time
                remaining_time = max(0.0, LOCK_DURATION - elapsed_time)
                
                cv2.putText(frame, f"MENGUNCI... {remaining_time:.1f}s", (30, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                if elapsed_time >= LOCK_DURATION:
                    data_counter += 1
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Simpan langsung ke file CSV
                    with open(CSV_FILENAME, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp, data_counter, current_d, current_theta])
                    
                    print(f">> [REKAM DATA #{data_counter}] Jarak: {current_d}m | Sudut: {current_theta}° tersimpan ke CSV.")
                    
                    targets_queue.append({'jarak': current_d, 'sudut': current_theta})
                    targets_queue = sorted(targets_queue, key=lambda x: x['jarak'])
                    
                    lock_start_time = 0
                    time.sleep(0.5)
                    break

    if not detected_open_palm:
        lock_start_time = 0
        cv2.putText(frame, "STATUS: MENUNGGU GESTUR OPEN PALM", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    y_offset = 380
    cv2.putText(frame, f"Total Data Tercatat: {len(targets_queue)}", (30, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    for idx, t in enumerate(targets_queue[-4:]): # Tampilkan 4 data terakhir di layar
        txt = f" Data #{idx+1}: Jarak={t['jarak']}m, Sudut={t['sudut']} deg"
        cv2.putText(frame, txt, (30, y_offset + 25 * (idx + 1)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Perekam Data Pengujian Kinect v1", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. TUTUP STREAM & GENERATE GRAFIK PNG OTOMATIS ---
color_stream.stop()
depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()

print("\nProgram dihentikan. Memproses data CSV menjadi Grafik PNG...")

# Membaca data dari CSV untuk digambar menggunakan Matplotlib
timestamps = []
nomor_data = []
jarak_list = []

if os.path.exists(CSV_FILENAME):
    with open(CSV_FILENAME, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # Lewati header
        for row in reader:
            if len(row) >= 4:
                timestamps.append(row[0])
                nomor_data.append(int(row[1]))
                jarak_list.append(float(row[2]))

    if jarak_list:
        plt.figure(figsize=(10, 5))
        plt.plot(nomor_data, jarak_list, marker='o', color='b', linestyle='-', linewidth=2, label='Jarak Terbaca (m)')
        plt.title('Grafik Pengujian Batas Pembacaan Jarak Sensor Kinect v1', fontsize=14)
        plt.xlabel('Urutan Percobaan (Ke-', fontsize=12)
        plt.ylabel('Jarak Sensor (Meter)', fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Simpan sebagai file PNG
        png_filename = "grafik_pengujian_kinect.png"
        plt.savefig(png_filename)
        print(f">> Berhasil! Grafik telah disimpan otomatis sebagai '{png_filename}' di folder proyek Anda.")
        print(f">> Ringkasan: Jarak Minimum tercatat = {min(jarak_list)} m | Jarak Maksimum tercatat = {max(jarak_list)} m")
    else:
        print("Tidak ada data jarak yang terekam selama percobaan.")