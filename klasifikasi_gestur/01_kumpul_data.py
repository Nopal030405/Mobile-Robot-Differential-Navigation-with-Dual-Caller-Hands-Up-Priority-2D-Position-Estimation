# ==============================================================================
# 01_kumpul_data.py - Perekaman Dataset Gestur Hands Up (Kinect/Webcam + YOLO + MediaPipe)
# ==============================================================================

import os
import cv2
import csv
import time
from datetime import datetime
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

import config

# Pastikan direktori dataset dan foto tersedia
os.makedirs(config.DIR_DATASET, exist_ok=True)
os.makedirs(config.DIR_IMAGES_RAW, exist_ok=True)

CSV_COLUMNS = [
    "id", "timestamp", "subjek", "sesi", "jarak_m", "pencahayaan",
    "kelas", "f_L", "f_R", "y_LS", "y_RS", "y_LW", "y_RW", "path_foto", "sumber"
]

def inisialisasi_csv():
    """Membuat file dataset.csv jika belum ada atau memuat ID terakhir."""
    if not os.path.exists(config.FILE_DATASET_CSV):
        with open(config.FILE_DATASET_CSV, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
        print(f">> [INFO] File baru dibuat: {config.FILE_DATASET_CSV}")

def dapatkan_next_id():
    """Menghitung ID urut berikutnya dari file CSV."""
    if not os.path.exists(config.FILE_DATASET_CSV):
        return 1
    with open(config.FILE_DATASET_CSV, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        if len(reader) <= 1:
            return 1
        return len(reader)

def hitung_statistik_dataset():
    """Menghitung jumlah data per kelas dan per subjek saat ini."""
    if not os.path.exists(config.FILE_DATASET_CSV):
        return 0, 0, {}
    count_0 = 0
    count_1 = 0
    per_subjek = {}
    with open(config.FILE_DATASET_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kelas = row.get("kelas", "")
            subjek = row.get("subjek", "tidak_diketahui")
            if kelas == "1":
                count_1 += 1
            elif kelas == "0":
                count_0 += 1
            per_subjek[subjek] = per_subjek.get(subjek, 0) + 1
    return count_0, count_1, per_subjek

def impor_data_lama():
    """Mengimpor 400 data lama dari dataset_gestur_pose.csv jika belum pernah diimpor."""
    if not os.path.exists(config.FILE_DATASET_LAMA):
        print(">> [INFO] File dataset lama tidak ditemukan, melewati tahap impor.")
        return 0

    inisialisasi_csv()
    sudah_ada_lama = False
    with open(config.FILE_DATASET_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("sumber") == "lama":
                sudah_ada_lama = True
                break

    if sudah_ada_lama:
        print(">> [INFO] Dataset lama sudah pernah diimpor sebelumnya.")
        return 0

    next_id = dapatkan_next_id()
    diimpor = 0
    with open(config.FILE_DATASET_LAMA, mode="r", encoding="utf-8") as f_lama:
        reader_lama = csv.DictReader(f_lama)
        with open(config.FILE_DATASET_CSV, mode="a", newline="", encoding="utf-8") as f_baru:
            writer_baru = csv.writer(f_baru)
            for row in reader_lama:
                # Kolom lama: left_wrist_shoulder_y, right_wrist_shoulder_y, label
                f_L = float(row["left_wrist_shoulder_y"])
                f_R = float(row["right_wrist_shoulder_y"])
                label = int(row["label"])
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Format: id, timestamp, subjek, sesi, jarak_m, pencahayaan, kelas, f_L, f_R, y_LS, y_RS, y_LW, y_RW, path_foto, sumber
                writer_baru.writerow([
                    next_id, ts, "subjek_1", "sesi_lama", 1.5, "normal",
                    label, f_L, f_R, "", "", "", "", "", "lama"
                ])
                next_id += 1
                diimpor += 1

    print(f">> [SUKSES] Berhasil mengimpor {diimpor} sampel data lama ke dataset.csv (label: subjek_1, sumber: lama).")
    return diimpor

def inisialisasi_kamera():
    """Inisialisasi sensor kamera Kinect v1 secara eksklusif menggunakan OpenNI2."""
    print(">> Menghubungkan ke sensor Kinect v1 via OpenNI2...")
    try:
        from openni import openni2
        # Cari dll di folder saat ini atau path instalasi
        openni2.initialize(config.BASE_DIR)
        dev = openni2.Device.open_any()
        color_stream = dev.create_color_stream()
        color_stream.start()
        print(">> [BERHASIL] Sensor Kinect v1 RGB aktif (640x480)!")
        return "kinect", color_stream, openni2
    except Exception as e:
        print("\n" + "=" * 70)
        print(">> [ERROR] GAGAL MENGHUBUNGKAN KE SENSOR KINECT V1!")
        print(f">> Detail Kendala: {e}")
        print(">> Pastikan hal berikut:")
        print("   1. Kabel USB Kinect v1 dan Adaptor Daya sudah tercolok ke laptop.")
        print("   2. Lampu indikator hijau di sensor Kinect menyala/berkedip.")
        print("=" * 70 + "\n")
        return None, None, None

def baca_frame(tipe_kamera, kamera_obj):
    """Membaca satu frame gambar RGB (640x480) langsung dari sensor Kinect v1."""
    if tipe_kamera == "kinect":
        frame = kamera_obj.read_frame()
        frame_data = frame.get_buffer_as_uint8()
        img = np.frombuffer(frame_data, dtype=np.uint8).reshape((480, 640, 3))
        # Kinect stream berupa RGB, OpenCV membutuhkan BGR untuk visualisasi & penyimpanan
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return True, img
    return False, None

def main():
    print("=" * 70)
    print("   PENGUMPULAN DATASET GESTUR HANDS UP (YOLOv8 + MediaPipe)")
    print("=" * 70)

    # Inisialisasi CSV (file kosong/bersih untuk data baru)
    inisialisasi_csv()

    c0, c1, per_subjek = hitung_statistik_dataset()
    print(f">> Statistik Dataset Baru Saat Ini: Kelas 0 = {c0} | Kelas 1 = {c1} | Total = {c0 + c1}")
    if per_subjek:
        print(f">> Distribusi Subjek Baru: {per_subjek}\n")
    else:
        print(">> Status: Dataset masih BERSIH/KOSONG (Siap rekam sampel baru)\n")

    # Input metadata sesi interaktif
    subjek = input(">> Masukkan nama subjek (contoh: subjek_1 / subjek_2 / subjek_3) [default: subjek_1]: ").strip()
    if not subjek:
        subjek = "subjek_1"

    jarak_input = input(">> Masukkan jarak pengambilan data dalam meter (0.5 - 3.5) [default: 1.5]: ").strip()
    try:
        jarak_m = float(jarak_input) if jarak_input else 1.5
    except ValueError:
        jarak_m = 1.5

    pencahayaan = input(">> Masukkan kondisi pencahayaan (terang / redup / normal) [default: normal]: ").strip()
    if not pencahayaan:
        pencahayaan = "normal"

    sesi = f"sesi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n>> Sesi Aktif: Subjek={subjek} | Jarak={jarak_m}m | Cahaya={pencahayaan} | Sesi={sesi}")
    pilih_kelas = input(">> Rekam Kelas apa? [1: Hands Up / 0: No Gesture] [default: 1]: ").strip()
    target_kelas = 0 if pilih_kelas == "0" else 1
    nama_kelas = "HANDS UP (Kelas 1)" if target_kelas == 1 else "NO GESTURE (Kelas 0)"

    sampel_str = input(f">> Target berapa sampel untuk {nama_kelas}? [default: 200]: ").strip()
    try:
        target_sampel = int(sampel_str) if sampel_str else 200
    except ValueError:
        target_sampel = 200

    print("\n" + "=" * 70)
    print(f">> SISTEM GESTURE-TRIGGERED RECORDING AKTIF:")
    if target_kelas == 1:
        print("   >> Mode: HANDS UP")
        print("   >> Sistem HANYA akan memotret & menyimpan ketika tangan terangkat (Hands Up terdeteksi).")
        print("   >> Jika tangan turun, kamera AKAN DIAM / TIDAK MENYIMPAN APAPUN.")
    else:
        print("   >> Mode: NO GESTURE")
        print("   >> Sistem HANYA akan memotret ketika posisi santai (tangan di bawah bahu).")
        print("   >> Jika tangan terangkat, kamera AKAN DIAM / TIDAK MENYIMPAN.")
    print("=" * 70)

    sample_interval = 0.08  # Interval antar potret (detik) agar gambar sedikit variatif
    last_save_time = 0
    saved_count = 0

    # Inisialisasi Model YOLOv8n
    print(f">> Memuat YOLOv8n dari: {config.YOLO_MODEL_PATH}")
    yolo_model = YOLO(config.YOLO_MODEL_PATH)

    # Inisialisasi MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=config.MP_MODEL_COMPLEXITY,
        min_detection_confidence=config.MP_MIN_DETECTION_CONF,
        min_tracking_confidence=config.MP_MIN_TRACKING_CONF
    )

    tipe_kamera, kamera_obj, openni_mod = inisialisasi_kamera()
    if tipe_kamera is None:
        return

    next_id = dapatkan_next_id()

    try:
        while True:
            ret, frame = baca_frame(tipe_kamera, kamera_obj)
            if not ret or frame is None:
                continue

            display_frame = frame.copy()
            h_img, w_img, _ = frame.shape

            # 1. Deteksi orang menggunakan YOLOv8n (hanya kelas 0 = person)
            results = yolo_model(frame, classes=[0], conf=config.YOLO_CONF_THRESHOLD, verbose=False)

            detected_features = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bw = x2 - x1
                    bh = y2 - y1

                    # Filter batas ukuran bbox
                    if bw < config.MIN_BBOX_WIDTH or bh < config.MIN_BBOX_HEIGHT:
                        continue

                    # Crop area bbox untuk MediaPipe Pose
                    crop_y1 = max(0, y1)
                    crop_y2 = min(h_img, y2)
                    crop_x1 = max(0, x1)
                    crop_x2 = min(w_img, x2)
                    crop_person = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                    if crop_person.size == 0:
                        continue

                    crop_rgb = cv2.cvtColor(crop_person, cv2.COLOR_BGR2RGB)
                    pose_res = pose_detector.process(crop_rgb)

                    if pose_res.pose_landmarks:
                        lms = pose_res.pose_landmarks.landmark

                        # Ambil koordinat landmark 11, 12, 15, 16
                        # 11: Left Shoulder, 12: Right Shoulder, 15: Left Wrist, 16: Right Wrist
                        y_ls = lms[11].y
                        y_rs = lms[12].y
                        y_lw = lms[15].y
                        y_rw = lms[16].y

                        # Fitur selisih posisi vertikal (negatif = tangan di atas bahu)
                        f_L = y_lw - y_ls
                        f_R = y_rw - y_rs

                        detected_features.append({
                            "bbox": (x1, y1, x2, y2),
                            "f_L": f_L,
                            "f_R": f_R,
                            "y_ls": y_ls,
                            "y_rs": y_rs,
                            "y_lw": y_lw,
                            "y_rw": y_rw,
                            "raw_lms": lms,
                            "crop_offset": (crop_x1, crop_y1, bw, bh)
                        })

                        # Gambar visualisasi bbox dan landmark kunci pada display
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        for idx_lm, color_lm in [(11, (255, 0, 0)), (12, (0, 0, 255)), (15, (255, 255, 0)), (16, (0, 255, 255))]:
                            lm_pt = lms[idx_lm]
                            pt_x = int(crop_x1 + lm_pt.x * bw)
                            pt_y = int(crop_y1 + lm_pt.y * bh)
                            cv2.circle(display_frame, (pt_x, pt_y), 5, color_lm, -1)

                        cv2.putText(display_frame, f"f_L:{f_L:.2f} f_R:{f_R:.2f}",
                                    (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Evaluasi gestur pada orang yang terdeteksi
            gestur_handsup_terdeteksi = False
            sample_terpilih = None

            if detected_features:
                sample_terpilih = detected_features[0]
                # Syarat Hands Up: Minimal salah satu tangan (atau keduanya) berada di atas bahu
                # (f_L < 0 berarti pergelangan kiri di atas bahu kiri, f_R < 0 berarti pergelangan kanan di atas bahu kanan)
                if sample_terpilih["f_L"] < 0 or sample_terpilih["f_R"] < 0:
                    gestur_handsup_terdeteksi = True

            # Tentukan apakah kondisi saat ini memenuhi syarat pemotretan
            siap_potret = False
            if target_kelas == 1:
                # Kelas Hands Up: HANYA potret jika gestur Hands Up terdeteksi
                siap_potret = gestur_handsup_terdeteksi
            else:
                # Kelas No Gesture: Bebas / hanya potret jika posisi santai (tidak sedang Hands Up)
                siap_potret = (detected_features and not gestur_handsup_terdeteksi)

            # --- TAMPILKAN STATUS PADA OVERLAY LAYAR ---
            overlay_h = 80
            cv2.rectangle(display_frame, (0, 0), (640, overlay_h), (25, 25, 25), -1)
            cv2.putText(display_frame, f"Subjek: {subjek} | Jarak: {jarak_m}m | Cahaya: {pencahayaan}",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            if target_kelas == 1:
                if gestur_handsup_terdeteksi:
                    status_text = f"[MEMOTRET] HANDS UP TERDETEKSI! [{saved_count}/{target_sampel}]"
                    status_color = (0, 255, 0)
                else:
                    status_text = f"[MENUNGGU GESTUR] Silakan Angkat Tangan Anda [{saved_count}/{target_sampel}]"
                    status_color = (0, 165, 255)
            else:
                status_text = f"[MEMOTRET] NO GESTURE (Posisi Santai) [{saved_count}/{target_sampel}]"
                status_color = (0, 255, 0) if siap_potret else (0, 0, 255)

            cv2.putText(display_frame, status_text, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(display_frame, "Otomatis berhenti saat target tercapai. Tekan 'q' untuk selesai lebih awal.",
                        (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # --- LOGIKA PENYIMPANAN OTOMATIS BERBASIS TRIGGER GESTUR ---
            now_time = time.time()
            if siap_potret and sample_terpilih and (now_time - last_save_time >= sample_interval):
                if saved_count < target_sampel:
                    photo_filename = f"{next_id}_{target_kelas}_{jarak_m}m.jpg"
                    photo_rel_path = os.path.join("dataset", "images", "raw", photo_filename)
                    photo_abs_path = os.path.join(config.DIR_IMAGES_RAW, photo_filename)
                    cv2.imwrite(photo_abs_path, frame)

                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    with open(config.FILE_DATASET_CSV, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            next_id, ts, subjek, sesi, jarak_m, pencahayaan,
                            target_kelas,
                            f"{sample_terpilih['f_L']:.6f}", f"{sample_terpilih['f_R']:.6f}",
                            f"{sample_terpilih['y_ls']:.6f}", f"{sample_terpilih['y_rs']:.6f}",
                            f"{sample_terpilih['y_lw']:.6f}", f"{sample_terpilih['y_rw']:.6f}",
                            photo_rel_path, "kamera"
                        ])

                    if target_kelas == 1:
                        c1 += 1
                    else:
                        c0 += 1

                    saved_count += 1
                    last_save_time = now_time
                    print(f">> [TRIGGER {saved_count}/{target_sampel}] ID#{next_id} Kelas={target_kelas} | f_L={sample_terpilih['f_L']:.2f}, f_R={sample_terpilih['f_R']:.2f} -> {photo_filename}")
                    next_id += 1

                    if saved_count >= target_sampel:
                        print(f"\n>> [SUKSES] Target {target_sampel} sampel untuk {nama_kelas} BERHASIL DIPEROLEH!")
                        break

            cv2.imshow("01 - Pengumpulan Data Gestur", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('j'):
                cv2.destroyAllWindows()
                baru = input(">> Masukkan jarak baru (meter): ").strip()
                try:
                    jarak_m = float(baru)
                except ValueError:
                    pass
                print(f">> Jarak aktif sekarang: {jarak_m}m")

            elif key == ord('s'):
                cv2.destroyAllWindows()
                baru_s = input(">> Masukkan nama subjek baru: ").strip()
                if baru_s:
                    subjek = baru_s
                print(f">> Subjek aktif sekarang: {subjek}")

    finally:
        cv2.destroyAllWindows()
        if tipe_kamera == "kinect" and openni_mod:
            try:
                kamera_obj.stop()
                openni_mod.unload()
            except Exception:
                pass
        elif tipe_kamera == "webcam":
            kamera_obj.release()

        print("\n" + "=" * 70)
        print("   RINGKASAN AKHIR PENGUMPULAN DATA")
        print("=" * 70)
        c0, c1, per_subjek = hitung_statistik_dataset()
        print(f"Total Sampel Kelas 0 : {c0}")
        print(f"Total Sampel Kelas 1 : {c1}")
        print(f"Total Keseluruhan    : {c0 + c1}")
        print(f"Sebaran Subjek       : {per_subjek}")
        print(f"File Dataset         : {config.FILE_DATASET_CSV}")
        print("=" * 70)

if __name__ == "__main__":
    main()

