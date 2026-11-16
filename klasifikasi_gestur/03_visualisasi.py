# ==============================================================================
# 03_visualisasi.py - Visualisasi Foto Klasifikasi (TP, TN, FP, FN) & Kolase Jurnal
# ==============================================================================

import os
import cv2
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import config

def buat_kanvas_pose_ilustrasi(f_L, f_R, y_true, y_pred, subjek, jarak, width=480, height=480):
    """
    Membuat visualisasi pose skeleton jika foto asli kamera tidak tersedia (misal data lama).
    Menggambar sketsa anatomis bahu dan tangan berdasarkan nilai fitur f_L dan f_R.
    """
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)

    # Titik tengah
    cx, cy = width // 2, height // 2

    # Bahu kiri dan kanan
    shoulder_y = cy - 20
    ls_x, ls_y = cx - 70, shoulder_y
    rs_x, rs_y = cx + 70, shoulder_y

    # Pergelangan tangan: f_L = y_LW - y_LS => y_LW = y_LS + f_L * scale
    scale = 160.0
    lw_x = ls_x - 30
    lw_y = int(ls_y + f_L * scale)

    rw_x = rs_x + 30
    rw_y = int(rs_y + f_R * scale)

    # Batasi koordinat agar tetap di kanvas
    lw_y = max(40, min(height - 40, lw_y))
    rw_y = max(40, min(height - 40, rw_y))

    # Gambar kepala & leher
    cv2.circle(canvas, (cx, cy - 80), 35, (100, 100, 100), 2)
    cv2.line(canvas, (cx, cy - 45), (cx, cy + 80), (120, 120, 120), 3)

    # Gambar garis bahu
    cv2.line(canvas, (ls_x, ls_y), (rs_x, rs_y), (80, 80, 80), 3)

    # Gambar lengan kiri
    cv2.line(canvas, (ls_x, ls_y), (lw_x, lw_y), (255, 120, 0), 3)
    cv2.circle(canvas, (ls_x, ls_y), 7, (255, 0, 0), -1)  # Bahu kiri (Biru)
    cv2.circle(canvas, (lw_x, lw_y), 8, (255, 200, 0), -1) # Pergelangan kiri

    # Gambar lengan kanan
    cv2.line(canvas, (rs_x, rs_y), (rw_x, rw_y), (0, 120, 255), 3)
    cv2.circle(canvas, (rs_x, rs_y), 7, (0, 0, 255), -1)  # Bahu kanan (Merah)
    cv2.circle(canvas, (rw_x, rw_y), 8, (0, 200, 255), -1) # Pergelangan kanan

    # Label status kategori
    warna_teks = (0, 180, 0) if y_true == y_pred else (0, 0, 200)
    status_label = "BENAR" if y_true == y_pred else "SALAH"

    cv2.rectangle(canvas, (0, 0), (width, 85), (35, 35, 35), -1)
    cv2.putText(canvas, f"Aktual: {y_true} | Prediksi: {y_pred} ({status_label})",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna_teks, 2)
    cv2.putText(canvas, f"f_L: {f_L:.3f} | f_R: {f_R:.3f}",
                (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(canvas, f"Subjek: {subjek} | Jarak: {jarak}m",
                (15, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return canvas

def gambar_anotasi_foto_asli(img, row, y_true, y_pred):
    """Menggambar bounding box, landmark bahu-tangan, dan informasi klasifikasi pada foto asli."""
    annotated = img.copy()
    h, w, _ = annotated.shape

    # Jika ada koordinat landmark tersimpan
    try:
        y_ls = float(row["y_ls"]) if pd.notna(row.get("y_ls")) else None
        y_rs = float(row["y_rs"]) if pd.notna(row.get("y_rs")) else None
        y_lw = float(row["y_lw"]) if pd.notna(row.get("y_lw")) else None
        y_rw = float(row["y_rw"]) if pd.notna(row.get("y_rw")) else None
    except Exception:
        y_ls, y_rs, y_lw, y_rw = None, None, None, None

    f_L = float(row["f_L"])
    f_R = float(row["f_R"])
    subjek = row.get("subjek", "unknown")
    jarak = row.get("jarak_m", "1.5")

    # Header panel informasi
    cv2.rectangle(annotated, (0, 0), (w, 75), (25, 25, 25), -1)
    warna_status = (0, 220, 0) if y_true == y_pred else (0, 0, 240)
    nama_kelas = {1: "Hands Up (1)", 0: "No Gesture (0)"}

    cv2.putText(annotated, f"Aktual: {nama_kelas.get(y_true, y_true)} | Pred: {nama_kelas.get(y_pred, y_pred)}",
                (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, warna_status, 2)
    cv2.putText(annotated, f"f_L: {f_L:.3f} | f_R: {f_R:.3f} | Subjek: {subjek} | Jarak: {jarak}m",
                (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated

def buat_kolase_jurnal(kategori_dict, output_path):
    """
    Membuat kolase gambar (grid 2x2 atau 4 kategori) untuk kebutuhan figur jurnal/skripsi.
    Menampilkan representasi TP, TN, FP, FN.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    mapping = [
        ("True Positive (TP - Hands Up Terdeteksi)", kategori_dict["TP"], axes[0, 0]),
        ("True Negative (TN - No Gesture Terdeteksi)", kategori_dict["TN"], axes[0, 1]),
        ("False Positive (FP - Salah Deteksi)", kategori_dict["FP"], axes[1, 0]),
        ("False Negative (FN - Gagal Deteksi)", kategori_dict["FN"], axes[1, 1]),
    ]

    for judul, img_list, ax in mapping:
        if img_list:
            # Ambil contoh pertama
            img_bgr = cv2.imread(img_list[0])
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                ax.set_title(f"{judul}\n(Total: {len(img_list)} sampel)", fontsize=11, fontweight="bold")
                ax.axis("off")
                continue

        # Jika tidak ada sampel di kategori tersebut (misal FP=0 atau FN=0, yang berarti model sempurna)
        blank = np.full((300, 300, 3), 240, dtype=np.uint8)
        cv2.putText(blank, "Nol Kasus (0 Sampel)", (35, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
        ax.imshow(blank)
        ax.set_title(f"{judul}\n(0 Kasus - Performa Ideal)", fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.DPI_GRAFIK)
    plt.close()
    print(f">> [SUKSES] Kolase figur jurnal disimpan di: {output_path}")

def proses_visualisasi_test_set():
    print("=" * 70)
    print("   PROSES VISUALISASI FOTO KLASIFIKASI (TP, TN, FP, FN)")
    print("=" * 70)

    # 1. Pastikan model KNN ada
    if not os.path.exists(config.FILE_MODEL_KNN):
        print(">> [ERROR] Model KNN belum dilatih! Jalankan 02_latih_evaluasi.py terlebih dahulu.")
        return

    knn = joblib.load(config.FILE_MODEL_KNN)

    # 2. Muat test set teranotasi
    file_test_csv = os.path.join(config.DIR_TABEL, "test_set_annotated.csv")
    if not os.path.exists(file_test_csv):
        print(">> [ERROR] File test_set_annotated.csv tidak ditemukan! Jalankan 02_latih_evaluasi.py terlebih dahulu.")
        return

    df_test = pd.read_csv(file_test_csv)
    print(f">> Memproses visualisasi untuk {len(df_test)} sampel data uji...")

    kategori_files = {"TP": [], "TN": [], "FP": [], "FN": []}

    for idx, row in df_test.iterrows():
        y_true = int(row["kelas"])
        f_L = float(row["f_L"])
        f_R = float(row["f_R"])
        y_pred = int(knn.predict([[f_L, f_R]])[0])

        # Tentukan Kategori
        if y_true == 1 and y_pred == 1:
            kategori = "TP"
            target_dir = config.DIR_TP
        elif y_true == 0 and y_pred == 0:
            kategori = "TN"
            target_dir = config.DIR_TN
        elif y_true == 0 and y_pred == 1:
            kategori = "FP"
            target_dir = config.DIR_FP
        else:
            kategori = "FN"
            target_dir = config.DIR_FN

        sample_id = row.get("id", idx)
        filename_out = f"{kategori}_id{sample_id}_pred{y_pred}_true{y_true}.jpg"
        path_out = os.path.join(target_dir, filename_out)

        # Cek apakah ada foto asli di path_foto
        path_foto_rel = str(row.get("path_foto", "")).strip()
        path_foto_abs = os.path.join(config.BASE_DIR, path_foto_rel) if path_foto_rel else ""

        if path_foto_abs and os.path.exists(path_foto_abs):
            img_raw = cv2.imread(path_foto_abs)
            vis_img = gambar_anotasi_foto_asli(img_raw, row, y_true, y_pred)
        else:
            # Generate kanvas visualisasi pose skeleton
            subjek = str(row.get("subjek", "subjek_1"))
            jarak = str(row.get("jarak_m", "1.5"))
            vis_img = buat_kanvas_pose_ilustrasi(f_L, f_R, y_true, y_pred, subjek, jarak)

        cv2.imwrite(path_out, vis_img)
        kategori_files[kategori].append(path_out)

    print(f">> Hasil Distribusi Foto:")
    print(f"   True Positives  (TP) : {len(kategori_files['TP'])} foto di {config.DIR_TP}")
    print(f"   True Negatives  (TN) : {len(kategori_files['TN'])} foto di {config.DIR_TN}")
    print(f"   False Positives (FP) : {len(kategori_files['FP'])} foto di {config.DIR_FP}")
    print(f"   False Negatives (FN) : {len(kategori_files['FN'])} foto di {config.DIR_FN}")

    # Buat kolase gambar untuk jurnal
    kolase_path = os.path.join(config.DIR_GRAFIK, "kolase_klasifikasi_jurnal.png")
    buat_kolase_jurnal(kategori_files, kolase_path)

def mode_realtime_kamera():
    """Mode opsional: Uji klasifikasi langsung menggunakan kamera Kinect v1 via OpenNI2."""
    print("\n>> Memulai Mode Real-Time Kinect v1 untuk Uji Langsung...")
    print(">> Tekan 's' untuk menyimpan screenshot, 'q' untuk keluar.")

    if not os.path.exists(config.FILE_MODEL_KNN):
        print(">> [ERROR] Model KNN belum ada! Latih model terlebih dahulu di 02_latih_evaluasi.py")
        return

    knn = joblib.load(config.FILE_MODEL_KNN)

    import mediapipe as mp
    from ultralytics import YOLO
    from openni import openni2

    yolo = YOLO(config.YOLO_MODEL_PATH)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    try:
        openni2.initialize(config.BASE_DIR)
        dev = openni2.Device.open_any()
        color_stream = dev.create_color_stream()
        color_stream.start()
        print(">> [BERHASIL] Sensor Kinect v1 RGB aktif!")
    except Exception as e:
        print(f">> [ERROR] Gagal membuka Kinect v1: {e}")
        return

    screenshot_count = 0

    try:
        while True:
            cframe = color_stream.read_frame()
            cdata = cframe.get_buffer_as_uint8()
            frame = np.frombuffer(cdata, dtype=np.uint8).reshape((480, 640, 3))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            h, w, _ = frame.shape
            results = yolo(frame, classes=[0], conf=0.5, verbose=False)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if (x2 - x1) < config.MIN_BBOX_WIDTH or (y2 - y1) < config.MIN_BBOX_HEIGHT:
                        continue

                    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    if crop.size == 0:
                        continue

                    res = pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    if res.pose_landmarks:
                        lms = res.pose_landmarks.landmark
                        f_L = lms[15].y - lms[11].y
                        f_R = lms[16].y - lms[12].y

                        pred = knn.predict([[f_L, f_R]])[0]
                        label = "HANDS UP" if pred == 1 else "NO GESTURE"
                        color = (0, 255, 0) if pred == 1 else (0, 0, 255)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label} (f_L:{f_L:.2f}, f_R:{f_R:.2f})",
                                    (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Real-Time Klasifikasi Gestur Kinect v1 (Tekan 's': Simpan, 'q': Keluar)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('s'):
                screenshot_count += 1
                ss_path = os.path.join(config.DIR_FOTO_KLASIFIKASI, f"screenshot_realtime_{screenshot_count}.jpg")
                cv2.imwrite(ss_path, frame)
                print(f">> [SCREENSHOT] Tersimpan: {ss_path}")

    finally:
        color_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    proses_visualisasi_test_set()
    pilihan = input("\n>> Apakah ingin menjalankan mode Real-Time Kamera sekarang? (y/n) [default: n]: ").strip().lower()
    if pilihan == 'y':
        mode_realtime_kamera()

