import os
import glob
import cv2
import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO
import mediapipe as mp

# 1. SETUP
BASE_DIR = "multi_target_v2/hasil_percobaan"
OUTPUT_DIR = "analisis_knn_sistem"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = "klasifikasi_gestur/model/knn_k5.pkl"
knn_model = joblib.load(MODEL_PATH)
yolo_model = YOLO("yolov8n.pt")

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

EXPERIMENT_MAP = {
    "percobaan_20261118_011430": 1,
    "percobaan_20261118_014632": 2,
    "percobaan_20261118_015047": 3,
    "percobaan_20261118_015357": 4,
    "percobaan_20261118_015429": 5,
    "percobaan_20261118_015534": 6,
    "percobaan_20261118_015731": 7,
    "percobaan_20261118_015837": 8,
    "percobaan_20261118_015934": 9
}

print("Running inventory and image re-inference...")
records = []

# Iterate over experiments 2-9
for exp_folder, exp_num in sorted(EXPERIMENT_MAP.items(), key=lambda x: x[1]):
    if exp_num == 1:
        continue # Percobaan 1 is dry-run, no photos
    
    # For percobaan 3, check if foto_original_backup exists to get the unadulterated photo
    exp_path = os.path.join(BASE_DIR, exp_folder)
    foto_dir = os.path.join(exp_path, "foto_original_backup")
    if not os.path.exists(foto_dir):
        foto_dir = os.path.join(exp_path, "foto")
    
    # We will also inspect the actual files in foto/
    active_foto_dir = os.path.join(exp_path, "foto")
    
    if not os.path.exists(active_foto_dir):
        continue
        
    foto_files = sorted(os.listdir(active_foto_dir))
    
    # Read log_event.txt for timestamps and context
    log_event_file = os.path.join(exp_path, "log", "log_event.txt")
    log_events = []
    if os.path.exists(log_event_file):
        with open(log_event_file, "r", encoding="utf-8", errors="ignore") as f:
            log_events = f.readlines()
            
    for fname in foto_files:
        if not fname.endswith(".png"):
            continue
            
        # Determine status_mesin from fname
        # format: <status_mesin>_<counter>.png
        parts = fname.rsplit("_", 1)
        status_mesin = parts[0]
        
        # Load image (prefer unadulterated backup if exists, else active)
        img_load_path = os.path.join(foto_dir, fname) if os.path.exists(os.path.join(foto_dir, fname)) else os.path.join(active_foto_dir, fname)
        img = cv2.imread(img_load_path)
        if img is None:
            records.append({
                "percobaan": exp_num,
                "nama_file": fname,
                "status_mesin": status_mesin,
                "id_orang": "NONE",
                "box_xyxy": "NONE",
                "f_L": np.nan,
                "f_R": np.nan,
                "label_knn": "TIDAK DAPAT DINILAI",
                "proba_hands_up": np.nan,
                "jarak_log_bila_ada": "",
                "penilaian_visual": "TIDAK DAPAT DINILAI",
                "catatan": "File gambar gagal dibaca atau rusak"
            })
            continue
            
        fh, fw = img.shape[:2]
        
        # YOLO detection
        yolo_res = yolo_model(img, conf=0.5, verbose=False)
        boxes_kept = []
        for r in yolo_res:
            for b in r.boxes:
                # check class is person (class 0)
                if int(b.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    x1 = max(0, min(x1, fw-1))
                    y1 = max(0, min(y1, fh-1))
                    x2 = max(0, min(x2, fw))
                    y2 = max(0, min(y2, fh))
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw >= 35 and bh >= 70:
                        boxes_kept.append((x1, y1, x2, y2, float(b.conf[0])))
                        
        # Sort boxes left-to-right
        boxes_kept.sort(key=lambda b: b[0])
        
        if len(boxes_kept) == 0:
            records.append({
                "percobaan": exp_num,
                "nama_file": fname,
                "status_mesin": status_mesin,
                "id_orang": 0,
                "box_xyxy": "NONE",
                "f_L": np.nan,
                "f_R": np.nan,
                "label_knn": "TIDAK DAPAT DINILAI",
                "proba_hands_up": np.nan,
                "jarak_log_bila_ada": "",
                "penilaian_visual": "TIDAK DAPAT DINILAI",
                "catatan": "YOLO tidak mendeteksi orang dengan kriteria (w>=35, h>=70, conf>=0.5)"
            })
            continue
            
        for person_idx, (x1, y1, x2, y2, conf) in enumerate(boxes_kept, start=1):
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pose_res = pose_detector.process(crop_rgb)
            
            if not pose_res.pose_landmarks:
                records.append({
                    "percobaan": exp_num,
                    "nama_file": fname,
                    "status_mesin": status_mesin,
                    "id_orang": person_idx,
                    "box_xyxy": f"({x1},{y1},{x2},{y2})",
                    "f_L": np.nan,
                    "f_R": np.nan,
                    "label_knn": "TIDAK DAPAT DINILAI",
                    "proba_hands_up": np.nan,
                    "jarak_log_bila_ada": "",
                    "penilaian_visual": "TIDAK DAPAT DINILAI",
                    "catatan": f"YOLO mendeteksi orang (conf={conf:.2f}), namun MediaPipe Pose gagal mengekstrak landmark"
                })
                continue
                
            lm = pose_res.pose_landmarks.landmark
            ls, rs = lm[11], lm[12]
            lw, rw = lm[15], lm[16]
            
            # Check visibility of wrists and shoulders
            # If wrists or shoulders have very low visibility, note it
            vis_min = min(ls.visibility, rs.visibility, lw.visibility, rw.visibility)
            
            f_L = lw.y - ls.y
            f_R = rw.y - rs.y
            
            pred = knn_model.predict(np.array([[f_L, f_R]]))[0]
            proba = knn_model.predict_proba(np.array([[f_L, f_R]]))[0][1] # proba of class 1 (hands_up)
            
            label_knn = "HANDS_UP" if pred == 1 else "NO_GESTURE"
            
            # Preliminary visual evaluation based on relative geometry & visibility:
            # In normalized crop coordinates, negative diff means wrist is above shoulder
            # We will store raw features and do careful classification
            catatan = f"conf={conf:.2f}, vis_min={vis_min:.2f}"
            
            records.append({
                "percobaan": exp_num,
                "nama_file": fname,
                "status_mesin": status_mesin,
                "id_orang": person_idx,
                "box_xyxy": f"({x1},{y1},{x2},{y2})",
                "f_L": round(f_L, 4),
                "f_R": round(f_R, 4),
                "label_knn": label_knn,
                "proba_hands_up": round(proba, 4),
                "jarak_log_bila_ada": "",
                "penilaian_visual": "",
                "catatan": catatan
            })

df = pd.DataFrame(records)
df.to_csv(os.path.join(OUTPUT_DIR, "raw_inferensi_foto.csv"), index=False)
print(f"Total person instances detected: {len(df)}")

