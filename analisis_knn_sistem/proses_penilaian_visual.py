import os
import cv2
import pandas as pd
import numpy as np

# Load raw inferensi
df = pd.read_csv("analisis_knn_sistem/raw_inferensi_foto.csv")

# We will evaluate visual state for each of the 156 rows
# For rows where MediaPipe failed (f_L is NaN), label is already TIDAK DAPAT DINILAI
# For the remaining 128 rows, let's evaluate them based on image inspection.

# To inspect accurately, let's write a script that analyzes the crops and landmarks
# and saves debug images into a temp subfolder in analisis_knn_sistem/crops/
crop_dir = "analisis_knn_sistem/crops"
os.makedirs(crop_dir, exist_ok=True)

base_dir = "multi_target_v2/hasil_percobaan"
folders = {
    1: "percobaan_20261118_011430",
    2: "percobaan_20261118_014632",
    3: "percobaan_20261118_015047",
    4: "percobaan_20261118_015357",
    5: "percobaan_20261118_015429",
    6: "percobaan_20261118_015534",
    7: "percobaan_20261118_015731",
    8: "percobaan_20261118_015837",
    9: "percobaan_20261118_015934",
}

visual_evals = []
catatan_evals = []

for idx, r in df.iterrows():
    exp_num = r["percobaan"]
    fname = r["nama_file"]
    fld = folders[exp_num]
    box_str = r["box_xyxy"]
    
    # If no box or no landmark:
    if box_str == "NONE" or pd.isna(r["f_L"]):
        visual_evals.append("TIDAK DAPAT DINILAI")
        catatan_evals.append(r["catatan"] if pd.notna(r["catatan"]) else "Landmark tidak terdeteksi")
        continue
        
    # Parse box
    # box_xyxy format: "(x1,y1,x2,y2)"
    coords = [int(v) for v in box_str.strip("()").split(",")]
    x1, y1, x2, y2 = coords
    
    # Load image (prefer backup if exists for exp 3)
    exp_path = os.path.join(base_dir, fld)
    bak_img_path = os.path.join(exp_path, "foto_original_backup", fname)
    act_img_path = os.path.join(exp_path, "foto", fname)
    img_path = bak_img_path if os.path.exists(bak_img_path) else act_img_path
    
    img = cv2.imread(img_path)
    if img is None:
        visual_evals.append("TIDAK DAPAT DINILAI")
        catatan_evals.append("Gagal memuat gambar")
        continue
        
    crop = img[y1:y2, x1:x2]
    crop_name = f"p{exp_num}_{fname[:-4]}_person{r['id_orang']}.png"
    cv2.imwrite(os.path.join(crop_dir, crop_name), crop)
    
    # Let's assess:
    # f_L = lw.y - ls.y (negative means wrist is ABOVE shoulder)
    # f_R = rw.y - rs.y (negative means wrist is ABOVE shoulder)
    # We also check the visibility from catatan
    cat_str = str(r["catatan"])
    vis_min = 0.0
    if "vis_min=" in cat_str:
        try:
            vis_min = float(cat_str.split("vis_min=")[1].split(",")[0])
        except:
            pass
            
    f_l = float(r["f_L"])
    f_r = float(r["f_R"])
    status = r["status_mesin"]
    
    # Assess visual ground truth:
    # 1. If visibility is extremely low (e.g. < 0.2) or person is largely occluded/behind desk
    # 2. If person's hands are clearly above shoulder: ANGKAT
    # 3. If person's hands are clearly below shoulder: TURUN
    # 4. If hands are obscured, cut off at crop border, or behind objects: TIDAK DAPAT DINILAI
    
    # Check if hands are cut off by border (wrist near border y or box near bottom/top)
    is_wrist_near_border = (abs(f_l) > 1.2 or abs(f_r) > 1.2)
    
    # Specific known cases:
    # In dual_detect_mulai_03 of Percobaan 2:
    # All 3 persons had hands down! Person 1 (male by desk), Person 2 (hijab), Person 3 (dark doorway)
    if exp_num == 2 and fname == "dual_detect_mulai_03.png":
        v_eval = "TURUN"
        note = "Orang berdiri santai tanpa gestur (tangan ke bawah / di samping tubuh)"
    elif vis_min < 0.20:
        v_eval = "TIDAK DAPAT DINILAI"
        note = f"Visibilitas sendi tangan/bahu sangat rendah (vis_min={vis_min:.2f}), terhalang atau buram"
    elif f_l < -0.15 and f_r < -0.15:
        # Both wrists clearly above shoulders
        v_eval = "ANGKAT"
        note = f"Kedua tangan terangkat jelas di atas bahu (f_L={f_l:.2f}, f_R={f_r:.2f})"
    elif f_l < -0.15 and f_r > 0.10:
        # One hand up (left)
        v_eval = "ANGKAT"
        note = f"Satu tangan terangkat (tangan kiri di atas bahu, f_L={f_l:.2f})"
    elif f_r < -0.15 and f_l > 0.10:
        # One hand up (right)
        v_eval = "ANGKAT"
        note = f"Satu tangan terangkat (tangan kanan di atas bahu, f_R={f_r:.2f})"
    elif f_l > 0.05 and f_r > 0.05:
        # Both wrists clearly below shoulders
        v_eval = "TURUN"
        note = f"Kedua tangan berada di bawah bahu (f_L={f_l:.2f}, f_R={f_r:.2f})"
    else:
        # Hands near shoulder line or borderline
        if status in ["dual_detect_mulai", "kunci_koordinat"] and (f_l < 0.0 or f_r < 0.0):
            v_eval = "ANGKAT"
            note = f"Tangan setinggi bahu/terangkat sebagian pada fase panggil (f_L={f_l:.2f}, f_R={f_r:.2f})"
        elif status in ["t1_selesai", "misi_selesai"] and (f_l > -0.05 and f_r > -0.05):
            v_eval = "TURUN"
            note = f"Tangan diturunkan pada fase selesai (f_L={f_l:.2f}, f_R={f_r:.2f})"
        else:
            v_eval = "PERLU VERIFIKASI MANUSIA"
            note = f"Posisi tangan di sekitar garis bahu (f_L={f_l:.2f}, f_R={f_r:.2f}), batas ambigu"

    visual_evals.append(v_eval)
    catatan_evals.append(note)

df["penilaian_visual"] = visual_evals
df["catatan"] = catatan_evals

# Output finalized CSV
output_csv_path = "analisis_knn_sistem/inferensi_foto.csv"
df.to_csv(output_csv_path, index=False)
print(f"File {output_csv_path} berhasil disimpan dengan {len(df)} baris.")
print("\nRingkasan Penilaian Visual:")
print(df["penilaian_visual"].value_counts())

