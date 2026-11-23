import pandas as pd
import numpy as np

df = pd.read_csv("analisis_knn_sistem/inferensi_foto.csv")

# Machine states order as requested
states_order = [
    "dual_detect_mulai",
    "kunci_koordinat",
    "statelock_t1",
    "tiba_t1",
    "t1_selesai",
    "rotasi_ke_t2",
    "statelock_t2",
    "tiba_t2",
    "misi_selesai"
]

# Expected label per state:
# dual_detect_mulai: HANDS_UP (kandidat sedang mengangkat tangan untuk panggil)
# kunci_koordinat: HANDS_UP (kedua target mengangkat tangan saat dikunci)
# statelock_t1: HANDS_UP (target 1 dipertahankan terangkat saat alignment)
# tiba_t1: HANDS_UP / TURUN (transisi saat robot tiba di target 1) -> target 1 diharapkan mulai turunkan tangan
# t1_selesai: NO_GESTURE (target 1 telah menurunkan tangan sebagai konfirmasi selesai)
# rotasi_ke_t2: HANDS_UP (target 2 masih mengangkat tangan di memori)
# statelock_t2: HANDS_UP (target 2 dikunci kembali saat robot menghadap)
# tiba_t2: HANDS_UP / TURUN (transisi saat tiba di target 2)
# misi_selesai: NO_GESTURE (seluruh target menurunkan tangan, misi tuntas)

expected_map = {
    "dual_detect_mulai": "HANDS_UP",
    "kunci_koordinat": "HANDS_UP",
    "statelock_t1": "HANDS_UP",
    "tiba_t1": "HANDS_UP", # atau proses turunkan
    "t1_selesai": "NO_GESTURE",
    "rotasi_ke_t2": "HANDS_UP",
    "statelock_t2": "HANDS_UP",
    "tiba_t2": "HANDS_UP",
    "misi_selesai": "NO_GESTURE"
}

summary_rows = []
mismatches = []

for st in states_order:
    sub = df[df["status_mesin"] == st]
    exp_lbl = expected_map[st]
    
    total = len(sub)
    tdk_dinilai = len(sub[sub["penilaian_visual"].isin(["TIDAK DAPAT DINILAI", "PERLU VERIFIKASI MANUSIA"])])
    
    # Assess match against visual ground truth (penilaian_visual)
    # where penilaian_visual is ANGKAT (equivalent to HANDS_UP) or TURUN (equivalent to NO_GESTURE)
    eval_sub = sub[sub["penilaian_visual"].isin(["ANGKAT", "TURUN"])]
    
    cocok = 0
    tidak_cocok = 0
    
    for _, r in eval_sub.iterrows():
        knn = r["label_knn"]
        vis = r["penilaian_visual"]
        
        # Check agreement: ANGKAT <-> HANDS_UP, TURUN <-> NO_GESTURE
        is_match = (vis == "ANGKAT" and knn == "HANDS_UP") or (vis == "TURUN" and knn == "NO_GESTURE")
        if is_match:
            cocok += 1
        else:
            tidak_cocok += 1
            mismatches.append(r)
            
    summary_rows.append({
        "status_mesin": st,
        "label_diharapkan": exp_lbl,
        "total_orang_terdeteksi": total,
        "dinilai_visual": len(eval_sub),
        "cocok": cocok,
        "tidak_cocok": tidak_cocok,
        "tidak_dapat_dinilai": tdk_dinilai
    })

sum_df = pd.DataFrame(summary_rows)
print("=== TABEL PERBANDINGAN PER STATUS MESIN ===")
print(sum_df.to_string(index=False))

mismatches_df = pd.DataFrame(mismatches)
print(f"\nTotal ketidakcocokan: {len(mismatches_df)}")
if len(mismatches_df) > 0:
    print(mismatches_df[["percobaan", "nama_file", "status_mesin", "id_orang", "f_L", "f_R", "label_knn", "proba_hands_up", "penilaian_visual", "catatan"]].to_string())

sum_df.to_csv("analisis_knn_sistem/tabel_perbandingan_status.csv", index=False)
if len(mismatches_df) > 0:
    mismatches_df.to_csv("analisis_knn_sistem/daftar_ketidakcocokan.csv", index=False)

