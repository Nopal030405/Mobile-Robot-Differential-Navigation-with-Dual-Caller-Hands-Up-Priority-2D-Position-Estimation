import os
import re
import pandas as pd
from datetime import datetime

base = "multi_target_v2/hasil_percobaan"
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

jurnal_durasi = {
    2: 28.8,
    3: 30.3,
    4: 23.3,
    5: 28.1,
    6: 29.5,
    7: 31.2,
    8: 35.5,
    9: 50.2,
}

print("=== ANALISIS LOG & TIMING PERCOBAAN 1-9 ===")

results = []

for num, fld in sorted(folders.items()):
    p = os.path.join(base, fld, "log")
    event_file = os.path.join(p, "log_event.txt")
    ringkasan_file = os.path.join(p, "ringkasan.txt")
    csv_file = os.path.join(p, f"log_frame{num}.csv" if os.path.exists(os.path.join(p, f"log_frame{num}.csv")) else "log_frame.csv")
    
    events = []
    if os.path.exists(event_file):
        with open(event_file, "r", encoding="utf-8", errors="ignore") as f:
            events = [line.strip() for line in f if line.strip()]
            
    # Parse events
    start_time = None
    lock_time = None
    t1_done_time = None
    t2_done_time = None
    end_time = None
    dual_reentry_count = 0
    
    for ev in events:
        # Format: [HH:MM:SS.mmm] [TAG] message
        m = re.match(r"\[(\d{2}:\d{2}:\d{2}\.\d{3})\]\s+\[(.*?)\]\s+(.*)", ev)
        if m:
            ts_str, tag, msg = m.groups()
            t_obj = datetime.strptime(ts_str, "%H:%M:%S.%f")
            if tag == "SISTEM" and "Sesi percobaan dimulai" in msg:
                start_time = t_obj
            elif tag == "KUNCI":
                lock_time = t_obj
            elif tag == "T1_SELESAI":
                t1_done_time = t_obj
            elif tag == "T2_SELESAI":
                t2_done_time = t_obj
            elif tag == "AKHIR":
                end_time = t_obj
            elif tag == "FOTO" and "dual_detect_mulai" in msg:
                dual_reentry_count += 1
                
    # Read CSV
    state_counts = {}
    total_frames = 0
    csv_durasi_wait_t1 = 0.0
    csv_durasi_wait_t2 = 0.0
    
    if os.path.exists(csv_file):
        df_csv = pd.read_csv(csv_file)
        total_frames = len(df_csv)
        col_state = "Fase_State" if "Fase_State" in df_csv.columns else ("state" if "state" in df_csv.columns else None)
        col_time = "Waktu_Detik" if "Waktu_Detik" in df_csv.columns else ("waktu_detik" if "waktu_detik" in df_csv.columns else None)
        
        if col_state:
            state_counts = df_csv[col_state].value_counts().to_dict()
            if col_time:
                df_w1 = df_csv[df_csv[col_state] == "WAIT_T1"]
                if len(df_w1) > 1:
                    csv_durasi_wait_t1 = df_w1[col_time].iloc[-1] - df_w1[col_time].iloc[0]
                elif len(df_w1) == 1:
                    csv_durasi_wait_t1 = 0.1
                
                df_w2 = df_csv[df_csv[col_state] == "WAIT_T2"]
                if len(df_w2) > 1:
                    csv_durasi_wait_t2 = df_w2[col_time].iloc[-1] - df_w2[col_time].iloc[0]
                elif len(df_w2) == 1:
                    csv_durasi_wait_t2 = 0.1
                    
    # Calculate durations from event timestamps
    dur_dual_detect = (lock_time - start_time).total_seconds() if (lock_time and start_time) else None
    dur_murni_nav = (t2_done_time - lock_time).total_seconds() if (t2_done_time and lock_time) else None
    dur_total = (end_time - start_time).total_seconds() if (end_time and start_time) else None
    
    results.append({
        "no": num,
        "folder": fld,
        "start": start_time.strftime("%H:%M:%S.%f")[:-3] if start_time else "-",
        "lock": lock_time.strftime("%H:%M:%S.%f")[:-3] if lock_time else "-",
        "t1_done": t1_done_time.strftime("%H:%M:%S.%f")[:-3] if t1_done_time else "-",
        "t2_done": t2_done_time.strftime("%H:%M:%S.%f")[:-3] if t2_done_time else "-",
        "dur_dual_detect_s": round(dur_dual_detect, 2) if dur_dual_detect else "-",
        "dur_navigasi_murni_s": round(dur_murni_nav, 2) if dur_murni_nav else "-",
        "dur_jurnal_s": jurnal_durasi.get(num, "-"),
        "selisih_durasi_jurnal": round(dur_murni_nav - jurnal_durasi[num], 2) if (dur_murni_nav and num in jurnal_durasi) else "-",
        "dur_total_s": round(dur_total, 2) if dur_total else "-",
        "dual_detect_triggers": dual_reentry_count,
        "total_frames": total_frames,
        "dur_wait_t1_csv_s": round(csv_durasi_wait_t1, 2),
        "dur_wait_t2_csv_s": round(csv_durasi_wait_t2, 2),
        "state_counts": state_counts
    })

res_df = pd.DataFrame(results)
print(res_df[["no", "dur_dual_detect_s", "dur_navigasi_murni_s", "dur_jurnal_s", "selisih_durasi_jurnal", "dur_total_s", "dual_detect_triggers", "dur_wait_t1_csv_s", "dur_wait_t2_csv_s"]].to_string())

with open("analisis_knn_sistem/rekap_log_timing.txt", "w", encoding="utf-8") as f:
    for r in results:
        f.write(f"Percobaan #{r['no']} ({r['folder']}):\n")
        f.write(f"  Waktu Mulai         : {r['start']}\n")
        f.write(f"  Waktu Lock          : {r['lock']}\n")
        f.write(f"  Waktu T1 Selesai    : {r['t1_done']}\n")
        f.write(f"  Waktu T2 Selesai    : {r['t2_done']}\n")
        f.write(f"  Durasi DUAL_DETECT  : {r['dur_dual_detect_s']} s\n")
        f.write(f"  Durasi Navigasi Murni: {r['dur_navigasi_murni_s']} s (Jurnal: {r['dur_jurnal_s']} s, Selisih: {r['selisih_durasi_jurnal']} s)\n")
        f.write(f"  Durasi Total Sesi   : {r['dur_total_s']} s\n")
        f.write(f"  Trigger Dual Detect : {r['dual_detect_triggers']} kali\n")
        f.write(f"  Durasi WAIT_T1 (CSV): {r['dur_wait_t1_csv_s']} s\n")
        f.write(f"  Durasi WAIT_T2 (CSV): {r['dur_wait_t2_csv_s']} s\n")
        f.write(f"  Frame Counts by State: {r['state_counts']}\n\n")
