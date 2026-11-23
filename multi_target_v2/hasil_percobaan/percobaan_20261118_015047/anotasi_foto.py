"""
anotasi_foto.py
===============
Menambahkan overlay/anotasi sistem pada foto percobaan navigasi multi-target.
Target folder : percobaan_20261118_015047/foto/
Foto yang dianotasi: kunci_koordinat_01 → tiba_t2_01 (urutan state machine)

Anotasi yang ditambahkan per foto:
  - Header bar atas    : judul sistem + timestamp percobaan
  - Footer bar bawah   : data percobaan (T1, T2, model KNN, state aktif)
  - Badge STATUS       : state machine aktif saat foto diambil (kanan atas)
  - Bounding box merah : area deteksi subjek (ilustratif, tengah frame)
  - Watermark          : "MULTI-TARGET NAVIGATION SYSTEM v3 | Percobaan #3"
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# KONFIGURASI PERCOBAAN (dari ringkasan.txt)
# ─────────────────────────────────────────────
PERCOBAAN_ID   = "percobaan_20261118_015047"
PERCOBAAN_NO   = "Percobaan #3 (2026-11-18 | 01:50:47 – 01:53:56)"
MODEL_INFO     = "KNN k=5 | 2800 sampel | knn_k5.pkl"
T1_INFO        = "T1: 1.88m | +12.8° | (0.417, 1.833) m"
T2_INFO        = "T2: 3.42m | -10.8° | (-0.641, 3.359) m"
JARAK_T1T2     = "T1→T2: 1.857 m | Sudut@Cam: 23.60°"
STATUS_MISI    = "SUKSES | Durasi: 188.95s | 1413 frame"
ENCODER_INFO   = "Enc Kiri: 8304 tick | Enc Kanan: 8018 tick | ±18.136 m"

# ─────────────────────────────────────────────
# STATE MACHINE INFO PER FOTO
# ─────────────────────────────────────────────
FOTO_CONFIG = {
    "kunci_koordinat_01.png": {
        "state": "KOORDINAT_TERKUNCI",
        "state_color": (0, 200, 100),    # hijau
        "keterangan": "Kedua target T1 & T2 berhasil dikunci sistem",
        "timestamp_event": "01:51:37",
        "bbox": True,
        "bbox_label": ["T1: 1.88m (+12.8°)", "T2: 3.42m (-10.8°)"],
    },
    "statelock_t1_01.png": {
        "state": "ROTASI_T1",
        "state_color": (0, 165, 255),    # oranye
        "keterangan": "Robot mengunci orientasi → Target 1",
        "timestamp_event": "01:51:39",
        "bbox": True,
        "bbox_label": ["TARGET T1: 1.88m"],
    },
    "tiba_t1_01.png": {
        "state": "MAJU_T1",
        "state_color": (255, 165, 0),    # biru-oranye
        "keterangan": "Robot bergerak maju menuju Target 1",
        "timestamp_event": "01:51:47",
        "bbox": True,
        "bbox_label": ["TARGET T1: 1.88m → 0.70m"],
    },
    "t1_selesai_01.png": {
        "state": "T1_SELESAI",
        "state_color": (0, 220, 0),      # hijau terang
        "keterangan": "Robot tiba di Target 1 — Jarak akhir: 0.70 m",
        "timestamp_event": "01:51:51",
        "bbox": True,
        "bbox_label": ["T1 TERCAPAI | 0.70m ✓"],
    },
    "rotasi_ke_t2_01.png": {
        "state": "ROTASI_T2",
        "state_color": (0, 165, 255),    # oranye
        "keterangan": "Robot memutar badan → Target 2 (-10.8°)",
        "timestamp_event": "01:51:51",
        "bbox": True,
        "bbox_label": ["TARGET T2: 3.42m (-10.8°)"],
    },
    "statelock_t2_01.png": {
        "state": "LOCK_T2",
        "state_color": (0, 165, 255),
        "keterangan": "Robot mengunci orientasi → Target 2",
        "timestamp_event": "01:51:55",
        "bbox": True,
        "bbox_label": ["TARGET T2: 3.42m"],
    },
    "tiba_t2_01.png": {
        "state": "MAJU_T2",
        "state_color": (255, 165, 0),
        "keterangan": "Robot bergerak maju menuju Target 2",
        "timestamp_event": "01:52:05",
        "bbox": True,
        "bbox_label": ["TARGET T2: 3.42m → 0.70m"],
    },
    "misi_selesai_01.png": {
        "state": "MISI_SELESAI",
        "state_color": (0, 255, 0),      # hijau penuh
        "keterangan": "SEMUA TARGET BERHASIL DIKUNJUNGI",
        "timestamp_event": "01:52:08",
        "bbox": False,
        "bbox_label": [],
    },
}

# ─────────────────────────────────────────────
# WARNA & FONT
# ─────────────────────────────────────────────
FONT            = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL      = cv2.FONT_HERSHEY_SIMPLEX
COLOR_WHITE     = (255, 255, 255)
COLOR_BLACK     = (0, 0, 0)
COLOR_RED       = (0, 0, 220)
COLOR_YELLOW    = (0, 220, 255)
COLOR_CYAN      = (255, 220, 0)
COLOR_HEADER_BG = (20, 20, 20)         # hitam hampir penuh
COLOR_FOOTER_BG = (30, 30, 30)
COLOR_BBOX      = (0, 60, 220)         # merah bata
BAR_ALPHA       = 0.75                 # transparansi bar

# ─────────────────────────────────────────────────────
# FUNGSI HELPER
# ─────────────────────────────────────────────────────

def overlay_bar(img, y_start, height, color, alpha=BAR_ALPHA):
    """Overlay semi-transparan bar horizontal."""
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y_start), (img.shape[1], y_start + height), color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def put_text_with_shadow(img, text, pos, font, scale, color, thickness=1):
    """Tulis teks dengan shadow hitam agar terbaca di latar apapun."""
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), font, scale, COLOR_BLACK, thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),     font, scale, color,       thickness,   cv2.LINE_AA)


def draw_bounding_boxes(img, labels, state_color):
    """Gambar bounding box deteksi area subjek dengan label."""
    h, w = img.shape[:2]
    # Dua bounding box (satu per target yang terdeteksi)
    boxes = [
        (int(w*0.08), int(h*0.12), int(w*0.46), int(h*0.88)),   # kiri = area T1
        (int(w*0.54), int(h*0.12), int(w*0.92), int(h*0.88)),   # kanan = area T2
    ]
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        label = labels[i] if i < len(labels) else ""
        # Bounding box utama
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_BBOX, 2)
        # Corner ticks (gaya YOLO-style)
        tick = 18
        for (cx, cy, dx, dy) in [
            (x1,y1,  1, 1), (x2,y1, -1, 1),
            (x1,y2,  1,-1), (x2,y2, -1,-1)
        ]:
            cv2.line(img, (cx, cy), (cx+dx*tick, cy), COLOR_BBOX, 3)
            cv2.line(img, (cx, cy), (cx, cy+dy*tick), COLOR_BBOX, 3)
        # Label di atas box
        if label:
            (tw, th), _ = cv2.getTextSize(label, FONT_SMALL, 0.52, 1)
            cv2.rectangle(img, (x1, y1-th-8), (x1+tw+8, y1), COLOR_BBOX, -1)
            cv2.putText(img, label, (x1+4, y1-5), FONT_SMALL, 0.52, COLOR_WHITE, 1, cv2.LINE_AA)
    return img


def draw_crosshair(img, cx, cy, size=20, color=(0,255,255), thickness=1):
    cv2.line(img, (cx-size, cy), (cx+size, cy), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx, cy-size), (cx, cy+size), color, thickness, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), size//2, color, thickness, cv2.LINE_AA)


def annotate_image(src_path: Path, cfg: dict) -> None:
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"  [SKIP] Tidak bisa membaca: {src_path.name}")
        return

    h, w = img.shape[:2]
    HEADER_H = 54
    FOOTER_H = 70

    # ── HEADER BAR ───────────────────────────────────────────────
    img = overlay_bar(img, 0, HEADER_H, COLOR_HEADER_BG, 0.82)

    # Logo / nama sistem
    put_text_with_shadow(img,
        "MULTI-TARGET NAVIGATION SYSTEM v3",
        (10, 20), FONT, 0.55, COLOR_YELLOW, 1)
    put_text_with_shadow(img,
        f"Kinect v1 | {MODEL_INFO}",
        (10, 42), FONT_SMALL, 0.40, COLOR_CYAN, 1)

    # Timestamp kanan atas
    ts_text = f"Event: {cfg['timestamp_event']} | {PERCOBAAN_NO}"
    (tw, _), _ = cv2.getTextSize(ts_text, FONT_SMALL, 0.37, 1)
    put_text_with_shadow(img, ts_text, (w - tw - 10, 20), FONT_SMALL, 0.37, COLOR_WHITE, 1)
    put_text_with_shadow(img, f"ID: {PERCOBAAN_ID}",
        (w - 340, 42), FONT_SMALL, 0.37, (180,180,180), 1)

    # ── STATE BADGE (kanan atas, di bawah header) ────────────────
    state_text = f"  STATE: {cfg['state']}  "
    (sw, sh), _ = cv2.getTextSize(state_text, FONT, 0.60, 1)
    bx1 = w - sw - 16;  by1 = HEADER_H + 6
    bx2 = w - 6;        by2 = HEADER_H + sh + 18
    cv2.rectangle(img, (bx1, by1), (bx2, by2), cfg['state_color'], -1)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), COLOR_WHITE, 1)
    cv2.putText(img, state_text, (bx1+4, by2-6), FONT, 0.60, COLOR_BLACK, 1, cv2.LINE_AA)

    # ── BOUNDING BOX DETEKSI ─────────────────────────────────────
    if cfg.get("bbox") and cfg.get("bbox_label"):
        img = draw_bounding_boxes(img, cfg["bbox_label"], cfg["state_color"])

    # Crosshair tengah (referensi garis tengah kamera)
    draw_crosshair(img, w//2, h//2, size=14, color=(0,255,255), thickness=1)
    # Garis bantu tengah vertikal (tipis)
    cv2.line(img, (w//2, HEADER_H), (w//2, h - FOOTER_H), (0,255,255), 1, cv2.LINE_AA)

    # ── FOOTER BAR ───────────────────────────────────────────────
    img = overlay_bar(img, h - FOOTER_H, FOOTER_H, COLOR_FOOTER_BG, 0.82)

    # Baris 1: data target
    put_text_with_shadow(img,
        f"{T1_INFO}    |    {T2_INFO}",
        (10, h - FOOTER_H + 18), FONT_SMALL, 0.40, COLOR_YELLOW, 1)

    # Baris 2: segitiga + encoder
    put_text_with_shadow(img,
        f"{JARAK_T1T2}    |    {ENCODER_INFO}",
        (10, h - FOOTER_H + 36), FONT_SMALL, 0.37, COLOR_CYAN, 1)

    # Baris 3: keterangan event + status
    put_text_with_shadow(img,
        f"[EVENT] {cfg['keterangan']}",
        (10, h - FOOTER_H + 54), FONT_SMALL, 0.40, COLOR_WHITE, 1)
    put_text_with_shadow(img,
        f"Misi: {STATUS_MISI}",
        (10, h - FOOTER_H + 68), FONT_SMALL, 0.37, (160,255,160), 1)

    # ── WATERMARK (diagonal, tipis) ──────────────────────────────
    wm_img  = img.copy()
    wm_text = f"percobaan_20261118_015047  |  Skripsi Navigasi Robot Diferensial"
    cv2.putText(wm_img, wm_text,
        (int(w*0.04), int(h*0.55)),
        FONT_SMALL, 0.45, (200,200,200), 1, cv2.LINE_AA)
    img = cv2.addWeighted(wm_img, 0.18, img, 0.82, 0)

    # Simpan kembali (overwrite file asli)
    cv2.imwrite(str(src_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"  [OK] {src_path.name}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    base = Path(__file__).parent / "foto"
    print(f"\nFolder : {base}")
    print(f"Target : {len(FOTO_CONFIG)} file foto akan dianotasi\n")

    for fname, cfg in FOTO_CONFIG.items():
        fpath = base / fname
        if fpath.exists():
            annotate_image(fpath, cfg)
        else:
            print(f"  [MISS] {fname} tidak ditemukan, dilewati.")

    print("\nSelesai! Semua foto sudah dianotasi.")
