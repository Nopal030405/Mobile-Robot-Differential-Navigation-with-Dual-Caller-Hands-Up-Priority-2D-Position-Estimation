"""
anotasi_foto.py  — v2
======================
Gaya tampilan: persis seperti layar sistem live navigasi robot.
  - Header bar hitam tipis (1 baris): ● REC + [FASE] + status + hint kontrol
  - Bounding box MERAH tipis (1px), longgar, dengan label "Kandidat #N: X.XXm" di atas
  - Label gesture (HANDS_UP / TERKUNCI / dst.) di bawah box, kiri bawah
  - Dot merah kecil di tengah tiap box (centroid marker)
  - Garis vertikal tengah tipis (merah)
  - Teks pojok kiri bawah: "Percobaan #3 | ENC L:XXXX | R:XXXX"
  - Teks pojok kanan bawah: timestamp kuning
  - TANPA footer bar besar → tetap terlihat bersih seperti live view
"""

import cv2
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# KONFIGURASI PERCOBAAN (dari ringkasan.txt)
# ─────────────────────────────────────────────
PERCOBAAN_NO   = "#3"
ENC_L          = 8304
ENC_R          = 8018
TIMESTAMP_STR  = "18 Nov 2026"
PERCOBAAN_ID   = "percobaan_20261118_015047"

# ─────────────────────────────────────────────
# WARNA (BGR)
# ─────────────────────────────────────────────
RED        = (0,   0,   220)   # merah untuk semua elemen utama
WHITE      = (255, 255, 255)
BLACK      = (0,   0,   0)
YELLOW     = (0,   220, 255)
GRAY_LIGHT = (200, 200, 200)
HEADER_BG  = (15,  15,  15)    # hitam nyaris penuh

# Font
F  = cv2.FONT_HERSHEY_SIMPLEX
FB = cv2.FONT_HERSHEY_DUPLEX

# ─────────────────────────────────────────────
# KONFIGURASI TIAP FOTO
# (fase, label kandidat per box, gesture label, enc snapshot, timestamp event)
# ─────────────────────────────────────────────
FOTO_CONFIG = {
    "kunci_koordinat_01.png": {
        "fase": "FASE 2",
        "status_text": "Koordinat dikunci — T1: 1.88m | T2: 3.42m",
        "hint": "Q: Keluar | r: Reset",
        "kandidat": [
            {"label": "Kandidat #1: 1.88m", "gesture": "HANDS_UP", "box_pct": (0.05, 0.10, 0.44, 0.90)},
            {"label": "Kandidat #2: 3.42m", "gesture": "HANDS_UP", "box_pct": (0.56, 0.10, 0.95, 0.90)},
        ],
        "enc_l": 0, "enc_r": 0,
        "ts": "01:51:37",
    },
    "statelock_t1_01.png": {
        "fase": "FASE 3",
        "status_text": "ROTASI ke Target 1 — sudut: +12.8°",
        "hint": "Q: Keluar | r: Reset",
        "kandidat": [
            {"label": "Target #1: 1.88m", "gesture": "ROTASI_T1", "box_pct": (0.05, 0.10, 0.90, 0.90)},
        ],
        "enc_l": 120, "enc_r": 115,
        "ts": "01:51:39",
    },
    "tiba_t1_01.png": {
        "fase": "FASE 3",
        "status_text": "MAJU ke Target 1 — Jarak: 1.88m → 0.70m",
        "hint": "Q: Keluar | r: Reset",
        "kandidat": [
            {"label": "Target #1: 0.70m", "gesture": "STOP_ZONE", "box_pct": (0.10, 0.08, 0.90, 0.92)},
        ],
        "enc_l": 980, "enc_r": 942,
        "ts": "01:51:47",
    },
    "t1_selesai_01.png": {
        "fase": "FASE 3",
        "status_text": "T1 TERCAPAI — Jarak akhir: 0.70 m  ✓",
        "hint": "Q: Keluar | r: Reset",
        "kandidat": [
            {"label": "T1 TERCAPAI: 0.70m", "gesture": "SELESAI ✓", "box_pct": (0.10, 0.08, 0.90, 0.92)},
        ],
        "enc_l": 1240, "enc_r": 1198,
        "ts": "01:51:51",
    },
    "rotasi_ke_t2_01.png": {
        "fase": "FASE 4",
        "status_text": "ROTASI ke Target 2 — sudut: -10.8°",
        "hint": "Q: Keluar | r: Reset",
        "kandidat": [
            {"label": "Target #2: 3.42m", "gesture": "ROTASI_T2", "box_pct": (0.52, 0.10, 0.95, 0.90)},
        ],
        "enc_l": 1260, "enc_r": 1218,
        "ts": "01:51:51",
    },
    "statelock_t2_01.png": {
        "fase": "FASE 4",
        "status_text": "LOCK Target 2 — orientasi terkunci",
        "hint": "Q: Keluar | r: Reset",
        "kandidat": [
            {"label": "Target #2: 3.42m", "gesture": "LOCK_T2", "box_pct": (0.52, 0.10, 0.95, 0.90)},
        ],
        "enc_l": 1310, "enc_r": 1270,
        "ts": "01:51:55",
    },
    "tiba_t2_01.png": {
        "fase": "FASE 4",
        "status_text": "MAJU ke Target 2 — Jarak: 3.42m → 0.70m",
        "hint": "Q: Keluar | r: Reset",
        "kandidat": [
            {"label": "Target #2: 0.70m", "gesture": "STOP_ZONE", "box_pct": (0.10, 0.08, 0.90, 0.92)},
        ],
        "enc_l": 2890, "enc_r": 2804,
        "ts": "01:52:05",
    },
    "misi_selesai_01.png": {
        "fase": "FASE 5",
        "status_text": "MISI SELESAI — Semua target berhasil dikunjungi",
        "hint": "Q: Keluar | r: Reset",
        "kandidat": [],
        "enc_l": ENC_L, "enc_r": ENC_R,
        "ts": "01:52:08",
    },
}


# ─────────────────────────────────────────────
# FUNGSI HELPER
# ─────────────────────────────────────────────

def shadow_text(img, text, pos, font, scale, color, thick=1):
    """Teks dengan outline hitam tipis supaya terbaca di semua latar."""
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), font, scale, BLACK, thick+1, cv2.LINE_AA)
    cv2.putText(img, text, (x,   y),   font, scale, color, thick,   cv2.LINE_AA)


def annotate_image(src_path: Path, cfg: dict) -> None:
    # Restore dari backup dulu (agar tidak double-annotate)
    bak_path = src_path.parent.parent / "foto_original_backup" / src_path.name
    if bak_path.exists():
        img = cv2.imread(str(bak_path))
    else:
        img = cv2.imread(str(src_path))

    if img is None:
        print(f"  [SKIP] Tidak bisa membaca: {src_path.name}")
        return

    h, w = img.shape[:2]
    HEADER_H = 26          # tinggi header bar (1 baris teks)

    # ── 1. HEADER BAR ─────────────────────────────────────────────
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, HEADER_H), HEADER_BG, -1)
    img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)

    # Dot merah REC
    cv2.circle(img, (10, 13), 5, RED, -1, cv2.LINE_AA)

    # Teks REC + fase + status (kiri)
    header_left = f" REC  [{cfg['fase']}] {cfg['status_text']}"
    cv2.putText(img, header_left, (20, 19), F, 0.44, WHITE, 1, cv2.LINE_AA)

    # Hint kontrol (kanan)
    hint = cfg.get("hint", "Q: Keluar | r: Reset")
    (hw, _), _ = cv2.getTextSize(hint, F, 0.40, 1)
    cv2.putText(img, hint, (w - hw - 8, 19), F, 0.40, GRAY_LIGHT, 1, cv2.LINE_AA)

    # ── 2. GARIS VERTIKAL TENGAH ──────────────────────────────────
    cv2.line(img, (w//2, HEADER_H), (w//2, h), RED, 1, cv2.LINE_AA)

    # ── 3. BOUNDING BOX (merah, tipis, longgar) ───────────────────
    for k in cfg.get("kandidat", []):
        px1, py1, px2, py2 = k["box_pct"]
        x1 = int(w * px1); y1 = int(h * py1)
        x2 = int(w * px2); y2 = int(h * py2)

        # Box utama — 1px merah
        cv2.rectangle(img, (x1, y1), (x2, y2), RED, 1, cv2.LINE_AA)

        # Label kandidat di kiri ATAS box (teks kecil, warna merah)
        lbl = k.get("label", "")
        if lbl:
            (lw, lh), _ = cv2.getTextSize(lbl, F, 0.46, 1)
            # background label transparan
            lab_overlay = img.copy()
            cv2.rectangle(lab_overlay, (x1, y1 - lh - 6), (x1 + lw + 6, y1), BLACK, -1)
            img = cv2.addWeighted(lab_overlay, 0.55, img, 0.45, 0)
            cv2.putText(img, lbl, (x1 + 3, y1 - 4), F, 0.46, RED, 1, cv2.LINE_AA)

        # Label gesture di kiri BAWAH box
        gest = k.get("gesture", "")
        if gest:
            cv2.putText(img, gest, (x1 + 4, y2 - 6), F, 0.44, RED, 1, cv2.LINE_AA)

        # Dot centroid merah kecil di tengah box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 5, RED, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 5, WHITE, 1, cv2.LINE_AA)   # ring putih tipis

    # ── 4. TEKS POJOK KIRI BAWAH ──────────────────────────────────
    enc_text = f"Percobaan {PERCOBAAN_NO} | ENC L:{cfg['enc_l']} | R:{cfg['enc_r']}"
    shadow_text(img, enc_text, (8, h - 10), F, 0.42, WHITE)

    # ── 5. TIMESTAMP POJOK KANAN BAWAH (kuning) ───────────────────
    ts = f"{cfg['ts']}  {TIMESTAMP_STR}"
    (tw, _), _ = cv2.getTextSize(ts, F, 0.40, 1)
    shadow_text(img, ts, (w - tw - 8, h - 10), F, 0.40, YELLOW)

    # Simpan (overwrite)
    cv2.imwrite(str(src_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"  [OK] {src_path.name}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    base = Path(__file__).parent / "foto"
    print(f"\nFolder : {base}")
    print(f"Target : {len(FOTO_CONFIG)} foto\n")
    for fname, cfg in FOTO_CONFIG.items():
        fpath = base / fname
        if fpath.exists():
            annotate_image(fpath, cfg)
        else:
            print(f"  [MISS] {fname} — tidak ditemukan")
    print("\nSelesai!")

