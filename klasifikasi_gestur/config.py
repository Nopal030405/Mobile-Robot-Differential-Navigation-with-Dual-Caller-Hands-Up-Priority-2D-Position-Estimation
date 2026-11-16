# ==============================================================================
# config.py - Konfigurasi Terpusat untuk Evaluasi Klasifikasi Gestur Hands Up
# ==============================================================================

import os

# --- 1. REPRODUSIBILITAS ---
SEED = 42

# --- 2. DIREKTORI & PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DIR_DATASET = os.path.join(BASE_DIR, "dataset")
DIR_IMAGES_RAW = os.path.join(DIR_DATASET, "images", "raw")
FILE_DATASET_CSV = os.path.join(DIR_DATASET, "dataset.csv")

# Path dataset lama untuk migrasi/impor awal
FILE_DATASET_LAMA = os.path.join(os.path.dirname(BASE_DIR), "Klasifikasi Gesture", "dataset_gestur_pose.csv")

DIR_MODEL = os.path.join(BASE_DIR, "model")
FILE_MODEL_KNN = os.path.join(DIR_MODEL, "knn_k5.pkl")

DIR_HASIL = os.path.join(BASE_DIR, "hasil")
DIR_LOG = os.path.join(DIR_HASIL, "log")
DIR_GRAFIK = os.path.join(DIR_HASIL, "grafik")
DIR_TABEL = os.path.join(DIR_HASIL, "tabel")
DIR_FOTO_KLASIFIKASI = os.path.join(DIR_HASIL, "foto_klasifikasi")

DIR_TP = os.path.join(DIR_FOTO_KLASIFIKASI, "TP")
DIR_TN = os.path.join(DIR_FOTO_KLASIFIKASI, "TN")
DIR_FP = os.path.join(DIR_FOTO_KLASIFIKASI, "FP")
DIR_FN = os.path.join(DIR_FOTO_KLASIFIKASI, "FN")

# --- 3. PARAMETER DETEKSI & POSE ---
YOLO_MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "yolov8n.pt")
YOLO_CONF_THRESHOLD = 0.5
MIN_BBOX_WIDTH = 35    # Abaikan deteksi orang jika lebar bbox < 35 px
MIN_BBOX_HEIGHT = 70   # Abaikan deteksi orang jika tinggi bbox < 70 px

MP_MODEL_COMPLEXITY = 1
MP_MIN_DETECTION_CONF = 0.5
MP_MIN_TRACKING_CONF = 0.5

# Landmark MediaPipe Pose Index:
# 11: Left Shoulder, 12: Right Shoulder, 15: Left Wrist, 16: Right Wrist
LANDMARKS_USED = [11, 12, 15, 16]

# --- 4. PARAMETER TRAINING & EVALUASI ---
KNN_K_MIN = 1
KNN_K_MAX = 15
DEFAULT_K = 5
CV_FOLDS = 5

TEST_SIZE = 0.2
LEARNING_CURVE_STEPS = [50, 100, 150, 200, 250, 320]
LEARNING_CURVE_TRIALS = 10  # Evaluasi variansi seed (10 kali pengulangan)
BENCHMARK_ITERATIONS = 1000 # Pengulangan untuk benchmark waktu inferensi

DPI_GRAFIK = 300

