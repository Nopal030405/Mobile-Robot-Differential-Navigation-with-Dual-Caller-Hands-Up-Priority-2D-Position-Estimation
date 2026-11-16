# ==============================================================================
# 02_latih_evaluasi.py - Pelatihan, Tuning, Evaluasi Menyeluruh Model KNN Gestur
# ==============================================================================

import os
import sys
import time
import math
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import config

# Pastikan seluruh direktori hasil tersedia
os.makedirs(config.DIR_MODEL, exist_ok=True)
os.makedirs(config.DIR_LOG, exist_ok=True)
os.makedirs(config.DIR_GRAFIK, exist_ok=True)
os.makedirs(config.DIR_TABEL, exist_ok=True)

# Inisialisasi Logger
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
FILE_LOG_TXT = os.path.join(config.DIR_LOG, f"run_{RUN_TIMESTAMP}.txt")
FILE_LOG_CSV = os.path.join(config.DIR_LOG, f"run_{RUN_TIMESTAMP}.csv")

log_entries = []

def catat_log(langkah, parameter, hasil):
    """Mencatat log terstruktur ke console, file txt, dan memori untuk csv."""
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pesan = f"[{waktu}] [{langkah}] {parameter} -> {hasil}"
    print(pesan)
    with open(FILE_LOG_TXT, mode="a", encoding="utf-8") as f:
        f.write(pesan + "\n")
    log_entries.append({
        "timestamp": waktu,
        "langkah": langkah,
        "parameter": parameter,
        "hasil": hasil
    })

def simpan_log_csv():
    df_log = pd.DataFrame(log_entries)
    df_log.to_csv(FILE_LOG_CSV, index=False)

def simpan_environment():
    """Menyimpan versi seluruh library ke hasil/log/environment.txt."""
    env_file = os.path.join(config.DIR_LOG, "environment.txt")
    try:
        res = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
        with open(env_file, "w", encoding="utf-8") as f:
            f.write(f"# Environment Dump: {datetime.now()}\n")
            f.write(res.stdout)
        catat_log("ENV", "pip freeze", f"Berhasil disimpan di {env_file}")
    except Exception as e:
        catat_log("ENV", "pip freeze", f"Gagal mengekstrak environment: {e}")

def wilson_score_interval(successes, total, confidence=0.95):
    """Menghitung Wilson Score Confidence Interval 95% untuk akurasi."""
    if total == 0:
        return 0.0, 0.0
    z = 1.95996  # untuk 95% confidence
    p_hat = successes / total
    denominator = 1 + z**2 / total
    centre_adjusted_probability = p_hat + z**2 / (2 * total)
    adjusted_std_error = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total)
    lower_bound = (centre_adjusted_probability - adjusted_std_error) / denominator
    upper_bound = (centre_adjusted_probability + adjusted_std_error) / denominator
    return max(0.0, lower_bound), min(1.0, upper_bound)

def main():
    print("=" * 70)
    print("   EVALUASI KLASIFIKASI GESTUR HANDS UP BERBASIS KNN")
    print(f"   Waktu Eksekusi: {RUN_TIMESTAMP} | SEED: {config.SEED}")
    print("=" * 70)

    simpan_environment()

    # 1. Pemuatan Dataset
    if not os.path.exists(config.FILE_DATASET_CSV):
        catat_log("LOAD_DATA", "dataset.csv", "ERROR: File dataset tidak ditemukan! Jalankan 01_kumpul_data.py dulu.")
        return

    df = pd.read_csv(config.FILE_DATASET_CSV)
    df["kelas"] = df["kelas"].astype(int)
    total_data = len(df)
    catat_log("LOAD_DATA", "Ukuran Dataset", f"{total_data} baris total")

    # Filter data yang memiliki fitur valid (f_L dan f_R tidak NaN)
    df_clean = df.dropna(subset=["f_L", "f_R"]).copy()
    df_clean["f_L"] = df_clean["f_L"].astype(float)
    df_clean["f_R"] = df_clean["f_R"].astype(float)

    # a) Ringkasan Dataset
    ringkasan_kelas = df_clean["kelas"].value_counts().to_dict()
    ringkasan_subjek = df_clean["subjek"].value_counts().to_dict() if "subjek" in df_clean else {}
    ringkasan_jarak = df_clean["jarak_m"].value_counts().to_dict() if "jarak_m" in df_clean else {}

    catat_log("EDA", "Distribusi Kelas", str(ringkasan_kelas))
    catat_log("EDA", "Distribusi Subjek", str(ringkasan_subjek))
    catat_log("EDA", "Distribusi Jarak", str(ringkasan_jarak))

    # Simpan Tabel Ringkasan Dataset
    df_ringkasan = pd.DataFrame([
        {"Kategori": "Total Data", "Detail": "Semua Sampel", "Jumlah": len(df_clean)},
        {"Kategori": "Kelas", "Detail": "1 (Hands Up)", "Jumlah": ringkasan_kelas.get(1, 0)},
        {"Kategori": "Kelas", "Detail": "0 (No Gesture)", "Jumlah": ringkasan_kelas.get(0, 0)},
    ])
    for s_name, s_count in ringkasan_subjek.items():
        df_ringkasan = pd.concat([df_ringkasan, pd.DataFrame([{"Kategori": "Subjek", "Detail": str(s_name), "Jumlah": s_count}])], ignore_index=True)
    for j_val, j_count in ringkasan_jarak.items():
        df_ringkasan = pd.concat([df_ringkasan, pd.DataFrame([{"Kategori": "Jarak (m)", "Detail": f"{j_val}m", "Jumlah": j_count}])], ignore_index=True)

    tabel_ringkasan_path = os.path.join(config.DIR_TABEL, "ringkasan_dataset.csv")
    df_ringkasan.to_csv(tabel_ringkasan_path, index=False)

    # Plot Grafik Batang Distribusi Dataset
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.countplot(data=df_clean, x="kelas", palette=["#e74c3c", "#2ecc71"])
    plt.title("Distribusi Kelas (0: No Gesture, 1: Hands Up)")
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah Sampel")

    plt.subplot(1, 2, 2)
    if "subjek" in df_clean:
        sns.countplot(data=df_clean, x="subjek", palette="Blues_d")
        plt.title("Distribusi Sampel per Subjek")
        plt.xlabel("Subjek")
        plt.ylabel("Jumlah Sampel")
    plt.tight_layout()
    plt.savefig(os.path.join(config.DIR_GRAFIK, "distribusi_dataset.png"), dpi=config.DPI_GRAFIK)
    plt.close()

    # Ekstraksi Fitur dan Label
    X = df_clean[["f_L", "f_R"]].values
    y = df_clean["kelas"].values

    # --------------------------------------------------------------------------
    # b) Pemilihan k Terbaik: Stratified 5-Fold Cross Validation (k = 1..15)
    # --------------------------------------------------------------------------
    print("\n>> Menjalankan Stratified 5-Fold Cross-Validation untuk Tuning k (1..15)...")
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    k_range = list(range(config.KNN_K_MIN, config.KNN_K_MAX + 1))
    k_scores_mean = []
    k_scores_std = []
    tabel_k = []

    for k in k_range:
        fold_scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_cv_train, X_cv_val = X[train_idx], X[val_idx]
            y_cv_train, y_cv_val = y[train_idx], y[val_idx]

            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(X_cv_train, y_cv_train)
            pred = clf.predict(X_cv_val)
            fold_scores.append(accuracy_score(y_cv_val, pred))

        mean_acc = np.mean(fold_scores)
        std_acc = np.std(fold_scores)
        k_scores_mean.append(mean_acc)
        k_scores_std.append(std_acc)
        tabel_k.append({"k": k, "mean_accuracy": mean_acc, "std_accuracy": std_acc})

    df_k_tuning = pd.DataFrame(tabel_k)
    tabel_k_path = os.path.join(config.DIR_TABEL, "pemilihan_k.csv")
    df_k_tuning.to_csv(tabel_k_path, index=False)

    best_k_idx = int(np.argmax(k_scores_mean))
    best_k = k_range[best_k_idx]
    best_k_acc = k_scores_mean[best_k_idx]
    acc_k5 = k_scores_mean[config.DEFAULT_K - 1]

    is_k5_optimal = (best_k == 5) or (abs(best_k_acc - acc_k5) < 0.005)
    catat_log("TUNING_K", f"Rentang k={config.KNN_K_MIN}..{config.KNN_K_MAX}",
              f"k Terbaik = {best_k} (Akurasi CV: {best_k_acc*100:.2f}%). Akurasi k=5: {acc_k5*100:.2f}%. k=5 optimal: {is_k5_optimal}")

    # Plot Hasil Tuning k
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, k_scores_mean, marker='o', color='#2980b9', lw=2, label="Mean Accuracy")
    plt.fill_between(k_range,
                     np.array(k_scores_mean) - np.array(k_scores_std),
                     np.array(k_scores_mean) + np.array(k_scores_std),
                     alpha=0.2, color='#2980b9', label="±1 Std Dev")
    plt.axvline(x=5, color='orange', linestyle='--', label=f"k=5 (Default, Acc={acc_k5*100:.1f}%)")
    plt.axvline(x=best_k, color='green', linestyle=':', label=f"Best k={best_k} (Acc={best_k_acc*100:.1f}%)")
    plt.title("Pemilihan Parameter k pada KNN (Stratified 5-Fold CV)")
    plt.xlabel("Nilai k (Jumlah Tetangga)")
    plt.ylabel("Akurasi Rata-rata")
    plt.xticks(k_range)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.DIR_GRAFIK, "pemilihan_k_cv.png"), dpi=config.DPI_GRAFIK)
    plt.close()

    # --------------------------------------------------------------------------
    # d) Dua Skema Pembagian Data (Train-Test Split)
    # --------------------------------------------------------------------------
    print("\n>> Menguji Dua Skema Pembagian Data...")

    # Skema 1: Acak Stratified 80/20 (Sesuai Jurnal Utama)
    X_train_s1, X_test_s1, y_train_s1, y_test_s1, idx_tr_s1, idx_ts_s1 = train_test_split(
        X, y, df_clean.index, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED
    )

    knn_model = KNeighborsClassifier(n_neighbors=config.DEFAULT_K)
    knn_model.fit(X_train_s1, y_train_s1)
    y_pred_s1 = knn_model.predict(X_test_s1)

    acc_s1 = accuracy_score(y_test_s1, y_pred_s1)
    f1_s1 = f1_score(y_test_s1, y_pred_s1, average="weighted")
    catat_log("SPLIT_SKEMA_1", "Stratified 80/20 (seed=42)", f"Akurasi: {acc_s1*100:.2f}%, F1-Score: {f1_s1:.4f}")

    # Skema 2: Split Berdasarkan Subjek / Blok Sesi (Mencegah Kebocoran Data Sesi)
    skema_2_berhasil = False
    acc_s2, f1_s2 = None, None
    penjelasan_selisih = ""

    unique_subjek = df_clean["subjek"].unique() if "subjek" in df_clean else []
    if len(unique_subjek) >= 2:
        # Split train subjek[0..n-2], test subjek[last]
        test_subjek_target = unique_subjek[-1]
        train_mask = df_clean["subjek"] != test_subjek_target
        test_mask = df_clean["subjek"] == test_subjek_target

        X_train_s2 = df_clean.loc[train_mask, ["f_L", "f_R"]].values
        y_train_s2 = df_clean.loc[train_mask, "kelas"].values
        X_test_s2 = df_clean.loc[test_mask, ["f_L", "f_R"]].values
        y_test_s2 = df_clean.loc[test_mask, "kelas"].values

        if len(y_test_s2) > 0 and len(np.unique(y_test_s2)) >= 1:
            knn_s2 = KNeighborsClassifier(n_neighbors=config.DEFAULT_K)
            knn_s2.fit(X_train_s2, y_train_s2)
            y_pred_s2 = knn_s2.predict(X_test_s2)

            acc_s2 = accuracy_score(y_test_s2, y_pred_s2)
            f1_s2 = f1_score(y_test_s2, y_pred_s2, average="weighted", zero_division=0)
            skema_2_berhasil = True
            selisih = (acc_s1 - acc_s2) * 100
            penjelasan_selisih = f"Akurasi Skema 2 ({acc_s2*100:.2f}%) memiliki selisih {selisih:.2f}% dibanding Skema 1 ({acc_s1*100:.2f}%). Penurunan kecil wajar karena subjek uji ({test_subjek_target}) belum pernah dilihat pada data latih (uji generalisasi postur baru)."
            catat_log("SPLIT_SKEMA_2", f"Leave-One-Subject-Out ({test_subjek_target})", f"Akurasi: {acc_s2*100:.2f}%, F1: {f1_s2:.4f}. {penjelasan_selisih}")
    else:
        # Jika baru 1 subjek, split berdasarkan sesi/urutan kronologis (50% awal vs 50% akhir)
        n_mid = int(len(df_clean) * 0.8)
        X_train_s2 = X[:n_mid]
        y_train_s2 = y[:n_mid]
        X_test_s2 = X[n_mid:]
        y_test_s2 = y[n_mid:]

        knn_s2 = KNeighborsClassifier(n_neighbors=config.DEFAULT_K)
        knn_s2.fit(X_train_s2, y_train_s2)
        y_pred_s2 = knn_s2.predict(X_test_s2)
        acc_s2 = accuracy_score(y_test_s2, y_pred_s2)
        f1_s2 = f1_score(y_test_s2, y_pred_s2, average="weighted", zero_division=0)
        skema_2_berhasil = True
        selisih = (acc_s1 - acc_s2) * 100
        penjelasan_selisih = f"Split sekuensial kronologis waktu (80% awal latih, 20% akhir uji) menghasilkan akurasi {acc_s2*100:.2f}% (selisih {selisih:.2f}% dari acak stratified)."
        catat_log("SPLIT_SKEMA_2", "Split Sekuensial Sesi Waktu", f"Akurasi: {acc_s2*100:.2f}%, F1: {f1_s2:.4f}. {penjelasan_selisih}")

    # Simpan Perbandingan Skema
    df_skema = pd.DataFrame([
        {"Skema": "1. Stratified Random 80/20 (Sesuai Jurnal)", "Akurasi": acc_s1, "F1_Score": f1_s1, "Keterangan": "Distribusi kelas seimbang pada train & test"},
        {"Skema": "2. Split Blok Sesi / Subjek Independen", "Akurasi": acc_s2, "F1_Score": f1_s2, "Keterangan": penjelasan_selisih}
    ])
    df_skema.to_csv(os.path.join(config.DIR_TABEL, "perbandingan_skema_split.csv"), index=False)

    # --------------------------------------------------------------------------
    # c) Kurva Jumlah Data Latih (Learning Curve: 50..320 sampel, 10 seed)
    # --------------------------------------------------------------------------
    print("\n>> Menguji Kurva Jumlah Data Latih (Learning Curve)...")
    learning_results = []
    max_train_available = len(X_train_s1)

    steps = [s for s in config.LEARNING_CURVE_STEPS if s <= max_train_available]
    if not steps or steps[-1] != max_train_available:
        steps.append(max_train_available)

    for n_samples in steps:
        acc_trials = []
        f1_trials = []
        for trial_seed in range(config.LEARNING_CURVE_TRIALS):
            # Sampling acak bertingkat sejumlah n_samples dari data latih
            sample_indices = np.random.RandomState(trial_seed + 100).choice(
                len(X_train_s1), size=n_samples, replace=False
            )
            X_sub = X_train_s1[sample_indices]
            y_sub = y_train_s1[sample_indices]

            clf_lc = KNeighborsClassifier(n_neighbors=config.DEFAULT_K)
            clf_lc.fit(X_sub, y_sub)
            pred_sub = clf_lc.predict(X_test_s1)

            acc_trials.append(accuracy_score(y_test_s1, pred_sub))
            f1_trials.append(f1_score(y_test_s1, pred_sub, average="weighted", zero_division=0))

        m_acc, s_acc = np.mean(acc_trials), np.std(acc_trials)
        m_f1, s_f1 = np.mean(f1_trials), np.std(f1_trials)

        learning_results.append({
            "n_samples": n_samples,
            "mean_accuracy": m_acc,
            "std_accuracy": s_acc,
            "mean_f1": m_f1,
            "std_f1": s_f1
        })
        catat_log("LEARNING_CURVE", f"Sampel Latih={n_samples}", f"Akurasi: {m_acc*100:.2f}% (±{s_acc*100:.2f}%), F1: {m_f1:.4f}")

    df_lc = pd.DataFrame(learning_results)
    df_lc.to_csv(os.path.join(config.DIR_TABEL, "learning_curve.csv"), index=False)

    # Plot Learning Curve
    plt.figure(figsize=(8, 5))
    plt.plot(df_lc["n_samples"], df_lc["mean_accuracy"], marker='s', color='#8e44ad', lw=2, label="Akurasi Uji")
    plt.fill_between(df_lc["n_samples"],
                     df_lc["mean_accuracy"] - df_lc["std_accuracy"],
                     df_lc["mean_accuracy"] + df_lc["std_accuracy"],
                     alpha=0.2, color='#8e44ad', label="±1 Std Dev")
    plt.title("Pengaruh Jumlah Data Latih terhadap Akurasi Uji (10 Pengulangan Seed)")
    plt.xlabel("Jumlah Sampel Latih (N)")
    plt.ylabel("Akurasi pada Data Uji (Held-out 20%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.DIR_GRAFIK, "learning_curve.png"), dpi=config.DPI_GRAFIK)
    plt.close()

    # --------------------------------------------------------------------------
    # e) Metrik Evaluasi Komprehensif (Skema 1: Test Set 20%)
    # --------------------------------------------------------------------------
    print("\n>> Menghitung Seluruh Metrik Evaluasi pada Test Set...")
    cm = confusion_matrix(y_test_s1, y_pred_s1, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_test_s1, y_pred_s1)
    prec_0 = precision_score(y_test_s1, y_pred_s1, pos_label=0, zero_division=0)
    prec_1 = precision_score(y_test_s1, y_pred_s1, pos_label=1, zero_division=0)
    rec_0 = recall_score(y_test_s1, y_pred_s1, pos_label=0, zero_division=0)
    rec_1 = recall_score(y_test_s1, y_pred_s1, pos_label=1, zero_division=0)
    f1_0 = f1_score(y_test_s1, y_pred_s1, pos_label=0, zero_division=0)
    f1_1 = f1_score(y_test_s1, y_pred_s1, pos_label=1, zero_division=0)

    prec_w = precision_score(y_test_s1, y_pred_s1, average="weighted", zero_division=0)
    rec_w = recall_score(y_test_s1, y_pred_s1, average="weighted", zero_division=0)
    f1_w = f1_score(y_test_s1, y_pred_s1, average="weighted", zero_division=0)

    correct_predictions = int(tp + tn)
    total_test = int(len(y_test_s1))
    wilson_lower, wilson_upper = wilson_score_interval(correct_predictions, total_test)

    catat_log("METRIK", "Akurasi", f"{acc*100:.2f}% (Wilson 95% CI: [{wilson_lower*100:.2f}%, {wilson_upper*100:.2f}%])")
    catat_log("METRIK", "Confusion Matrix", f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    catat_log("METRIK", "Hands Up (Kelas 1)", f"Precision: {prec_1:.4f}, Recall: {rec_1:.4f}, F1: {f1_1:.4f}")
    catat_log("METRIK", "No Gesture (Kelas 0)", f"Precision: {prec_0:.4f}, Recall: {rec_0:.4f}, F1: {f1_0:.4f}")

    # Simpan Tabel Metrik Utama
    df_metrik = pd.DataFrame([
        {"Metrik": "Akurasi Keseluruhan", "Nilai": acc, "Keterangan": f"Wilson 95% CI: [{wilson_lower:.4f}, {wilson_upper:.4f}]"},
        {"Metrik": "True Positives (TP)", "Nilai": tp, "Keterangan": "Hands Up benar terdeteksi"},
        {"Metrik": "True Negatives (TN)", "Nilai": tn, "Keterangan": "No Gesture benar terdeteksi"},
        {"Metrik": "False Positives (FP)", "Nilai": fp, "Keterangan": "No Gesture salah diprediksi Hands Up"},
        {"Metrik": "False Negatives (FN)", "Nilai": fn, "Keterangan": "Hands Up gagal terdeteksi"},
        {"Metrik": "Precision - Kelas 1 (Hands Up)", "Nilai": prec_1, "Keterangan": ""},
        {"Metrik": "Recall - Kelas 1 (Hands Up)", "Nilai": rec_1, "Keterangan": ""},
        {"Metrik": "F1-Score - Kelas 1 (Hands Up)", "Nilai": f1_1, "Keterangan": ""},
        {"Metrik": "Precision - Kelas 0 (No Gesture)", "Nilai": prec_0, "Keterangan": ""},
        {"Metrik": "Recall - Kelas 0 (No Gesture)", "Nilai": rec_0, "Keterangan": ""},
        {"Metrik": "F1-Score - Kelas 0 (No Gesture)", "Nilai": f1_0, "Keterangan": ""},
        {"Metrik": "Weighted Precision", "Nilai": prec_w, "Keterangan": "Rata-rata terbobot"},
        {"Metrik": "Weighted Recall", "Nilai": rec_w, "Keterangan": "Rata-rata terbobot"},
        {"Metrik": "Weighted F1-Score", "Nilai": f1_w, "Keterangan": "Rata-rata terbobot"},
    ])
    df_metrik.to_csv(os.path.join(config.DIR_TABEL, "metrik_utama.csv"), index=False)

    # Plot Confusion Matrix (Raw & Normalized)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["No Gesture (0)", "Hands Up (1)"],
                yticklabels=["No Gesture (0)", "Hands Up (1)"])
    axes[0].set_title(f"Confusion Matrix (Total Sampel: {total_test})")
    axes[0].set_xlabel("Prediksi")
    axes[0].set_ylabel("Aktual")

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens", ax=axes[1],
                xticklabels=["No Gesture (0)", "Hands Up (1)"],
                yticklabels=["No Gesture (0)", "Hands Up (1)"])
    axes[1].set_title("Confusion Matrix Ternormalisasi")
    axes[1].set_xlabel("Prediksi")
    axes[1].set_ylabel("Aktual")

    plt.tight_layout()
    plt.savefig(os.path.join(config.DIR_GRAFIK, "confusion_matrix.png"), dpi=config.DPI_GRAFIK)
    plt.close()

    # --------------------------------------------------------------------------
    # f) Analisis Akurasi per Rentang Jarak
    # --------------------------------------------------------------------------
    df_test_full = df_clean.loc[idx_ts_s1].copy()
    df_test_full["prediksi"] = y_pred_s1
    df_test_full["benar"] = (df_test_full["kelas"] == df_test_full["prediksi"]).astype(int)

    jarak_analisis = []
    if "jarak_m" in df_test_full and df_test_full["jarak_m"].notna().sum() > 0:
        unique_distances = sorted(df_test_full["jarak_m"].unique())
        for dist in unique_distances:
            sub = df_test_full[df_test_full["jarak_m"] == dist]
            acc_d = accuracy_score(sub["kelas"], sub["prediksi"])
            rec_d = recall_score(sub["kelas"], sub["prediksi"], pos_label=1, zero_division=0)
            prec_d = precision_score(sub["kelas"], sub["prediksi"], pos_label=1, zero_division=0)
            f1_d = f1_score(sub["kelas"], sub["prediksi"], pos_label=1, zero_division=0)
            jarak_analisis.append({
                "Jarak_Meter": f"{dist}m",
                "Jumlah_Sampel": len(sub),
                "Akurasi": acc_d,
                "Precision_HandsUp": prec_d,
                "Recall_HandsUp": rec_d,
                "F1_HandsUp": f1_d
            })
    else:
        jarak_analisis.append({
            "Jarak_Meter": "Semua Jarak",
            "Jumlah_Sampel": len(df_test_full),
            "Akurasi": acc,
            "Precision_HandsUp": prec_1,
            "Recall_HandsUp": rec_1,
            "F1_HandsUp": f1_1
        })

    df_jarak = pd.DataFrame(jarak_analisis)
    df_jarak.to_csv(os.path.join(config.DIR_TABEL, "metrik_per_jarak.csv"), index=False)
    catat_log("JARAK", "Evaluasi Per Titik Jarak", df_jarak.to_dict(orient="records"))

    # Plot Grafik Per Jarak
    if len(df_jarak) > 1:
        plt.figure(figsize=(9, 4.5))
        sns.barplot(data=df_jarak, x="Jarak_Meter", y="Akurasi", color="#2980b9")
        plt.title("Akurasi Klasifikasi Gestur Hands Up pada Setiap Jarak (0.5m - 3.5m)")
        plt.ylim(0, 1.1)
        plt.ylabel("Akurasi (0 - 1.0)")
        plt.xlabel("Jarak Pengujian (Meter)")
        for idx_b, row_b in df_jarak.iterrows():
            plt.text(idx_b, row_b["Akurasi"] + 0.02, f"{row_b['Akurasi']*100:.1f}%",
                     ha="center", fontsize=9, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(config.DIR_GRAFIK, "akurasi_per_jarak.png"), dpi=config.DPI_GRAFIK)
        plt.close()

    # --------------------------------------------------------------------------
    # g) Benchmark Waktu Inferensi (ms per prediksi, 1000 iterasi)
    # --------------------------------------------------------------------------
    print(f"\n>> Menjalankan Benchmark Waktu Inferensi ({config.BENCHMARK_ITERATIONS} iterasi)...")
    single_sample = X_test_s1[0:1]

    # Warmup
    for _ in range(50):
        knn_model.predict(single_sample)

    latencies = []
    for _ in range(config.BENCHMARK_ITERATIONS):
        t0 = time.perf_counter()
        knn_model.predict(single_sample)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

    avg_ms = np.mean(latencies)
    std_ms = np.std(latencies)
    fps_est = 1000.0 / avg_ms if avg_ms > 0 else 0

    catat_log("BENCHMARK", f"{config.BENCHMARK_ITERATIONS} Iterasi",
              f"Rata-rata: {avg_ms:.4f} ms (±{std_ms:.4f} ms) | Est. Throughput: {fps_est:.1f} FPS")

    pd.DataFrame([{
        "Pengulangan": config.BENCHMARK_ITERATIONS,
        "Waktu_Rata_Rata_ms": avg_ms,
        "Std_Dev_ms": std_ms,
        "Estimasi_FPS": fps_est
    }]).to_csv(os.path.join(config.DIR_TABEL, "inferensi_waktu.csv"), index=False)

    # --------------------------------------------------------------------------
    # h) Grafik Sebaran Fitur & Decision Boundary KNN
    # --------------------------------------------------------------------------
    print("\n>> Menggambar Batas Keputusan (Decision Boundary) KNN...")
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', alpha=0.8)
    plt.title(f"Batas Keputusan KNN (k={config.DEFAULT_K}) pada Ruang Fitur (f_L vs f_R)")
    plt.xlabel("f_L: Selisih Vertikal Pergelangan-Bahu Kiri (y_LW - y_LS)")
    plt.ylabel("f_R: Selisih Vertikal Pergelangan-Bahu Kanan (y_RW - y_RS)")
    plt.axhline(0, color='gray', linestyle=':', alpha=0.6)
    plt.axvline(0, color='gray', linestyle=':', alpha=0.6)
    legend1 = plt.legend(*scatter.legend_elements(), title="Kelas (0: No, 1: HandsUp)", loc="lower left")
    plt.gca().add_artist(legend1)
    plt.tight_layout()
    plt.savefig(os.path.join(config.DIR_GRAFIK, "decision_boundary.png"), dpi=config.DPI_GRAFIK)
    plt.close()

    # --------------------------------------------------------------------------
    # i) Simpan Model Terlatih ke model/knn_k5.pkl
    # --------------------------------------------------------------------------
    joblib.dump(knn_model, config.FILE_MODEL_KNN)
    catat_log("SAVE_MODEL", "KNeighborsClassifier(k=5)", f"Model disimpan di {config.FILE_MODEL_KNN}")

    # Simpan hasil test set teranotasi untuk digunakan oleh 03_visualisasi.py
    file_test_annotated = os.path.join(config.DIR_TABEL, "test_set_annotated.csv")
    df_test_full.to_csv(file_test_annotated, index=False)
    catat_log("SAVE_TEST", "Test Set Anotasi", f"Disimpan di {file_test_annotated}")

    simpan_log_csv()

    print("\n" + "=" * 70)
    print("   RINGKASAN AKHIR PELATIHAN & EVALUASI")
    print("=" * 70)
    print(f"Total Sampel Data       : {len(df_clean)}")
    print(f"Nilai k Optimal (CV)    : k={best_k} (Akurasi: {best_k_acc*100:.2f}%)")
    print(f"Akurasi Test Set (k=5)  : {acc*100:.2f}% (Wilson 95%: [{wilson_lower*100:.2f}%, {wilson_upper*100:.2f}%])")
    print(f"Confusion Matrix        : TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Waktu Inferensi Rata2   : {avg_ms:.4f} ms per frame (~{fps_est:.0f} FPS)")
    print(f"Lokasi Model Disimpan   : {config.FILE_MODEL_KNN}")
    print(f"Grafik & Tabel Hasil    : {config.DIR_HASIL}")
    print("=" * 70)

if __name__ == "__main__":
    main()

