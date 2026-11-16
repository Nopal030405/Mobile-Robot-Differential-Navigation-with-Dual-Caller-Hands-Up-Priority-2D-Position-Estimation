# ==============================================================================
# 04_pembanding.py - Lampiran Pembanding Model (GNB, Logistic Regression, Perceptron)
# ==============================================================================

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

import config

def uji_mcnemar(y_true, y_pred_knn, y_pred_lain):
    """
    Menghitung uji McNemar berpasangan antara KNN dan model lain.
    b: KNN benar, model lain salah
    c: KNN salah, model lain benar
    Statistik Chi-squared = (|b - c| - 1)^2 / (b + c)
    """
    b = 0  # KNN Benar, Lain Salah
    c = 0  # KNN Salah, Lain Benar
    a = 0  # Keduanya Benar
    d = 0  # Keduanya Salah

    for yt, y_knn, y_other in zip(y_true, y_pred_knn, y_pred_lain):
        knn_correct = (yt == y_knn)
        other_correct = (yt == y_other)
        if knn_correct and not other_correct:
            b += 1
        elif not knn_correct and other_correct:
            c += 1
        elif knn_correct and other_correct:
            a += 1
        else:
            d += 1

    total_discordant = b + c
    if total_discordant == 0:
        stat_chi2 = 0.0
        p_val = 1.0
    else:
        # Koreksi kontinuitas Edwards
        stat_chi2 = ((abs(b - c) - 1.0) ** 2) / float(total_discordant)
        p_val = 1.0 - stats.chi2.cdf(stat_chi2, df=1)

    signifikan = p_val < 0.05
    return {
        "b (KNN Benar, Model Salah)": b,
        "c (KNN Salah, Model Benar)": c,
        "Chi2_Statistic": stat_chi2,
        "p_value": p_val,
        "Signifikan_alpha_0.05": signifikan
    }

def main():
    print("=" * 70)
    print("   LAMPIRAN: EVALUASI MODEL PEMBANDING & UJI MCNEMAR")
    print("   (Data khusus lampiran naskah, bukan badan utama jurnal)")
    print("=" * 70)

    if not os.path.exists(config.FILE_DATASET_CSV):
        print(">> [ERROR] File dataset.csv tidak ditemukan!")
        return

    df = pd.read_csv(config.FILE_DATASET_CSV)
    df_clean = df.dropna(subset=["f_L", "f_R"]).copy()
    X = df_clean[["f_L", "f_R"]].values
    y = df_clean["kelas"].astype(int).values

    # Pembagian data sama persis dengan 02_latih_evaluasi.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED
    )

    models = {
        "K-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=config.DEFAULT_K),
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(random_state=config.SEED),
        "Perceptron": Perceptron(random_state=config.SEED)
    }

    hasil_evaluasi = []
    prediksi_models = {}

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        prediksi_models[name] = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        hasil_evaluasi.append({
            "Model": name,
            "Akurasi": acc,
            "Precision_Weighted": prec_w,
            "Recall_Weighted": rec_w,
            "F1_Weighted": f1_w,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        })

    df_hasil = pd.DataFrame(hasil_evaluasi)
    file_tabel_pembanding = os.path.join(config.DIR_TABEL, "lampiran_evaluasi_model_pembanding.csv")
    df_hasil.to_csv(file_tabel_pembanding, index=False)
    print(f"\n>> Tabel Evaluasi Model Pembanding disimpan di: {file_tabel_pembanding}")
    print(df_hasil[["Model", "Akurasi", "Precision_Weighted", "Recall_Weighted", "F1_Weighted"]].to_string(index=False))

    # Uji McNemar KNN vs Tiap Model Lain
    print("\n>> Menghitung Uji McNemar (KNN vs Model Pembanding)...")
    y_pred_knn = prediksi_models["K-Nearest Neighbors (k=5)"]
    mcnemar_rows = []

    for name, y_pred_other in prediksi_models.items():
        if name == "K-Nearest Neighbors (k=5)":
            continue
        res_mc = uji_mcnemar(y_test, y_pred_knn, y_pred_other)
        mcnemar_rows.append({
            "Perbandingan": f"KNN vs {name}",
            **res_mc
        })

    df_mcnemar = pd.DataFrame(mcnemar_rows)
    file_mcnemar = os.path.join(config.DIR_TABEL, "lampiran_mcnemar_test.csv")
    df_mcnemar.to_csv(file_mcnemar, index=False)
    print(f">> Hasil Uji McNemar disimpan di: {file_mcnemar}")
    print(df_mcnemar.to_string(index=False))

    # Plot Grafik Perbandingan Akurasi & F1-Score
    plt.figure(figsize=(9, 5))
    x_indices = np.arange(len(df_hasil))
    bar_width = 0.35

    plt.bar(x_indices - bar_width/2, df_hasil["Akurasi"] * 100, width=bar_width, label="Akurasi (%)", color="#3498db")
    plt.bar(x_indices + bar_width/2, df_hasil["F1_Weighted"] * 100, width=bar_width, label="F1-Score Weighted (%)", color="#2ecc71")

    plt.title("Perbandingan Performa Model Klasifikasi Gestur (Lampiran)")
    plt.xticks(x_indices, df_hasil["Model"], rotation=15, ha="right")
    plt.ylabel("Persentase (%)")
    plt.ylim(0, 110)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.legend()
    plt.tight_layout()
    grafik_path = os.path.join(config.DIR_GRAFIK, "lampiran_grafik_pembanding.png")
    plt.savefig(grafik_path, dpi=config.DPI_GRAFIK)
    plt.close()
    print(f">> Grafik pembanding disimpan di: {grafik_path}")

    print("\n" + "=" * 70)
    print("   SELESAI MENGEKSEKUSI PENGUJIAN LAMPIRAN PEMBANDING")
    print("=" * 70)

if __name__ == "__main__":
    main()

