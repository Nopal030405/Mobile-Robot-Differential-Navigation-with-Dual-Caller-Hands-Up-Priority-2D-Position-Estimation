import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Perceptron
import joblib

# Load Dataset Pose
DATASET_FILE = "dataset_gestur_pose.csv"

try:
    df = pd.read_csv(DATASET_FILE)
    print(f">> Dataset pose berhasil dimuat! Total sampel: {len(df)} baris.")
except FileNotFoundError:
    print(f">> [ERROR] File '{DATASET_FILE}' tidak ditemukan! Jalankan collect_data_2.py terlebih dahulu.")
    exit()

X = df[['left_wrist_shoulder_y', 'right_wrist_shoulder_y']].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(),
    "Perceptron": Perceptron(random_state=42)
}

results = []
confusion_matrices = {}

print("\n" + "="*60)
print("     EVALUASI MODEL POSE (HANDS UP vs NO GESTURE)")
print("="*60)

best_knn_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })
    
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)
    
    if name == "K-Nearest Neighbors":
        best_knn_model = model

    print(f"\n--- Metode: {name} ---")
    print(f"  • Akurasi  : {acc:.4f}")
    print(f"  • Presisi  : {prec:.4f}")
    print(f"  • Recall   : {rec:.4f}")
    print(f"  • F1-Score : {f1:.4f}")
    print("-" * 35)

results_df = pd.DataFrame(results)

# --- 1. MEMBUAT GRAFIK PERBANDINGAN 4 METRIK (Subplots 2x2) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Analisis Perbandingan Kinerja Model Klasifikasi Pose (Hands Up)', fontsize=16, fontweight='bold')

metrics = [('Accuracy', 'Akurasi', 'skyblue'), 
           ('Precision', 'Presisi', 'lightgreen'), 
           ('Recall', 'Recall', 'orange'), 
           ('F1-Score', 'F1-Score', 'salmon')]

for idx, (col, title, color) in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    bars = ax.bar(results_df['Model'], results_df[col], color=color, edgecolor='black', alpha=0.8)
    ax.set_title(f'Perbandingan Berdasarkan {title}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Nilai Skor (0.0 - 1.0)', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', rotation=10)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
chart_filename = "grafik_perbandingan_model_pose.png"
plt.savefig(chart_filename, dpi=300)
print(f">> Grafik perbandingan model berhasil disimpan sebagai '{chart_filename}'.")

# --- 2. MEMBUAT GAMBAR CONFUSION MATRIX (Grid 2x2) ---
fig_cm, axes_cm = plt.subplots(2, 2, figsize=(12, 10))
fig_cm.suptitle('Confusion Matrix untuk Keempat Model Klasifikasi Pose', fontsize=16, fontweight='bold')

axes_cm_flat = axes_cm.flatten()
for idx, (name, cm) in enumerate(confusion_matrices.items()):
    ax = axes_cm_flat[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False, 
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Prediksi Model', fontsize=10)
    ax.set_ylabel('Data Aktual (Asli)', fontsize=10)
    ax.xaxis.set_ticklabels(['No Gesture (0)', 'Hands Up (1)'])
    ax.yaxis.set_ticklabels(['No Gesture (0)', 'Hands Up (1)'])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
cm_filename = "confusion_matrix_pose.png"
plt.savefig(cm_filename, dpi=300)
print(f">> Grafik Confusion Matrix Pose disimpan sebagai '{cm_filename}'.")

# Simpan model KNN terbaik agar siap dipakai di eksperimen utama
joblib.dump(best_knn_model, 'model_knn_gesture.pkl')
print(">> Model KNN Pose berhasil disimpan ke 'model_knn_gesture.pkl'")