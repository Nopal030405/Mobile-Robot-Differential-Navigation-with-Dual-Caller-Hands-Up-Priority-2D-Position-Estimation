import pandas as pd
import numpy as np

df = pd.read_csv("analisis_knn_sistem/raw_inferensi_foto.csv")

print("=== DISTRIBUSI STATUS MESIN & DETEKSI ===")
print(df.groupby(["status_mesin", "label_knn"]).size())

print("\n=== DISTRIBUSI f_L, f_R ===")
valid = df[df["f_L"].notna()]
print(f"Total instances dengan landmark: {len(valid)}")
print(f"Total instances tanpa landmark (TIDAK DAPAT DINILAI): {len(df) - len(valid)}")

# Check instances where label_knn is HANDS_UP vs NO_GESTURE
print("\n=== LABEL KNN COUNTS ===")
print(valid["label_knn"].value_counts())

