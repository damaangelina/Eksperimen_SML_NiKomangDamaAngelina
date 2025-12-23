import pandas as pd
import os
from sklearn.preprocessing import StandardScaler


def run_automation():
    # Base directory (root project)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ===============================
    # 1. Load RAW Dataset 
    # ===============================
    raw_path = os.path.join(base_dir, "cancer_raw", "breast_cancer.csv")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            "File raw dataset tidak ditemukan! Pastikan ada di folder cancer_raw/"
        )

    df = pd.read_csv(raw_path)

    # ===============================
    # 2. Data Preprocessing
    # ===============================
    # Drop duplicates
    df = df.drop_duplicates()

    # Pisahkan fitur dan target
    X = df.drop(columns=["target"])
    y = df["target"]

    # Standarisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Gabungkan kembali
    df_final = pd.DataFrame(X_scaled, columns=X.columns)
    df_final["target"] = y.values

    # ===============================
    # 3. Simpan Dataset Preprocessing
    # ===============================
    output_dir = os.path.join(
        base_dir,
        "preprocessing",
        "cancer_preprocessing"
    )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "breast_cancer_preprocessing.csv"
    )

    df_final.to_csv(output_path, index=False)

    # ===============================
    # 4. Logs
    # ===============================
    print("=== PREPROCESSING BERHASIL ===")
    print(f"Raw dataset dibaca dari : {raw_path}")
    print(f"Preprocessed dataset   : {output_path}")
    print(f"Shape data akhir       : {df_final.shape}")


if __name__ == "__main__":
    run_automation()