import pandas as pd
import mlflow
import mlflow.sklearn

# PENTING: Sesuaikan path dataset agar terbaca oleh MLProject
dataset_path = "dataset_preprocessing/cleaned_diabetes.csv"
df = pd.read_csv(dataset_path)

with mlflow.start_run():
    # ... (kode pelatihan model Anda)
    mlflow.sklearn.autolog()
    print("Re-training model selesai!")

# Triggering GitHub Actions re-run