import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

def train_basic():
    # 1. Muat Data
    dataset_path = 'dataset_preprocessing/cleaned_diabetes.csv'
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset tidak ditemukan di {dataset_path}")
        exit(1)

    try:
        df = pd.read_csv(dataset_path)
        print("Dataset berhasil dimuat!")
        
        X = df.drop(columns=['Outcome'])
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. Pelatihan (Tanpa with mlflow.start_run)
        # Aktifkan autolog, MLflow akan otomatis mencatat ke Run yang sedang aktif
        mlflow.sklearn.autolog() 
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        print("Re-training model sukses dilakukan!")

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
        exit(1)

if __name__ == "__main__":
    train_basic()