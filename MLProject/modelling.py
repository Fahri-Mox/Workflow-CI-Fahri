import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def train_basic():
    # 1. Muat Data
    # Menggunakan relative path karena folder dataset_preprocessing 
    # berada tepat di samping file modelling.py ini
    dataset_path = 'dataset_preprocessing/cleaned_diabetes.csv'
    
    if not os.path.exists(dataset_path):
        print(f"Error: File tidak ditemukan di {os.path.abspath(dataset_path)}")
        exit(1) # Sinyal error untuk GitHub Actions

    try:
        df = pd.read_csv(dataset_path)
        print("Dataset berhasil dimuat!")
        
        X = df.drop(columns=['Outcome'])
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. Setup MLflow
        # Di GitHub Actions, biarkan default atau hapus set_experiment 
        # agar tidak bentrok dengan path lokal
        mlflow.sklearn.autolog() 

        with mlflow.start_run(run_name="CI_Retraining_Run"):
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            print(f"Model Retraining Selesai. Akurasi: {acc}")

    except Exception as e:
        print(f"Terjadi kesalahan saat eksekusi: {e}")
        exit(1)

if __name__ == "__main__":
    train_basic()