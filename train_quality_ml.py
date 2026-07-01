import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def train_and_compare_models(dataset_path):
    print(f"Membaca dataset dari: {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # Fitur input (menggunakan Min-Max Scaled features)
    features = ['persentase_kecacatan_minmax', 'diameter_rata_rata_minmax', 'kadar_air_minmax']
    X = df[features]
    y = df['apakah_ekspor']
    
    # Split data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Jumlah Data Latih: {X_train.shape[0]}")
    print(f"Jumlah Data Uji: {X_test.shape[0]}\n")
    
    # Inisialisasi model
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine (SVM)": SVC(probability=True, random_state=42)
    }
    
    best_model_name = None
    best_accuracy = 0.0
    best_model_obj = None
    
    results = []
    
    for name, model in models.items():
        print(f"=== Melatih Model: {name} ===")
        model.fit(X_train, y_train)
        
        # Prediksi data uji
        y_pred = model.predict(X_test)
        
        # Hitung metrik
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(classification_report(y_test, y_pred, target_names=["Lokal", "Ekspor"]))
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })
        
        # Cek jika ini model terbaik
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model_obj = model
            
    # Tampilkan perbandingan metrik dalam bentuk tabel/summary
    summary_df = pd.DataFrame(results)
    print("\n=== RINGKASAN PERBANDINGAN MODEL ===")
    print(summary_df.to_string(index=False))
    
    # Simpan model terbaik ke folder backend/models/
    model_dir = "./backend/models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_save_path = os.path.join(model_dir, "coffee_quality_classifier.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(best_model_obj, f)
        
    print(f"\nModel terbaik: {best_model_name} dengan Akurasi: {best_accuracy:.4f}")
    print(f"Berhasil menyimpan model terbaik ke '{model_save_path}'")

if __name__ == "__main__":
    dataset = "./Coffee_Normalized_Dataset.csv"
    if os.path.exists(dataset):
        train_and_compare_models(dataset)
    else:
        print(f"Error: Berkas '{dataset}' tidak ditemukan. Silakan jalankan 'normalize_dataset.py' terlebih dahulu.")
