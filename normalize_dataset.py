import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def normalize_coffee_dataset(input_csv, output_csv):
    # 1. Membaca dataset hasil ekstraksi citra
    df = pd.read_csv(input_csv)
    
    print(f"Membaca {len(df)} baris data dari {input_csv}...")
    
    # 2. Kalibrasi Fisik (Mengoreksi diameter agar realistis di kisaran 5-9 mm)
    # Kita bagi dengan 3 karena skala sebelumnya (0.15) terlalu besar. 
    # Dengan membaginya, diameter berkisar di angka realistis ~6.3 s/d ~8.0 mm.
    df['diameter_rata_rata'] = round(df['diameter_rata_rata'] / 3.0, 2)
    
    # 3. Definisikan fitur yang akan dinormalisasi
    features = ['persentase_kecacatan', 'diameter_rata_rata', 'kadar_air']
    
    # 4. Melakukan Normalisasi Statistik (Min-Max Scaling ke rentang 0 s/d 1)
    # Ini sangat direkomendasikan untuk melatih model klasifikasi Machine Learning.
    scaler_minmax = MinMaxScaler()
    minmax_scaled = scaler_minmax.fit_transform(df[features])
    
    # 5. Melakukan Standarisasi (Z-score Scaling: rata-rata=0, std=1)
    # Kami buatkan kedua versi agar Anda bisa memilih tipe normalisasi yang diinginkan dosen.
    scaler_std = StandardScaler()
    std_scaled = scaler_std.fit_transform(df[features])
    
    # 6. Menambahkan kolom baru hasil normalisasi ke DataFrame
    df['persentase_kecacatan_minmax'] = round(pd.Series(minmax_scaled[:, 0]), 4)
    df['diameter_rata_rata_minmax'] = round(pd.Series(minmax_scaled[:, 1]), 4)
    df['kadar_air_minmax'] = round(pd.Series(minmax_scaled[:, 2]), 4)
    
    df['persentase_kecacatan_std'] = round(pd.Series(std_scaled[:, 0]), 4)
    df['diameter_rata_rata_std'] = round(pd.Series(std_scaled[:, 1]), 4)
    df['kadar_air_std'] = round(pd.Series(std_scaled[:, 2]), 4)
    
    # Menyimpan DataFrame baru yang sudah terkalibrasi dan dinormalisasi
    df.to_csv(output_csv, index=False)
    print(f"Dataset berhasil dikalibrasi dan dinormalisasi!")
    print(f"Hasil disimpan di: {output_csv}")

if __name__ == "__main__":
    input_file = "./Coffee_Extracted_From_Images.csv"
    output_file = "./Coffee_Normalized_Dataset.csv"
    
    try:
        normalize_coffee_dataset(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' tidak ditemukan.")
        print("Silakan jalankan 'extract_features.py' terlebih dahulu untuk membuat file tersebut.")
    except Exception as e:
        print(f"Terjadi kesalahan: {str(e)}")
