import os
import cv2
import numpy as np
import pandas as pd

def extract_coffee_features(image_path):
    """
    Ekstraksi fitur fisik biji kopi menggunakan Pengolahan Citra Digital (OpenCV):
    1. Diameter Rata-rata (diestimasi dari area kontur)
    2. Persentase Kecacatan (diestimasi dari bentuk kontur yang tidak bulat/pecah)
    3. Kadar Air (diestimasi dari nilai intensitas warna/kecerahan biji)
    """
    # Membaca gambar
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # 1. Konversi ke Grayscale dan Thresholding untuk Segmentasi Biji Kopi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Thresholding Otsu untuk memisahkan biji kopi dari background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Opsional: Operasi morfologi untuk menghilangkan noise kecil
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Mendeteksi kontur (butiran biji kopi)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    diameters = []
    defects_count = 0
    total_beans = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100: # Abaikan noise kecil
            continue
            
        total_beans += 1
        
        # Hitung diameter setara (equivalent diameter) dalam pixel
        equiv_diameter = 2 * np.sqrt(area / np.pi)
        
        # Konversi skala pixel ke mm (asumsi faktor skala misalnya 0.15 mm per pixel)
        diameter_mm = equiv_diameter * 0.15
        diameters.append(diameter_mm)
        
        # Hitung kebulatan (circularity) untuk mendeteksi kecacatan (pecah/berlubang)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            # Jika circularity jauh dari 1 (misal < 0.75), dikategorikan sebagai biji cacat/pecah
            if circularity < 0.75:
                defects_count += 1
                
    if total_beans == 0:
        return None
        
    avg_diameter = np.mean(diameters)
    defect_percentage = (defects_count / total_beans) * 100
    
    # 2. Estimasi Kadar Air Berdasarkan Kecerahan & Warna Biji Kopi
    # Biji kopi basah biasanya lebih gelap/memiliki intensitas warna tertentu.
    # Kita menggunakan rata-rata intensitas warna RGB pada area biji kopi (menggunakan mask)
    mask = thresh
    mean_val = cv2.mean(img, mask=mask)[:3] # BGR mean
    b_mean, g_mean, r_mean = mean_val
    
    # Rumus heuristik untuk mensimulasikan kadar air (korelasi kecerahan)
    # Rata-rata intensitas cahaya dikonversi ke skala persentase kadar air realistis (9% - 18%)
    brightness = (r_mean + g_mean + b_mean) / 3.0
    moisture_content = 9.0 + (brightness / 255.0) * 9.0
    
    return {
        'persentase_kecacatan': round(defect_percentage, 1),
        'diameter_rata_rata': round(avg_diameter, 2),
        'kadar_air': round(moisture_content, 1)
    }

def generate_csv_from_images(dataset_dir, output_csv_path):
    print(f"Mulai memproses gambar di folder: {dataset_dir} ...")
    
    data_list = []
    
    # Cari subfolder kelas (misalnya: Dark, Green, Light, Medium atau Ekspor, Lokal)
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    for class_name in classes:
        class_path = os.path.join(dataset_dir, class_name)
        print(f"Memproses kelas: {class_name}...")
        
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(class_path, img_name)
            
            # Ekstrak fitur dari gambar
            features = extract_coffee_features(img_path)
            
            if features is not None:
                # Menentukan label kualitas akhir berdasarkan aturan SNI
                is_export = (features['kadar_air'] <= 12.5 and 
                             features['persentase_kecacatan'] <= 11.0 and 
                             features['diameter_rata_rata'] >= 5.5)
                
                label_kualitas = "Ekspor" if is_export else "Lokal"
                apakah_ekspor = 1 if is_export else 0
                
                features['nama_file'] = img_name
                features['kelas_asli'] = class_name
                features['label_kualitas'] = label_kualitas
                features['apakah_ekspor'] = apakah_ekspor
                
                data_list.append(features)
                
    # Buat DataFrame dan simpan ke CSV
    if len(data_list) > 0:
        df = pd.DataFrame(data_list)
        # Reorder kolom agar rapi
        df = df[['nama_file', 'kelas_asli', 'persentase_kecacatan', 'diameter_rata_rata', 'kadar_air', 'label_kualitas', 'apakah_ekspor']]
        df.to_csv(output_csv_path, index=False)
        print(f"Ekstraksi selesai! Berhasil menyimpan {len(df)} baris data ke {output_csv_path}")
    else:
        print("Tidak ada data gambar yang berhasil diproses.")

if __name__ == "__main__":
    # Tentukan folder dataset gambar kopi lokal
    dataset_train_dir = "./train"
    if not os.path.exists(dataset_train_dir):
        dataset_train_dir = "./coffee-bean-dataset-resized-224-x-224/train"
        
    output_csv = "./Coffee_Extracted_From_Images.csv"
    
    if os.path.exists(dataset_train_dir):
        generate_csv_from_images(dataset_train_dir, output_csv)
    else:
        print(f"Direktori dataset gambar tidak ditemukan di '{dataset_train_dir}'.")
        print("Silakan jalankan 'python train_model.py' terlebih dahulu untuk mengunduh dataset gambar dari Kaggle.")
