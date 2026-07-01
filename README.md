# RoastCode: Coffee Roast & Quality Classifier Web App

Aplikasi web full-stack responsif (Desktop & Mobile) untuk klasifikasi tingkat kematangan sangrai (*roast level*) biji kopi menggunakan **Convolutional Neural Network (CNN)** serta evaluasi mutu fisik biji kopi pasca-panen menggunakan algoritma **Machine Learning (Decision Tree)** berdasarkan standar **SNI 01-2907-2008**.

---

## 🌟 Fitur Utama
1. **Klasifikasi Kematangan Sangrai (CNN)**:
   - Klasifikasi gambar biji kopi menjadi 4 kelas: *Dark, Green, Light, dan Medium*.
   - Dilengkapi *Mock Mode* otomatis jika model belum dilatih (menganalisis berdasarkan kecerahan warna gambar rata-rata).
   - Akses kamera perangkat (HP/Webcam) langsung dari browser untuk memotret biji kopi.
2. **Evaluasi Mutu Pasca-Panen (Machine Learning)**:
   - Input manual parameter fisik: **Kadar Air (%)**, **Persentase Kecacatan (%)**, dan **Diameter Rata-rata (mm)**.
   - Klasifikasi mutu menjadi **Kualitas Ekspor** atau **Kualitas Lokal** menggunakan model **Decision Tree Classifier** terlatih.
   - Ulasan deskripsi alasan keputusan model sesuai standar SNI.
3. **Statistik Dataset**:
   - Visualisasi distribusi kelas data langsung dari file metadata [Coffee Bean.csv](file:///e:/SEMESTER4/ai/machineLearning/Coffee%20Bean.csv).
4. **Vercel Ready**:
   - Konfigurasi deployment serverless instan ke platform Vercel.

---

## 🛠️ Teknologi & Pustaka
* **Frontend**: HTML5, Vanilla CSS3 (Custom Coffee Theme, Glassmorphism, Responsive Grid), JavaScript ES6.
* **Backend Options**: FastAPI (Python) & Flask (Python).
* **AI/Machine Learning**: TensorFlow (Keras), Scikit-Learn, Pandas, NumPy, OpenCV, Pillow.

---

## 📂 Struktur Direktori Proyek
```text
machineLearning/
│
├── api/
│   └── index.py            # Entry point WSGI untuk Vercel Serverless (Flask)
│
├── backend/                # Kode Program Backend
│   ├── models/             # Folder penyimpan model AI terlatih (*.keras, *.pkl)
│   ├── main.py             # Server API versi FastAPI (Default)
│   ├── main_flask.py       # Server API versi Flask (Alternatif)
│   └── requirements.txt    # Kebutuhan pustaka Python lokal
│
├── frontend/               # Kode Program Frontend (Website)
│   ├── index.html          # Kerangka web
│   ├── style.css           # Styling responsif bertema kopi
│   └── app.js              # Logika interaksi API dan kamera
│
├── Coffee_Normalized_Dataset.csv   # Dataset fisik biji kopi ternormalisasi
├── extract_features.py     # Skrip ekstraksi fitur gambar dengan OpenCV ke CSV
├── normalize_dataset.py    # Skrip kalibrasi diameter & normalisasi statistik (MinMax/Z-Score)
├── train_quality_ml.py     # Skrip latih & perbandingan 4 algoritma ML (Decision Tree, dll.)
├── train_model.py          # Skrip latih model CNN tingkat sangrai gambar
├── vercel.json             # Konfigurasi deploy Vercel
├── requirements.txt        # Dependensi khusus untuk Vercel (Tanpa TensorFlow)
└── .gitignore              # Pengecualian berkas besar/temp saat push ke Git
```

---

## 🚀 Panduan Menjalankan Secara Lokal

### 1. Pasang Dependensi Lokal
Buka terminal pada folder proyek ini, jalankan:
```bash
pip install -r backend/requirements.txt
```

### 2. Melatih Model ML (Kualitas Fisik)
Lakukan pelatihan model klasifikasi mutu menggunakan algoritma *Decision Tree*:
```bash
python train_quality_ml.py
```
*Model terbaik otomatis tersimpan di folder `backend/models/coffee_quality_classifier.pkl`.*

### 3. Jalankan Server API
Anda dapat memilih salah satu versi server berikut:
* **Menggunakan Flask**:
  ```bash
  python backend/main_flask.py
  ```
* **Menggunakan FastAPI**:
  ```bash
  python backend/main.py
  ```
Setelah server berjalan, akses website melalui browser di alamat:
`http://127.0.0.1:8000`

---

## ☁️ Panduan Deploy ke Vercel
Proyek ini sudah dilengkapi konfigurasi serverless tanpa menyertakan TensorFlow (karena batas ukuran Vercel 250MB). Fitur klasifikasi gambar akan berjalan dalam *Mock Mode*, namun fitur Evaluasi Mutu ML (Decision Tree) tetap berjalan asli menggunakan `scikit-learn`.

1. Pasang Vercel CLI jika belum ada:
   ```bash
   npm install -g vercel
   ```
2. Jalankan perintah deploy di folder proyek:
   ```bash
   vercel
   ```
3. Ikuti langkah-langkah di terminal untuk menyelesaikan deployment.
"# coffeBeanify" 
