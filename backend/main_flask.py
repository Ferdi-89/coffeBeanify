import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageStat
import io
import pickle

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app) # Mengaktifkan CORS untuk pengembangan lokal

# Konstanta
CATEGORIES = ['Dark', 'Green', 'Light', 'Medium']
ROAST_DETAILS = {
    'Green': {
        'title': 'Green Beans (Belum Disangrai)',
        'description': 'Biji kopi mentah sebelum proses pemanggangan. Berwarna hijau pucat hingga kekuningan, memiliki kadar air tinggi, dan aroma herba/dedaunan segar.',
        'brew_recommendation': 'Tidak disarankan untuk diseduh langsung. Harus melalui proses roasting terlebih dahulu.',
        'flavor_notes': 'Rumput segar, kayu, asam mentah.'
    },
    'Light': {
        'title': 'Light Roast (Cinnamon / New England)',
        'description': 'Dipanggang dalam waktu singkat (suhu sekitar 196°C - 205°C). Biji kopi berwarna cokelat muda terang, tidak berminyak, dengan keasaman (acidity) yang menonjol.',
        'brew_recommendation': 'Sangat cocok untuk metode seduh manual (Manual Brew / Pour Over V60) untuk menonjolkan note rasa buah dan bunga.',
        'flavor_notes': 'Buah-buahan (fruity), bunga (floral), keasaman tinggi, body ringan.'
    },
    'Medium': {
        'title': 'Medium Roast (American / City)',
        'description': 'Roast level paling populer (suhu sekitar 210°C - 219°C). Warna cokelat sedang, bodi rasa lebih mantap, tingkat keasaman seimbang, dan mulai muncul rasa manis karamel.',
        'brew_recommendation': 'Sangat serbaguna. Cocok untuk Pour Over, AeroPress, Syphon, maupun Espresso berbasis susu.',
        'flavor_notes': 'Kacang-kacangan (nutty), cokelat, karamel manis, bodi seimbang.'
    },
    'Dark': {
        'title': 'Dark Roast (French / Italian)',
        'description': 'Dipanggang lama hingga mengeluarkan minyak di permukaan biji (suhu di atas 225°C). Berwarna cokelat sangat gelap hingga kehitaman. Keasaman hampir hilang sepenuhnya, digantikan rasa pahit manis yang pekat.',
        'brew_recommendation': 'Sangat cocok untuk Espresso murni, Kopi Tubruk klasik, Vietnam Drip, atau French Press.',
        'flavor_notes': 'Pahit pekat (bitter-sweet), asap (smoky), cokelat hitam, bodi tebal.'
    }
}

# 1. Memuat Model CNN Tingkat Sangrai (Keras)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'FineTunedCoffeeBeanCNN.keras')
model = None
is_mock = True

try:
    import tensorflow as tf
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        is_mock = False
        print(f"Berhasil memuat model CNN dari {MODEL_PATH}")
    else:
        print(f"Model CNN tidak ditemukan di {MODEL_PATH}. Berjalan dalam MOCK MODE.")
except Exception as e:
    print(f"Gagal memuat model CNN Keras: {e}. Berjalan dalam MOCK MODE.")

# 2. Memuat Model Klasifikasi Mutu Kopi (Pickle)
QUALITY_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'coffee_quality_classifier.pkl')
quality_model = None

if os.path.exists(QUALITY_MODEL_PATH):
    try:
        with open(QUALITY_MODEL_PATH, 'rb') as f:
            quality_model = pickle.load(f)
        print(f"Berhasil memuat model ML Kualitas dari {QUALITY_MODEL_PATH}")
    except Exception as e:
        print(f"Gagal memuat model ML Kualitas: {e}")
else:
    print(f"Model ML Kualitas tidak ditemukan di {QUALITY_MODEL_PATH}. Evaluator menggunakan logika aturan manual.")


def validate_coffee_image(image_bytes: bytes):
    """
    Verifikasi dasar untuk memastikan gambar yang diunggah memiliki profil warna biji kopi:
    - Biji kopi (Green, Light, Medium, Dark) memiliki rentang warna hijau, cokelat, hitam.
    - Menolak gambar berwarna neon ekstrem (seperti dominan biru/dingin) atau gambar polos satu warna.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        stat = ImageStat.Stat(img)
        r_mean, g_mean, b_mean = stat.mean[:3]
        r_std, g_std, b_std = stat.stddev[:3]
        
        # 1. Tolak gambar polos satu warna (solid color / blank screenshot)
        if max(r_std, g_std, b_std) < 8.0:
            raise ValueError("Gambar terlalu polos atau kosong. Pastikan mengunggah foto biji kopi yang jelas.")
            
        # 2. Tolak gambar berwarna dominan biru/ungu dingin (kopi tidak berwarna biru)
        if b_mean > r_mean + 15 and b_mean > g_mean + 15:
            raise ValueError("Warna gambar tidak sesuai dengan karakteristik warna biji kopi (terlalu dominan biru/dingin).")
            
        # 3. Tolak gambar dengan saturasi warna tidak wajar (misal pink/purple neon)
        if r_mean > 240 and g_mean < 50 and b_mean > 200:
            raise ValueError("Warna gambar terdeteksi tidak wajar untuk biji kopi.")
            
    except ValueError as ve:
        raise ve
    except Exception:
        raise ValueError("Format gambar tidak valid atau berkas rusak.")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Resize gambar ke 128x128 dan normalisasi."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_mock_prediction(image_bytes: bytes):
    """Prediksi tiruan cerdas berdasarkan warna gambar rata-rata."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        stat = ImageStat.Stat(img)
        r, g, b = stat.mean[:3]
        mean_brightness = (r + g + b) / 3.0
        
        if mean_brightness < 70:
            pred_class = 'Dark'
            confidence = np.random.uniform(94.0, 99.8)
        elif g > r - 10 and g > b + 15 and mean_brightness > 100:
            pred_class = 'Green'
            confidence = np.random.uniform(92.0, 98.5)
        elif mean_brightness > 160:
            pred_class = 'Light'
            confidence = np.random.uniform(90.0, 97.0)
        else:
            if r > g + 25:
                pred_class = 'Medium'
            else:
                pred_class = 'Light'
            confidence = np.random.uniform(93.0, 99.2)
        return pred_class, confidence
    except Exception:
        return np.random.choice(CATEGORIES), np.random.uniform(85.0, 99.9)


# Route untuk Halaman Utama Frontend
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada berkas gambar yang dikirim"}), 400
        
    file = request.files['file']
    try:
        image_bytes = file.read()
        
        # Pengaman/validasi: Cek apakah gambar yang diunggah benar-benar gambar biji kopi
        try:
            validate_coffee_image(image_bytes)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
            
        if not is_mock and model is not None:
            processed = preprocess_image(image_bytes)
            predictions = model.predict(processed)
            class_idx = np.argmax(predictions[0])
            pred_class = CATEGORIES[class_idx]
            confidence = float(predictions[0][class_idx]) * 100
        else:
            pred_class, confidence = get_mock_prediction(image_bytes)
            
        details = ROAST_DETAILS[pred_class]
        
        return jsonify({
            "class": pred_class,
            "confidence": round(confidence, 2),
            "is_mock": is_mock,
            "title": details['title'],
            "description": details['description'],
            "brew_recommendation": details['brew_recommendation'],
            "flavor_notes": details['flavor_notes']
        })
    except Exception as e:
        return jsonify({"error": f"Gagal memproses gambar: {str(e)}"}), 500


# Endpoint API Evaluasi Kualitas Mutu Fisik
@app.route('/api/quality-evaluate', methods=['POST'])
def evaluate_quality():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Data parameter tidak lengkap"}), 400
        
    try:
        defect_percentage = float(data.get('defect_percentage', 0))
        average_diameter = float(data.get('average_diameter', 0))
        moisture_content = float(data.get('moisture_content', 0))
        
        reasons = []
        is_export = True
        
        # Check moisture
        if moisture_content > 12.5:
            reasons.append(f"Kadar air terlalu tinggi ({moisture_content}%), melampaui batas maksimal ekspor 12.5%. Berisiko berjamur saat pengapalan.")
        elif moisture_content < 9.0:
            reasons.append(f"Kadar air sangat rendah ({moisture_content}%), biji kopi mungkin terlalu rapuh.")
            
        # Check defect
        if defect_percentage > 11.0:
            reasons.append(f"Persentase kecacatan fisik terlalu tinggi ({defect_percentage}%), batas kualitas ekspor premium adalah maksimal 11.0%.")
            
        # Check diameter
        if average_diameter < 6.5:
            reasons.append(f"Ukuran diameter rata-rata ({average_diameter} mm) termasuk kecil. Untuk ekspor, biji ukuran sedang-besar (>= 6.5 mm) lebih diutamakan.")
            if average_diameter < 5.5:
                reasons.append("Ukuran biji terlalu kecil (< 5.5 mm) untuk standar perdagangan ekspor umum.")

        # Prediksi Menggunakan Model ML (Pickle)
        if quality_model is not None:
            # Min-Max Scaling manual
            defect_scaled = np.clip(defect_percentage / 100.0, 0.0, 1.0)
            diameter_scaled = np.clip((average_diameter - 1.99) / 6.18, 0.0, 1.0)
            moisture_scaled = np.clip((moisture_content - 10.0) / 3.7, 0.0, 1.0)
            
            features_input = np.array([[defect_scaled, diameter_scaled, moisture_scaled]])
            prediction = quality_model.predict(features_input)[0]
            is_export = (prediction == 1)
        else:
            is_export = (moisture_content <= 12.5 and 
                         defect_percentage <= 11.0 and 
                         average_diameter >= 5.5)

        if is_export:
            quality_class = "Kualitas Ekspor"
            title = "Biji Kopi Mutu Ekspor (Premium Grade)"
            description = "Biji kopi memenuhi standar SNI untuk perdagangan internasional. Kadar air optimal, kecacatan minimal, dan ukuran biji seragam/besar."
            recommendation = "Layak dikemas dalam kantong grainpro kedap udara untuk pengapalan ekspor."
        else:
            quality_class = "Kualitas Lokal"
            title = "Biji Kopi Mutu Lokal (Commercial Grade)"
            description = "Biji kopi memiliki parameter fisik yang belum memenuhi standar ekspor premium. Masih sangat layak dikonsumsi dan diolah untuk pasar lokal."
            recommendation = "Lakukan pemilahan ulang (sorting) secara manual untuk mengurangi kecacatan fisik, atau keringkan kembali jika kadar air masih tinggi."

        if is_export and not reasons:
            reasons.append("Semua parameter fisik memenuhi standar mutu ekspor utama.")

        return jsonify({
            "quality_class": quality_class,
            "title": title,
            "description": description,
            "recommendation": recommendation,
            "reasons": reasons if reasons else ["Biji kopi diklasifikasikan sebagai Kualitas Lokal oleh model Machine Learning."]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Endpoint API Statistik Dataset
@app.route('/api/stats', methods=['GET'])
def get_stats():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Coffee Bean.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            distribution = df['labels'].value_counts().to_dict()
            total_images = len(df)
            dataset_splits = df['data set'].value_counts().to_dict()
            return jsonify({
                "success": True,
                "total_images": total_images,
                "distribution": distribution,
                "splits": dataset_splits
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    else:
        return jsonify({
            "success": True,
            "total_images": 1600,
            "distribution": {"Dark": 400, "Green": 400, "Light": 400, "Medium": 400},
            "splits": {"train": 1200, "test": 400}
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
