import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageStat
from pydantic import BaseModel
import io


app = FastAPI(title="Coffee Bean Roast Classifier API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
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

# Try loading the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'FineTunedCoffeeBeanCNN.keras')
model = None
is_mock = True

try:
    import tensorflow as tf
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        is_mock = False
        print(f"Successfully loaded model from {MODEL_PATH}")
    else:
        print(f"Model not found at {MODEL_PATH}. Starting in MOCK MODE.")
except Exception as e:
    print(f"Failed to load TensorFlow model due to error: {e}. Starting in MOCK MODE.")

# Load ImageNet model for out-of-distribution verification
imagenet_model = None
try:
    if not is_mock:
        print("Loading ImageNet model for out-of-distribution validation...")
        imagenet_model = tf.keras.applications.MobileNetV2(weights='imagenet')
        print("ImageNet model loaded successfully.")
except Exception as e:
    print(f"Failed to load ImageNet model: {e}")

QUALITY_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'coffee_quality_classifier.pkl')
quality_model = None

if os.path.exists(QUALITY_MODEL_PATH):
    try:
        import pickle
        with open(QUALITY_MODEL_PATH, 'rb') as f:
            quality_model = pickle.load(f)
        print(f"Successfully loaded coffee quality ML model from {QUALITY_MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load quality ML model: {e}")
else:
    print(f"Quality ML model not found at {QUALITY_MODEL_PATH}. Quality evaluator will fallback to rule-based logic.")


def validate_coffee_image(image_bytes: bytes):
    """
    Verifikasi gambar untuk memastikan gambar yang diunggah adalah biji kopi.
    Menggunakan checks dasar warna dan model ImageNet untuk menolak gambar random (hewan, kendaraan, orang, dll.).
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
            
        # 4. Gunakan ImageNet model untuk menolak gambar random (hewan, manusia, kendaraan, furnitur, dll.)
        global imagenet_model
        if imagenet_model is not None:
            # Resize ke 224x224 untuk MobileNetV2 ImageNet
            img_224 = img.resize((224, 224))
            img_array = np.array(img_224).astype(np.float32)
            # MobileNetV2 preprocess_input: scale to [-1, 1]
            img_array = (img_array / 127.5) - 1.0
            img_array = np.expand_dims(img_array, axis=0)
            
            preds = imagenet_model.predict(img_array, verbose=0)
            from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
            decoded = decode_predictions(preds, top=5)[0]
            
            # Daftar kata kunci OOD yang dilarang
            OOD_KEYWORDS = [
                'cat', 'dog', 'hound', 'terrier', 'retriever', 'spaniel', 'collie', 'mastiff', 'shepherd', 'poodle',
                'pug', 'beagle', 'foxhound', 'dane', 'husky', 'dalmatian', 'boxer', 'rottweiler',
                'lion', 'tiger', 'leopard', 'bear', 'elephant', 'zebra', 'giraffe', 'deer', 'rabbit', 'squirrel',
                'bird', 'parrot', 'eagle', 'hawk', 'owl', 'chicken', 'duck', 'goose', 'swan',
                'fish', 'shark', 'whale', 'dolphin',
                'person', 'man', 'woman', 'child', 'baby', 'face', 'people', 'groom', 'bride', 'jersey', 't-shirt',
                'car', 'truck', 'bus', 'train', 'jeep', 'cab', 'limousine', 'sports_car', 'minivan', 'racer',
                'bicycle', 'motorcycle', 'airplane', 'airliner', 'ship', 'boat',
                'chair', 'sofa', 'couch', 'table', 'desk', 'bed', 'wardrobe', 'cabinet',
                'computer', 'monitor', 'laptop', 'keyboard', 'phone', 'television', 'screen',
                'house', 'building', 'church', 'castle', 'palace', 'monument', 'tower', 'bridge'
            ]
            
            for synset, label_name, prob in decoded:
                label_lower = label_name.lower().replace('_', ' ')
                for keyword in OOD_KEYWORDS:
                    if keyword in label_lower and prob > 0.05:
                        # Terjemahkan beberapa label umum ke Bahasa Indonesia untuk UX yang lebih baik
                        indonesian_label = label_name.replace('_', ' ')
                        if 'cat' in label_lower:
                            indonesian_label = "kucing"
                        elif 'dog' in label_lower or any(k in label_lower for k in ['hound', 'terrier', 'retriever', 'spaniel', 'collie', 'mastiff', 'shepherd', 'poodle', 'pug', 'beagle', 'foxhound', 'dane', 'husky', 'dalmatian', 'boxer', 'rottweiler']):
                            indonesian_label = "anjing"
                        elif 'car' in label_lower or 'jeep' in label_lower or 'sports_car' in label_lower:
                            indonesian_label = "mobil"
                        elif 'person' in label_lower or 'man' in label_lower or 'woman' in label_lower or 'child' in label_lower or 'baby' in label_lower or 'face' in label_lower or 'people' in label_lower or 'groom' in label_lower or 'bride' in label_lower:
                            indonesian_label = "orang/manusia"
                            
                        raise ValueError(f"Gambar yang diunggah terdeteksi sebagai {indonesian_label} ({prob*100:.1f}%). Harap unggah foto biji kopi yang valid.")
                        
    except ValueError as ve:
        raise ve
    except Exception as e:
        # Jika ada kesalahan internal, lewati agar alur utama tidak terganggu
        pass


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Resize image to 128x128 and normalize it to [-1, 1] for MobileNetV2."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((128, 128))
    img_array = (np.array(img) / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array


def get_mock_prediction(image_bytes: bytes):
    """Analyze image stats to give a realistic mock classification based on average color."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        stat = ImageStat.Stat(img)
        r, g, b = stat.mean[:3]
        
        # Simple color heuristics for green, light, medium, dark
        # Green beans tend to have closer green and red channels or distinct green hue
        # Dark roast is very dark (low values across all channels)
        # Light roast is bright brown/yellow (high values, R > G > B)
        
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
            # Check ratio
            if r > g + 25:
                pred_class = 'Medium'
            else:
                pred_class = 'Light'
            confidence = np.random.uniform(93.0, 99.2)
            
        return pred_class, confidence
    except Exception:
        # Fallback to random if error
        pred_class = np.random.choice(CATEGORIES)
        confidence = np.random.uniform(85.0, 99.9)
        return pred_class, confidence


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        
        # Pengaman/validasi: Cek apakah gambar yang diunggah benar-benar gambar biji kopi
        try:
            validate_coffee_image(image_bytes)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
            
        if not is_mock and model is not None:
            processed = preprocess_image(image_bytes)
            predictions = model.predict(processed)
            class_idx = np.argmax(predictions[0])
            pred_class = CATEGORIES[class_idx]
            confidence = float(predictions[0][class_idx]) * 100
        else:
            # Mock mode
            pred_class, confidence = get_mock_prediction(image_bytes)
            
        details = ROAST_DETAILS[pred_class]
        
        return {
            "class": pred_class,
            "confidence": round(confidence, 2),
            "is_mock": is_mock,
            "title": details['title'],
            "description": details['description'],
            "brew_recommendation": details['brew_recommendation'],
            "flavor_notes": details['flavor_notes']
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses gambar: {str(e)}")


class CoffeeQualityInput(BaseModel):
    defect_percentage: float
    average_diameter: float
    moisture_content: float

@app.post("/api/quality-evaluate")
async def evaluate_quality(data: CoffeeQualityInput):
    try:
        reasons = []
        is_export = True
        
        # Check moisture (SNI: Max 12.5% for export)
        if data.moisture_content > 12.5:
            reasons.append(f"Kadar air terlalu tinggi ({data.moisture_content}%), melampaui batas maksimal ekspor 12.5%. Berisiko berjamur saat pengapalan.")
        elif data.moisture_content < 9.0:
            reasons.append(f"Kadar air sangat rendah ({data.moisture_content}%), biji kopi mungkin terlalu rapuh.")
            
        # Check defect (SNI: low defect points, usually < 11% for premium)
        if data.defect_percentage > 11.0:
            reasons.append(f"Persentase kecacatan fisik terlalu tinggi ({data.defect_percentage}%), batas kualitas ekspor premium adalah maksimal 11.0%.")
            
        # Check diameter
        if data.average_diameter < 6.5:
            reasons.append(f"Ukuran diameter rata-rata ({data.average_diameter} mm) termasuk kecil. Untuk ekspor, biji ukuran sedang-besar (>= 6.5 mm) lebih diutamakan.")
            if data.average_diameter < 5.5:
                reasons.append("Ukuran biji terlalu kecil (< 5.5 mm) untuk standar perdagangan ekspor umum.")

        # Prediksi Menggunakan Model Machine Learning jika tersedia
        if quality_model is not None:
            # Lakukan Min-Max scaling manual menggunakan range dataset asli:
            # Defect: [0.0, 100.0], Diameter: [1.99, 8.17], Moisture: [10.0, 13.7]
            defect_scaled = np.clip(data.defect_percentage / 100.0, 0.0, 1.0)
            diameter_scaled = np.clip((data.average_diameter - 1.99) / 6.18, 0.0, 1.0)
            moisture_scaled = np.clip((data.moisture_content - 10.0) / 3.7, 0.0, 1.0)
            
            features_input = np.array([[defect_scaled, diameter_scaled, moisture_scaled]])
            prediction = quality_model.predict(features_input)[0]
            is_export = (prediction == 1)
        else:
            # Fallback ke aturan berbasis regulasi jika model belum dilatih
            is_export = (data.moisture_content <= 12.5 and 
                         data.defect_percentage <= 11.0 and 
                         data.average_diameter >= 5.5)

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

        # Tambahkan teks default jika lolos ekspor tanpa catatan buruk
        if is_export and not reasons:
            reasons.append("Semua parameter fisik memenuhi standar mutu ekspor utama.")

        return {
            "quality_class": quality_class,
            "title": title,
            "description": description,
            "recommendation": recommendation,
            "reasons": reasons if reasons else ["Biji kopi diklasifikasikan sebagai Kualitas Lokal oleh model Machine Learning."]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Coffee Bean.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Count distribution
            distribution = df['labels'].value_counts().to_dict()
            total_images = len(df)
            dataset_splits = df['data set'].value_counts().to_dict()
            return {
                "success": True,
                "total_images": total_images,
                "distribution": distribution,
                "splits": dataset_splits
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    else:
        # Fallback statistics if CSV is missing
        return {
            "success": True,
            "total_images": 1600,
            "distribution": {"Dark": 400, "Green": 400, "Light": 400, "Medium": 400},
            "splits": {"train": 1200, "test": 400}
        }

# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    @app.get("/")
    def read_root():
        return {"message": "API is working. Please create the frontend directory to view the website UI."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
