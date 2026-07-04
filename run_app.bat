@echo off
title RoastCode - Coffee Roast & Quality Classifier

:: Pastikan bekerja di direktori tempat berkas .bat berada
cd /d "%~dp0"

echo =======================================================================
echo           Menjalankan RoastCode Web App (FastAPI + ML/CNN)            
echo =======================================================================
echo.

:: 1. Cek apakah folder virtual environment .venv ada
if not exist ".venv" goto no_venv

:: 2. Aktivasi virtual environment
echo [*] Mengaktifkan virtual environment (.venv)...
call .venv\Scripts\activate.bat
if errorlevel 1 goto activation_failed

:: 3. Instal/perbarui dependensi jika diperlukan
echo [*] Memeriksa dan menginstal dependensi...
pip install -r backend/requirements.txt
if errorlevel 1 (
    echo [WARNING] Terjadi kesalahan saat menginstal dependensi.
    echo Mencoba melanjutkan...
)

:: 4. Latih model ML mutu fisik jika belum ada
if exist "backend\models\coffee_quality_classifier.pkl" goto model_exists

echo [*] Model ML Kualitas belum terdeteksi.
echo [*] Melatih model ML mutu fisik biji kopi...
python train_quality_ml.py
if errorlevel 1 (
    echo [WARNING] Gagal melatih model ML Kualitas. Backend akan berjalan menggunakan logika manual.
)
goto start_server

:model_exists
echo [*] Model ML Kualitas terdeteksi, melewati proses pelatihan.

:start_server
:: 5. Buka web browser otomatis
echo [*] Membuka web browser ke http://127.0.0.1:8000...
start "" "http://127.0.0.1:8000"

:: 6. Jalankan server FastAPI
echo [*] Memulai FastAPI Server (backend/main.py)...
echo [INFO] Tekan Ctrl+C untuk menghentikan server.
python backend/main.py
if errorlevel 1 goto server_failed

goto end

:no_venv
echo [ERROR] Virtual environment (.venv) tidak ditemukan di folder ini!
echo Silakan buat virtual environment terlebih dahulu dengan perintah:
echo python -m venv .venv
goto pause_and_exit

:activation_failed
echo [ERROR] Gagal mengaktifkan virtual environment!
goto pause_and_exit

:server_failed
echo [ERROR] Server backend berhenti karena ada masalah.
goto pause_and_exit

:pause_and_exit
pause
exit /b

:end
pause
