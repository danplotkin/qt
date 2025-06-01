echo "[INFO] Creating virtual environment..."
python -m venv venv

echo "[INFO] Activating virtual environment..."
source venv/bin/activate

echo "[INFO] Installing dependencies..."
pip install -r requirements.txt

echo "[INFO] Downloading training data from S3..."
python ./scripts/data_scripts/download_data_from_s3.py