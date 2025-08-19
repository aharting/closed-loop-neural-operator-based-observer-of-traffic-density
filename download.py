import gdown
import zipfile
import os

# -----------------------------
# 1. Download and extract data
# -----------------------------
zip_url = "https://drive.google.com/uc?export=download&id=1NNKMMrOo04uLgewvvPhLwhToJm8AT8RD"
zip_path = "ResearchProjectData.zip"
data_folder = "data"

# Create folder if it doesn't exist
os.makedirs(data_folder, exist_ok=True)

# Download ZIP
if not os.path.exists(zip_path):
    print("Downloading data folder...")
    gdown.download(zip_url, zip_path, quiet=False)
else:
    print(f"{zip_path} already exists, skipping download.")

# Extract ZIP
print("Extracting files...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_folder)
print(f"All CSV files extracted to '{data_folder}'")

# -----------------------------
# 2. Download model files
# -----------------------------
models_folder = "models"
os.makedirs(models_folder, exist_ok=True)

# Dictionary of models: filename -> Google Drive direct link
model_files = {
    "closedloop.pt": "https://drive.google.com/uc?export=download&id=1JIkY98QQYI9-ULlAhuxGc911ryCYdOal",
    "openloop.pt": "https://drive.google.com/uc?export=download&id=1XkXyZArWbJCqsCOWhpXf-NNX4P8NU9eX",
    # Add more models here
}

# Download each model
for filename, url in model_files.items():
    output_path = os.path.join(models_folder, filename)
    if not os.path.exists(output_path):
        print(f"Downloading {filename}...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{filename} already exists, skipping download.")

print("All models and data files are ready!")
