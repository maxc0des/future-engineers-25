import pandas as pd
import random
import os
import requests

# Anzahl der Zeilen im Sample DataFrame
num_rows = 80

# Verzeichnis zum Speichern von Testbildern
# Verzeichnis zum Speichern der Testbilder
image_dir = "samples"
os.makedirs(image_dir, exist_ok=True)  # Erstellt das Verzeichnis, falls es nicht existiert


# Beispiel-URL eines GitHub-Datasets mit Bildern
github_image_urls = [
    "https://yavuzceliker.github.io/sample-images/image-1.jpg",
    "https://yavuzceliker.github.io/sample-images/image-2.jpg",
    "https://yavuzceliker.github.io/sample-images/image-3.jpg",
    "https://yavuzceliker.github.io/sample-images/image-4.jpg",
    "https://yavuzceliker.github.io/sample-images/image-5.jpg",
    "https://yavuzceliker.github.io/sample-images/image-6.jpg",
    "https://yavuzceliker.github.io/sample-images/image-7.jpg",
    "https://yavuzceliker.github.io/sample-images/image-8.jpg",
    "https://yavuzceliker.github.io/sample-images/image-9.jpg",
]

# Funktion zum Herunterladen von Bildern
def download_image(url, save_dir):
    filename = os.path.join(save_dir, url.split("/")[-1])
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
        return filename
    else:
        print(f"Failed to download image from {url}")
        return None

# Herunterladen der Bilder und Erstellen einer Liste der lokalen Pfade
local_image_paths = [
    download_image(url, image_dir) for url in github_image_urls if download_image(url, image_dir)
]

# Sicherstellen, dass genügend Bilder vorhanden sind
if len(local_image_paths) < num_rows:
    local_image_paths *= (num_rows // len(local_image_paths)) + 1  # Wiederholen der Pfade
local_image_paths = local_image_paths[:num_rows]

# Erstellen des Sample DataFrames
data = {
    "gyro": [random.uniform(-100, 100) for _ in range(num_rows)],
    "cam_path": local_image_paths,
    "tof1": [random.randint(10, 100) for _ in range(num_rows)],
    "tof2": [random.randint(10, 100) for _ in range(num_rows)],
    "steering": [random.randint(-50, 50) for _ in range(num_rows)],
    "velocity": [random.randint(0, 120) for _ in range(num_rows)],
}

df = pd.DataFrame(data)

# Speichern des DataFrames als CSV
csv_path = "sample_data.csv"
df.to_csv(csv_path, index=False)

print(f"Sample DataFrame mit {num_rows} Zeilen wurde erstellt und als '{csv_path}' gespeichert.")
print(df.head())