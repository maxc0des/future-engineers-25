import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import numpy as np

# Configuration:
learning_rate = 1e-3
batch_size = 64
epochs = 100
loss_plt = []

class Data(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)

        # Entferne NaN-Werte
        self.data = self.data.dropna()

        # Entferne Lenkwinkel über 80
        self.data = self.data[self.data["steering_angle"] <= 80]

        # Falls negative Geschwindigkeit unerwünscht ist, setzen wir sie auf 0
        self.data["velocity"] = self.data["velocity"].clip(lower=0)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = f"{self.root_dir}/{self.data.iloc[idx]['cam_path']}"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        tof = torch.tensor([
            self.data.iloc[idx]["tof_1"],
            self.data.iloc[idx]["tof_2"]
        ], dtype=torch.float32)

        label = torch.tensor([
            self.data.iloc[idx]["steering_angle"],
            self.data.iloc[idx]["velocity"]
        ], dtype=torch.float32)

        return image, tof, label

# Transformationen mit Bildgröße 64x64
transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Modell für Bildverarbeitung
class ImageNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64*64*3, 512),  
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128)  
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

# Modell für TOF-Daten
class TOFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 128),  
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

# Gesamtmodell (Fusion von Bild- und TOF-Daten)
class FullNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_nn = ImageNN()
        self.tof_nn = TOFNN()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128 + 128, 512),  
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)  
        )

    def forward(self, image, tof):
        image_features = self.image_nn(image)
        tof_features = self.tof_nn(tof)
        combined = torch.cat((image_features, tof_features), dim=1)
        return self.linear_relu_stack(combined)

# Lade die Trainings- und Validierungsdaten
train_dataset = Data(csv_file="train_data.csv", root_dir="images", transform=transform)
val_dataset = Data(csv_file="validation_data.csv", root_dir="images", transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialisiere Modell, Verlustfunktion und Optimierer
model = FullNN()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Trainingsloop
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (images, tofs, labels) in enumerate(dataloader):
        pred = model(images, tofs)
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"Loss: {loss.item():>7f}")
            loss_plt.append(int(loss.item()))

# Testloop (Validierung)
def test_loop(dataloader, model, loss_fn):
    model.eval()
    test_loss, total_samples = 0, 0

    with torch.no_grad():
        for images, tofs, labels in dataloader:
            pred = model(images, tofs)
            test_loss += loss_fn(pred, labels).item()
            total_samples += labels.size(0)

    avg_loss = test_loss / len(dataloader)
    print(f"Test Error: Avg MSE Loss: {avg_loss:>8f}")

# Training ausführen
for e in range(epochs):
    print(f"-------------------------------\nEpoch {e+1}")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(val_dataloader, model, loss_fn)
print("Training abgeschlossen! ✅")

loss_plt = np.array(loss_plt)
plt.plot(loss_plt)
plt.show()

model.eval()

# Eine zufällige Probe aus dem Validierungsset auswählen
sample_idx = random.randint(0, len(val_dataset) - 1)
image, tof, label = val_dataset[sample_idx]

# Tensor auf Batch-Dimension erweitern (weil das Modell Batches erwartet)
image = image.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
tof = tof.unsqueeze(0)      # [2] -> [1, 2]

# Vorhersage berechnen
with torch.no_grad():
    prediction = model(image, tof)

# Werte zurück in numpy konvertieren
predicted_steering, predicted_velocity = prediction.squeeze().tolist()
true_steering, true_velocity = label.tolist()

# Ausgabe der Werte
print(f"📷 Sample {sample_idx}:")
print(f"🔹 Wahre Werte: Steering Angle = {true_steering:.2f}, Velocity = {true_velocity:.2f}")
print(f"🔮 Vorhersage:  Steering Angle = {predicted_steering:.2f}, Velocity = {predicted_velocity:.2f}")
