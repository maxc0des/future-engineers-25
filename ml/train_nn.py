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
max_epochs = 200
min_epochs = 20
patience = 10
loss_plt = []
test_plt = []
counter = 0  # Initialisierung

class Data(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file).dropna()
        self.data = self.data[self.data["steering_angle"] <= 80]
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

transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class ImageNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64*64*3, 512),  
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128)  
        )

    def forward(self, x):
        return self.linear_relu_stack(self.flatten(x))

class TOFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 128),  
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

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

train_dataset = Data(csv_file="train_data.csv", root_dir="images", transform=transform)
val_dataset = Data(csv_file="validation_data.csv", root_dir="images", transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = FullNN()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

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
            loss_plt.append(loss.item())

def test_loop(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for images, tofs, labels in dataloader:
            pred = model(images, tofs)
            test_loss += loss_fn(pred, labels).item()

    avg_loss = test_loss / len(dataloader)
    print(f"Test Error: Avg MSE Loss: {avg_loss:>8f}")
    test_plt.append(avg_loss)

e = 0
best_loss = float('inf')
plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot([], [], 'r-', label="Loss")
line2, = ax.plot([], [], 'b-', label="Test Loss")

while True:
    print(f"-------------------------------\nEpoch {e}")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(val_dataloader, model, loss_fn)
    
    if test_plt[-1] < best_loss:
        best_loss = test_plt[-1]
        counter = 0
    else:
        counter += 1
    
    line1.set_xdata(range(len(loss_plt)))
    line1.set_ydata(loss_plt)
    line2.set_xdata(range(len(test_plt)))
    line2.set_ydata(test_plt)

    ax.set_xlim(0, max(len(loss_plt), len(test_plt)))
    ax.set_ylim(min(min(loss_plt, default=0), min(test_plt, default=0)), 
                max(max(loss_plt, default=1), max(test_plt, default=1)))

    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.1)

    if counter == patience and e > min_epochs or e == max_epochs:
        print("Training beendet (Modell verbessert sich nicht mehr / max_epochs erreicht)!")
        break

    e += 1
print("Training abgeschlossen! ✅")
plt.ioff()


model.eval()
sample_idx = random.randint(0, len(val_dataset) - 1)
image, tof, label = val_dataset[sample_idx]
image = image.unsqueeze(0)
tof = tof.unsqueeze(0)

with torch.no_grad():
    prediction = model(image, tof)

predicted_steering, predicted_velocity = prediction.squeeze().tolist()
true_steering, true_velocity = label.tolist()

print(f"📷 Sample {sample_idx}:")
print(f"🔹 Wahre Werte: Steering Angle = {true_steering:.2f}, Velocity = {true_velocity:.2f}")
print(f"🔮 Vorhersage:  Steering Angle = {predicted_steering:.2f}, Velocity = {predicted_velocity:.2f}")

if input("Modell speichern? (y/n) ").lower() in ["y", "yes"]:
    torch.save(model.state_dict(), "model.pth")
    print("Modell gespeichert! 🎉")
plt.show()
