import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import numpy as np
import json
from models import FullNN, IntegratedNN, EfficientNet5Channel, IntegratedNN2
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from playsound import playsound

# Configuration:
print("loading the config now")
with open("config_nn.json", "r") as f:
    config = json.load(f)
    learning_rate = float(config["config"]["learning_rate"])
    batch_size = config["config"]["batch_size"]
    max_epochs = config["config"]["max_epochs"]
    min_epochs = config["config"]["min_epochs"]
    patience = config["config"]["patience"]
    window_size = config["config"]["window_size"]

loss_plt = []
test_plt = []
patience_counter = 0

class Data(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file).dropna()
        self.data = self.data[self.data["steering_angle"] <= 80]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = f"{self.root_dir}/{self.data.iloc[idx]['cam_path']}"
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        
        tof = torch.tensor([
            self.data.iloc[idx]["tof_1"],
            self.data.iloc[idx]["tof_2"]
        ], dtype=torch.float32)

        label = torch.tensor([
            self.data.iloc[idx]["steering_angle"],
        ], dtype=torch.float32)

        tof_expanded = tof.view(2, 1, 1).expand(2, 128, 128)

        # Bild (3, 128, 128) und ToF-Daten (2, 128, 128) kombinieren -> (5, 128, 128)
        combined_input = torch.cat((image, tof_expanded), dim=0)

        return combined_input, label

transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (combined_input, labels) in enumerate(dataloader):
        pred = model(combined_input)
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
        for combined_input, labels in dataloader:
            pred = model(combined_input)
            test_loss += loss_fn(pred, labels).item()

    avg_loss = test_loss / len(dataloader)
    print(f"Test Error: Avg MSE Loss: {avg_loss:>8f}")
    test_plt.append(avg_loss)

def earlystop(values: list):
    global patience_counter, window_size
    if len(values) > min_epochs:
        derivatives = np.diff(values[-window_size:])
        print(f"Avg Derivatives: {np.mean(derivatives)}")

        if np.mean(derivatives) > 0:  # Wenn der Test-Loss steigt
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")
        else:
            patience_counter = 0

        return patience_counter >= patience
    return False


train_dataset = Data(csv_file="train_data.csv", root_dir="images", transform=transform)
val_dataset = Data(csv_file="validation_data.csv", root_dir="images", transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = IntegratedNN()
#loss_fn = nn.MSELoss()
loss_fn = nn.SmoothL1Loss(beta=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4,)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
if input("model weitertraineren?") in ["yes", "y"]:
    filename = str(input("model name: "))
    if not os.path.exists(f"{filename}.pth"):
        print("model not found")
        exit()

        #layer einfrieren
        #for param in model.features.parameters():
        #param.requires_grad = False
        #model.fc = torch.nn.Linear(512, 1)

    model.load_state_dict(torch.load(f"{filename}.pth"))
    model.train()
    checkpoint = torch.load(f"checkpoint-{filename}.pth")

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"loaded model from epoch {start} with loss {loss}")
else:
    start = 0
#execution starts here:
e = start
best_loss = float('inf')
plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot([], [], 'r-', label="Loss")
line2, = ax.plot([], [], 'b-', label="Test Loss")

while True:
    try:
        e += 1
        print(f"-------------------------------\nEpoch {e}")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(val_dataloader, model, loss_fn)
        scheduler.step(test_plt[-1])
        #loss_plt.append(e)
        #test_plt.append(e+1)

        num_points = min(len(loss_plt), 6)  # Use up to last 6 points
        plot1 = loss_plt[-num_points:]
        plot2 = test_plt[-num_points:]
        x = list(range(e - num_points + 1, e + 1))

        # Update the plot
        line1.set_xdata(x)
        line1.set_ydata(plot1)
        line2.set_xdata(x)
        line2.set_ydata(plot2)

        ax.set_xlim(e - num_points + 1, e + 1)
        ax.set_ylim(min(plot1 + plot2, default=0), max(plot1 + plot2, default=1))

        fig.canvas.draw()   # Force redraw
        fig.canvas.flush_events()  # Process GUI events
        plt.pause(0.6)  # Pause for updates


        if earlystop(test_plt):
            print("earlystopping")
            break

        if e == max_epochs:
            print("reached max epochs")
            break

    except KeyboardInterrupt:
        break
print("Training abgeschlossen! ✅")
try:
    playsound("finished.mp3")
except:
    print("no sound")
plt.ioff()


model.eval()
right_predictions = 0
for i in range(10):
    sample_idx = random.randint(0, len(val_dataset) - 1)
    combined_input, label = val_dataset[sample_idx]
    test_input = combined_input.unsqueeze(0)

    with torch.no_grad():
        prediction = model(test_input)

    predicted_steering = prediction.squeeze().tolist()
    true_steering = label.item()  # convert tensor to float

    if true_steering - 5 < predicted_steering < true_steering + 5:
        right_predictions+= 1

    print(f"📷 Sample {sample_idx}:")
    print(f"🔹 Wahre Werte: Steering Angle = {true_steering:.2f}")
    print(f"🔮 Vorhersage:  Steering Angle = {predicted_steering:.2f}")

print("right predictions: ", right_predictions, "/10")
filename = str(input("filename for saving the model: "))
if not filename in ["", " "]:
    torch.save(model.state_dict(), f"{filename}.pth")
    checkpoint = {
        "epoch": e,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss_plt[-1] if loss_plt else float('inf'),
    }
    torch.save(checkpoint, f"checkpoint-{filename}.pth")
    print("Model saved! 📦")
plt.close()
plt.figure()
plt.plot(loss_plt, 'r-', label="Loss")
plt.plot(test_plt, 'b-', label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Test Loss over Epochs")
plt.show()