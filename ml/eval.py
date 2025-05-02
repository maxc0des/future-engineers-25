import torch
import random
import pandas as pd
from PIL import Image
from models import IntegratedNN
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import numpy as np


csv_path = "train_data.csv"
model_path = "clockwise.pth"
right_predictions = 0
calcs = 0
median = []

#load the model
model = IntegratedNN()
model.load_state_dict(torch.load(model_path))
model.eval()

#load the data
data = pd.read_csv(csv_path)

def get_data(index):
    original_row = data.iloc[index].copy()
    img_path = f"images/{original_row['cam_path']}"
    image = Image.open(img_path)
    image = transforms.Resize((128, 128))(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
    tof = torch.tensor([
        data.iloc[index]["tof_1"],
        data.iloc[index]["tof_2"]
    ], dtype=torch.float32)
    label = torch.tensor([
        data.iloc[index]["steering_angle"],
    ], dtype=torch.float32)

    tof_expanded = tof.view(2, 1, 1).expand(2, 128, 128)
    combined_input = torch.cat((image, tof_expanded), dim=0)

    return combined_input, label, img_path

def predict(combined_input, label):
    start = time.time()
    input = combined_input.unsqueeze(0)
    with torch.no_grad():
        prediction = model(input)
    predicted_steering = prediction.squeeze().tolist()
    end = time.time()
    time_taken = end - start
    true_steering = label.item()
    offset = predicted_steering - true_steering

    return predicted_steering, time_taken, offset, true_steering

while True:
    mode = input("mode(e/n/'enter'): ")

    if mode == "e":
        break
    elif mode == "n":
        i = int(input("irritation: "))
        index = int(input(f"index (0-{len(data)-1}): "))
        calc_median = True
    elif mode == "":
        i = 1
        index = None
        calc_median = False
    else:
        i = int(mode)
        index = None
        calc_median = True

    plt.close('all')

    for _ in range(i):
        if index == None:
            index = random.randint(0, len(data)-1)
        combined_input, label, img_path = get_data(index)
        prediction, time_taken, offset, label = predict(combined_input, label)
        if calc_median:
            median.append(prediction)
    img = Image.open(img_path)
    img.thumbnail((128, 128))
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, 128, 0, 128])
    x1, y1 = 64, 0
    alpha = np.radians(prediction-50)
    g = np.tan(alpha)*80 #ankathete = 80
    x2, y2 = 64+g, 80
    ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2)
    ax.plot([x1, 110], [y1, y2], 'r-', linewidth=2) #rechte begrenzung
    ax.plot([x1, 17], [y1, y2], 'r-', linewidth=2) #linke begrenzung
    #ax.set_xlim(0, 128)
    #ax.set_ylim(0, 128)
    plt.title(f"Sample {index}:"
            f"\nWahre Werte: Steering Angle = {label:.2f}"
            f"\nVorhersage:  Steering Angle = {prediction:.2f}")
    plt.show(block=False)

    print(f"sample {index}, angel: {label}, prediction: {prediction}, offset: {offset}, time: {time_taken}")
    if calc_median:
        print(f"median: {np.median(median)}, median offset {np.median(median)-label}")