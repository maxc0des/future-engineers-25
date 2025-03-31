import torch
import pandas as pd
from PIL import Image
from models import IntegratedNN
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from get_data import get_tof, take_photo_fast
from motor import servo, motor, setup, cleanup

#define the paths
csv_path = "train_data.csv"
model_path = "model.pth"

#load the model
model = IntegratedNN()
model.load_state_dict(torch.load(model_path))
model.eval()

def predict(combined_input):
    input = combined_input.unsqueeze(0)
    with torch.no_grad():
        prediction = model(input)
    predicted_steering = prediction.squeeze().tolist()

    return predicted_steering

#setup
setup()
print("start")
while True:
    try:
        tof = list(get_tof())
        img = take_photo_fast()

        image = Image.fromarray(img) #fix this soon
        image = transforms.Resize((128, 128))(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)

        tof = torch.tensor([
            tof[0],
            tof[1]
        ], dtype=torch.float32)

        tof_expanded = tof.view(2, 1, 1).expand(2, 128, 128)
        combined_input = torch.cat((image, tof_expanded), dim=0)
        steering = predict(combined_input)
        motor(speed=200)
        servo(steering)
        
        #debugging:
        print(f"predicted angel: {steering}, tof: {tof}")
    
    except KeyboardInterrupt:
        break
motor(0)
servo(50)
cleanup()