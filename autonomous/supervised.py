import torch
import pandas as pd
from PIL import Image
from models import IntegratedNN
from torchvision import transforms

import pygame #type: ignore
import threading
import time
import queue
import os
from datetime import datetime

from get_data import get_tof, take_photo_fast
from motor import servo, motor, setup, cleanup

#define the paths
csv_path = "train_data.csv"
model_path = "model.pth"

mode = "autonomous"
data_buffer = []
last_data_save = 0
last_df_save = 0
filepath = ""
data_saves = 0

photo_queue = queue.Queue()

def predict(combined_input):
    input = combined_input.unsqueeze(0)
    with torch.no_grad():
        prediction = model(input)
    predicted_steering = prediction.squeeze().tolist()

    return predicted_steering

def photo_worker():
    while True:
        item = photo_queue.get()
        if item is None:
            break  # Beenden
        img_path, img_array = item
        Image.fromarray(img_array).save(img_path, format="JPEG")
        photo_queue.task_done()

worker_thread = threading.Thread(target=photo_worker, daemon=True)
worker_thread.start()

def collect_data(velocity, steering):
    global data_saves
    
    full_path = os.path.join(filepath, f"cam-{data_saves}.jpg")
    img_array = take_photo_fast(filepath=full_path, index=data_saves)
    photo_queue.put((full_path, img_array))

    photo_rel_path = os.path.relpath(full_path, start=filepath)

    try:
        tof = get_tof()
    except Exception as e:
        print(f"ToF Error: {e}")
        tof = [None, None]

    data_buffer.append([photo_rel_path, *tof, steering, velocity])
    data_saves += 1  # Foto-Counter hochzählen

#load the model
model = IntegratedNN()
model.load_state_dict(torch.load(model_path))
model.eval()

#setup
now = datetime.now()
time_stamp = now.strftime("%d-%H-%M-%S")
root = os.getcwd()
filepath = os.path.join(root, f"supervised-{time_stamp}")
os.makedirs(filepath, exist_ok=True)

pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    print("Kein Controller gefunden!")
    exit(code=1)
controller = pygame.joystick.Joystick(0)
controller.init()
print(f"Verbunden mit: {controller.get_name()}")

df = pd.DataFrame(columns=['cam_path', 'tof_1', 'tof_2', 'steering_angle', 'velocity'])

pi = setup()
print("start")

while True:
    try:
        pygame.event.pump()
        if controller.get_button(0): #X-Button
            mode = "break"
            print(mode)
        if controller.get_button(1): #O-Button
            mode = "autonomous"
            print(f"now driving {mode}")
        if 55 < controller.get_axis(3)*30+50 < 45:
            mode = "manuel"
            print(f"now driving {mode}")

        if mode == "autonomous":
            tof = list(get_tof())
            img = take_photo_fast()

            image = Image.fromarray(img)
            image = transforms.Resize((128, 128))(image) #might change size
            image = transforms.ToTensor()(image)
            image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)

            tof = torch.tensor([
                tof[0],
                tof[1]
            ], dtype=torch.float32)

            tof_expanded = tof.view(2, 1, 1).expand(2, 128, 128)
            combined_input = torch.cat((image, tof_expanded), dim=0)
            steering = predict(combined_input)
            motor(speed=100)
            servo(steering)
            
            #debugging:
            print(f"predicted angel: {steering}, tof: {tof}")

        elif mode == "manuel":
            pygame.event.pump()
            steering = int((controller.get_axis(3) * 30) + 50)
            motor(speed=100)
            servo(steering)
            if time.time() - last_data_save >= 0.3:
                collect_data(velocity=100, steering=steering)
                last_data_save = time.time()

            if len(data_buffer) > 20:
                print("Speichere CSV-Daten...")
                new_df = pd.DataFrame(data=data_buffer, columns=df.columns)
                data_buffer.clear()
                df = pd.concat([df, new_df], ignore_index=True)
                last_df_save = time.time()

        elif mode == "break":
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        break

motor(0)
servo(50)
cleanup()
pygame.quit()

print("Warte auf die Abarbeitung der restlichen Fotoaufgaben...")
photo_queue.join()
photo_queue.put(None)
worker_thread.join()

if len(data_buffer) > 0:
    new_df = pd.DataFrame(data=data_buffer, columns=df.columns)
    data_buffer.clear()
    df = pd.concat([df, new_df], ignore_index=True)

filename = f'{filepath}/train-data{time_stamp}.csv'
df.to_csv(filename, index=False)
print("Daten gespeichert!")
print(df.head(5))