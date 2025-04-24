import torch #type: ignore
import pandas as pd
import numpy as np
from PIL import Image
from models import IntegratedNN
from torchvision import transforms #type: ignore

import pygame #type: ignore
import threading
import time
import queue
import os
from datetime import datetime

from get_data import get_tof, take_photo_fast
from motor import servo, motor, setup, cleanup

#define the paths
model_path = "8l.pth"

mode = "autonomous"
data_buffer = []
last_data_save = 0
last_df_save = 0
filepath = ""
data_saves = 0
prev_img = np.zeros((128, 128, 3), dtype=np.uint8)

basic_speed = 100
curve_speed = 120

photo_queue = queue.Queue()

def crop_image(image):
    height, width = image.shape[:2]
    cropped_image = image[height-1600:height, 0:width]

    return cropped_image

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
    img_array = take_photo_fast()
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

setup()
input("start")

while True:
    try:
        pygame.event.pump()
        if controller.get_button(0): #X-Button
            mode = "break"
            motor(0)
            print(mode)
        if controller.get_button(1): #O-Button
            mode = "autonomous"
            print(f"now driving {mode}")
        if controller.get_axis(3)*30+50 < 45 or controller.get_axis(3)*30+50 > 55:
            mode = "manuel"

        if mode == "autonomous":
            img = take_photo_fast()
            # Ensure prev_img is initialized with the correct shape on first run or if dimensions change.
            if prev_img.shape != img.shape:
                prev_img = img.copy()
                continue

            diff = np.abs(img.astype(np.int16) - prev_img.astype(np.int16))
            movement = np.mean(diff)
            threshold = 10
            num_changed_pixels = np.sum(diff > threshold)
            if num_changed_pixels < 100: #⚠️ parameter evtl anpassen
                print("keine Bewegung erkannt")
                motor(speed=-100)
                time.sleep(0.1)
                motor(speed=0)
                img = take_photo_fast()
                continue

            prev_img = img.copy()

            tof = list(get_tof())
            image = Image.fromarray(img)
            image = crop_image(image)
            image = transforms.Resize((128, 128))(image) #might change size
            image = transforms.ToTensor()(image)
            image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
            try:
                tof = torch.tensor([
                    tof[0],
                    tof[1]
                ], dtype=torch.float32)

                tof_expanded = tof.view(2, 1, 1).expand(2, 128, 128)
                combined_input = torch.cat((image, tof_expanded), dim=0)
                steering = predict(combined_input)

                servo(steering)
                
                servo(int(steering))
                if steering < 30 or steering > 70:
                    motor(speed=curve_speed)
                else:
                    motor(speed=basic_speed)

                #debugging:
                print(f"predicted angle: {steering}, tof: {tof}")
            except Exception as e:
                print(f"Error: {e}")
                motor(speed=0)
                servo(50)
                continue

        elif mode == "manuel":
            pygame.event.pump()
            steering = int((controller.get_axis(3) * 30) + 50)
            motor(speed=110)
            servo(steering)
            print(f"manual steering: {steering}")
            if time.time() - last_data_save >= 0.3:
                collect_data(velocity=110, steering=steering)
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