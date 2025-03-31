import pygame # type: ignore
import time
from motor import *
import pandas as pd
from datetime import datetime
from get_data import get_tof, take_photo_fast
import os
import threading
import queue
from PIL import Image

data_buffer = []
last_data_save = 0
last_df_save = 0
filepath = ""
data_saves = 0
record = False

photo_queue = queue.Queue()

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
<<<<<<< HEAD
    
    full_path = os.path.join(filepath, f"cam-{data_saves}.jpg")
    img_array = take_photo_fast(filepath=full_path, index=data_saves)
    photo_queue.put((full_path, img_array))

    photo_rel_path = os.path.relpath(full_path, start=filepath)

    try:
        tof = get_tof()
    except Exception as e:
        print(f"ToF Error: {e}")
        tof = [None, None]
=======
    current_save = data_saves  # lokalen Zähler speichern
    take_photo_async(filepath, current_save)
    full_path = os.path.join(filepath, f"cam-{current_save}.jpg")
    photo = os.path.relpath(full_path, start=filepath)
    try:
        tof = get_tof()
    except:
        tof = [None, None]
    data_buffer.append([photo, *tof, steering, velocity])
>>>>>>> 33aa668a80be1793db68ce74ac2ca0374e5dcc72

    data_buffer.append([photo_rel_path, *tof, steering, velocity])
    data_saves += 1  # Foto-Counter hochzählen

now = datetime.now()
time_stamp = now.strftime("%d-%H-%M-%S")
root = os.getcwd()
filepath = os.path.join(root, f"data-{time_stamp}")
os.makedirs(filepath, exist_ok=True)

pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    print("Kein Controller gefunden!")
    exit(1)
controller = pygame.joystick.Joystick(0)
controller.init()
print(f"Verbunden mit: {controller.get_name()}")
setup()

df = pd.DataFrame(columns=['cam_path', 'tof_1', 'tof_2', 'steering_angle', 'velocity'])

print("Recording bereit, drücke 'O' zum Starten.")

while True:
    try:
        pygame.event.pump()
        if controller.get_button(0):
            break
        if controller.get_button(1):
            record = True
            print("Recording gestartet")
        if controller.get_button(2):
            record = False
            print("Recording pausiert")
        if record:
            velocity = int(controller.get_axis(1) * 210 * -1)
            steering = int((controller.get_axis(3) * 30) + 50)

            if -10 < velocity < 10:
                velocity = 0  # Stick-Drift fixen

            motor(velocity)
            servo(steering)

            if time.time() - last_data_save >= 0.3:
                collect_data(velocity, steering)
                last_data_save = time.time()

            if time.time() - last_df_save >= 20:
                print("Speichere CSV-Daten...")
                new_df = pd.DataFrame(data=data_buffer, columns=df.columns)
                data_buffer.clear()
                df = pd.concat([df, new_df], ignore_index=True)
                last_df_save = time.time()

    except KeyboardInterrupt:
        break

motor(0)
servo(50)
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

cleanup()
pygame.quit()