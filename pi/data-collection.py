import pygame # type: ignore
import time
from motor import *
import pandas as pd
from datetime import datetime
from get_data import get_tof, take_photo
import os
import threading
import queue

data_buffer = []
last_data_save = 0
last_df_save = 0
filepath = ""
data_saves = 0
record = False

# Erstelle eine Queue für Fotoaufgaben
photo_queue = queue.Queue()

def photo_worker():
    while True:
        item = photo_queue.get()
        if item is None:
            break  # Exit-Signal erhalten
        photo_path, data_num = item
        take_photo(photo_path, data_num)
        photo_queue.task_done()

# Starte den Worker-Thread als Daemon, so dass er den Hauptthread nicht blockiert
worker_thread = threading.Thread(target=photo_worker, daemon=True)
worker_thread.start()

def take_photo_async(filepath, data_num):
    # Lege eine Fotoaufgabe in die Queue, der Hauptthread wartet hier nicht
    photo_queue.put((filepath, data_num))

def collect_data(velocity, steering):
    #gyro = get_gyro() #might add a gyro later
    take_photo_async(filepath, data_saves)
    full_path = os.path.join(filepath, f"cam-{data_saves}.jpg")
    photo = os.path.relpath(full_path, start=filepath)
    try:
        tof = get_tof()
    except:
        tof=[None, None]
    data_buffer.append([photo, *tof, steering, velocity])

now = datetime.now() #getting the time to add a time stamp to the file name
time_stamp = now.strftime("%d-%H-%M-%S")
root = os.getcwd()  # Pfad des aktuell laufenden Skripts (also vom USB-Stick)
filepath = os.path.join(root, f"data-{time_stamp}")
os.makedirs(filepath, exist_ok=True)  # Erstellt den Ordner (falls nicht vorhanden)
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
time.sleep(1)
print("recording is ready, press o to start")

while True:
    try:
        pygame.event.pump()
        if controller.get_button(0):
            break
        if controller.get_button(1):
            record = True
            print("recording started")
        if controller.get_button(2):
            record = False
            print("recording paused")
        if record:
            velocity = int(controller.get_axis(1) * 210 * -1) #the left stick controles the speed / we multiply by 255 because the controler returns values -1 <-> 1 and the motor takes values -255 - 255
            steering = int((controller.get_axis(3) * 30) + 50) #the right stick controles the steering / we multiply by 30 and add 50 because controler returns values -1 <-> 1 and the motor takes values 20 - 80
            #print("steering: ", steering )
            #print("velocity: ", velocity )
            if velocity < 10 and velocity > -10: #threashold to conter stick drift
                velocity = 0
            motor(velocity)
            servo(steering)
            if time.time() - last_data_save >= 0.3:
                collect_data(velocity, steering)
                last_data_save = time.time()
                data_saves += 1
            if time.time() - last_df_save >= 20:
                print("Saving data")
                new_df = pd.DataFrame(data=data_buffer, columns=df.columns)
                data_buffer.clear()
                df = pd.concat([df, new_df], ignore_index=True)
                last_df_save = time.time()
    except KeyboardInterrupt:
        
        break
motor(0)
servo(50)
print("Warte auf die Abarbeitung der restlichen Fotoaufgaben...")
photo_queue.join()  # Blockiert, bis alle Tasks finished sind

# Sende Stop-Signal an den Foto-Worker:
photo_queue.put(None)
worker_thread.join()
if len(data_buffer) > 0:
    new_df = pd.DataFrame(data=data_buffer, columns=df.columns)
    data_buffer.clear()
    df = pd.concat([df, new_df], ignore_index=True)
print("stopped the recording, saving the data now")
filename = f'{filepath}/train-data{time_stamp}.csv'
df.to_csv(filename, index=False) #saving the df as csv file
print("the data is now saved, exiting the program")
print(df.head(5))
cleanup()
pygame.quit()