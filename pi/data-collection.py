import pygame # type: ignore
import time
from motor import *
import pandas as pd
from datetime import datetime
from get_data import get_tof, take_photo
import os

data_buffer = []
last_data_save = 0
last_df_save = 0
filepath = ""
data_saves = 0
record = False

def get_input(): #get the controler inputs and convert it into commands for the motors
    print("")

def collect_data(velocity, steering):
    #gyro = get_gyro() #might add a gyro later
    photo = take_photo(filepath, data_saves)
    tof = get_tof()
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
            record = not record
            print("recording: ", record)
        if record:
            velocity = int(controller.get_axis(1) * 230 * -1) #the left stick controles the speed / we multiply by 255 because the controler returns values -1 <-> 1 and the motor takes values -255 - 255
            steering = int((controller.get_axis(3) * 30) + 50) #the right stick controles the steering / we multiply by 30 and add 50 because controler returns values -1 <-> 1 and the motor takes values 20 - 80
            print("steering: ", steering )
            print("velocity: ", velocity )
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