#this script handels the input of a controler to control the robot remotly
import pygame
from send_i2c import send_i2c, encode_data
import pandas as pd
import time
from datetime import datetime
from get_data import get_tof, get_gyro, take_photo, cleanup

data_buffer = []
last_data_save = 0
last_df_save = 0

def collect_data(velocity, steering):
    photo = take_photo()
    gyro = get_gyro()
    tof = get_tof()
    data_buffer.append([photo, gyro, *tof, steering, velocity])

def get_input(): #get the controler inputs and convert it into commands for the motors
    pygame.event.pump()
    velocity = joystick.get_axis(1) * 255 #the left stick controles the speed / we multiply by 255 because the controler returns values -1 - 1 and the motor takes values -255 - 255
    steering = (joystick.get_axis(2) * 30) + 50 #the right stick controles the steering / we multiply by 30 and add 50 because controler returns values -1 - 1 and the motor takes values 20 - 80
    return velocity, steering

def process_input(velocity, steering): #send data to the arduino 
    velocity = encode_data("speed", velocity)
    steering = encode_data("steering", steering)
    send_i2c(velocity)
    send_i2c(steering)

def save_data(df, data): # DataFrame als Parameter übergeben
    new_df = pd.DataFrame(data=data, columns=df.columns)
    data.clear()
    return pd.concat([df, new_df], ignore_index=True)

# Initialisierung von pygame und der Joystick-Modul
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("Kein Joystick gefunden!")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()

print("Controller verbunden:", joystick.get_name())

# Dataframe erstellen
df = pd.DataFrame(columns=['cam_path', 'gyro_angle', 'tof_1', 'tof_2', 'steering_angle', 'velocity'])

while True:
    try:
        velocity, steering = get_input()
        process_input(velocity, steering)
        current_time = time.time() * 1000
        if  current_time > last_data_save + 1000: #get the data once every second
            collect_data(velocity, steering)
            last_data_save = current_time
        if current_time > last_df_save + 20000:
            df = save_data(df=df, data=data_buffer)
            last_df_save = current_time
    except KeyboardInterrupt:
        pygame.quit()
        if data_buffer:
            df = save_data(df=df, data=data_buffer)
        print("stopped the recording, saving the data now")
        now = datetime.now() #getting the time to add a time stamp to the file name
        time_stamp = now.strftime("%d-%H-%M-%S")
        filename = f'data_{time_stamp}.csv'
        df.to_csv(filename, index=False) #saving the df as csv file
        print("the data is now saved, exiting the program")
        break
    finally:
        cleanup()
