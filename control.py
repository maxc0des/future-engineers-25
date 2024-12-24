#this script handels the input of a controler to control the robot remotly
import pygame
from send_i2c import send_i2c, encode_data
import pandas as pd
from datetime import date
import time
from get_data import take_photo, get_gyro, get_tof

data_buffer = []
last_saved = 0

def collect_data():
    photo = take_photo()
    gyro = get_gyro()
    tof = get_tof()
    velocity, steering = get_input()
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

# Initialisierung von pygame und der Joystick-Modul
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("Kein Joystick gefunden!")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()

print("Controller verbunden:", joystick.get_name())

while True:
    try:
        velocity, steering = get_input()
        process_input(velocity, steering)
        if time.time() * 1000 > last_saved + 1000:
            collect_data()
            last_saved = time.time()* 1000
    except KeyboardInterrupt:
        pygame.quit()
        print("stopped the recording, saving the data now")
        df = pd.DataFrame(data_buffer, columns=['cam_path', 'gyro_angle', 'tof_1', 'tof_2', 'tof_3', 'steering_angle', 'velocity'])
        filename = f'data_{date.today()}.csv'
        df.to_csv(filename, index=False)
        print("the data is now saved, exiting the program")
        break