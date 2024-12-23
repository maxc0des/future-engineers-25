import pygame
from send_i2c import send_i2c
from send_i2c import encode_data

# Initialisierung von pygame und der Joystick-Modul
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("Kein Joystick gefunden!")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()

print("Controller verbunden:", joystick.get_name())

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

while True:
    try:
        velocity, steering = get_input()
        process_input(velocity, steering)
    except KeyboardInterrupt:
        print("Programm beendet.")
        break