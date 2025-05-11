import pygame # type: ignore
import time # type: ignore
from motor import *

servo_old = 0
motor_old = 0

# Pygame initialisieren
pygame.init()

# Alle verfügbaren Joysticks auflisten
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("Kein Controller gefunden!")
    exit()

# DualSense Controller auswählen (erster gefundener Controller)
controller = pygame.joystick.Joystick(0)
controller.init()

print(f"Verbunden mit: {controller.get_name()}")
setup()
# Hauptloop zum Auslesen der Eingaben
while True:
    try:
        pygame.event.pump()
        velocity = int(controller.get_axis(1) * 230 * -1) #the left stick controles the speed / we multiply by 255 because the controler returns values -1 <-> 1 and the motor takes values -255 - 255
        steering = int((controller.get_axis(3) * 30) + 50) #the right stick controles the steering / we multiply by 30 and add 50 because controler returns values -1 <-> 1 and the motor takes values 20 - 80
        button = controller.get_button(2)
        print("steering: ", steering )
        print("velocity: ", velocity )
        print("button: ", button)
        time.sleep(0.1)
        #print(type(velocity))
        if velocity != motor_old:
            motor(velocity)
            motor_old = velocity
        if steering != servo_old:
            servo(steering)
            servo_old = steering

    except KeyboardInterrupt:
        break

pygame.quit()
