import pygame
import time
from motor import *

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
        velocity = int(controller.get_axis(1) * 255 * -1) #the left stick controles the speed / we multiply by 255 because the controler returns values -1 <-> 1 and the motor takes values -255 - 255
        steering = int((controller.get_axis(3) * 30) + 50) #the right stick controles the steering / we multiply by 30 and add 50 because controler returns values -1 <-> 1 and the motor takes values 20 - 80
        print("steering: ", steering )
        print("velocity: ", velocity )
        time.sleep(0.1)
        print(type(velocity))
        motor(velocity)

    except KeyboardInterrupt:
        break

pygame.quit()