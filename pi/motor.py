import pigpio
import json

# Globale Variablen
servo_pin = 0
motor_speed = 0
motor1 = 0
motor2 = 0
pi = None

def load_var():
    global servo_pin, motor_speed, motor1, motor2
    with open("config.json", "r") as file:
        data = json.load(file)

    servo_pin = data["servo"]
    motor_speed = data["motor-speed"]
    motor1 = data["motor1"]
    motor2 = data["motor2"]

def servo(angle: int):
    if pi is None:
        print("Fehler: `pi` wurde nicht initialisiert. Rufe `setup()` zuerst auf!")
        return
    
    pulse_width = 500 + (angle / 180.0) * 2000
    pi.set_servo_pulsewidth(servo_pin, pulse_width)

def motor(speed: int):
    if speed < 0: #rückwärts
        pi.write(motor1, 0)
        pi.write(motor2, 1)
        speed = speed * -1
    elif speed == 0: #null
        pi.write(motor1, 0)
        pi.write(motor2, 0)
    elif speed == 1: #vorwärts
        pi.write(motor1, 1)
        pi.write(motor2, 0)


    pi.set_PWM_dutycycle(motor_speed, speed)

    

def setup():
    global pi
    load_var()
    pi = pigpio.pi()

def cleanup():
    pi.stop()