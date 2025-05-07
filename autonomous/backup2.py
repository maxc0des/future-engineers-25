#backup:
import numpy as np
import time
import pigpio as gpio #type: ignore
import threading

from get_data import *
from motor import servo, motor, setup, cleanup

#thread for button interrupts
button_pressed_event = threading.Event()

#defining speed presets
basic_speed = 90
speed_boost = 60

#define other const
debounce_time_us = 20000
turns = 1
threshold = 20
pixel_threshold = 80
base_delay = 0.5

t = 0.8

BUTTON_PIN = 12

counter_degrees = 85
clock_degrees = 108

prev_img = None
clockwise = False
current_status = "off"

hindernissrennen = False

#callback func for button
def button_callback(gpio, level, tick):
    if level == 0:
        print("button pressed!")
        button_pressed_event.set()
    

#realign at corner
def turn():
    global turns, clockwise, counter_degrees, basic_speed, t
    print("turning")
    status("running")
    motor(0)
    tof = list(get_tof())
    r = tof[1]
    l = tof[0]
    tendency = ["none", 0]
    if clockwise:
        tendency[1] = ((int(l)/10)-50)/50
        if tendency[1] < 0:
            tendency[1]=tendency[1]*-1
            tendency[0]="forward"
        else:
            tendency[0]="backward"
        print(tendency[1])
        tendency[1]=(tendency[1]*3)+1
    else:
        tendency[1] = ((int(r)/10)-50)/50
        if tendency[1] < 0:
            tendency[1]=tendency[1]*-1
            tendency[0]="forward"
        else:
            tendency[0]="backward"
        print(tendency[1])
        tendency[1]=(tendency[1]*3)+1
    print(l)
    print(r)

    #debugging:
    #input(tendency)
    
    #needed_angle = get_gyro("gyro")
    time.sleep(0.1)
    #drive to start spot
    angle = get_gyro("gyro")
    forward = False #driving direction
    print("current angle",angle)
    if clockwise:
        needed_angle = (turns * clock_degrees)*-1
        print("needed angle",needed_angle)
        #needed_angle = -(turns * 70)
        while angle > needed_angle:
            remaining_angle = abs(needed_angle - angle)  # immer den Absolutwert
            if remaining_angle < 5:
                break
            print("remaining", remaining_angle)
            if remaining_angle > 30:
                steering = 30
            else:
                steering = remaining_angle
            if forward:
                steering += 50
                if remaining_angle > 0:
                    # Berechne den Speed mit Hilfe des Absolutwertes und setze den Offset
                    speed = (abs(remaining_angle) / abs(needed_angle + 0.1)) * speed_boost + basic_speed
            else:
                steering = 50 - steering
                if remaining_angle > 0:
                    # Rückwärtsbewegung: Speed negativ
                    speed = -((abs(remaining_angle) / abs(needed_angle + 0.1)) * speed_boost + basic_speed)
            print("steering", steering)
            print("speed:", speed)
            servo(steering)
            motor(speed)
            if (forward and tendency[0] == "forward") or (not forward and tendency[0] == "backward"):
                time.sleep(base_delay * tendency[1])
            else:
                time.sleep(base_delay)
            motor(0)
            angle = get_gyro("gyro")
            forward = not forward
            time.sleep(1)

    else:
        needed_angle = turns * counter_degrees
        print("needed angle",needed_angle)
        print("needed",needed_angle)
        print("currently",angle)
        while angle < needed_angle:
            remaining_angle = abs(needed_angle - angle)
            if remaining_angle < 5:
                break
            print("remaining", remaining_angle)
            if remaining_angle > 30:
                steering = 30
            else:
                steering = remaining_angle
            if forward:
                # Im Forwardfall: Steuerung um 50 reduzieren
                steering = 50 - steering
                if remaining_angle > 0:
                    speed = (abs(remaining_angle) / abs(needed_angle + 0.1)) * speed_boost + 100
            else:
                steering += 50
                if remaining_angle > 0:
                    speed = -((abs(remaining_angle) / abs(needed_angle + 0.1)) * speed_boost + 100)
            print("steering", steering)
            servo(steering)
            motor(speed)
            if (forward and tendency[0] == "forward") or (not forward and tendency[0] == "backward"):
                time.sleep(base_delay * tendency[1])
            else:
                time.sleep(base_delay)
            motor(0)
            angle = get_gyro("gyro")
            forward = not forward
    
    t += 0.1
    counter_degrees += 2
    servo(50)
    motor(100)
    time.sleep(2)
    motor(0)
    status("off")
    turns += 1


#display staus of the execution
def status(status: str):
    global current_status
    if status == "running":
        pi.write(22, 1)
        pi.write(27, 0)
        pi.write(17, 0)
    elif status == "setup":
        pi.write(22, 0)
        pi.write(27, 1)
        pi.write(17, 0)
    elif status == "error":
        pi.write(22, 0)
        pi.write(27, 0)
        pi.write(17, 1)
    else:
        pi.write(22, 0)
        pi.write(27, 0)
        pi.write(17, 0)

    current_status = status

#determine the direction
def start_sequence():
    i = 0
    servo(50)
    while True:
        try:
            tof = list(get_tof())
        except OSError:
            tof = [0, 0]
        if tof[0] > 800:
            clockwise = False
            motor(speed=0)
            break
        elif tof[1] > 800:
            clockwise = True
            motor(speed=0)
            break
        else:
            motor(speed=basic_speed)
    #motor(speed=-100)
    #time.sleep(2)
    #motor(speed=0)
    
    motor(speed=0)
    print(f"set direction to {clockwise}")
    return clockwise

def reset():
    status("error")
    print("weewoo reset")
    motor(0)
    servo(50)
    while True:
        if button_pressed_event.is_set():
            button_pressed_event.clear()
            break
        else:
            time.sleep(0.5)

    reset_gyro()

#EXECUTION STARTS HERE
#setup
print("starting setup")
pi = gpio.pi()
if pi.connected == 0:
    print("Error: PiGPIO not connected")
    exit(1)
status("setup")
pi.set_mode(BUTTON_PIN, gpio.INPUT)
pi.set_pull_up_down(BUTTON_PIN, gpio.PUD_UP)
pi.set_glitch_filter(BUTTON_PIN, debounce_time_us)
pi.callback(BUTTON_PIN, gpio.FALLING_EDGE, button_callback)
setup()

#wait for button press
while True:
    if button_pressed_event.is_set():
        button_pressed_event.clear() # Zurücksetzen des Events
        break
    else:
        if current_status == "setup":
            status("off")
        else:
            status("setup")
        time.sleep(0.5)

status("setup")
print("getting direction")
#load the needed model
clockwise = start_sequence() #maybe turn

print("switching to autonomous mode")

#main loop
while True:
    try:
        if turns > 12:
            motor(basic_speed)
            time.sleep(3)
            motor(0)
        #gyro = get_gyro("gyro")
        tof = get_tof()
        #print("tof0", tof[0], "tof1", tof[1])

        if tof[0] > 1000 or tof[1] > 1000:
            motor(basic_speed)
            print("turning")
            time.sleep(t)
            motor(0)
            turn()
        else:
            servo(50)
            motor(basic_speed)

        if button_pressed_event.is_set():
            print("Button Event erkannt, führe Reset aus.")
            button_pressed_event.clear() # Zurücksetzen des Events
            reset()
            continue

        if hindernissrennen:
            tof = list(get_tof())

            if tof[0] > 1500 or tof[1] > 1500:
                next_color = turn()
                if next_color == "red":
                    if clockwise:
                        model.load_state_dict(torch.load(redclock_model))
                    else:
                        model.load_state_dict(torch.load(redcounter_model))
                else:
                    if clockwise:
                        model.load_state_dict(torch.load(greenclock_model))
                    else:
                        model.load_state_dict(torch.load(greencounter_model))

            else:
                 continue

        status("running")
        img = take_photo_fast()
        
        #check if img has changed
        #print("checking pic")
        if prev_img is None or prev_img.shape != img.shape:
            prev_img = img.copy()
        else:
            diff = np.abs(img.astype(np.int16) - prev_img.astype(np.int16))
            movement = np.mean(diff)
            num_changed_pixels = np.sum(diff > threshold)
            #print(num_changed_pixels)
            if num_changed_pixels < pixel_threshold: #⚠️ parameter evtl anpassen
                status("error")
                print("keine Bewegung erkannt")
                z = get_gyro("gyro")
                if clockwise:
                    supposed_z = turns*counter_degrees
                else:
                    supposed_z = turns*clock_degrees
                steering = z - supposed_z
                if z-supposed_z < 0:
                    steering = 50 - steering
                else:
                    steering = 50 + steering
                if steering < 20:
                    steering = 20
                if steering > 80:
                    steering = 80
                #print("steering", steering)
                tof = get_tof()
                if tof[0]>tof[1] or tof[1] > 8000:
                    print
                    #wir stoßen rechts an
                    servo(65)
                elif tof[1]>tof[0] or tof[0] > 8000:
                    #wir stoßen links an
                    servo(35)
                
                motor(speed=basic_speed*-2)
                time.sleep(1)
                motor(speed=0)
                servo(50)
                motor(speed=basic_speed)
                time.sleep(1)
                motor(speed=0)
                img = take_photo_fast()

            prev_img = img.copy()
        
    
    except KeyboardInterrupt:
        break

    except Exception as e:
        print(f"weewoo {e}")
        status("error")
        time.sleep(0.5)
        continue

status("off")
motor(0)
servo(50)
cleanup()