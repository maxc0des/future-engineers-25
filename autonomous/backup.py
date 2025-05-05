#always have a backup!
import numpy as np
import time
import pigpio as gpio #type: ignore


from get_data import *
from motor import servo, motor, setup, cleanup

#defining speed presets
basic_speed = 100
curve_speed = 120

#define other const
turns = 0
threshold = 30
pixel_threshold = 100
base_delay = 1
speed_boost = 40
prev_img = None
clockwise = False

hindernissrennen = False

#button stopping
class ButtonPressed(Exception):
    pass

    def check_button():
        if pi.gpio_trigger():
            raise ButtonPressed

#crop image before processing it        
def crop_image(image):
    width, height = image.size
    start_y = max(0, height - 1600)
    cropped_image = image.crop((0, start_y, width, height))
    return cropped_image

#realign at corner
def turn():
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
    input(tendency)
    
    needed_angle = get_gyro("gyro")
    time.sleep(0.1)
    #drive to start spot
    angle = get_gyro("gyro")
    forward = False #driving direction
    if clockwise:
        needed_angle -= 85
        while angle > needed_angle:
            remaining_angle = (needed_angle - angle)*-1
            print("remaining", remaining_angle)
            #print(remaining_angle)
            if remaining_angle > 30:
                steering = 30
            else:
                steering = remaining_angle
            if forward:
                steering += 50
                if remaining_angle > 0:
                    speed = remaining_angle/(needed_angle+0.1)*speed_boost+basic_speed
            else:
                steering = 50 - steering
                if remaining_angle > 0:
                    speed = (remaining_angle/(needed_angle+0.1)*speed_boost+basic_speed)*-1
            print("steering",steering)
            print(speed)
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
        needed_angle += 85
        print("needed",needed_angle)
        print("currently",angle)
        while angle < needed_angle:
            remaining_angle = needed_angle - angle 
            print(remaining_angle)
            if remaining_angle > 30:
                steering = 30
            else:
                steering = remaining_angle
            if forward:
                steering = 50 - steering
                if remaining_angle > 0:
                    speed = remaining_angle/(needed_angle+0.1)*speed_boost+100
            else:
                steering += 50
                if remaining_angle > 0:
                    speed = 0-remaining_angle/(needed_angle+0.1)*speed_boost+100
            print(steering)
            servo(steering)
            motor(speed)
            if (forward and tendency[0] == "forward") or (not forward and tendency[0] == "backward"):
                time.sleep(base_delay * tendency[1])
            else:
                time.sleep(base_delay)
            motor(0)
            angle = get_gyro("gyro")
            forward = not forward

    servo(50)
    motor(-100)
    time.sleep(2)
    motor(0)
    reset_gyro()

    turns += 1

#display staus of the execution
def status(status: str):
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

def reset():
    print("2reset??")



#EXECUTION STARTS HERE
#setup
print("starting setup")
pi = gpio.pi()
status("setup")
setup()

print("switching to autonomous mode")

input("start")

#main loop
while True:
    try:
        if turns >= 13:
            break

        status("running")

        tof = list(get_tof())

        if tof[0] > 1000 or tof[1] > 1000:
            turn()
        else:
            servo(50)
            motor(basic_speed)
        
    
    except KeyboardInterrupt:
        break

    except ButtonPressed:
        reset()
        continue

    except Exception as e:
        print(f"weewoo {e}")
        status("error")
        time.sleep(0.5)
        continue

status("off")
motor(0)
servo(50)
cleanup()