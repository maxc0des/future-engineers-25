#full program for driving autonomous, make sure to set the flag correctly
import torch
from PIL import Image
from models import IntegratedNN
from torchvision import transforms
import numpy as np
import time
import pigpio as gpio #type: ignore
import cv2

from get_data import *
from motor import servo, motor, setup, cleanup

#define the paths
model_path = "v1.pth"
counterclock_model = ".pth" #model for going clockwise
clock_model = "v1.pth" #model for going counterclockwise
redclock_model = "v1.pth" #model for going clockwise on red
redcounter_model = "v1.pth" #model for going counterclockwise on red
greenclock_model = "v1.pth" #model for going clockwise on green
greencounter_model = "v1.pth" #model for going counterclockwise on green

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

    #also determine next color
    img = take_photo_fast()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    lower_green = np.array([35,  40,  40])   # H, S, V
    upper_green = np.array([85, 255, 255])

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    if np.count_nonzero(red_mask) > np.count_nonzero(green_mask):
        next_color="red"
    else:
        next_color="green"
    
    turns += 1
    return next_color

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

#predict steering angle
def predict(combined_input):
    input = combined_input.unsqueeze(0)
    with torch.no_grad():
        prediction = model(input)
    predicted_steering = prediction.squeeze().tolist()

    return predicted_steering

#determine the direction
def start_sequence():
    i = 0
    servo(50)
    while True:
        try:
            tof = list(get_tof())
        except OSError:
            tof = [0, 0]
        if tof[0] > tof[1]:
            clockwise = True
            motor(speed=0)
            break
        elif tof[1] > tof[0]:
            clockwise
            motor(speed=0)
            break
        else:
            motor(speed=100)
        i += 1
    for step in range(i):
        motor(speed=-100)
    
    motor(speed=0)
    print(f"set direction to {clockwise}")
    return clockwise

def reset():
    print("2reset??")
    motor(0)
    servo(50)
    while True:
        try:
            time.sleep(1) 
        except ButtonPressed:
            break

    reset_gyro()




#EXECUTION STARTS HERE
#setup
print("starting setup")
pi = gpio.pi()
status("setup")
setup()

#load the needed model
model = IntegratedNN()
clockwise = start_sequence() #maybe turn
if not clockwise:
    model.load_state_dict(torch.load(counterclock_model))
elif clockwise:
    model.load_state_dict(torch.load(clock_model))
model.eval()

print("switching to autonomous mode")

input("start")

#main loop
while True:
    try:
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
            
        z = get_gyro("gyro")

        if z < -90 or z > 90:
            turns += 1
            reset_gyro()

        if turns >= 13:
            break

        status("running")
        img = take_photo_fast()
        
        #check if img has changed
        print("checking pic")
        if prev_img is None or prev_img.shape != img.shape:
            prev_img = img.copy()
            continue
        diff = np.abs(img.astype(np.int16) - prev_img.astype(np.int16))
        movement = np.mean(diff)
        num_changed_pixels = np.sum(diff > threshold)
        #print(num_changed_pixels)
        if num_changed_pixels < pixel_threshold: #⚠️ parameter evtl anpassen
            status("error")
            print("keine Bewegung erkannt")
            motor(speed=-100)
            time.sleep(0.1)
            motor(speed=0)
            img = take_photo_fast()
            continue
        prev_img = img.copy()

        tof = list(get_tof())
        image = Image.fromarray(img)
        image = crop_image(image)
        image = transforms.Resize((128, 128))(image) #might change size
        image = transforms.ToTensor()(image)
        image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)

        
        tof = torch.tensor([
            tof[0],
            tof[1]
        ], dtype=torch.float32)

        tof_expanded = tof.view(2, 1, 1).expand(2, 128, 128)
        combined_input = torch.cat((image, tof_expanded), dim=0)
        steering = predict(combined_input)
        
        if steering < 20 or steering > 80:
            raise ValueError("steering out of bounds")
        else:
            servo(int(steering))

        speed = abs(basic_speed + (abs(50-steering)))
        motor(speed)

        #debugging:
        print(f"predicted angle: {steering}, tof: {tof}")
        
    
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