#full program for driving autonomous, make sure to set the flag correctly
import torch #type: ignore
from PIL import Image
from models import IntegratedNN
from torchvision import transforms #type: ignore
import numpy as np
import time
import pigpio as gpio #type: ignore
import cv2
import threading

from get_data import *
from motor import servo, motor, setup, cleanup

#define the paths
model_path = "v1.pth"
counterclock_model = "counterclockwise/counter_final.pth" #model for going clockwise
clock_model = "clockwise/clock_finalfinal.pth" #model for going counterclockwise
redclock_model = "clockwise/clock2.pth" #model for going clockwise on red
redcounter_model = "v1.pth" #model for going counterclockwise on red
greenclock_model = "clockwise/clock2.pth" #model for going clockwise on green
greencounter_model = "v1.pth" #model for going counterclockwise on green

#thread for button interrupts
button_pressed_event = threading.Event()

#defining speed presets
basic_speed = 80
speed_boost = 70

#define other const
debounce_time_us = 20000
turns = 0
threshold = 20
pixel_threshold = 80
base_delay = 1

BUTTON_PIN = 12

prev_img = None
clockwise = False
current_status = "off"

hindernissrennen = False

#crop image before processing it        
def crop_image(image):
    width, height = image.size
    start_y = max(0, height - 1600)
    cropped_image = image.crop((0, start_y, width, height))
    #save the cropped image for debugging
    #cropped_image.save("debugging.jpg")
    return cropped_image

#callback func for button
def button_callback(gpio, level, tick):
    if level == 0:
        print("Taster wurde gedrückt!")
        button_pressed_event.set()
    

#realign at corner
def turn():
    print("turning")
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
    input(tendency)
    
    needed_angle = get_gyro("gyro")
    time.sleep(0.1)
    #drive to start spot
    angle = get_gyro("gyro")
    forward = False #driving direction
    if clockwise:
        needed_angle -= 85
        while angle > needed_angle:
            remaining_angle = abs(needed_angle - angle)  # immer den Absolutwert
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
        needed_angle += 85
        print("needed",needed_angle)
        print("currently",angle)
        while angle < needed_angle:
            remaining_angle = abs(needed_angle - angle)
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
    global current_status
    if status == "running":
        pi.write(22, 1)
        pi.write(27, 0)
        pi.write(17, 0)
    elif status == "setup":
        pi.write(22, 0)
        pi.write(17, 1)
        pi.write(27, 0)
    elif status == "error":
        pi.write(22, 0)
        pi.write(17, 0)
        pi.write(27, 1)
    else:
        pi.write(22, 0)
        pi.write(27, 0)
        pi.write(17, 0)

    current_status = status

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
            print("setup tof", tof)
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
    motor(speed=-100)
    time.sleep(2)
    motor(speed=0)
    
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
model = IntegratedNN()
clockwise = start_sequence() #maybe turn
if not clockwise:
    model.load_state_dict(torch.load(counterclock_model))
elif clockwise:
    model.load_state_dict(torch.load(clock_model))
model.eval()

print("switching to autonomous mode")

#main loop
while True:
    try:
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
            
        z = get_gyro("gyro")

        if z < -90 or z > 90:
            turns += 1
            reset_gyro()

        if turns >= 13:
            break

        status("running")
        img = take_photo_fast()
        
        #check if img has changed
        #print("checking pic")
        if prev_img is None or prev_img.shape != img.shape:
            prev_img = img.copy()
        else:
            #drive backward if no movement is detected
            diff = np.abs(img.astype(np.int16) - prev_img.astype(np.int16))
            movement = np.mean(diff)
            num_changed_pixels = np.sum(diff > threshold)
            #print(num_changed_pixels)
            if num_changed_pixels < pixel_threshold: #⚠️ parameter evtl anpassen
                status("error")
                tof = get_tof()
                if tof[0]>tof[1] or tof[1] > 8000:
                    print
                    #wir stoßen rechts an
                    servo(65)
                elif tof[1]>tof[0] or tof[0] > 8000:
                    #wir stoßen links an
                    servo(35)
                
                print("keine Bewegung erkannt")
                motor(speed=basic_speed*-2)
                time.sleep(0.45)
                motor(speed=0)
                servo(50)
                motor(speed=basic_speed*-2)
                time.sleep(0.3)
                motor(speed=0)
                img = take_photo_fast()
            prev_img = img.copy()

        gyro = get_gyro("gyro")
        tof = list(get_tof())
        image = Image.fromarray(img)
        image = crop_image(image)
        image = transforms.Resize((128, 128))(image) #might change size
        image = transforms.ToTensor()(image)
        image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)

        gyro = torch.tensor([
            gyro
        ], dtype=torch.float32)

        tof = torch.tensor([
            tof[0],
            tof[1]
        ], dtype=torch.float32)

        gyro_expanded = gyro.view(1, 1, 1).expand(1, 128, 128)
        tof_expanded = tof.view(2, 1, 1).expand(2, 128, 128)
        combined_input = torch.cat((image, tof_expanded, gyro_expanded), dim=0)
        #combined_input = torch.cat((image, tof_expanded), dim=0)
        steering = predict(combined_input)
        
        if steering < 20 or steering > 80:
            raise ValueError("steering out of bounds")
        else:
            #if steering > 60:
             #   turn()
            #else:
            servo(int(steering))

        # Beispiel: Je größer der Unterschied von 50, desto höher der Speed-Zusatz.
        divisor = 30.0  # Mit diesem Divisor bestimmst du, wie stark der Speed ansteigt
        additional_speed = (abs(steering - 50) / divisor) * speed_boost
        speed = basic_speed + additional_speed

        motor(speed)

        #debugging:
        print(f"predicted angle: {steering}, speed: {speed}, tof: {tof}")
        time.sleep(0.1)
        
    
    except KeyboardInterrupt:
        break

    except Exception as e:
        print(f"weewoo {e}")
        status("error")
        #time.sleep(0.5)
        continue

status("off")
motor(0)
servo(50)
cleanup()