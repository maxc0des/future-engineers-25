#hindernissrennen
import torch
from PIL import Image
from models import IntegratedNN
from torchvision import transforms
import numpy as np
import time

from get_data import get_tof, take_photo_fast
from motor import servo, motor, setup, cleanup

#define the paths
model_path = "model.pth"
a_model = "model.pth" #model for going clockwise
c_model = "model.pth" #model for going counterclockwise

#defining speed presets
basic_speed = 100
curve_speed = 120

#define other const
threshold = 10

#button stopping
class ButtonPressed(Exception):
    pass

    def check_button():
        if pi.gpio_trigger():
            raise ButtonPressed

#crop image before processing it        
def crop_image(image):
    height, width = image.shape[:2]
    cropped_image = image[height-1600:height, 0:width]

    return cropped_image

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
        if tof[0] > 50:
            direction = "anticlockwise"
            motor(speed=0)
            break
        elif tof[1] > 50:
            direction = "clockwise"
            motor(speed=0)
            break
        else:
            motor(speed=100)
        i += 1
    for step in range(i):
        motor(speed=-100)
    
    motor(speed=0)
    print(f"set direction to {direction}")
    return direction

def reset():
    print("2reset??")

status("setup")

#EXECUTION STARTS HERE
#setup
print("starting setup")
pi = setup()

#load the needed model
model = IntegratedNN()
direction = start_sequence()
if direction == "anticlockwise":
    model.load_state_dict(torch.load(a_model))
elif direction == "clockwise":
    model.load_state_dict(torch.load(c_model))
model.eval()

print("switching to autonomous mode")

input("start")

#main loop
while True:
    try:
        status("running")
        img = take_photo_fast()
        
        #check if img has changed
        if prev_img.shape != img.shape:
            prev_img = img.copy()
            continue
        diff = np.abs(img.astype(np.int16) - prev_img.astype(np.int16))
        movement = np.mean(diff)
        num_changed_pixels = np.sum(diff > threshold)
        if num_changed_pixels < 100: #⚠️ parameter evtl anpassen
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

        try:
            tof = torch.tensor([
                tof[0],
                tof[1]
            ], dtype=torch.float32)

            tof_expanded = tof.view(2, 1, 1).expand(2, 128, 128)
            combined_input = torch.cat((image, tof_expanded), dim=0)
            steering = predict(combined_input)

            servo(steering)
                
            servo(int(steering))
            if steering < 30 or steering > 70:
                motor(speed=curve_speed)
            else:
                motor(speed=basic_speed)

            #debugging:
            print(f"predicted angle: {steering}, tof: {tof}")
        except Exception as e:
            print(f"Error: {e}")
            motor(speed=0)
            servo(50)
            continue
    
    except KeyboardInterrupt:
        break

    except ButtonPressed:
        reset()

    except Exception as e:
        print(f"weewoo {e}")
        time.sleep(0.5)
        continue

status("off")
motor(0)
servo(50)
cleanup()