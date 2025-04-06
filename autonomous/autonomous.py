#eröfnungsrennen
import torch
from PIL import Image
from models import IntegratedNN
from torchvision import transforms

from get_data import get_tof, take_photo_fast
from motor import servo, motor, setup, cleanup

#define the paths
model_path = "model.pth"
a_model = "model.pth"
c_model = "model.pth"

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
        tof = list(get_tof())
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

status("setup")

#setup
pi = setup()
print("starting setup")

#load the needed model
model = IntegratedNN()
direction = start_sequence()
if direction == "anticlockwise":
    model.load_state_dict(torch.load(a_model))
elif direction == "clockwise":
    model.load_state_dict(torch.load(c_model))
model.eval()

print("switching to autonomous mode")

#main loop
while True:
    try:
        status("running")
        tof = list(get_tof())
        img = take_photo_fast()

        image = Image.fromarray(img)
        image = transforms.Resize((128, 128))(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)

        tof = torch.tensor([
            tof[0],
            tof[1]
        ], dtype=torch.float32)

        tof_expanded = tof.view(2, 1, 1).expand(2, 128, 128)
        combined_input = torch.cat((image, tof_expanded), dim=0)
        steering = predict(combined_input)
        motor(speed=100)
        servo(int(steering))
        
        #debugging:
        print(f"predicted angel: {steering}, tof: {tof}")
    
    except KeyboardInterrupt:
        break

motor(0)
servo(50)
cleanup()