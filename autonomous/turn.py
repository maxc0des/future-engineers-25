from motor import *
from get_data import *
import cv2
import numpy as np

i = 1.2
b = 60
clockwise = True
setup()
#wait for the start
while True:
    tof = list(get_tof())
    if tof[0] > 1500 or tof[1] > 1500:    #still need to test how the tof behaves
        break
    else:
        time.sleep(0.4)

servo(50)
motor(160)
time.sleep(1.5)
motor(0)

#get the data for the algorithm
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
                speed = remaining_angle/remaining_angle*b+100
        else:
            steering = 50 - steering
            if remaining_angle > 0:
                speed = (remaining_angle/remaining_angle*b+100)*-1
        print("steering",steering)
        print(speed)
        servo(steering)
        motor(speed)
        if (forward and tendency[0] == "forward") or (not forward and tendency[0] == "backward"):
            time.sleep(i * tendency[1])
        else:
            time.sleep(i)
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
                speed = remaining_angle/remaining_angle*b+100
        else:
            steering += 50
            if remaining_angle > 0:
                speed = 0-remaining_angle/remaining_angle*b+100
        print(steering)
        servo(steering)
        motor(speed)
        if (forward and tendency[0] == "forward") or (not forward and tendency[0] == "backward"):
            time.sleep(i * tendency[1])
        else:
            time.sleep(i)
        motor(0)
        angle = get_gyro("gyro")
        forward = not forward

servo(50)
motor(-100)
time.sleep(2)
motor(0)
reset_gyro()

#determine next color
img = take_photo_fast()
#img = cv.imread(img)
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
    next="red"
else:
    next="green"
print(next)

#debugging out:
red_output = cv2.bitwise_and(img, img, mask=red_mask)
green_output = cv2.bitwise_and(img, img, mask=green_mask)

cv2.imwrite("red_masked.png", red_output)
cv2.imwrite("green_masked.png", green_output)