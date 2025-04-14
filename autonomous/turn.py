from motor import *
from get_data import *

i = 1.5
clockwise = True
setup()
#wait for the start
while True:
    tof = list(get_tof())
    if tof[0] > 1500 or tof[1] > 1500:    #still need to test how the tof behaves
        break
    else:
        time.sleep(0.4)

#get the data for the algorithm
tof = list(get_tof())
r = tof[1]
l = tof[0]

if clockwise:
    tendency = int(l)/10-50+1
else:
    tendency = int(r)/10-50+1
print(l)
print(r)
print(tendency)
exit()

reset_gyro()
time.sleep(0.1)
input("§")
#drive to start spot
angle = get_gyro("gyro")
forward = False #driving direction
if clockwise:
    while angle > -90:
        remaining_angle = angle + 90
        #print(remaining_angle)
        if remaining_angle > 30:
            steering = 30
        else:
            steering = angle-(-90)
        if forward:
            steering += 50
            speed = remaining_angle/90*100+100
        else:
            steering = 50 - steering
            speed = (remaining_angle/90*100+100)*-1
        print(steering)
        print(speed)
        servo(steering)
        motor(speed)
        time.sleep(i)
        motor(0)
        angle = get_gyro("gyro")
        forward = not forward
        #input("next")

else:
    while angle < 90:
        remaining_angle = 90 - angle 
        print(remaining_angle)
        if remaining_angle > 30:
            steering = 30
        else:
            steering = angle-90
        if forward:
            steering = 50 - steering
            speed = remaining_angle/90*100+100
        else:
            steering += 50
            speed = 0-remaining_angle/90*100+100
        print(steering)
        servo(steering)
        motor(speed)
        time.sleep(i)
        motor(0)
        angle = get_gyro("gyro")
        forward = not forward
        #input("next")