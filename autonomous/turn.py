from motor import *
from get_data import *

#wait for the start
while True:
    tof = list(get_tof())
    if tof[0] > 150 or tof[1] > 150:    #still need to test how the tof behaves
        break
    else:
        time.sleep(0.4)

#get the data for the algorithm
tof = list(get_tof())
l = tof[0]
r = tof[1]
reset_gyro()
time.sleep(0.1)

#drive to statr spot
remaining_distance = 0.8 #m
t_start = time.time()
while remaining_distance > 5: #assuming a stays near to constant
    angle, a = get_gyro()
    t = (t_start -time-time())/1000
    traveled_distance = 1/2 * a * t**2 #d=1/2*a*t**2
    remaining_distance -= traveled_distance
    print(remaining_distance)