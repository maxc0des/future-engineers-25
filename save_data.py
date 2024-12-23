#this script collects data during during driving the robot and converts it into a pandas dataframe
import pandas as pd
from datetime import date
import time

data_buffer = [] #hier werden die daten zwischengespeichert, bevor sie in den df übertragen werden

def take_photo():
    print("taking a photo")
    #take the photo
    #save it
    #return the path

def get_gyro():
    print("getting the gyro data")
    #get the data
    #return the value

def get_tof():
    print("getting the tof data")
    #get the data
    #return the values (3!!)

def get_steering():
    print("getting the steering angel")
    #get the angel (from the controler input)
    #return the angel

def get_velocity():
    print("getting the velocity")
    #get the speed (from the controler input)

def collect_data():
    photo = take_photo()
    gyro = get_gyro()
    tof = get_tof()
    steering = get_steering()
    velocity = get_velocity()

    data_buffer.append([photo, gyro, *tof, steering, velocity])



if __name__ == "__main__":
    print("started the recording, ^C to stop")
    while True:
        try:
            collect_data()
            time.sleep(1)
        except KeyboardInterrupt:
            print("stopped the recording, saving the data now")
            df = pd.DataFrame(data_buffer, columns=['cam_path', 'gyro_angel', 'tof_1', 'tof_2', 'tof_3', 'steering_angel', 'velocity'])
            df.to_csv('daten.csv'+ str(date.today()), index=False)
            print("the data is now saved, exiting the program")
            break