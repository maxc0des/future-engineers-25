import time
import board
import busio
import adafruit_vl53l0x
import pigpio as gpio

#define the pins
xshut0 = 17
xshut1 = 27

#set up the i2c and gpio
pi = gpio.pi()
i2c = busio.I2C(board.SCL, board.SDA)

# VL53L0X-Sensoren initialisieren
def initialize_sensors():
    #check if libary is running
    if not pi.connected:
        print("Fehler: pigpio-Daemon läuft nicht!")
        exit()
    #change the i2c adresse of the sensors so we can acces the seperatly
    pi.write(xshut0, 0)
    pi.write(xshut1, 0)
    time.sleep(0.1)
    pi.write(xshut0, 1)
    time.sleep(0.1)
    sensor1 = adafruit_vl53l0x.VL53L0X(i2c)
    sensor1.set_address(0x30)
    pi.write(xshut1, 1)
    time.sleep(0.1)
    sensor2 = adafruit_vl53l0x.VL53L0X(i2c)
    sensor2.set_address(0x31)

    return sensor1, sensor2

#initialize sensors
tof_sensor1, tof_sensor2 = initialize_sensors()

#get the distances
def get_tof():
    print("Getting the ToF data")
    sensor_data = {
        tof_sensor1.range, tof_sensor2.range
    }
    return sensor_data

#take the photo and return the path
def take_photo():
    print("Taking a photo")
    # Simuliere das Aufnehmen eines Fotos
    return "/path/to/photo.jpg"

#get the gyro data
def get_gyro():
    print("Getting the gyro data")
    # Simuliere Gyro-Daten
    return "gyro data"
