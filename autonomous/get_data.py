import time
import board
import busio
import adafruit_vl53l0x
import pigpio as gpio
import time
from picamera2 import Picamera2
import json
import os
import numpy as np
import smbus
import threading

#define the pins
xshut0 = 0
xshut1 = 0

#set up the i2c and gpio
pi = gpio.pi()
i2c = busio.I2C(board.SCL, board.SDA)

class mpu6050:

    # Global Variables
    GRAVITIY_MS2 = 9.80665
    address = None
    bus = None

    # Scale Modifiers
    ACCEL_SCALE_MODIFIER_2G = 16384.0
    ACCEL_SCALE_MODIFIER_4G = 8192.0
    ACCEL_SCALE_MODIFIER_8G = 4096.0
    ACCEL_SCALE_MODIFIER_16G = 2048.0

    GYRO_SCALE_MODIFIER_250DEG = 131.0
    GYRO_SCALE_MODIFIER_500DEG = 65.5
    GYRO_SCALE_MODIFIER_1000DEG = 32.8
    GYRO_SCALE_MODIFIER_2000DEG = 16.4

    # Pre-defined ranges
    ACCEL_RANGE_2G = 0x00
    ACCEL_RANGE_4G = 0x08
    ACCEL_RANGE_8G = 0x10
    ACCEL_RANGE_16G = 0x18

    GYRO_RANGE_250DEG = 0x00
    GYRO_RANGE_500DEG = 0x08
    GYRO_RANGE_1000DEG = 0x10
    GYRO_RANGE_2000DEG = 0x18

    # MPU-6050 Registers
    PWR_MGMT_1 = 0x6B
    PWR_MGMT_2 = 0x6C

    ACCEL_XOUT0 = 0x3B
    ACCEL_YOUT0 = 0x3D
    ACCEL_ZOUT0 = 0x3F

    TEMP_OUT0 = 0x41

    GYRO_XOUT0 = 0x43
    GYRO_YOUT0 = 0x45
    GYRO_ZOUT0 = 0x47

    ACCEL_CONFIG = 0x1C
    GYRO_CONFIG = 0x1B

    def __init__(self, address, bus=1):
        self.address = address
        self.bus = smbus.SMBus(bus)
        # Wake up the MPU-6050 since it starts in sleep mode
        self.bus.write_byte_data(self.address, self.PWR_MGMT_1, 0x00)

    # I2C communication methods

    def read_i2c_word(self, register):
        # Read the data from the registers
        high = self.bus.read_byte_data(self.address, register)
        low = self.bus.read_byte_data(self.address, register + 1)

        value = (high << 8) + low

        if (value >= 0x8000):
            return -((65535 - value) + 1)
        else:
            return value

    def set_accel_range(self, accel_range):
        # First change it to 0x00 to make sure we write the correct value later
        self.bus.write_byte_data(self.address, self.ACCEL_CONFIG, 0x00)

        # Write the new range to the ACCEL_CONFIG register
        self.bus.write_byte_data(self.address, self.ACCEL_CONFIG, accel_range)

    def read_accel_range(self, raw = False):
        raw_data = self.bus.read_byte_data(self.address, self.ACCEL_CONFIG)

        if raw is True:
            return raw_data
        elif raw is False:
            if raw_data == self.ACCEL_RANGE_2G:
                return 2
            elif raw_data == self.ACCEL_RANGE_4G:
                return 4
            elif raw_data == self.ACCEL_RANGE_8G:
                return 8
            elif raw_data == self.ACCEL_RANGE_16G:
                return 16
            else:
                return -1

    def get_accel_data(self, g = False):
        x = self.read_i2c_word(self.ACCEL_XOUT0)
        y = self.read_i2c_word(self.ACCEL_YOUT0)
        z = self.read_i2c_word(self.ACCEL_ZOUT0)

        accel_scale_modifier = None
        accel_range = self.read_accel_range(True)

        if accel_range == self.ACCEL_RANGE_2G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_2G
        elif accel_range == self.ACCEL_RANGE_4G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_4G
        elif accel_range == self.ACCEL_RANGE_8G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_8G
        elif accel_range == self.ACCEL_RANGE_16G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_16G
        else:
            print("Unknown range-accel_scale_modifier set to self.ACCEL_SCALE_MODIFIER_2G")
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_2G

        x = x / accel_scale_modifier
        y = y / accel_scale_modifier
        z = z / accel_scale_modifier

        if g is True:
            return {'x': x, 'y': y, 'z': z}
        elif g is False:
            x = x * self.GRAVITIY_MS2
            y = y * self.GRAVITIY_MS2
            z = z * self.GRAVITIY_MS2
            return {'x': x, 'y': y, 'z': z}

    def set_gyro_range(self, gyro_range):
        # First change it to 0x00 to make sure we write the correct value later
        self.bus.write_byte_data(self.address, self.GYRO_CONFIG, 0x00)

        # Write the new range to the ACCEL_CONFIG register
        self.bus.write_byte_data(self.address, self.GYRO_CONFIG, gyro_range)

    def read_gyro_range(self, raw = False):
        raw_data = self.bus.read_byte_data(self.address, self.GYRO_CONFIG)

        if raw is True:
            return raw_data
        elif raw is False:
            if raw_data == self.GYRO_RANGE_250DEG:
                return 250
            elif raw_data == self.GYRO_RANGE_500DEG:
                return 500
            elif raw_data == self.GYRO_RANGE_1000DEG:
                return 1000
            elif raw_data == self.GYRO_RANGE_2000DEG:
                return 2000

            else:
                return -1

    def get_gyro_data(self):
        x = self.read_i2c_word(self.GYRO_XOUT0)
        y = self.read_i2c_word(self.GYRO_YOUT0)
        z = self.read_i2c_word(self.GYRO_ZOUT0)

        gyro_scale_modifier = None
        gyro_range = self.read_gyro_range(True)

        if gyro_range == self.GYRO_RANGE_250DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_250DEG
        elif gyro_range == self.GYRO_RANGE_500DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_500DEG
        elif gyro_range == self.GYRO_RANGE_1000DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_1000DEG
        elif gyro_range == self.GYRO_RANGE_2000DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_2000DEG
        else:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_250DEG

        x = x / gyro_scale_modifier
        y = y / gyro_scale_modifier
        z = self.read_i2c_word(self.GYRO_ZOUT0)

        gyro_scale_modifier = None
        gyro_range = self.read_gyro_range(True)

        if gyro_range == self.GYRO_RANGE_250DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_250DEG
        elif gyro_range == self.GYRO_RANGE_500DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_500DEG
        elif gyro_range == self.GYRO_RANGE_1000DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_1000DEG
        elif gyro_range == self.GYRO_RANGE_2000DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_2000DEG
        else:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_250DEG

        x = x / gyro_scale_modifier
        y = y / gyro_scale_modifier
        z = z / gyro_scale_modifier

        return {'x': x, 'y': y, 'z': z}

z_axis = 0
z_offset = 0
last_time = time.time()
mpu = mpu6050(0x68)

def gyro_thread():
    global z_axis, last_time
    while True:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        gyro_data = mpu.get_gyro_data()
        
        z_axis += (gyro_data['z'] - z_offset) * dt
        time.sleep(0.05)

# VL53L0X-Sensoren initialisieren
def initialize_sensors():
    global xshut0, xshut1
    with open("config.json", "r") as file:
        data = json.load(file)
    xshut0 = int(data["sensor"]["xshut0"])
    xshut1 = int(data["sensor"]["xshut1"])
    #check if libary is running
    if not pi.connected:
        print("Fehler: pigpio-Daemon läuft nicht!")
        exit()
    #change the i2c adresse of the sensors so we can acces the seperatly
    pi.write(xshut0, 0)
    pi.write(xshut1, 0)
    time.sleep(0.1)
    pi.write(xshut0, 1)
    time.sleep(0.5)
    sensor1 = adafruit_vl53l0x.VL53L0X(i2c)
    sensor1.set_address(0x30)
    pi.write(xshut1, 1)
    time.sleep(0.5)
    sensor2 = adafruit_vl53l0x.VL53L0X(i2c)
    sensor2.set_address(0x31)
    picam = Picamera2()
    config = picam.create_still_configuration()
    picam.configure(config)
    picam.start()
    time.sleep(8)
    
    offset = []
    for i in range(10):
        gyro_data = mpu.get_gyro_data()
        offset.append(gyro_data['z'])
        time.sleep(0.2)
    z_offset = np.mean(offset)
    last_time = time.time()
    thread = threading.Thread(target=gyro_thread)
    thread.daemon = True  # Damit der Thread im Hintergrund läuft
    thread.start()

    return sensor1, sensor2, picam, z_offset, last_time

def initialize_sensor():
    global xshut0, xshut1
    with open("config.json", "r") as file:
        data = json.load(file)
    xshut0 = int(data["sensor"]["xshut0"])
    xshut1 = int(data["sensor"]["xshut1"])
    xshut2 = int(data["sensor"]["xshut2"])
    #check if libary is running
    if not pi.connected:
        print("Fehler: pigpio-Daemon läuft nicht!")
        exit()
    #change the i2c adresse of the sensors so we can acces the seperatly
    pi.write(xshut0, 0)
    pi.write(xshut1, 0)
    pi.write(xshut2, 0)
    time.sleep(0.1)
    pi.write(xshut0, 1)
    time.sleep(0.5)
    sensor1 = adafruit_vl53l0x.VL53L0X(i2c)
    sensor1.set_address(0x30)
    pi.write(xshut1, 1)
    time.sleep(0.5)
    sensor2 = adafruit_vl53l0x.VL53L0X(i2c)
    sensor2.set_address(0x31)
    time.sleep(0.5)
    sensor3 = adafruit_vl53l0x.VL53L0X(i2c)
    sensor3.set_address(0x32)

    offset = []
    for i in range(10):
        gyro_data = mpu.get_gyro_data()
        offset.append(gyro_data['z'])
        time.sleep(0.2)
    z_offset = np.mean(offset)
    last_time = time.time()
    thread = threading.Thread(target=gyro_thread)
    thread.daemon = True  # Damit der Thread im Hintergrund läuft
    thread.start()

    return sensor1, sensor2, sensor3, z_offset, last_time

#initialize sensors
#tof_sensor1, tof_sensor2, picam, z_offset, last_time = initialize_sensors()
tof_sensor1, tof_sensor2, tof_sensor3, z_offset, last_time = initialize_sensor()

#get the distances
def get_tof():
    sensor_data = [
        tof_sensor1.range, tof_sensor2.range
    ]
    return sensor_data

def get_tof2():
    sensor_data = [
        tof_sensor1.range, tof_sensor2.range, tof_sensor3.range
    ]
    return sensor_data


#take the photo and return the path
def take_photo(filepath: str, index: int):
    photo_path = os.path.join(filepath, f"cam-{index}.jpg")
    picam.capture_file(photo_path)

def take_photo_fast():
    array = picam.capture_array("main")

    return array

#get the gyro data
def get_gyro(value: str):
    global z_axis
    if value == "accel":
        accel = mpu.get_accel_data()
        a = accel['y']
        return a
    elif value == "gyro":
        return z_axis
    elif value == "2":
        accel = mpu.get_accel_data()
        a = accel['y']
        return z_axis, a
    else:
        return False
    

def reset_gyro():
    global z_axis
    z_axis = 0