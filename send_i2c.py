import smbus
import time

I2C_ADDRESS = 0x04 #arduino adresse

bus = smbus.SMBus(1)  #pi adresse

def send_i2c(data: list):
    try:
        for byte in data:
            bus.write_byte(I2C_ADDRESS, byte)
        response = bus.read_byte(I2C_ADDRESS)   #Der Arduino gibt den Status-Code 0 bei fehlgeschlagener und 1 bei erfolgreicher Übermittlung zurück
        print("Received response:", response)
        if not response:
            print(f"Error: Arduino returned 0, failed sending {data}")
            return False
        #print(f"sent: {data}") #uncomment for debugging
        return True
    except OSError as e:
        print(f"I2C Error: {e}")
        return False

def encode_data(target: str, value: int):
    if target == 'speed':
        if value < 0:
            value = value * -1
            value = str(value).zfill(3)
            out = "3" + str(value)
        else:
            value = str(value).zfill(3)
            out = "2" + str(value)
    elif target == 'steering':
        value = str(value).zfill(3)
        out = "1" + str(value)
    return out