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