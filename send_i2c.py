import smbus
import time

I2C_ADDRESS = 0x04 #arduino adresse

bus = smbus.SMBus(1)  #pi adresse

def send_i2c(data: list):
    try:
        bus.write_i2c_block_data(I2C_ADDRESS, 0, data)
        print(bus.read_byte(I2C_ADDRESS))
        if bus.read_byte(I2C_ADDRESS):
            return True
        else:
            return False
    except OSError as e:
            print(f"I2C Fehler: {e}")


def main():
    while True:
        try:
            data=[1, 2, 3, 4] #may be changed in the future
            if send_i2c(data):
                print("it works")
            else:
                print("error while sending data")
            time.sleep(1)
        except KeyboardInterrupt:
            print("\nProgramm beendet.")
            break

if __name__ == "__main__":
    main()